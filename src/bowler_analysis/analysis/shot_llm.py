"""Batter shot analysis via a vision LLM.

Sample a handful of frames spanning one shot (backlift -> contact -> follow-through),
send them to the configured Claude backend, and return a structured coaching
assessment. No ball tracking or calibration needed — the model reasons over the
frames directly, which is what makes this work zero-shot on clips that the
geometry pipeline can't handle.
"""

from __future__ import annotations

import base64
import subprocess
from pathlib import Path

import cv2
import numpy as np

from ..config import Config
from ..ingest.video_reader import probe_video, read_frame_at
from ..llm import Usage, estimate_cost, make_client
from ..models.batter_schemas import (
    BatterReportData,
    SessionReportData,
    SessionShot,
)
from .audio_contact import detect_contact_time, detect_contacts
from .swing import find_swings, motion_energy

_SYSTEM = (
    "You are an expert cricket batting coach analysing a single shot from a short "
    "image sequence. The frames are consecutive moments of ONE shot in time order "
    "(earliest first), spanning backlift through contact to follow-through. The "
    "batter to assess is the player on strike (facing the bowler, bat in hand). "
    "Judge only what is visible across the frames; be specific and reference what "
    "you see. Cover head position, foot movement and stride, balance, bat-swing "
    "path and bat face, and follow-through. Be honest about faults — a coaching "
    "report is only useful if it flags what to work on."
)

# Min audio-onset peak-to-mean ratio to trust the knock as the contact anchor.
# A clean isolated crack scores far higher than a soft/commentary-buried contact.
STRONG_KNOCK_STRENGTH = 60.0


def _select_frame_indices(clip_path: str, n_frames: int) -> tuple[list[int], float, int]:
    """Pick ``n_frames`` frame indices spanning the shot, plus (fps, total_frames)."""
    energy, fps = motion_energy(clip_path)
    total = len(energy) + 1
    duration = total / fps
    contact = detect_contact_time(clip_path)  # (time_s, strength) | None
    # Only trust the audio anchor when the knock is clearly impulsive; a soft or
    # commentary-buried contact (low strength) drifts onto the wrong transient.
    strong_knock = contact is not None and contact[1] >= STRONG_KNOCK_STRENGTH

    if duration <= 5.0:
        # Already scoped to ~one shot: sample evenly across the whole clip so
        # backlift -> contact -> follow-through are all represented. This is the
        # reliable path — prefer it (e.g. via --clip) over trusting one peak.
        lo, hi = 0, total - 1
    elif strong_knock:
        t = contact[0]                       # bias earlier to catch the backlift
        lo = max(0, int((t - 1.0) * fps))
        hi = min(total - 1, int((t + 0.6) * fps))
    else:
        swings = find_swings(energy, fps)
        if swings:
            center = max(swings, key=lambda f: energy[min(f, len(energy) - 1)])
            lo = max(0, int(center - 1.2 * fps))
            hi = min(total - 1, int(center + 1.2 * fps))
        else:
            lo, hi = 0, total - 1

    if hi - lo < 2:
        lo, hi = 0, total - 1
    idx = np.linspace(lo, hi, num=min(n_frames, max(1, hi - lo + 1)))
    return sorted({int(round(i)) for i in idx}), fps, total, contact


def _encode_jpeg_b64(frame: np.ndarray, long_edge: int) -> str:
    h, w = frame.shape[:2]
    scale = long_edge / max(h, w)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("Failed to JPEG-encode frame")
    return base64.b64encode(buf).decode("ascii")


def _contact_sheet(frames: list[np.ndarray], out_path: str, cols: int = 5,
                   thumb_h: int = 240) -> str:
    """Tile the sampled frames into one PNG for the report."""
    thumbs = []
    for f in frames:
        h, w = f.shape[:2]
        thumbs.append(cv2.resize(f, (int(w * thumb_h / h), thumb_h)))
    width = max(t.shape[1] for t in thumbs)
    thumbs = [cv2.copyMakeBorder(t, 0, 0, 0, width - t.shape[1],
                                 cv2.BORDER_CONSTANT, value=(20, 20, 20)) for t in thumbs]
    rows = []
    for i in range(0, len(thumbs), cols):
        row = thumbs[i:i + cols]
        while len(row) < cols:
            row.append(np.full_like(thumbs[0], 20))
        rows.append(np.hstack(row))
    cv2.imwrite(out_path, np.vstack(rows))
    return out_path


def _trim(src: str, start: float, end: float, out: str) -> str:
    """Trim ``[start, end]`` seconds, keeping audio (needed for knock detection)."""
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-i", str(src),
         "-t", f"{max(0.2, end - start):.3f}", "-c:v", "libx264", "-preset", "fast",
         "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", out],
        check=True,
    )
    return out


def analyze_clip(clip_path: str, cfg: Config, run_id: str, out_dir: str | Path,
                 clip_s: tuple[float, float] | None = None) -> BatterReportData:
    """Run the full LLM batter analysis on one shot clip.

    ``clip_s`` scopes to a ``(start, end)`` second window before analysis.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    work_clip = clip_path
    if clip_s is not None:
        work_clip = _trim(clip_path, clip_s[0], clip_s[1], str(out_dir / "_clip.mp4"))

    video = probe_video(work_clip)
    indices, fps, _, contact = _select_frame_indices(work_clip, cfg.llm.n_frames)

    frames, images_b64, times = [], [], []
    for i in indices:
        frame = read_frame_at(work_clip, i)
        if frame is None:
            continue
        frames.append(frame)
        images_b64.append(_encode_jpeg_b64(frame, cfg.llm.image_long_edge_px))
        times.append(round(i / fps, 3))
    if not images_b64:
        raise RuntimeError(f"Could not read any frames from {work_clip}")

    sheet = _contact_sheet(frames, str(out_dir / "frames.jpg"))

    contact_note = ""
    if contact is not None:
        contact_note = (f" Audio analysis places bat-ball contact at about "
                        f"{contact[0]:.2f}s into this clip, so the middle frames "
                        "should capture the moment of contact.")

    user_text = (
        f"These {len(images_b64)} frames are consecutive moments of ONE batting "
        "shot in time order (earliest first), from backlift through contact to "
        "follow-through. Identify the shot type and family, assess technique across "
        "the key dimensions, list strengths and faults, give an overall rating, and "
        "a short coaching summary." + contact_note
    )

    client = make_client(cfg.llm)
    analysis, usage = client.analyze_images(images_b64, _SYSTEM, user_text)

    return BatterReportData(
        run_id=run_id,
        clip_path=str(clip_path),
        video=video,
        backend=cfg.llm.backend,
        model=cfg.llm.model,
        frame_times_s=times,
        contact_sheet_png=sheet,
        contact_detected=contact is not None,
        contact_time_s=round(contact[0], 3) if contact else None,
        contact_strength=round(contact[1], 1) if contact else None,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=round(estimate_cost(cfg.llm.model, usage), 4),
        analysis=analysis,
    )


def _user_text(n_images: int, contact_s: float | None = None) -> str:
    note = ""
    if contact_s is not None:
        note = (f" Audio places bat-ball contact near the middle of the sequence "
                f"({contact_s:.2f}s), so a middle frame should show contact.")
    return (
        f"These {n_images} frames are consecutive moments of ONE batting shot in "
        "time order (earliest first), from backlift through contact to "
        "follow-through. Identify the shot type and family, assess technique across "
        "the key dimensions, list strengths and faults, give an overall rating, and "
        "a short coaching summary." + note
    )


def analyze_session(clip_path: str, cfg: Config, run_id: str, out_dir: str | Path,
                    pre: float = 1.2, post: float = 1.0, max_shots: int | None = None,
                    progress=None) -> SessionReportData:
    """Detect every shot via the audio knock and analyse each into one session.

    Each contact anchors a ``[t-pre, t+post]`` window sampled evenly (backlift ->
    contact -> follow-through). ``progress(i, n, msg)`` is called per shot.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video = probe_video(clip_path)
    fps = video.fps or 30.0

    contacts = detect_contacts(clip_path)
    if not contacts:
        raise RuntimeError("No bat-ball contacts found in the audio — is there a "
                           "clear knock? (this needs the session's own audio).")
    contacts.sort(key=lambda c: c[1], reverse=True)
    if max_shots:
        contacts = contacts[:max_shots]
    contacts.sort(key=lambda c: c[0])  # chronological in the report

    client = make_client(cfg.llm)
    total_frames = video.n_frames or 10 ** 9
    shots: list[SessionShot] = []
    tot_in = tot_out = 0
    for i, (t, strength) in enumerate(contacts, 1):
        lo = max(0, int((t - pre) * fps))
        hi = min(total_frames - 1, int((t + post) * fps))
        idx = sorted({int(round(x)) for x in
                      np.linspace(lo, hi, num=cfg.llm.n_frames)})
        frames, b64 = [], []
        for fi in idx:
            fr = read_frame_at(clip_path, fi)
            if fr is None:
                continue
            frames.append(fr)
            b64.append(_encode_jpeg_b64(fr, cfg.llm.image_long_edge_px))
        if not b64:
            continue
        sheet = _contact_sheet(frames, str(out_dir / f"shot_{i:02d}.jpg"),
                               cols=len(frames))

        # One bad shot must not abort the session: retry once, then skip.
        result = None
        for attempt in (1, 2):
            try:
                result = client.analyze_images(b64, _SYSTEM, _user_text(len(b64), pre))
                break
            except Exception as exc:  # noqa: BLE001 — keep the session going
                if attempt == 2 and progress:
                    progress(i, len(contacts), f"shot at {t:.1f}s SKIPPED ({type(exc).__name__})")
        if result is None:
            continue
        analysis, usage = result
        tot_in += usage.input_tokens
        tot_out += usage.output_tokens
        if progress:
            progress(i, len(contacts),
                     f"shot at {t:.1f}s — {usage.input_tokens}+{usage.output_tokens} tok")
        shots.append(SessionShot(index=i, time_s=round(t, 1),
                                 contact_strength=round(strength, 0), frame_png=sheet,
                                 input_tokens=usage.input_tokens,
                                 output_tokens=usage.output_tokens, analysis=analysis))

    return SessionReportData(
        run_id=run_id, clip_path=str(clip_path), video=video,
        backend=cfg.llm.backend, model=cfg.llm.model,
        input_tokens=tot_in, output_tokens=tot_out,
        cost_usd=round(estimate_cost(cfg.llm.model, Usage(tot_in, tot_out)), 4),
        shots=shots)

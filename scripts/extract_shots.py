"""Extract per-shot clips and frames from a batting video — no LLM, no cost.

Detects every bat-ball contact from the audio *knock* (the same detector the
``session`` command uses), then for each shot writes:

* a short trimmed **video clip** (the exact shot, with audio), and
* a set of **frames** spanning backlift -> contact -> follow-through (saved both as
  individual JPEGs and a single strip image).

Plus a ``manifest.csv``. Useful for building a labelled dataset or eyeballing
shots without paying for the vision model.

Usage:
    python scripts/extract_shots.py INPUT.mp4 --out data/shots/myvid
    python scripts/extract_shots.py INPUT.mp4 --out OUT --frames 6 --no-clips
Tunables: --pre/--post (window around contact), --frames, --max-shots, and the
detector knobs --min-gap / --height.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import cv2
import numpy as np

from bowler_analysis.analysis.audio_contact import detect_contacts


def _trim(video: str, start: float, end: float, out: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-i", str(video),
         "-t", f"{max(0.2, end - start):.3f}", "-c:v", "libx264", "-preset", "fast",
         "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", str(out)],
        check=True,
    )


def _strip(frames: list[np.ndarray], out: Path, thumb_h: int = 240) -> None:
    thumbs = [cv2.resize(f, (int(f.shape[1] * thumb_h / f.shape[0]), thumb_h))
              for f in frames]
    w = max(t.shape[1] for t in thumbs)
    thumbs = [cv2.copyMakeBorder(t, 0, 0, 0, w - t.shape[1], cv2.BORDER_CONSTANT,
                                 value=(20, 20, 20)) for t in thumbs]
    cv2.imwrite(str(out), np.hstack(thumbs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--out", required=True, help="Output directory.")
    ap.add_argument("--pre", type=float, default=1.2, help="Seconds before contact.")
    ap.add_argument("--post", type=float, default=1.0, help="Seconds after contact.")
    ap.add_argument("--frames", type=int, default=9, help="Frames to save per shot.")
    ap.add_argument("--max-shots", type=int, default=None, help="Cap to N strongest.")
    ap.add_argument("--min-gap", type=float, default=1.0, help="Min seconds between knocks.")
    ap.add_argument("--height", type=float, default=0.25, help="Detection threshold 0-1.")
    ap.add_argument("--no-clips", action="store_true", help="Skip video clips.")
    ap.add_argument("--no-frames", action="store_true", help="Skip frame extraction.")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(parents=True, exist_ok=True)

    contacts = detect_contacts(args.video, min_gap_s=args.min_gap, rel_height=args.height)
    if not contacts:
        raise SystemExit("No bat-ball contacts found in the audio — is there a clear "
                         "knock? (needs the video's own audio track).")
    contacts.sort(key=lambda c: c[1], reverse=True)
    if args.max_shots:
        contacts = contacts[:args.max_shots]
    contacts.sort(key=lambda c: c[0])  # chronological

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 10 ** 9

    print(f"{len(contacts)} shots detected (fps {fps:.1f}).")
    rows = []
    for i, (t, strength) in enumerate(contacts, 1):
        start, end = max(0.0, t - args.pre), t + args.post
        clip_name = ""
        if not args.no_clips:
            clip_name = f"shot_{i:02d}.mp4"
            _trim(args.video, start, end, out / "clips" / clip_name)
        if not args.no_frames:
            shot_dir = out / "frames" / f"shot_{i:02d}"
            shot_dir.mkdir(exist_ok=True)
            idx = sorted({int(round(x)) for x in
                          np.linspace(start * fps, min(end * fps, n_total - 1), args.frames)})
            frames = []
            for j, fi in enumerate(idx):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, fr = cap.read()
                if not ok:
                    continue
                frames.append(fr)
                cv2.imwrite(str(shot_dir / f"f{j:02d}.jpg"), fr)
            if frames:
                _strip(frames, out / "frames" / f"shot_{i:02d}_strip.jpg")
        rows.append({"shot": i, "time_s": round(t, 2), "strength": round(strength, 0),
                     "clip": clip_name})
        print(f"  shot {i:02d}: {t:6.1f}s  (strength {strength:.0f})")
    cap.release()

    with (out / "manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["shot", "time_s", "strength", "clip"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} shots to {out}/  (clips/, frames/, manifest.csv)")


if __name__ == "__main__":
    main()

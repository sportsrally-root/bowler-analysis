"""Video metadata + frame access.

fps is read via ``ffprobe`` because OpenCV frequently under/mis-reports it for
phone footage — and a wrong fps silently scales every speed estimate. We surface
both ``r_frame_rate`` and ``avg_frame_rate`` and flag when they disagree (a common
slow-mo tell).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Iterator

import cv2

from ..models.schemas import VideoInfo


def _parse_rational(value: str | None) -> float:
    """Parse ffprobe rationals like '120000/1001' into a float."""
    if not value or value in ("0/0", "N/A"):
        return 0.0
    if "/" in value:
        num, den = value.split("/")
        den_f = float(den)
        return float(num) / den_f if den_f else 0.0
    return float(value)


def probe_video(path: str | Path, fps_override: float | None = None) -> VideoInfo:
    """Read container metadata via ffprobe (with an OpenCV fallback)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    container_fps = avg_fps = 0.0
    width = height = n_frames = 0
    duration = 0.0

    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration"
            ":format=duration",
            "-of", "json", str(path),
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        meta = json.loads(out.stdout)
        stream = (meta.get("streams") or [{}])[0]
        fmt = meta.get("format") or {}
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        container_fps = _parse_rational(stream.get("r_frame_rate"))
        avg_fps = _parse_rational(stream.get("avg_frame_rate"))
        n_frames = int(stream.get("nb_frames") or 0)
        duration = float(stream.get("duration") or fmt.get("duration") or 0.0)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError, KeyError):
        pass  # fall through to OpenCV

    # OpenCV fallback / fill-in for anything ffprobe missed.
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    if not width:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if not height:
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not container_fps:
        container_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not n_frames:
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if avg_fps <= 0:
        avg_fps = container_fps
    if not duration and container_fps:
        duration = n_frames / container_fps if n_frames else 0.0

    fps = fps_override if fps_override else container_fps
    slowmo = bool(container_fps and avg_fps and abs(container_fps - avg_fps) / container_fps > 0.05)

    return VideoInfo(
        path=str(path),
        width=width,
        height=height,
        n_frames=n_frames,
        duration_s=round(duration, 3),
        fps=round(fps, 4),
        container_fps=round(container_fps, 4),
        avg_fps=round(avg_fps, 4),
        fps_overridden=fps_override is not None,
        slowmo_suspected=slowmo,
    )


def read_frames(
    path: str | Path,
    start: int = 0,
    end: int | None = None,
) -> Iterator[tuple[int, "cv2.typing.MatLike"]]:
    """Yield ``(frame_index, frame_bgr)`` for frames in ``[start, end)``."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        if start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        idx = start
        while True:
            if end is not None and idx >= end:
                break
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()


def read_frame_at(path: str | Path, frame_index: int) -> "cv2.typing.MatLike | None":
    """Read a single frame by index (used for calibration / thumbnails)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()

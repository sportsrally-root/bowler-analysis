"""Swing detection from frame-to-frame motion energy.

A batter's downswing accelerates the bat through contact, producing a sharp,
short-lived burst of motion. We measure per-frame motion energy (downscaled
grayscale frame differencing) and high-pass it so those bursts stand out from the
slower body movement of the stance/follow-through. This drives both shot
segmentation (``scripts/segment_shots.py``) and frame sampling for the LLM
analyzer (``analysis/shot_llm.py``).
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.signal import find_peaks


def motion_energy(video: str) -> tuple[np.ndarray, float]:
    """Per-frame count of changed pixels (downscaled grayscale frame differencing).

    Returns ``(energy, fps)`` where ``energy`` has one entry per frame transition
    (i.e. ``len(energy) == n_frames - 1``).
    """
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    prev, energy = None, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (270, 480))
        if prev is not None:
            energy.append(float((cv2.absdiff(g, prev) > 25).sum()))
        prev = g
    cap.release()
    return np.asarray(energy, dtype=np.float64), fps


def find_swings(energy: np.ndarray, fps: float, min_gap_s: float = 1.3,
                height: float = 0.30) -> list[int]:
    """Frame indices of downswings = peaks in the high-pass ('spiky') motion signal."""
    if energy.max() <= 0:
        return []
    e = energy / energy.max()
    # High-pass: subtract a wide moving average so only short, sharp bursts survive.
    win = max(3, int(fps * 0.5) | 1)
    baseline = np.convolve(e, np.ones(win) / win, mode="same")
    spiky = np.clip(e - baseline, 0, None)
    spiky = spiky / (spiky.max() or 1.0)
    peaks, _ = find_peaks(spiky, height=height, distance=int(fps * min_gap_s))
    return [int(p) for p in peaks]

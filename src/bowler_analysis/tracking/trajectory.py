"""Trajectory helpers: smoothing and array conversion."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from ..models.schemas import Detection, Trajectory


def to_arrays(traj: Trajectory) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (frames, xs, ys) as numpy arrays sorted by frame."""
    dets = sorted(traj.detections, key=lambda d: d.frame)
    frames = np.array([d.frame for d in dets], dtype=np.float64)
    xs = np.array([d.x for d in dets], dtype=np.float64)
    ys = np.array([d.y for d in dets], dtype=np.float64)
    return frames, xs, ys


def smooth_xy(xs: np.ndarray, ys: np.ndarray, window: int = 7, poly: int = 2):
    """Savitzky-Golay smoothing of the pixel path; no-op for very short tracks."""
    n = len(xs)
    if n < 5:
        return xs.copy(), ys.copy()
    win = min(window, n if n % 2 == 1 else n - 1)
    if win < poly + 2:
        return xs.copy(), ys.copy()
    if win % 2 == 0:
        win -= 1
    return savgol_filter(xs, win, poly), savgol_filter(ys, win, poly)


def trajectory_from_points(frames, xs, ys, radii=None) -> Trajectory:
    radii = radii if radii is not None else [0.0] * len(frames)
    dets = [
        Detection(frame=int(f), x=float(x), y=float(y), radius=float(r))
        for f, x, y, r in zip(frames, xs, ys, radii)
    ]
    return Trajectory(detections=dets)

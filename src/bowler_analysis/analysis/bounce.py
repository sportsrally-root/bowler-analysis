"""Bounce detection from a ball trajectory.

Key insight: the bounce is the ONE instant the ball is on the ground plane, so it is
the only point the ground homography maps accurately. When every ball pixel is
ground-projected, the apparent down-pitch distance (world-Y) shows a sharp "knee" at
the bounce: pre-bounce the descending ball appears to approach rapidly (world-Y falls
steeply); post-bounce the rising ball's airborne projection inflates, so world-Y
flattens out. The bounce is therefore the point of maximum DECELERATION of world-Y —
its largest positive second derivative. This lives in world space, so it is robust to
camera orientation (the image-space path may show no visible corner at all).
"""

from __future__ import annotations

import numpy as np

from ..geometry.projection import image_to_world
from ..models.schemas import BounceResult, Trajectory
from ..tracking.trajectory import smooth_xy, to_arrays

# Only trust ground-projected points within a sane band around the pitch; the airborne
# ball near the bowler projects to wildly inflated distances we must ignore.
SANE_MAX_Y_PAD = 6.0
MIN_DECEL = 0.05  # m/frame^2; below this there is no clear bounce knee (full toss)


def detect_bounce(traj: Trajectory, H: np.ndarray, pitch_length: float = 20.12) -> BounceResult | None:
    """Locate the bounce and map it to world coordinates. None if too few points."""
    frames, xs, ys = to_arrays(traj)
    if len(frames) < 5:
        return None
    sx, sy = smooth_xy(xs, ys)

    world = image_to_world(H, np.stack([sx, sy], axis=1))  # (N, 2) metres
    wx, wy = world[:, 0], world[:, 1]

    # Restrict to the near-pitch region where ground projection is meaningful.
    sane = (wy >= -2.0) & (wy <= pitch_length + SANE_MAX_Y_PAD)
    sane_idx = np.where(sane)[0]
    if len(sane_idx) < 5:
        return None
    lo, hi = sane_idx[0], sane_idx[-1]

    # Second derivative (deceleration) of world-Y over the sane window.
    wy_s = wy[lo:hi + 1]
    f_s = frames[lo:hi + 1]
    second = np.full(len(wy_s), -np.inf)
    for i in range(1, len(wy_s) - 1):
        second[i] = wy_s[i + 1] - 2 * wy_s[i] + wy_s[i - 1]
    k = int(np.argmax(second))
    p = lo + k
    bounced = 1 <= k <= len(wy_s) - 2 and second[k] >= MIN_DECEL

    bframe = float(frames[p])
    bwx, bwy = float(wx[p]), float(wy[p])

    # Sub-frame refinement: parabola through world-Y around the knee.
    if bounced and 1 <= p <= len(wy) - 2:
        f3, y3 = frames[p - 1:p + 2], wy[p - 1:p + 2]
        try:
            a, b, c = np.polyfit(f3, y3, 2)
            if abs(a) > 1e-9:
                vertex = -b / (2 * a)
                if f3[0] <= vertex <= f3[-1]:
                    bframe = float(vertex)
                    bwy = float(np.interp(vertex, frames, wy))
                    bwx = float(np.interp(vertex, frames, wx))
        except (np.linalg.LinAlgError, ValueError):
            pass

    bimg_x = float(np.interp(bframe, frames, sx))
    bimg_y = float(np.interp(bframe, frames, sy))

    return BounceResult(
        frame=bframe,
        image_xy=[bimg_x, bimg_y],
        world_xy=[bwx, bwy],
        detected=bool(bounced),
    )

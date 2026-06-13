"""Ball speed estimation from the pre-bounce flight in world coordinates.

We project the ball's pixels to the ground plane and measure displacement per frame
over the descending (pre-bounce) segment, then convert to km/h using the true fps.
Because the ball is above the ground during flight, ground-plane projection is an
approximation — we report a speed band, not a single exact number. The biggest error
source is fps (slow-mo); see ingest.video_reader.
"""

from __future__ import annotations

import numpy as np

from ..geometry.projection import image_to_world
from ..models.schemas import BounceResult, Trajectory
from ..tracking.trajectory import to_arrays

MS_TO_KPH = 3.6


def estimate_speed(
    traj: Trajectory,
    H: np.ndarray,
    fps: float,
    bounce: BounceResult | None,
    cfg=None,
) -> tuple[float, float] | None:
    """Return (speed_kph, uncertainty_kph) or None if not estimable.

    Primary method (single airborne camera, height-robust): average down-pitch speed
    from release to bounce. The bounce is on the ground plane so its world position is
    accurate; the release down-pitch position is taken as a known reference (over the
    bowler's popping crease). Flight time = (bounce_frame - first_tracked_frame)/fps.
    This sidesteps the fact that the ground homography mis-projects the airborne ball.
    It assumes the ball is tracked from near release — report a band, not an exact figure.

    Fallback (no bounce / no cfg): per-frame ground-plane displacement, which is only
    valid when motion is in the ground plane (e.g. unit tests, or a rolling ball).
    """
    if fps <= 0:
        return None
    frames, xs, ys = to_arrays(traj)
    if len(frames) < 3:
        return None

    if bounce is not None and bounce.detected and cfg is not None:
        g = cfg.geometry
        release_y = g.pitch_length_m - g.popping_crease_offset_m  # ~over the crease
        first_frame = float(frames.min())
        flight_time = (bounce.frame - first_frame) / fps
        distance = abs(release_y - float(bounce.world_xy[1]))
        if flight_time > 0 and distance > 0:
            speed_kph = (distance / flight_time) * MS_TO_KPH
            # Band: combine a one-frame timing error with the release-position assumption.
            n = max(1.0, bounce.frame - first_frame)
            rel_err = max(1.0 / n, 0.08)
            return round(speed_kph, 1), round(speed_kph * rel_err, 1)

    # Fallback: per-frame ground-plane median speed (in-plane motion only).
    world = image_to_world(H, np.stack([xs, ys], axis=1))
    dpos = np.diff(world, axis=0)
    dframe = np.diff(frames)
    valid = dframe > 0
    if valid.sum() < 2:
        return None
    seg_speed = np.linalg.norm(dpos[valid], axis=1) / (dframe[valid] / fps)
    speed_kph = float(np.median(seg_speed)) * MS_TO_KPH
    uncertainty_kph = max(float(np.std(seg_speed)) * MS_TO_KPH, 0.05 * speed_kph)
    return round(speed_kph, 1), round(uncertainty_kph, 1)

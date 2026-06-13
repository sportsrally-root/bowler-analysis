"""Per-delivery segmentation.

MVP: treat the clip as one delivery (optionally bounded by --clip). The multi-track
path picks plausible delivery trajectories out of all tracks the tracker produced,
for videos containing several balls.
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from ..models.schemas import Trajectory


def filter_deliveries(trajectories: list[Trajectory], cfg: Config) -> list[Trajectory]:
    """Keep tracks long enough and with enough pixel travel to be a delivery."""
    out = []
    for traj in trajectories:
        if len(traj.detections) < cfg.quality.min_track_points:
            continue
        xs = np.array([d.x for d in traj.detections])
        ys = np.array([d.y for d in traj.detections])
        span = float(np.hypot(np.ptp(xs), np.ptp(ys)))
        if span < 40:  # negligible movement -> not a ball flight
            continue
        out.append(traj)
    # Order by first frame so delivery indices are chronological.
    out.sort(key=lambda t: t.detections[0].frame)
    return out

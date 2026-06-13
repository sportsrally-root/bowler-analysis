"""Multi-hypothesis Kalman tracking that stitches ball candidates into a flight.

Several blobs move each frame (ball, bowler, batter). We keep multiple candidate
tracks, associate each frame's detections by nearest-neighbour gating around a
constant-velocity Kalman prediction, coast through gaps, and finally pick the track
that best looks like a delivery (longest, fastest, travels down the pitch corridor).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from filterpy.kalman import KalmanFilter

from ..config import Config
from ..models.schemas import Detection, Trajectory
from .detector_classical import Candidate


def _new_kf(x: float, y: float, dt: float = 1.0) -> KalmanFilter:
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
    kf.x = np.array([x, y, 0, 0], float)
    kf.P *= 500.0
    kf.R *= 9.0
    kf.Q *= 0.5
    return kf


@dataclass
class _Track:
    kf: KalmanFilter
    detections: list[Detection] = field(default_factory=list)
    missed: int = 0
    last_frame: int = -1

    @property
    def pred_xy(self) -> tuple[float, float]:
        return float(self.kf.x[0]), float(self.kf.x[1])


class BallTracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tracks: list[_Track] = []
        self.finished: list[_Track] = []

    def _retire_stale(self):
        keep = []
        for tr in self.tracks:
            if tr.missed > self.cfg.tracking.max_missed_frames:
                self.finished.append(tr)
            else:
                keep.append(tr)
        self.tracks = keep

    def update(self, frame_idx: int, candidates: list[Candidate]):
        gate = self.cfg.tracking.max_assoc_dist_px
        for tr in self.tracks:
            tr.kf.predict()

        unmatched = list(candidates)
        # Greedy nearest-neighbour association (few tracks/candidates per frame).
        for tr in self.tracks:
            if not unmatched:
                break
            px, py = tr.pred_xy
            dists = [np.hypot(c.x - px, c.y - py) for c in unmatched]
            j = int(np.argmin(dists))
            if dists[j] <= gate:
                c = unmatched.pop(j)
                tr.kf.update([c.x, c.y])
                tr.detections.append(
                    Detection(frame=frame_idx, x=c.x, y=c.y, radius=c.radius, score=1.0)
                )
                tr.missed = 0
                tr.last_frame = frame_idx
            else:
                tr.missed += 1

        # Tracks that matched nothing this frame also age.
        for tr in self.tracks:
            if tr.last_frame != frame_idx:
                tr.missed += 1

        # Spawn new tracks from leftover candidates.
        for c in unmatched:
            kf = _new_kf(c.x, c.y)
            tr = _Track(kf=kf, last_frame=frame_idx)
            tr.detections.append(
                Detection(frame=frame_idx, x=c.x, y=c.y, radius=c.radius, score=1.0)
            )
            self.tracks.append(tr)

        self._retire_stale()

    def finalize(self) -> Trajectory:
        """Pick the best track and return it as a Trajectory."""
        self.finished.extend(self.tracks)
        self.tracks = []
        if not self.finished:
            return Trajectory(detections=[])
        best = max(self.finished, key=self._track_score)
        best.detections.sort(key=lambda d: d.frame)
        return Trajectory(detections=best.detections)

    def all_trajectories(self, min_len: int = 3) -> list[Trajectory]:
        """All tracks meeting a length threshold (for multi-delivery segmentation)."""
        pool = self.finished + self.tracks
        out = []
        for tr in pool:
            if len(tr.detections) >= min_len:
                dets = sorted(tr.detections, key=lambda d: d.frame)
                out.append(Trajectory(detections=dets))
        return out

    def _track_score(self, tr: _Track) -> float:
        """Heuristic: long tracks that move a lot in pixels look like deliveries."""
        if len(tr.detections) < 2:
            return -1.0
        xs = np.array([d.x for d in tr.detections])
        ys = np.array([d.y for d in tr.detections])
        span = np.hypot(xs.max() - xs.min(), ys.max() - ys.min())
        return len(tr.detections) * 2.0 + span * 0.05


def track_candidates(cfg: Config, per_frame: list[tuple[int, list[Candidate]]]) -> Trajectory:
    """Convenience: run the tracker over pre-collected per-frame candidates."""
    tracker = BallTracker(cfg)
    for frame_idx, cands in per_frame:
        tracker.update(frame_idx, cands)
    return tracker.finalize()

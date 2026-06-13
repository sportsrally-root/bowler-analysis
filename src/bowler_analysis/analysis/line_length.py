"""Assemble per-delivery line & length metrics from a bounce + calibration."""

from __future__ import annotations

import numpy as np

from ..config import Config
from ..geometry.calibration import homography_arrays
from ..geometry.pitch_model import classify_length, classify_line, signed_line_x
from ..models.schemas import CalibrationResult, Delivery, Trajectory
from .bounce import detect_bounce
from .speed import estimate_speed


def analyze_delivery(
    index: int,
    traj: Trajectory,
    calib: CalibrationResult,
    cfg: Config,
    fps: float,
) -> Delivery:
    """Run bounce -> line/length -> speed for one trajectory."""
    H, _ = homography_arrays(calib)
    delivery = Delivery(index=index, trajectory=traj, handedness=cfg.batter.handedness)

    bounce = detect_bounce(traj, H, cfg.geometry.pitch_length_m)
    delivery.bounce = bounce

    if bounce is not None:
        length_m = float(bounce.world_xy[1])         # distance from striker stumps
        delivery.length_m = round(length_m, 2)
        delivery.length_label = classify_length(length_m, cfg, bounced=bounce.detected)
        delivery.line_m = round(signed_line_x(bounce.world_xy[0], cfg), 3)
        delivery.line_label = classify_line(bounce.world_xy[0], cfg)
        if not bounce.detected:
            delivery.notes.append("No clear bounce detected — possible full toss.")

    speed = estimate_speed(traj, H, fps, bounce, cfg)
    if speed is not None:
        delivery.speed_kph, delivery.speed_uncertainty_kph = speed

    if len(traj.detections) < cfg.quality.min_track_points:
        delivery.notes.append(
            f"Short track ({len(traj.detections)} points) — metrics low confidence."
        )
    return delivery

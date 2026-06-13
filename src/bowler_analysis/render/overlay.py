"""Annotated video output: ball trail, bounce marker, perspective pitch overlay, HUD.

The perspective (broadcast-style) pitch map is the world-space length zones and
bounce dots warped back onto the real pitch via the inverse homography — the
Sky-Sports look comes "for free" from the calibration.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import Config
from ..geometry.calibration import homography_arrays
from ..geometry.projection import world_to_image
from ..models.schemas import CalibrationResult, Delivery
from ..ingest.video_reader import read_frames


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def draw_perspective_zones(frame: np.ndarray, calib: CalibrationResult, cfg: Config,
                           alpha: float = 0.25) -> np.ndarray:
    """Shade the length-zone bands on the real pitch in correct perspective."""
    _, H_inv = homography_arrays(calib)
    g = cfg.geometry
    rc = g.return_crease_half_width_m
    max_len = min(g.pitch_length_m, 12.0)
    overlay = frame.copy()
    for z in cfg.length_zones:
        if z.min >= max_len:
            continue
        top = min(z.max, max_len)
        world = np.array([[-rc, z.min], [rc, z.min], [rc, top], [-rc, top]])
        poly = world_to_image(H_inv, world).astype(np.int32)
        cv2.fillConvexPoly(overlay, poly, _hex_to_bgr(z.color or "#888888"))
    out = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    # Pitch rails.
    for sx in (-rc, rc):
        rail = world_to_image(H_inv, np.array([[sx, 0], [sx, max_len]])).astype(np.int32)
        cv2.line(out, tuple(rail[0]), tuple(rail[1]), (255, 255, 255), 1)
    return out


def _draw_bounce_marker(frame, calib, world_xy):
    _, H_inv = homography_arrays(calib)
    p = world_to_image(H_inv, np.array([world_xy]))[0].astype(int)
    cv2.drawMarker(frame, tuple(p), (0, 0, 255), cv2.MARKER_CROSS, 22, 3)
    cv2.circle(frame, tuple(p), 10, (0, 0, 255), 2)


def _hud(frame, lines: list[str], origin=(16, 30)):
    for i, txt in enumerate(lines):
        y = origin[1] + i * 26
        cv2.putText(frame, txt, (origin[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, txt, (origin[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)


def annotate_video(
    video_path: str,
    out_path: str,
    calib: CalibrationResult,
    cfg: Config,
    delivery: Delivery,
    fps: float,
    trail_len: int = 18,
) -> str:
    """Render an annotated MP4 for a single delivery."""
    w, h = calib.image_size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps if fps > 0 else 30.0, (w, h))

    # Per-frame ball positions for trail drawing.
    by_frame = {d.frame: (int(d.x), int(d.y)) for d in delivery.trajectory.detections}
    frames_sorted = sorted(by_frame)
    start = frames_sorted[0] if frames_sorted else 0
    end = (frames_sorted[-1] + 15) if frames_sorted else None
    bounce_frame = delivery.bounce.frame if delivery.bounce else None

    hud_base = [
        f"Delivery #{delivery.index}",
        f"Speed: {delivery.speed_kph:.0f} km/h" if delivery.speed_kph else "Speed: n/a",
        f"Length: {delivery.length_label or 'n/a'}"
        + (f" ({delivery.length_m:.1f} m)" if delivery.length_m else ""),
        f"Line: {delivery.line_label or 'n/a'}",
        f"fps used: {fps:g}",
    ]

    trail: list[tuple[int, int]] = []
    for idx, frame in read_frames(video_path, start=start, end=end):
        frame = draw_perspective_zones(frame, calib, cfg)
        if idx in by_frame:
            trail.append(by_frame[idx])
        trail = trail[-trail_len:]
        for j, (tx, ty) in enumerate(trail):
            fade = int(255 * (j + 1) / len(trail))
            cv2.circle(frame, (tx, ty), 3, (0, fade, 255), -1)
        if trail:
            cv2.circle(frame, trail[-1], 6, (0, 255, 255), 2)
        if bounce_frame is not None and idx >= bounce_frame and delivery.bounce:
            _draw_bounce_marker(frame, calib, delivery.bounce.world_xy)
        _hud(frame, hud_base)
        writer.write(frame)

    writer.release()
    return out_path

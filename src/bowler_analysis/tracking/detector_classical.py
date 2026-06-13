"""Classical ball-candidate detector: background subtraction + blob filtering.

A static tripod makes MOG2 background subtraction strong: the moving foreground is
mostly the ball, the bowler, and the batter. We keep small, roughly circular blobs
and (optionally) reject anything far outside the pitch corridor using the
homography. The tracker downstream stitches candidates into a flight path.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import Config


@dataclass
class Candidate:
    frame: int
    x: float
    y: float
    radius: float
    area: float


class ClassicalDetector:
    """Per-frame ball candidate detector with a persistent background model."""

    def __init__(self, cfg: Config, corridor_mask: np.ndarray | None = None):
        self.cfg = cfg
        t = cfg.tracking
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=t.bg_history, varThreshold=t.bg_var_threshold, detectShadows=False
        )
        self.corridor_mask = corridor_mask  # uint8 (H, W), 255 inside corridor

    def detect(self, frame_idx: int, frame_bgr: np.ndarray) -> list[Candidate]:
        t = self.cfg.tracking
        fg = self.bg.apply(frame_bgr)
        # Clean the mask: drop shadow speckle, close small gaps.
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg = cv2.dilate(fg, np.ones((3, 3), np.uint8), iterations=1)
        if self.corridor_mask is not None:
            fg = cv2.bitwise_and(fg, self.corridor_mask)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out: list[Candidate] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < t.min_blob_area_px or area > t.max_blob_area_px:
                continue
            perim = cv2.arcLength(c, True)
            if perim <= 0:
                continue
            circularity = 4.0 * np.pi * area / (perim * perim)
            if circularity < t.min_circularity:
                continue
            (x, y), radius = cv2.minEnclosingCircle(c)
            out.append(Candidate(frame_idx, float(x), float(y), float(radius), float(area)))
        return out


def build_corridor_mask(cfg: Config, H_inv: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """A binary mask of the pitch corridor (a lateral band) in image space.

    The corridor is the band between the two projected pitch rails (left/right return
    creases, plus a lateral margin), spanning the full length of the pitch. It is an
    IMAGE-space band, not a ground-projection test: an airborne ball travels roughly
    over the pitch centreline, so it stays within the band at every height — whereas a
    ground-projection test wrongly throws out the airborne ball near the bowler's end.
    Side clutter (crowd, sightscreen, square-leg umpire) still falls outside.
    ``H_inv`` is the world->image homography.
    """
    from ..geometry.projection import world_to_image

    g = cfg.geometry
    margin = cfg.tracking.corridor_margin_m
    half = g.return_crease_half_width_m + margin
    L = g.pitch_length_m
    w, h = image_size

    ys = np.linspace(-2.0, L, 60)
    left = world_to_image(H_inv, np.stack([np.full_like(ys, -half), ys], axis=1))
    right = world_to_image(H_inv, np.stack([np.full_like(ys, half), ys], axis=1))

    # Fit each rail as a line col = m*row + b, then extrapolate across the FULL image
    # height. The airborne ball can sit above or below the ground-pitch band, so we
    # only constrain it laterally (between the rails); vertical extent is unbounded.
    def rail_line(pts):
        rows, cols = pts[:, 1], pts[:, 0]
        ok = np.isfinite(rows) & np.isfinite(cols)
        m, b = np.polyfit(rows[ok], cols[ok], 1)
        return m, b

    ml, bl = rail_line(left)
    mr, br = rail_line(right)
    rows_full = np.arange(h)
    xl = ml * rows_full + bl
    xr = mr * rows_full + br
    lo = np.minimum(xl, xr)
    hi = np.maximum(xl, xr)

    mask = np.zeros((h, w), dtype=np.uint8)
    pad = 25  # lateral pixel margin
    for r in range(h):
        a = max(0, int(lo[r]) - pad)
        z = min(w, int(hi[r]) + pad)
        if z > a:
            mask[r, a:z] = 255
    return mask

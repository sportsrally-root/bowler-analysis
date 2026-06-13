"""Pixel <-> pitch-plane coordinate transforms via a homography.

A homography maps points on the ground plane (z=0, world metres) to image pixels
and back. This is sufficient for everything that happens on the ground: bounce
location, line and length, and ground-plane speed.
"""

from __future__ import annotations

import numpy as np


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to an (N, 2) array of points.

    Returns an (N, 2) array. Works for image->world or world->image depending on
    which matrix is passed.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1))
    homog = np.hstack([pts, ones])               # (N, 3)
    out = (H @ homog.T).T                         # (N, 3)
    w = out[:, 2:3]
    # Guard against division by zero for points on the line at infinity.
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return out[:, :2] / w


def image_to_world(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Map image pixels -> world metres on the ground plane."""
    return apply_homography(H, pts)


def world_to_image(H_inv: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Map world metres -> image pixels (pass the inverse homography)."""
    return apply_homography(H_inv, pts)


def reprojection_error(H: np.ndarray, image_pts: np.ndarray, world_pts: np.ndarray) -> float:
    """Mean pixel error when mapping world points back into the image.

    Uses the inverse of H (world->image) and compares against the clicked pixels.
    """
    H_inv = np.linalg.inv(H)
    projected = world_to_image(H_inv, world_pts)
    image_pts = np.asarray(image_pts, dtype=np.float64).reshape(-1, 2)
    errors = np.linalg.norm(projected - image_pts, axis=1)
    return float(np.mean(errors))

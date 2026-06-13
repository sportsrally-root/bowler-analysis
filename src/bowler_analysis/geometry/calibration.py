"""Solve and persist the image<->pitch homography from clicked reference points."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ..models.schemas import CalibrationResult
from .projection import reprojection_error


def solve_homography(
    image_points: np.ndarray,
    world_points: np.ndarray,
    point_names: list[str],
    image_size: tuple[int, int],
) -> CalibrationResult:
    """Compute the image->world homography from >=4 correspondences.

    ``image_points`` and ``world_points`` are (N, 2) arrays. Returns a
    CalibrationResult with both H and its inverse plus the reprojection error.
    """
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    world_points = np.asarray(world_points, dtype=np.float64).reshape(-1, 2)
    if len(image_points) < 4:
        raise ValueError("Need at least 4 point correspondences for a homography.")
    if len(image_points) != len(world_points):
        raise ValueError("image_points and world_points must have equal length.")

    H, _ = cv2.findHomography(image_points, world_points, method=cv2.RANSAC)
    if H is None:
        raise RuntimeError("Homography estimation failed (points may be collinear).")

    err = reprojection_error(H, image_points, world_points)
    H_inv = np.linalg.inv(H)

    return CalibrationResult(
        homography=H.tolist(),
        homography_inv=H_inv.tolist(),
        image_points=image_points.tolist(),
        world_points=world_points.tolist(),
        point_names=list(point_names),
        reprojection_error_px=err,
        image_size=[int(image_size[0]), int(image_size[1])],
    )


def save_calibration(result: CalibrationResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(result.model_dump(), fh, indent=2)


def load_calibration(path: str | Path) -> CalibrationResult:
    with Path(path).open("r") as fh:
        return CalibrationResult.model_validate(json.load(fh))


def homography_arrays(result: CalibrationResult) -> tuple[np.ndarray, np.ndarray]:
    """Return (H, H_inv) as numpy arrays from a stored result."""
    return (
        np.array(result.homography, dtype=np.float64),
        np.array(result.homography_inv, dtype=np.float64),
    )

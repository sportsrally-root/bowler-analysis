"""Geometry tests: homography round-trips pixel<->world and error is accurate."""

import numpy as np

from bowler_analysis.config import load_config
from bowler_analysis.geometry.calibration import homography_arrays, solve_homography
from bowler_analysis.geometry.pitch_model import (
    classify_length,
    classify_line,
    reference_points,
)
from bowler_analysis.geometry.projection import image_to_world, world_to_image


def _synthetic_correspondences():
    """A plausible perspective view of the return-crease box, looking down-pitch."""
    cfg = load_config()
    rp = {p.name: p for p in reference_points(cfg)}
    names = ["striker_pop_left", "striker_pop_right", "bowler_pop_left", "bowler_pop_right"]
    world = np.array([[rp[n].x, rp[n].y] for n in names])
    # Hand-placed pixels: near (striker) wide at bottom, far (bowler) narrow at top.
    image = np.array([
        [400, 1000],   # striker left
        [1520, 1000],  # striker right
        [840, 300],    # bowler left
        [1080, 300],   # bowler right
    ], dtype=float)
    return cfg, names, world, image


def test_homography_roundtrip():
    cfg, names, world, image = _synthetic_correspondences()
    result = solve_homography(image, world, names, (1920, 1080))
    H, H_inv = homography_arrays(result)

    # image -> world should recover the clicked world points.
    recovered = image_to_world(H, image)
    assert np.allclose(recovered, world, atol=1e-3)

    # world -> image should recover the clicked pixels.
    back = world_to_image(H_inv, world)
    assert np.allclose(back, image, atol=1e-2)

    assert result.reprojection_error_px < 1e-2


def test_length_classification():
    cfg = load_config()
    assert classify_length(0.5, cfg) == "Yorker"
    assert classify_length(6.0, cfg) == "Good"
    assert classify_length(11.0, cfg) == "Short"
    assert classify_length(3.0, cfg, bounced=False) == cfg.full_toss_label


def test_line_classification_handedness():
    cfg = load_config()
    cfg.batter.handedness = "rhb"
    # +x is OFF for RHB -> a positive offset reads as an off-side line.
    assert "off" in classify_line(0.30, cfg).lower()
    # Flip to LHB: the same physical +x now reads as leg side.
    cfg.batter.handedness = "lhb"
    assert "leg" in classify_line(0.20, cfg).lower()

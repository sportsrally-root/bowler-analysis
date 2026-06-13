"""Speed tests: known world displacement + fps -> known km/h, and fps scaling."""

import numpy as np

from bowler_analysis.analysis.speed import estimate_speed
from bowler_analysis.tracking.trajectory import trajectory_from_points


def _identity_homography():
    # Image pixels already in 'metres' -> trivial mapping for a controlled test.
    return np.eye(3, dtype=float)


def test_speed_from_known_displacement():
    # Ball moves 0.5 m per frame along +y; at 100 fps -> 50 m/s -> 180 km/h.
    H = _identity_homography()
    frames = np.arange(10)
    xs = np.zeros(10)
    ys = np.arange(10) * 0.5
    traj = trajectory_from_points(frames, xs, ys)
    speed_kph, _ = estimate_speed(traj, H, fps=100.0, bounce=None)
    assert abs(speed_kph - 180.0) < 1.0


def test_speed_scales_with_fps():
    H = _identity_homography()
    frames = np.arange(10)
    xs = np.zeros(10)
    ys = np.arange(10) * 0.5
    traj = trajectory_from_points(frames, xs, ys)
    s100, _ = estimate_speed(traj, H, fps=100.0, bounce=None)
    s50, _ = estimate_speed(traj, H, fps=50.0, bounce=None)
    # Halving fps halves the implied speed for the same pixel motion.
    assert abs(s100 - 2 * s50) < 1.0

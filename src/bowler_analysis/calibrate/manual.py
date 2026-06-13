"""Interactive manual calibration: click known pitch features to solve a homography.

Usage is driven from the CLI. The user is shown a reference frame and clicks each
named point in order; points can be skipped (only >=4 are required). For headless
runs or tests, ``points_from_file`` loads pre-recorded clicks instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ..config import Config
from ..geometry.calibration import solve_homography
from ..geometry.pitch_model import ReferencePoint, reference_points
from ..models.schemas import CalibrationResult


def _draw_overlay(img, ref_pts: list[ReferencePoint], clicks: dict[str, tuple[float, float]],
                  active_idx: int):
    canvas = img.copy()
    for i, rp in enumerate(ref_pts):
        if rp.name in clicks:
            x, y = clicks[rp.name]
            cv2.circle(canvas, (int(x), int(y)), 6, (0, 255, 0), -1)
            cv2.putText(canvas, str(i + 1), (int(x) + 8, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    active = ref_pts[active_idx] if active_idx < len(ref_pts) else None
    banner = (f"[{active_idx + 1}/{len(ref_pts)}] Click: {active.label}"
              if active else "All points placed.")
    help_line = "Left-click=place  n=skip  u=undo  ENTER=done  ESC=cancel"
    for j, txt in enumerate((banner, help_line)):
        cv2.putText(canvas, txt, (16, 32 + 28 * j), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, txt, (16, 32 + 28 * j), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


def collect_clicks_interactive(frame, cfg: Config) -> dict[str, tuple[float, float]]:
    """Open a window and let the user click reference points. Returns name->pixel."""
    ref_pts = reference_points(cfg)
    clicks: dict[str, tuple[float, float]] = {}
    state = {"active": 0}
    win = "bowler-analysis: calibration"

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and state["active"] < len(ref_pts):
            clicks[ref_pts[state["active"]].name] = (float(x), float(y))
            state["active"] += 1

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    try:
        while True:
            cv2.imshow(win, _draw_overlay(frame, ref_pts, clicks, state["active"]))
            key = cv2.waitKey(20) & 0xFF
            if key == 27:  # ESC
                clicks.clear()
                break
            if key in (13, 10):  # ENTER
                break
            if key == ord("n") and state["active"] < len(ref_pts):
                state["active"] += 1
            if key == ord("u") and state["active"] > 0:
                state["active"] -= 1
                clicks.pop(ref_pts[state["active"]].name, None)
    finally:
        cv2.destroyWindow(win)
    return clicks


def points_from_file(path: str | Path) -> dict[str, tuple[float, float]]:
    """Load clicked pixel points from JSON: {"name": [x, y], ...}."""
    with Path(path).open("r") as fh:
        data = json.load(fh)
    return {k: (float(v[0]), float(v[1])) for k, v in data.items()}


def build_calibration(
    clicks: dict[str, tuple[float, float]],
    cfg: Config,
    image_size: tuple[int, int],
) -> CalibrationResult:
    """Turn clicked points into a homography by matching names to world coords."""
    ref_by_name = {rp.name: rp for rp in reference_points(cfg)}
    image_pts, world_pts, names = [], [], []
    for name, (px, py) in clicks.items():
        rp = ref_by_name.get(name)
        if rp is None:
            continue
        image_pts.append([px, py])
        world_pts.append([rp.x, rp.y])
        names.append(name)
    if len(image_pts) < 4:
        raise ValueError(
            f"Only {len(image_pts)} valid points; need >=4 for calibration."
        )
    return solve_homography(np.array(image_pts), np.array(world_pts), names, image_size)


def render_qa_overlay(frame, result: CalibrationResult, cfg: Config):
    """Draw the reprojected pitch model on the frame so the user can sanity-check.

    Returns a new image with the world grid (creases + a length grid) projected
    back into the image via the inverse homography.
    """
    H_inv = np.array(result.homography_inv, dtype=np.float64)
    from ..geometry.projection import world_to_image

    canvas = frame.copy()
    g = cfg.geometry
    rc = g.return_crease_half_width_m
    L = g.pitch_length_m

    # Pitch side rails + every-2m length grid line.
    def line_world(p0, p1, color, thick=2):
        pts = world_to_image(H_inv, np.array([p0, p1]))
        cv2.line(canvas, tuple(pts[0].astype(int)), tuple(pts[1].astype(int)), color, thick)

    line_world([-rc, 0], [-rc, L], (255, 200, 0))
    line_world([rc, 0], [rc, L], (255, 200, 0))
    for y in range(0, int(L) + 1, 2):
        line_world([-rc, y], [rc, y], (180, 180, 180), 1)

    # Clicked points (green) for reference.
    for (px, py) in result.image_points:
        cv2.circle(canvas, (int(px), int(py)), 6, (0, 255, 0), -1)

    txt = f"reprojection error: {result.reprojection_error_px:.2f}px"
    cv2.putText(canvas, txt, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(canvas, txt, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
    return canvas

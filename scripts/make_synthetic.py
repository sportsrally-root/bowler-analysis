"""Generate a synthetic delivery video + calibration points for end-to-end testing.

Builds a pinhole camera behind the striker looking down the pitch, renders a ball
flying from the bowler's end, bouncing at a known world (line, length), then rising
to the batsman. Because the bounce is on the ground plane (z=0), the ground
homography recovers its true world position — so the pipeline's line/length output
can be checked against the ground truth printed at the end.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from bowler_analysis.config import load_config
from bowler_analysis.geometry.pitch_model import reference_points

W, H = 1280, 720
FPS = 120.0
LEAD_IN = 15           # static frames so MOG2 learns the background

# Ground-truth delivery (world metres): bounce at good length, just outside off.
RELEASE = np.array([0.05, 18.0, 2.10])
BOUNCE = np.array([0.12, 6.0, 0.0])
AFTER = np.array([0.16, 1.0, 0.75])
FRAMES_A = 44          # release -> bounce
FRAMES_B = 16          # bounce -> past the bat


def look_at_camera(C, target, up=(0, 0, 1)):
    """OpenCV camera basis: x=right, y=down, z=forward (right-handed)."""
    C = np.asarray(C, float)
    z = np.asarray(target, float) - C
    z /= np.linalg.norm(z)
    up = np.asarray(up, float)
    x = np.cross(z, up)          # right
    x /= np.linalg.norm(x)
    y = np.cross(z, x)           # down
    R = np.stack([x, y, z], axis=0)   # world -> camera
    return R, C


def project(P, R, C, K):
    pc = R @ (np.asarray(P, float) - C)
    if pc[2] <= 1e-6:
        return None
    u = K[0, 0] * pc[0] / pc[2] + K[0, 2]
    v = K[1, 1] * pc[1] / pc[2] + K[1, 2]
    return np.array([u, v])


def build_flight():
    """A realistic single-bounce flight: monotonic projectile descent hand->pitch,
    then a rising arc off the pitch. The bounce is a clean corner."""
    pts = []
    for i in range(FRAMES_A):
        t = i / (FRAMES_A - 1)
        xy = (1 - t) * RELEASE[:2] + t * BOUNCE[:2]
        z = RELEASE[2] * (1 - t ** 2)          # apex at release, drops to 0 at bounce
        pts.append([xy[0], xy[1], max(z, 0.0)])
    for i in range(1, FRAMES_B):
        t = i / (FRAMES_B - 1)
        xy = (1 - t) * BOUNCE[:2] + t * AFTER[:2]
        z = AFTER[2] * t                        # rises linearly off the pitch
        pts.append([xy[0], xy[1], max(z, 0.0)])
    return np.array(pts)


def make_background():
    rng = np.random.default_rng(7)
    bg = np.full((H, W, 3), (60, 120, 70), np.uint8)        # green field
    # A brown pitch strip down the middle + fixed speckle so the bg is textured/static.
    cv2.rectangle(bg, (W // 2 - 120, 80), (W // 2 + 120, H), (120, 160, 200), -1)
    noise = rng.integers(-12, 12, (H, W, 3), dtype=np.int16)
    return np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def main():
    cfg = load_config()
    K = np.array([[1300, 0, W / 2], [0, 1300, H / 2], [0, 0, 1]], float)
    R, C = look_at_camera([0.0, -4.0, 2.2], [0.0, 8.0, 0.2])

    bg = make_background()
    out_video = Path("data/raw/synthetic.mp4")
    out_video.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    for _ in range(LEAD_IN):
        vw.write(bg.copy())

    flight = build_flight()
    for P in flight:
        frame = bg.copy()
        p = project(P, R, C, K)
        if p is not None and 0 <= p[0] < W and 0 <= p[1] < H:
            # Ball radius grows as it nears the (close) batsman for realism.
            radius = int(np.interp(P[1], [1, 18], [9, 4]))
            cv2.circle(frame, (int(p[0]), int(p[1])), radius, (235, 235, 245), -1)
        vw.write(frame)
    for _ in range(8):
        vw.write(bg.copy())
    vw.release()

    # Calibration points: project the named z=0 reference points to pixels.
    clicks = {}
    for rp in reference_points(cfg):
        p = project([rp.x, rp.y, 0.0], R, C, K)
        if p is not None and 0 <= p[0] < W and 0 <= p[1] < H:
            clicks[rp.name] = [float(p[0]), float(p[1])]
    out_points = Path("data/calibration/synthetic_points.json")
    with out_points.open("w") as fh:
        json.dump(clicks, fh, indent=2)

    print(f"Wrote {out_video} ({LEAD_IN + len(flight) + 8} frames @ {FPS:g} fps)")
    print(f"Wrote {out_points} ({len(clicks)} reference points)")
    print(f"GROUND TRUTH bounce: line x={BOUNCE[0]:.2f} m, length y={BOUNCE[1]:.2f} m")


if __name__ == "__main__":
    main()

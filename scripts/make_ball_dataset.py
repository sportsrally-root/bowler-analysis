"""Generate a YOLO-format cricket-ball detection dataset, no external data needed.

Strategy: composite synthetically-rendered cricket balls (varied colour, size, shading
and — crucially — motion blur) onto REAL background frames sampled from the project's
videos. Using real backgrounds keeps the domain gap small; heavy motion-blur + size
variation teaches the model to find the fast, smeared ball that 30 fps footage produces.
Bounding-box labels are derived exactly from each rendered ball's alpha mask.

Output: data/ball_dataset/{images,labels}/{train,val} + data.yaml
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

OUT = Path("data/ball_dataset")
N_TRAIN = 1400
N_VAL = 200
VIDEOS = ["data/raw/raw.mp4", "data/raw/synthetic.mp4"]
SEED = 13


def sample_backgrounds(max_per_video=200) -> list[np.ndarray]:
    bgs = []
    for v in VIDEOS:
        if not Path(v).exists():
            continue
        cap = cv2.VideoCapture(v)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, max(n - 1, 0), min(max_per_video, max(n, 1))).astype(int)
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if ok:
                bgs.append(fr)
        cap.release()
    return bgs


def render_ball(radius: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Render a shaded cricket ball; return (bgr patch, alpha mask) before blur."""
    size = radius * 2 + 4
    patch = np.zeros((size, size, 3), np.float32)
    alpha = np.zeros((size, size), np.float32)
    c = size // 2
    # Red or white ball.
    if rng.random() < 0.7:
        base = np.array([30, 30, 150], np.float32)   # red (BGR)
    else:
        base = np.array([225, 225, 230], np.float32)  # white
    # Light direction for a simple spherical shading gradient.
    lx, ly = rng.uniform(-1, 1), rng.uniform(-1, 1)
    for y in range(size):
        for x in range(size):
            d = ((x - c) ** 2 + (y - c) ** 2) ** 0.5
            if d <= radius:
                shade = 0.55 + 0.45 * max(0.0, (lx * (c - x) + ly * (c - y)) / (radius + 1e-6))
                patch[y, x] = np.clip(base * shade, 0, 255)
                alpha[y, x] = 1.0
    return patch, alpha


def motion_blur(patch, alpha, length, angle_deg):
    if length <= 1:
        return patch, alpha
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((length / 2, length / 2), angle_deg, 1.0)
    k = cv2.warpAffine(k, M, (length, length))
    s = k.sum()
    k = k / s if s > 0 else k
    pb = cv2.filter2D(patch, -1, k)
    ab = cv2.filter2D(alpha, -1, k)
    return pb, ab


def composite(bg, rng) -> tuple[np.ndarray, list[tuple[float, float, float, float]]]:
    h, w = bg.shape[:2]
    img = bg.copy().astype(np.float32)
    boxes = []
    n_balls = rng.choices([0, 1, 1, 1, 2], k=1)[0]  # mostly 1, some 0 (neg) / 2
    for _ in range(n_balls):
        radius = rng.randint(3, 18)
        patch, alpha = render_ball(radius, rng)
        if rng.random() < 0.75:  # most balls are moving -> motion blur
            patch, alpha = motion_blur(patch, alpha,
                                       rng.randint(4, max(5, radius * 3)),
                                       rng.uniform(0, 180))
        ph, pw = alpha.shape
        # Bias placement toward the central/lower pitch region.
        cx = int(rng.uniform(0.2, 0.8) * w)
        cy = int(rng.uniform(0.25, 0.95) * h)
        x0, y0 = cx - pw // 2, cy - ph // 2
        x1, y1 = x0 + pw, y0 + ph
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            continue
        a = np.clip(alpha, 0, 1)[..., None]
        img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - a) + patch * a
        ys, xs = np.where(alpha > 0.15)
        if len(xs) == 0:
            continue
        bx0, bx1 = x0 + xs.min(), x0 + xs.max()
        by0, by1 = y0 + ys.min(), y0 + ys.max()
        bcx = ((bx0 + bx1) / 2) / w
        bcy = ((by0 + by1) / 2) / h
        bw_ = (bx1 - bx0 + 1) / w
        bh_ = (by1 - by0 + 1) / h
        boxes.append((bcx, bcy, bw_, bh_))
    return np.clip(img, 0, 255).astype(np.uint8), boxes


def main():
    rng = random.Random(SEED)
    bgs = sample_backgrounds()
    if not bgs:
        raise SystemExit("No background frames found — check VIDEOS paths.")
    print(f"{len(bgs)} background frames sampled.")

    for split, count in (("train", N_TRAIN), ("val", N_VAL)):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(count):
            bg = rng.choice(bgs)
            img, boxes = composite(bg, rng)
            stem = f"{split}_{i:05d}"
            cv2.imwrite(str(OUT / "images" / split / f"{stem}.jpg"), img)
            with (OUT / "labels" / split / f"{stem}.txt").open("w") as fh:
                for (cx, cy, bw, bh) in boxes:
                    fh.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        print(f"wrote {count} {split} images")

    with (OUT / "data.yaml").open("w") as fh:
        fh.write(
            f"path: {OUT.resolve()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "nc: 1\n"
            "names: [ball]\n"
        )
    print(f"dataset ready: {OUT}/data.yaml")


if __name__ == "__main__":
    main()

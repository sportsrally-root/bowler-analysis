"""Fine-tune a small YOLO to detect cricket balls, then export to models/ball.pt.

Trains on the dataset built by make_ball_dataset.py (or any YOLO-format data.yaml you
pass). Uses Apple MPS / CUDA if available. Small objects -> larger imgsz helps.

Usage:
    python scripts/train_ball.py [--data data/ball_dataset/data.yaml] [--epochs 40]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ball_dataset/data.yaml")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--base", default="yolo11n.pt")
    ap.add_argument("--out", default="models/ball.pt")
    args = ap.parse_args()

    from ultralytics import YOLO

    device = pick_device()
    print(f"Training on device: {device}")
    model = YOLO(args.base)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        name="ball",
        patience=12,
        # Small fast object: bias augmentation toward scale/translation, keep mosaic.
        scale=0.5,
        translate=0.1,
        fliplr=0.5,
        hsv_v=0.4,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, out)
    print(f"Best weights -> {out}")


if __name__ == "__main__":
    main()

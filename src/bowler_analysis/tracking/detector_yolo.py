"""Learned ball-candidate detector (YOLO via ultralytics).

Unlike background subtraction, a learned detector can pick the ball out of a cluttered
scene (moving bowler/batter, swaying nets) and at lower frame rates — which is exactly
where the classical detector fails. Two modes:

* **COCO baseline** (zero training): a stock YOLO model emits a "sports ball" class
  (COCO id 32). Works with no dataset, but a small/blurred cricket ball is hard for it,
  so we run at low confidence and keep small detections.
* **Custom model**: point ``model_path`` at a cricket-ball-fine-tuned ``.pt`` for real
  robustness. Same interface, so the rest of the pipeline is unchanged.

Shares the ``Candidate`` type and corridor-mask filtering with the classical detector,
so the Kalman tracker downstream is identical.
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from .detector_classical import Candidate

COCO_SPORTS_BALL = 32


class YoloBallDetector:
    """Per-frame ball candidate detector backed by an ultralytics YOLO model."""

    def __init__(
        self,
        cfg: Config,
        corridor_mask: np.ndarray | None = None,
        model_path: str | None = None,
        conf: float | None = None,
        classes: list[int] | None = None,
    ):
        # Imported lazily so the package works without the optional [yolo] extra.
        from ultralytics import YOLO

        self.cfg = cfg
        self.corridor_mask = corridor_mask
        y = cfg.yolo
        self.model = YOLO(model_path or y.model)
        self.conf = conf if conf is not None else y.conf
        # COCO baseline -> only "sports ball"; a custom single-class model -> all classes.
        if classes is not None:
            self.classes = classes
        elif model_path or y.model_is_custom:
            self.classes = None
        else:
            self.classes = [COCO_SPORTS_BALL]

    def detect(self, frame_idx: int, frame_bgr: np.ndarray) -> list[Candidate]:
        t = self.cfg.tracking
        res = self.model.predict(
            frame_bgr, conf=self.conf, classes=self.classes, verbose=False
        )[0]
        out: list[Candidate] = []
        if res.boxes is None:
            return out
        h, w = frame_bgr.shape[:2]
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            bw, bh = (x2 - x1), (y2 - y1)
            area = bw * bh
            if area < t.min_blob_area_px or area > t.max_blob_area_px * 8:
                continue  # YOLO boxes run larger than blobs; relax the upper bound
            if self.corridor_mask is not None:
                ix, iy = int(min(max(cx, 0), w - 1)), int(min(max(cy, 0), h - 1))
                if self.corridor_mask[iy, ix] == 0:
                    continue
            out.append(
                Candidate(frame_idx, float(cx), float(cy),
                          float(max(bw, bh) / 2), float(area))
            )
        return out

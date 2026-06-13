"""End-to-end Phase 1 pipeline: video -> trajectories -> metrics -> outputs.

Each stage writes a JSON artifact under ``data/output/<run_id>/`` so runs are
debuggable and individual stages can be re-run.
"""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from .analysis.line_length import analyze_delivery
from .analysis.segment import filter_deliveries
from .config import Config
from .geometry.calibration import homography_arrays, load_calibration
from .ingest.video_reader import probe_video
from .models.schemas import CalibrationResult, Delivery, ReportData, VideoInfo
from .render.overlay import annotate_video
from .render.report import build_report
from .tracking.detector_classical import ClassicalDetector, build_corridor_mask
from .tracking.tracker import BallTracker
from .ingest.video_reader import read_frames


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(obj, fh, indent=2)


def _make_detector(name: str, cfg: Config, corridor, model_path: str | None):
    if name == "yolo":
        from .tracking.detector_yolo import YoloBallDetector
        return YoloBallDetector(cfg, corridor_mask=corridor, model_path=model_path)
    return ClassicalDetector(cfg, corridor_mask=corridor)


def track_video(
    video: VideoInfo,
    calib: CalibrationResult,
    cfg: Config,
    start: int = 0,
    end: int | None = None,
    multi: bool = False,
    detector_name: str = "classical",
    model_path: str | None = None,
) -> list:
    """Detect + track the ball across frames; return candidate trajectories."""
    _, H_inv = homography_arrays(calib)
    corridor = build_corridor_mask(cfg, H_inv, (video.width, video.height))
    detector = _make_detector(detector_name, cfg, corridor, model_path)
    tracker = BallTracker(cfg)

    total = (end or video.n_frames) - start if (end or video.n_frames) else None
    for idx, frame in tqdm(read_frames(video.path, start=start, end=end),
                           total=total, desc="tracking", unit="f"):
        cands = detector.detect(idx, frame)
        tracker.update(idx, cands)

    if multi:
        return filter_deliveries(tracker.all_trajectories(cfg.quality.min_track_points), cfg)
    traj = tracker.finalize()
    return [traj] if traj.detections else []


def run_pipeline(
    video_path: str,
    calibration_path: str,
    out_dir: str,
    cfg: Config,
    run_id: str,
    fps_override: float | None = None,
    clip: tuple[int, int] | None = None,
    multi: bool = False,
    make_video: bool = True,
    detector_name: str = "classical",
    model_path: str | None = None,
) -> ReportData:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    video = probe_video(video_path, fps_override=fps_override)
    calib = load_calibration(calibration_path)
    _write_json(out / "video_info.json", video.model_dump())

    start, end = (clip if clip else (0, None))
    trajectories = track_video(video, calib, cfg, start=start, end=end, multi=multi,
                               detector_name=detector_name, model_path=model_path)

    deliveries: list[Delivery] = []
    for i, traj in enumerate(trajectories, start=1):
        deliveries.append(analyze_delivery(i, traj, calib, cfg, video.fps))

    _write_json(out / "deliveries.json", [d.model_dump() for d in deliveries])

    report_data = ReportData(run_id=run_id, video=video, calibration=calib,
                             deliveries=deliveries)

    if make_video and deliveries:
        # Annotate the first delivery (full demo); extend to all when multi.
        targets = deliveries if multi else deliveries[:1]
        for d in targets:
            vid_out = str(out / f"annotated_delivery_{d.index}.mp4")
            annotate_video(video_path, vid_out, calib, cfg, d, video.fps)

    build_report(report_data, cfg, out)
    return report_data

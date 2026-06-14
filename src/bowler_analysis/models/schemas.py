"""Serializable data schemas shared across pipeline stages.

Every stage reads/writes these as JSON under ``data/output/<run_id>/`` so stages
can be re-run independently and the pipeline stays debuggable.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class VideoInfo(BaseModel):
    """Metadata extracted from the input video container."""

    path: str
    width: int
    height: int
    n_frames: int
    duration_s: float
    fps: float                       # the fps used by the pipeline (may be overridden)
    container_fps: float             # r_frame_rate reported by ffprobe
    avg_fps: float                   # avg_frame_rate reported by ffprobe
    fps_overridden: bool = False
    slowmo_suspected: bool = False   # container vs avg fps disagree -> investigate


class Detection(BaseModel):
    """A single ball candidate accepted into a trajectory."""

    frame: int
    x: float          # image pixel coords (centre of blob)
    y: float
    radius: float = 0.0
    score: float = 1.0
    interpolated: bool = False  # filled in by the Kalman coast step, not a real detection


class Trajectory(BaseModel):
    """An ordered set of ball detections forming one delivery's flight."""

    detections: list[Detection] = Field(default_factory=list)

    @property
    def frames(self) -> list[int]:
        return [d.frame for d in self.detections]


class CalibrationResult(BaseModel):
    """Image<->pitch-plane homography plus calibration quality."""

    homography: list[list[float]]            # 3x3, image -> world (metres, ground plane)
    homography_inv: list[list[float]]        # 3x3, world -> image
    image_points: list[list[float]]          # clicked pixel points
    world_points: list[list[float]]          # corresponding world points (x, y) metres
    point_names: list[str] = Field(default_factory=list)
    reprojection_error_px: float = 0.0
    image_size: list[int] = Field(default_factory=list)  # [w, h]


class BounceResult(BaseModel):
    frame: float                # sub-frame interpolated bounce frame
    image_xy: list[float]       # bounce point in pixels
    world_xy: list[float]       # bounce point in metres (pitch plane)
    detected: bool = True


class Delivery(BaseModel):
    """All computed metrics for a single delivery."""

    index: int
    trajectory: Trajectory
    bounce: BounceResult | None = None
    length_m: float | None = None      # distance from striker stumps to bounce
    length_label: str | None = None
    line_m: float | None = None        # signed lateral offset (+ = off for handedness)
    line_label: str | None = None
    speed_kph: float | None = None
    speed_uncertainty_kph: float | None = None
    handedness: str = "rhb"
    notes: list[str] = Field(default_factory=list)


class ReportData(BaseModel):
    """Top-level bundle handed to the renderers."""

    run_id: str
    video: VideoInfo
    calibration: CalibrationResult
    deliveries: list[Delivery] = Field(default_factory=list)

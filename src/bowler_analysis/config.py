"""Configuration loading and typed settings.

Loads ``config/default.yaml`` and exposes it as validated pydantic models so the
rest of the pipeline gets autocomplete + validation instead of dict spelunking.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

# Repo root = three parents up from this file (src/bowler_analysis/config.py).
PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


class Geometry(BaseModel):
    pitch_length_m: float = 20.12
    popping_crease_offset_m: float = 1.22
    return_crease_half_width_m: float = 1.32
    stump_height_m: float = 0.711
    wicket_width_m: float = 0.2286


class Zone(BaseModel):
    name: str
    min: float
    max: float
    color: str | None = None


class Batter(BaseModel):
    handedness: str = "rhb"
    rhb_off_sign: int = 1


class Tracking(BaseModel):
    bg_history: int = 200
    bg_var_threshold: float = 25
    min_blob_area_px: float = 4
    max_blob_area_px: float = 1200
    min_circularity: float = 0.45
    corridor_margin_m: float = 1.5
    max_assoc_dist_px: float = 120
    max_missed_frames: int = 12


class Calibration(BaseModel):
    max_reprojection_error_px: float = 8.0


class Quality(BaseModel):
    min_track_points: int = 6


class Yolo(BaseModel):
    model: str = "yolo11n.pt"     # stock COCO model (has a 'sports ball' class)
    model_is_custom: bool = False  # set true when 'model' is a cricket-ball fine-tune
    conf: float = 0.10             # low: a small/blurred cricket ball is low-confidence


class Llm(BaseModel):
    """Settings for the vision-LLM batter analysis path.

    ``backend`` picks the client: ``aws`` (Claude Platform on AWS, default),
    ``bedrock`` (Amazon Bedrock), ``anthropic`` (direct API), or ``databricks``
    (OpenAI-compatible serving endpoint). The first three share the ``anthropic``
    SDK and differ only by client + model-id prefix; ``databricks`` uses the
    ``openai`` client. See ``llm/client.py``.
    """

    backend: str = "nova"              # nova | aws | bedrock | anthropic | databricks
    model: str = "us.amazon.nova-pro-v1:0"  # claude backends: e.g. claude-opus-4-8
    n_frames: int = 9                  # frames sampled around the swing per shot
    max_tokens: int = 4096
    image_long_edge_px: int = 768      # downscale frames before encoding (cost)
    # Databricks-only: serving endpoint name + host (token via env DATABRICKS_TOKEN).
    databricks_endpoint: str | None = None
    databricks_host: str | None = None  # else env DATABRICKS_HOST


class Config(BaseModel):
    geometry: Geometry = Field(default_factory=Geometry)
    length_zones: list[Zone] = Field(default_factory=list)
    line_zones: list[Zone] = Field(default_factory=list)
    full_toss_label: str = "Full toss"
    batter: Batter = Field(default_factory=Batter)
    tracking: Tracking = Field(default_factory=Tracking)
    calibration: Calibration = Field(default_factory=Calibration)
    quality: Quality = Field(default_factory=Quality)
    yolo: Yolo = Field(default_factory=Yolo)
    llm: Llm = Field(default_factory=Llm)


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from YAML, falling back to packaged defaults."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r") as fh:
        data = yaml.safe_load(fh) or {}
    return Config.model_validate(data)

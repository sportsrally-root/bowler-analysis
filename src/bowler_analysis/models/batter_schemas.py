"""Schemas for the LLM-based batter shot analysis path.

``BatterShotAnalysis`` is the structured object the vision model returns — kept to
JSON-schema-friendly types (strings, enums, lists, plain numbers) so it works
both with the Anthropic SDK's ``messages.parse`` and an OpenAI-compatible
``response_format`` (Databricks). ``extra="forbid"`` makes pydantic emit
``additionalProperties: false``, which structured-output backends require.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .schemas import VideoInfo

Rating = Literal["poor", "needs work", "solid", "excellent"]


class ShotDimension(BaseModel):
    """One coaching dimension of the shot, rated and explained."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Technique dimension, e.g. 'Head position', "
                                  "'Front-foot stride', 'Balance', 'Bat-swing path', "
                                  "'Follow-through'.")
    rating: Rating
    observation: str = Field(description="What is visible across the frames that "
                                         "justifies the rating.")


class BatterShotAnalysis(BaseModel):
    """The vision model's structured assessment of one batting shot."""

    model_config = ConfigDict(extra="forbid")

    shot_type: str = Field(description="Specific shot name, e.g. 'straight drive', "
                                       "'cover drive', 'pull', 'cut', 'defensive push'.")
    shot_family: Literal["drive", "cut", "pull/hook", "sweep", "defensive",
                         "leave", "glance/flick", "other"] = Field(
        description="Coarse shot family the shot_type belongs to.")
    confidence: float = Field(description="0..1 confidence in the shot_type call.")
    dimensions: list[ShotDimension] = Field(
        description="Per-dimension technique assessment (aim for 4-6 dimensions).")
    strengths: list[str] = Field(default_factory=list,
                                 description="What the batter did well.")
    faults: list[str] = Field(default_factory=list,
                              description="Technique faults / areas to improve.")
    overall_rating: Rating
    coaching_summary: str = Field(description="2-4 sentence coaching takeaway.")


class BatterReportData(BaseModel):
    """Top-level bundle handed to the batter report renderer."""

    run_id: str
    clip_path: str
    video: VideoInfo
    backend: str
    model: str
    frame_times_s: list[float] = Field(default_factory=list)
    contact_sheet_png: str | None = None
    # Bat-ball contact located from the audio 'knock' (used to anchor frames).
    contact_detected: bool = False
    contact_time_s: float | None = None
    contact_strength: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    analysis: BatterShotAnalysis


class SessionShot(BaseModel):
    """One shot within a session (located by its audio contact)."""

    index: int
    time_s: float                       # contact time in the source video
    contact_strength: float | None = None
    frame_png: str | None = None        # per-shot frames strip
    input_tokens: int = 0
    output_tokens: int = 0
    analysis: BatterShotAnalysis


class SessionReportData(BaseModel):
    """All shots in one session, for the consolidated report."""

    run_id: str
    clip_path: str
    video: VideoInfo
    backend: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    shots: list[SessionShot] = Field(default_factory=list)

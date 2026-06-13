"""Canonical cricket-pitch world model and line/length zone classification.

World coordinate frame (ground plane, metres):
    origin = base of the striker's middle stump
    +y     = down the pitch toward the bowler's end (bowler stumps at y = pitch_length)
    +x     = lateral across the pitch; +x is the OFF side for a right-handed batter
             when ``Batter.rhb_off_sign == 1`` (configurable / flippable).

The reference points below are painted crease/stump features the user clicks during
manual calibration. They form a known rectangle (the return-crease box at each end)
which is excellent geometry for a homography.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Config


@dataclass(frozen=True)
class ReferencePoint:
    name: str
    label: str          # human prompt shown in the calibration tool
    x: float            # world metres
    y: float


def reference_points(cfg: Config) -> list[ReferencePoint]:
    """The ordered set of clickable calibration reference points.

    Uses the return-crease/popping-crease intersections at both ends plus the
    middle-stump bases. The four crease corners alone are enough for a homography;
    the stump bases add accuracy and a centre constraint.
    """
    g = cfg.geometry
    L = g.pitch_length_m
    pc = g.popping_crease_offset_m          # popping crease offset from stumps
    rc = g.return_crease_half_width_m        # return crease half-width
    half = g.wicket_width_m / 2.0

    near_pop = pc                            # striker popping crease at y = +pc
    far_pop = L - pc                         # bowler popping crease

    return [
        ReferencePoint("striker_pop_left", "Striker popping crease — LEFT corner", -rc, near_pop),
        ReferencePoint("striker_pop_right", "Striker popping crease — RIGHT corner", rc, near_pop),
        ReferencePoint("bowler_pop_left", "Bowler popping crease — LEFT corner", -rc, far_pop),
        ReferencePoint("bowler_pop_right", "Bowler popping crease — RIGHT corner", rc, far_pop),
        ReferencePoint("striker_off_stump", "Striker stumps — LEFT (off) stump base", -half, 0.0),
        ReferencePoint("striker_leg_stump", "Striker stumps — RIGHT (leg) stump base", half, 0.0),
        ReferencePoint("bowler_middle_stump", "Bowler stumps — middle stump base", 0.0, L),
    ]


def classify_length(length_m: float, cfg: Config, bounced: bool = True) -> str:
    """Map distance-from-striker-stumps to a length label."""
    if not bounced:
        return cfg.full_toss_label
    for z in cfg.length_zones:
        if z.min <= length_m < z.max:
            return z.name
    # Beyond the last band (shouldn't happen within pitch) -> nearest extreme.
    if cfg.length_zones and length_m >= cfg.length_zones[-1].max:
        return cfg.length_zones[-1].name
    return "Unknown"


def signed_line_x(world_x: float, cfg: Config) -> float:
    """Convert raw world x to a handedness-aware signed line offset.

    Positive result == OFF side for the configured batter.
    """
    sign = cfg.batter.rhb_off_sign
    if cfg.batter.handedness.lower() == "lhb":
        sign = -sign
    return world_x * sign


def classify_line(world_x: float, cfg: Config) -> str:
    """Map a bounce's world x to a line label (off/leg aware)."""
    sx = signed_line_x(world_x, cfg)
    for z in cfg.line_zones:
        if z.min <= sx < z.max:
            return z.name
    return "Unknown"

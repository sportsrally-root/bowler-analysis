"""Top-down schematic pitch map (matplotlib) for the PDF report.

Length runs up the page (distance from the striker's stumps); line runs across.
Length-zone bands are shaded with the configured colours and each bounce is plotted
as a dot coloured by its length zone.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless / no display required
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from ..config import Config  # noqa: E402
from ..models.schemas import Delivery  # noqa: E402


def _zone_color(label: str | None, cfg: Config, default="#888888") -> str:
    for z in cfg.length_zones:
        if z.name == label:
            return z.color or default
    return default


def draw_pitch_map(deliveries: list[Delivery], cfg: Config, out_path: str,
                   title: str = "Pitch map — line & length") -> str:
    g = cfg.geometry
    rc = g.return_crease_half_width_m
    max_len = min(g.pitch_length_m, 12.0)  # only the batsman's half matters for L&L

    fig, ax = plt.subplots(figsize=(4.5, 7.5))

    # Length zone bands.
    for z in cfg.length_zones:
        if z.min >= max_len:
            continue
        top = min(z.max, max_len)
        ax.axhspan(z.min, top, facecolor=z.color or "#cccccc", alpha=0.35, zorder=0)
        ax.text(-rc + 0.05, (z.min + top) / 2, z.name, va="center", ha="left",
                fontsize=8, color="#222222")

    # Pitch rails + stumps.
    ax.axvline(-rc, color="#666", lw=1)
    ax.axvline(rc, color="#666", lw=1)
    half = g.wicket_width_m / 2
    for sx in (-half, 0, half):
        ax.plot([sx], [0], marker="|", markersize=14, color="black")
    ax.axhline(0, color="black", lw=1.5)
    ax.text(0, -0.4, "Striker stumps", ha="center", va="top", fontsize=8)

    # Bounce dots, coloured by length zone.
    for d in deliveries:
        if d.bounce is None or d.line_m is None or d.length_m is None:
            continue
        # line_m is signed (+ = off). Plot directly; off side to the right.
        ax.scatter(d.line_m, d.length_m, s=70,
                   color=_zone_color(d.length_label, cfg),
                   edgecolors="black", linewidths=0.6, zorder=5)
        ax.annotate(str(d.index), (d.line_m, d.length_m), fontsize=6,
                    ha="center", va="center", color="white", zorder=6)

    ax.set_xlim(-rc - 0.2, rc + 0.2)
    ax.set_ylim(max_len, -0.8)  # near (striker) at top, away down the page
    ax.set_xlabel("Line  (- leg   |   off +)  metres")
    ax.set_ylabel("Length from striker stumps (m)")
    ax.set_title(title, fontsize=11)
    ax.set_aspect("auto")

    handles = [mpatches.Patch(color=z.color or "#ccc", label=z.name, alpha=0.5)
               for z in cfg.length_zones if z.min < max_len]
    ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def draw_distributions(deliveries: list[Delivery], cfg: Config, out_path: str) -> str:
    """Length histogram, line bar chart, and speed-by-delivery chart in one figure."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))

    # Length distribution by zone.
    len_labels = [z.name for z in cfg.length_zones]
    len_counts = {n: 0 for n in len_labels}
    for d in deliveries:
        if d.length_label in len_counts:
            len_counts[d.length_label] += 1
    axes[0].bar(len_labels, [len_counts[n] for n in len_labels],
                color=[z.color or "#888" for z in cfg.length_zones])
    axes[0].set_title("Length distribution")
    axes[0].tick_params(axis="x", rotation=45, labelsize=7)

    # Line distribution by zone.
    line_labels = [z.name for z in cfg.line_zones]
    line_counts = {n: 0 for n in line_labels}
    for d in deliveries:
        if d.line_label in line_counts:
            line_counts[d.line_label] += 1
    axes[1].bar(line_labels, [line_counts[n] for n in line_labels], color="#3498db")
    axes[1].set_title("Line distribution")
    axes[1].tick_params(axis="x", rotation=45, labelsize=7)

    # Speed per delivery.
    idxs = [d.index for d in deliveries if d.speed_kph is not None]
    spds = [d.speed_kph for d in deliveries if d.speed_kph is not None]
    if spds:
        axes[2].plot(idxs, spds, marker="o", color="#e74c3c")
        axes[2].set_ylim(0, max(spds) * 1.2)
    axes[2].set_title("Speed by delivery (km/h)")
    axes[2].set_xlabel("Delivery #")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

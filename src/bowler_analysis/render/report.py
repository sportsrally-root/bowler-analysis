"""Assemble the PDF report: matplotlib charts + reportlab layout (no system deps)."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ..config import Config
from ..models.schemas import ReportData
from .pitchmap import draw_distributions, draw_pitch_map


def build_report(data: ReportData, cfg: Config, out_dir: str | Path) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pitch_png = str(out_dir / "pitch_map.png")
    dist_png = str(out_dir / "distributions.png")
    draw_pitch_map(data.deliveries, cfg, pitch_png)
    draw_distributions(data.deliveries, cfg, dist_png)

    pdf_path = str(out_dir / "report.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                            leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    styles = getSampleStyleSheet()
    story = []

    v = data.video
    story.append(Paragraph("Bowler Performance Report", styles["Title"]))
    story.append(Paragraph(f"Run: {data.run_id}", styles["Normal"]))
    story.append(Spacer(1, 0.3 * cm))

    # Header / provenance block.
    cal_err = data.calibration.reprojection_error_px
    cal_flag = "OK" if cal_err <= cfg.calibration.max_reprojection_error_px else "HIGH — recalibrate"
    fps_note = " (slow-mo suspected)" if v.slowmo_suspected else ""
    header_rows = [
        ["Video", Path(v.path).name],
        ["Resolution", f"{v.width} x {v.height}"],
        ["fps used", f"{v.fps:g}{' (overridden)' if v.fps_overridden else ''}{fps_note}"],
        ["Container / avg fps", f"{v.container_fps:g} / {v.avg_fps:g}"],
        ["Batter", cfg.batter.handedness.upper()],
        ["Calibration error", f"{cal_err:.2f} px  [{cal_flag}]"],
        ["Deliveries analysed", str(len(data.deliveries))],
    ]
    t = Table(header_rows, colWidths=[5 * cm, 11 * cm])
    t.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.4 * cm))

    # Pitch map.
    story.append(Paragraph("Pitch map (line &amp; length)", styles["Heading2"]))
    story.append(Image(pitch_png, width=9 * cm, height=15 * cm))
    story.append(Spacer(1, 0.3 * cm))

    # Per-delivery table.
    story.append(Paragraph("Per-delivery metrics", styles["Heading2"]))
    rows = [["#", "Speed (km/h)", "Length", "Len (m)", "Line", "Notes"]]
    for d in data.deliveries:
        spd = f"{d.speed_kph:.0f} ± {d.speed_uncertainty_kph:.0f}" if d.speed_kph else "—"
        rows.append([
            str(d.index), spd, d.length_label or "—",
            f"{d.length_m:.1f}" if d.length_m is not None else "—",
            d.line_label or "—",
            "; ".join(d.notes) if d.notes else "",
        ])
    dt = Table(rows, colWidths=[1 * cm, 3 * cm, 3 * cm, 1.8 * cm, 3 * cm, 4.2 * cm])
    dt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.whitesmoke]),
    ]))
    story.append(dt)
    story.append(Spacer(1, 0.4 * cm))

    # Distributions.
    story.append(Paragraph("Distributions", styles["Heading2"]))
    story.append(Image(dist_png, width=17 * cm, height=5.1 * cm))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph(
        "<i>Notes: line &amp; length are measured at the ball's bounce via a "
        "ground-plane homography from the clicked stump/crease references. Speed is "
        "an estimate from ground-projected flight and is sensitive to the frame rate "
        "(confirm fps for slow-mo footage). Biomechanics are not included in this "
        "Phase 1 report.</i>", styles["Normal"]))

    doc.build(story)
    return pdf_path

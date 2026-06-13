"""Command-line interface for bowler-analysis.

Commands:
  info       Print video metadata (fps, slow-mo flag, resolution, frames).
  calibrate  Click pitch references -> homography (saved to JSON) + QA overlay.
  run        Full Phase 1 pipeline: track -> line/length/speed -> video + PDF.
  batter     Vision-LLM analysis of a single batting shot -> coaching PDF + JSON.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import typer

from .calibrate.manual import (
    build_calibration,
    collect_clicks_interactive,
    points_from_file,
    render_qa_overlay,
)
from .config import load_config
from .geometry.calibration import save_calibration
from .ingest.video_reader import probe_video, read_frame_at

app = typer.Typer(add_completion=False, help="Cricket bowler analysis from video.")


@app.command()
def info(video: str, fps_override: Optional[float] = typer.Option(None, "--fps")):
    """Print video metadata."""
    v = probe_video(video, fps_override=fps_override)
    typer.echo(f"Path:        {v.path}")
    typer.echo(f"Resolution:  {v.width} x {v.height}")
    typer.echo(f"Frames:      {v.n_frames}")
    typer.echo(f"Duration:    {v.duration_s:.2f} s")
    typer.echo(f"fps (used):  {v.fps:g}{'  [overridden]' if v.fps_overridden else ''}")
    typer.echo(f"container fps: {v.container_fps:g}   avg fps: {v.avg_fps:g}")
    if v.slowmo_suspected:
        typer.secho("  ! container/avg fps disagree — check for slow-mo; "
                    "use --fps to set the true capture rate.", fg="yellow")


@app.command()
def calibrate(
    video: str,
    out: str = typer.Option(..., "--out", help="Path to write calibration JSON."),
    frame: int = typer.Option(0, "--frame", help="Frame index to calibrate on."),
    points_file: Optional[str] = typer.Option(
        None, "--points-file", help="JSON of clicked points {name:[x,y]} (headless)."),
    qa_out: Optional[str] = typer.Option(None, "--qa-out", help="Write QA overlay PNG."),
    config: Optional[str] = typer.Option(None, "--config"),
):
    """Calibrate the image<->pitch homography from clicked references."""
    cfg = load_config(config)
    img = read_frame_at(video, frame)
    if img is None:
        raise typer.BadParameter(f"Could not read frame {frame} from {video}")
    h, w = img.shape[:2]

    clicks = (points_from_file(points_file) if points_file
              else collect_clicks_interactive(img, cfg))
    if not clicks:
        typer.secho("Calibration cancelled / no points.", fg="red")
        raise typer.Exit(1)

    result = build_calibration(clicks, cfg, (w, h))
    save_calibration(result, out)
    color = "green" if result.reprojection_error_px <= cfg.calibration.max_reprojection_error_px else "yellow"
    typer.secho(f"Saved calibration -> {out}", fg="green")
    typer.secho(f"Reprojection error: {result.reprojection_error_px:.2f} px", fg=color)

    if qa_out:
        cv2.imwrite(qa_out, render_qa_overlay(img, result, cfg))
        typer.echo(f"QA overlay -> {qa_out}")


@app.command()
def run(
    video: str,
    calibration: str = typer.Option(..., "--calibration", help="Calibration JSON path."),
    out: str = typer.Option("data/output/run", "--out", help="Output directory."),
    fps_override: Optional[float] = typer.Option(None, "--fps"),
    clip: Optional[str] = typer.Option(None, "--clip", help="start:end frame range."),
    handedness: Optional[str] = typer.Option(None, "--batter", help="rhb | lhb"),
    multi: bool = typer.Option(False, "--multi", help="Detect multiple deliveries."),
    no_video: bool = typer.Option(False, "--no-video", help="Skip annotated video."),
    detector: str = typer.Option("classical", "--detector",
                                 help="Ball detector: classical | yolo"),
    model: Optional[str] = typer.Option(None, "--model",
                                        help="Path to a YOLO .pt (custom ball model)."),
    config: Optional[str] = typer.Option(None, "--config"),
):
    """Run the full Phase 1 analysis and write the report."""
    from .pipeline import run_pipeline

    cfg = load_config(config)
    if handedness:
        cfg.batter.handedness = handedness.lower()

    clip_range = None
    if clip:
        a, b = clip.split(":")
        clip_range = (int(a), int(b))

    run_id = Path(out).name
    data = run_pipeline(
        video_path=video, calibration_path=calibration, out_dir=out, cfg=cfg,
        run_id=run_id, fps_override=fps_override, clip=clip_range, multi=multi,
        make_video=not no_video, detector_name=detector, model_path=model,
    )

    typer.secho(f"\nAnalysed {len(data.deliveries)} delivery(ies).", fg="green")
    for d in data.deliveries:
        spd = f"{d.speed_kph:.0f} km/h" if d.speed_kph else "n/a"
        typer.echo(f"  #{d.index}: {d.length_label or '?'} / {d.line_label or '?'}  "
                   f"({d.length_m if d.length_m is not None else '?'} m)  speed {spd}")
    typer.secho(f"Report + outputs in: {out}", fg="green")


@app.command()
def batter(
    video: str,
    out: str = typer.Option("data/output/batter", "--out", help="Output directory."),
    clip: Optional[str] = typer.Option(
        None, "--clip", help="Scope to a START:END second window (e.g. 7.5:11.5)."),
    backend: Optional[str] = typer.Option(
        None, "--backend", help="nova | aws | bedrock | anthropic | databricks (overrides config)."),
    model: Optional[str] = typer.Option(None, "--model", help="Claude model id."),
    config: Optional[str] = typer.Option(None, "--config"),
):
    """Analyse a single batting-shot clip with a vision LLM -> coaching PDF + JSON."""
    from .analysis.shot_llm import analyze_clip
    from .render.batter_report import build_batter_report

    cfg = load_config(config)
    if backend:
        cfg.llm.backend = backend.lower()
    if model:
        cfg.llm.model = model

    clip_s = None
    if clip:
        a, b = clip.split(":")
        clip_s = (float(a), float(b))

    run_id = Path(out).name
    typer.echo(f"Analysing shot with {cfg.llm.model} ({cfg.llm.backend}) ...")
    try:
        data = analyze_clip(video, cfg, run_id=run_id, out_dir=out, clip_s=clip_s)
    except Exception as exc:  # surface a clean prerequisite hint, not a traceback
        hints = {
            "nova": "Amazon Nova on Bedrock needs AWS credentials + a region (AWS_REGION, "
                    "default us-east-1) and Bedrock model access to amazon.nova-* models.",
            "aws": "Claude Platform on AWS needs AWS_REGION + ANTHROPIC_AWS_WORKSPACE_ID "
                   "(plus AWS credentials on the standard chain).",
            "bedrock": "Amazon Bedrock needs AWS credentials + AWS_REGION.",
            "anthropic": "Direct Anthropic needs ANTHROPIC_API_KEY.",
            "databricks": "Databricks needs DATABRICKS_HOST + DATABRICKS_TOKEN and "
                          "llm.databricks_endpoint (the serving endpoint name).",
        }
        typer.secho(f"\nLLM call failed: {exc}", fg="red")
        typer.secho(hints.get(cfg.llm.backend, ""), fg="yellow")
        raise typer.Exit(1)
    pdf = build_batter_report(data, out)

    a = data.analysis
    typer.secho(f"\n{a.shot_type}  [{a.shot_family}]  "
                f"({a.confidence:.0%} confidence) — overall: {a.overall_rating}",
                fg="green")
    if data.contact_detected:
        typer.echo(f"  bat-ball contact (audio): ~{data.contact_time_s:.2f}s")
    for d in a.dimensions:
        typer.echo(f"  {d.name}: {d.rating}")
    typer.secho(f"Report: {pdf}", fg="green")


if __name__ == "__main__":
    app()

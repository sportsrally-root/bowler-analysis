"""Segment a single-camera batting clip into one trimmed clip per shot.

The input is a continuous, static-camera video of a batter playing the SAME shot
repeatedly (e.g. a net session of straight drives). We find each shot by its
*downswing*: the bat accelerates through contact, producing a sharp, short-lived
burst of frame-to-frame motion. Background subtraction-style motion energy with a
high-pass ("spikiness") filter isolates those bursts from the slower body movement
of the stance/follow-through, which keeps us from splitting on resets or footwork.

Each detected swing becomes a clip spanning [swing - PRE, swing + POST] so the clip
contains backlift -> contact -> follow-through. A CSV manifest (clip, label, start,
end) is written alongside for building a labelled shot-classification dataset.

Usage:
    python scripts/segment_shots.py INPUT.mp4 --label straight_drive \
        --out data/batter_shots/straight_drive
Tunables (--pre/--post/--min-gap/--height) let you adapt to faster/slower cadences.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

# Swing detection lives in the package so the LLM analyzer can reuse it.
from bowler_analysis.analysis.swing import find_swings, motion_energy


def cut_clip(video: str, start_s: float, dur_s: float, out: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{start_s:.3f}", "-i", video,
         "-t", f"{dur_s:.3f}", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
         "-pix_fmt", "yuv420p", "-an", str(out)],
        check=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--label", required=True, help="Shot type, used for filenames + manifest.")
    ap.add_argument("--out", required=True, help="Output directory for clips + manifest.")
    ap.add_argument("--pre", type=float, default=0.5, help="Seconds before swing to include.")
    ap.add_argument("--post", type=float, default=1.2, help="Seconds after swing to include.")
    ap.add_argument("--min-gap", type=float, default=1.3, help="Min seconds between swings.")
    ap.add_argument("--height", type=float, default=0.30, help="Swing-detection sensitivity 0-1.")
    args = ap.parse_args()

    energy, fps = motion_energy(args.video)
    swings = find_swings(energy, fps, args.min_gap, args.height)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = (len(energy) + 1) / fps

    print(f"{len(swings)} swing(s) detected at: "
          f"{', '.join(f'{s / fps:.2f}s' for s in swings)}")

    rows = []
    for i, s in enumerate(swings, 1):
        start = max(0.0, s / fps - args.pre)
        end = min(duration, s / fps + args.post)
        name = f"{args.label}_{Path(args.video).stem}_shot{i:02d}.mp4"
        cut_clip(args.video, start, end - start, out_dir / name)
        rows.append({"clip": name, "label": args.label,
                     "start_s": f"{start:.2f}", "end_s": f"{end:.2f}"})
        print(f"  shot {i:02d}: {start:.2f}-{end:.2f}s -> {name}")

    manifest = out_dir / "manifest.csv"
    write_header = not manifest.exists()
    with manifest.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["clip", "label", "start_s", "end_s"])
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"Manifest -> {manifest}")


if __name__ == "__main__":
    main()

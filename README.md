# Bowler Analysis

Cricket **bowler performance analysis** from a single phone-camera video — in the
style of [fulltrack.ai](https://www.fulltrack.ai). Phase 1 delivers **line & length**
(pitch map), **bounce point**, and a **speed** estimate, rendered as an **annotated
video** + **PDF report**. Phase 2 (action & biomechanics) is planned.

## What it produces

- **Top-down pitch map** with length zones (Yorker / Full / Good / Back of length /
  Short) and bounce dots coloured by zone.
- **Perspective "broadcast-style" overlay** drawn onto the real pitch in the video
  (length zones in 3D perspective + ball trail + bounce marker + HUD).
- **PDF report**: provenance header (fps, calibration quality), pitch map,
  per-delivery metrics table (speed, length, line), and distribution charts.
- **JSON artifacts** for every stage under `data/output/<run_id>/`.

## How it works

1. **Calibration** — you click known pitch features (return-crease corners, stump
   bases) on one frame; this solves an image↔pitch-plane **homography**. Because the
   tripod is static, you calibrate once per video.
2. **Tracking** — MOG2 background subtraction + blob filtering + a Kalman tracker
   stitch the ball into a flight path, constrained to the pitch corridor.
3. **Bounce** — the bounce is found from the *deceleration knee* of the ball's
   ground-projected down-pitch distance (robust to camera angle), and mapped to
   accurate world coordinates (the ball is on the ground plane only at the bounce).
4. **Line & length / speed** — the bounce world position gives line & length; speed is
   estimated from the release→bounce flight (height-robust).

## Setup

Requires **Python 3.10–3.12** (3.12 recommended; MediaPipe/3.14 wheels lag) and
`ffmpeg`/`ffprobe` on PATH.

```bash
pyenv local 3.12.9                 # or any 3.10–3.12
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

## Usage

```bash
# 1. Inspect the video (confirm fps — critical for speed; slow-mo is flagged)
bowler-analyze info data/raw/my_clip.mp4

# 2. Calibrate (opens a window; click the prompted points, ENTER when done)
bowler-analyze calibrate data/raw/my_clip.mp4 --out data/calibration/my_clip.json \
    --qa-out data/output/qa.png          # QA overlay to sanity-check alignment

# 3. Run the analysis
bowler-analyze run data/raw/my_clip.mp4 --calibration data/calibration/my_clip.json \
    --out data/output/my_run --batter rhb
#   --fps 120        override the capture fps (for slow-mo)
#   --clip 30:120     analyse only frames [30,120)
#   --multi           detect multiple deliveries in one video
#   --no-video        skip the annotated video (faster)
#   --detector yolo   use the learned detector (needs a cricket-ball model, see below)
#   --model ball.pt   path to a cricket-ball fine-tuned YOLO .pt
```

### Ball detectors

- **`classical`** (default): background subtraction + blobs. Best on **high-fps**
  (120–240) footage with a static camera and little other motion.
- **`yolo`**: a learned detector for cluttered / lower-fps clips. The stock COCO model
  does **not** detect cricket balls (too small) — you must supply a cricket-ball
  fine-tuned model via `--model` (or set `yolo.model` + `yolo.model_is_custom: true`
  in config). With that, `--detector yolo` picks the ball out of bowler/net motion.

Outputs land in `data/output/my_run/`: `report.pdf`, `annotated_delivery_*.mp4`,
`pitch_map.png`, `distributions.png`, and the stage JSONs.

### Headless calibration (no GUI)

Provide pre-recorded clicks as JSON (`{"striker_pop_left": [x, y], ...}`; point names
are in `geometry/pitch_model.py`):

```bash
bowler-analyze calibrate clip.mp4 --out cal.json --points-file points.json
```

## Recommended recording protocol

For best results (and to keep the analysis honest):

- **Static tripod**, taller than ~150 cm, **directly behind the stumps, down the
  pitch** — the camera must not move (calibration is per-video).
- **Highest frame rate** available (120/240 fps slow-mo) at **≥1080p**. Confirm the
  true capture fps — a wrong fps scales the speed estimate linearly.
- **Even lighting**; both sets of stumps and the crease markings fully visible and
  unobstructed.
- **One delivery per clip** for the MVP (or use `--clip` / `--multi`).
- Record ~2 s of the static scene before bowling so the background model settles.

## Accuracy caveats

- **Line & length** are measured at the bounce via the ground homography — the most
  reliable outputs.
- **Speed** is an estimate from a single camera (no true 3D); it assumes the ball is
  tracked from near release and is sensitive to fps. Treat it as indicative ± a band.
- 3D ball track, beehive, and biomechanics need full camera calibration / side-on
  footage and are out of scope for Phase 1.

## Development

```bash
.venv/bin/python -m pytest             # unit tests (geometry, speed, line/length)
.venv/bin/python scripts/make_synthetic.py   # generate a synthetic test clip
```

`scripts/make_synthetic.py` renders a synthetic delivery with known ground-truth
bounce/speed, used to validate the whole pipeline end-to-end without real footage.

## Project layout

```
src/bowler_analysis/
  geometry/    pitch model, homography calibration, projection (pure, tested)
  ingest/      video metadata (ffprobe fps) + frame access
  calibrate/   interactive manual calibration tool
  tracking/    classical detector + Kalman tracker + trajectory utils
  analysis/    bounce, line/length, speed, segmentation
  render/      pitch map, perspective overlay, PDF report
  pipeline.py  end-to-end orchestration
  cli.py       `bowler-analyze` commands: info / calibrate / run
```

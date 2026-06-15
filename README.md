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
  > Note: run stock COCO and it will report a "sports ball" — but on real footage that
  > is usually a **false positive** (a stationary round object such as a helmet), not
  > the delivery. A fine-tuned model is mandatory for the YOLO path to mean anything.

Outputs land in `data/output/my_run/`: `report.pdf`, `annotated_delivery_*.mp4`,
`pitch_map.png`, `distributions.png`, and the stage JSONs.

### Headless calibration (no GUI)

Provide pre-recorded clicks as JSON (`{"striker_pop_left": [x, y], ...}`; point names
are in `geometry/pitch_model.py`):

```bash
bowler-analyze calibrate clip.mp4 --out cal.json --points-file points.json
```

## Batter shot analysis (vision LLM)

`bowler-analyze batter` analyses a **single batting-shot clip** with a Claude
vision model — it classifies the shot and writes a coaching report (PDF + JSON),
**zero-shot, no training data**. Use it on clean, single-batter clips (e.g. the
per-shot files from `scripts/segment_shots.py`).

```bash
pip install -e ".[llm]"            # anthropic[aws] (+ boto3 for the Nova/Bedrock path)
export AWS_REGION=us-east-1         # default backend: Amazon Nova on Bedrock
bowler-analyze batter data/batter_shots/straight_drive/clip_shot01.mp4 --out data/output/shot01
#   --backend nova|aws|bedrock|anthropic|databricks   override the configured backend
#   --model us.amazon.nova-pro-v1:0                    override the model
```

It samples frames around the swing (reusing the motion detector), sends them in
one structured call, and writes `report.pdf`, `analysis.json`, and a `frames.png`
contact sheet. Backends are config-driven (`llm:` in `config/default.yaml`):

| Backend | Model | Notes |
|---|---|---|
| `nova` (default) | `us.amazon.nova-pro-v1:0` | Amazon's first-party multimodal model on Bedrock — **no third-party Marketplace subscription** needed, works on a plain Bedrock-enabled AWS account. Uses boto3 `converse`. |
| `aws` | `claude-opus-4-8` | Claude Platform on AWS (`AnthropicAWS`); needs `ANTHROPIC_AWS_WORKSPACE_ID` + a completed Marketplace subscription. Best quality. |
| `bedrock` | `us.anthropic.claude-opus-4-8` | Claude on Bedrock; needs the Anthropic Marketplace subscription (valid AWS payment instrument). |
| `anthropic` | `claude-opus-4-8` | Direct Claude API (`ANTHROPIC_API_KEY`), bypasses AWS. |
| `databricks` | endpoint name | OpenAI-compatible serving endpoint (`DATABRICKS_HOST`/`DATABRICKS_TOKEN`). |

Structured output degrades gracefully to prompt-based JSON on backends without
native support (Nova, Databricks).

### Whole-session report (`session`)

`bowler-analyze session` turns a **full net video** (many shots) into **one
consolidated PDF**. It runs three stages — only the last uses the LLM:

1. **Shot detection (no LLM):** every bat-ball **knock** is found from the audio
   (high-pass > 2 kHz → onset peaks, `analysis/audio_contact.py`). Each knock is a
   shot. Free, local, deterministic. (Motion-based detection exists but is *not*
   used here — on real footage it latched onto the batter walking past the camera.)
2. **Frame sampling (no LLM):** ~9 frames are grabbed around each contact.
3. **Shot analysis (LLM, one call per shot):** the frames go to the configured
   backend → shot type + technique.

```bash
export AWS_REGION=us-east-1                          # default backend: Nova on Bedrock
bowler-analyze session data/raw/<session>.mov --out data/output/session
#   --max-shots N     cap to the N strongest knocks (default: all)
#   --backend / --model   same overrides as `batter` (e.g. --backend anthropic for Claude)
```

Outputs in the run dir: **`session_report.pdf`** (summary + per-shot breakdown),
`session.json` (per-shot analysis + per-shot/total **token counts and an estimated
cost**), and a frame strip per shot. Token + cost totals are also printed and shown
on the report.

> Detection caveat: audio finds *contacts*, so it misses leaves / play-and-misses
> and can false-positive on bat taps or ball-into-net. Needs clean, audible knocks
> (great for net sessions, poor for commentary-heavy broadcast).

See **`docs/running-batter-analysis.md`** for the full runbook (prerequisites, AWS
setup, troubleshooting, switching to Claude).

### Extract per-shot clips & frames (no LLM, no cost)

`scripts/extract_shots.py` runs only the detection + extraction stages: it finds
each shot from the audio knock and writes a **trimmed clip** (the exact shot) and
its **frames** per shot — handy for building a labelled dataset or reviewing shots
without paying for the model.

```bash
.venv/bin/python scripts/extract_shots.py data/raw/<session>.mp4 --out data/shots/myvid
#   --frames 9   frames per shot   --max-shots N   cap to N strongest
#   --no-clips   frames only        --no-frames    clips only
#   --pre/--post window, --min-gap/--height detector knobs
```

Produces `clips/shot_NN.mp4`, `frames/shot_NN/fNN.jpg` (+ a `shot_NN_strip.jpg`),
and `manifest.csv` (shot, time, knock strength, clip).

## Getting a clip

Any `.mp4`/`.mov` works. To pull one from a URL:

```bash
.venv/bin/python -m pip install yt-dlp
.venv/bin/yt-dlp -f "bv*+ba/b" --merge-output-format mp4 \
    -o "data/raw/%(title).60s.%(ext)s" "<video-url>"
```

`data/raw/`, `data/output/` and model weights are git-ignored, so downloaded clips
and run outputs stay local.

## Working with match / broadcast footage

The pipeline assumes **one continuous clip from a single static camera, calibrated
once**. Edited highlights — a montage that cuts between deliveries, camera angles,
zoom levels, or matches — **break this** and will produce meaningless numbers: a
single homography is only valid for one fixed camera, and the tracker jumps across
cuts. Telltale signs the source is unsuitable as-is: a title/intro animation, a baked-in
watermark, a "CAM N"/scoreboard overlay, or aspect-ratio changes mid-video.

If you only have such a video, isolate **one delivery from one fixed camera** first:

```bash
# 1. Find hard cuts (scene changes) to spot continuous segments
ffmpeg -i data/raw/montage.mp4 -vf "select='gt(scene,0.25)',metadata=print:file=-" \
    -an -f null -                                    # prints pts_time of each cut

# 2. Cut out a single delivery (~2 s lead-in lets the bg model settle), keep resolution
ffmpeg -ss 53 -i data/raw/montage.mp4 -t 5.5 -c:v libx264 -crf 18 -an \
    data/raw/one_delivery.mp4

# 3. Calibrate + run on that clip as usual (headless calibration works on a still frame
#    if a GUI is unavailable — read crease/stump pixels off a frame you crop & zoom).
```

Even then, expect trouble: broadcast frames are **low-res, motion-blurred and
watermarked**, and the ball is only a few pixels across moving away from a wide camera.
Both detectors struggle (`classical` tends to latch onto the bowler's body; stock
`yolo` onto static round false positives). Reliable line/length/speed needs footage
that follows the protocol below — not match/TV footage.

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

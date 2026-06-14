# Running the batter shot analysis

A step-by-step runbook for turning a batting video into a coaching report using
the vision-LLM path (default backend: **Amazon Nova Pro on Bedrock**, which needs
no third-party Marketplace subscription).

---

## 1. Prerequisites

- **Python 3.10–3.12** and **ffmpeg/ffprobe** on `PATH`.
- An **AWS account** with **Bedrock model access** to the Amazon Nova models in
  **us-east-1** (Console → Bedrock → *Model access* → enable `amazon.nova-*`).
  Nova is Amazon-first-party, so it does **not** require a Marketplace subscription
  or a verified payment instrument (unlike Claude on Bedrock).
- **AWS credentials** on the standard chain (`aws configure`, SSO, or a role).
  Verify with `aws sts get-caller-identity`.

## 2. One-time setup

```bash
cd bowler-analysis
python -m venv .venv
.venv/bin/python -m pip install -e ".[llm]"     # installs anthropic[aws] + boto3
export AWS_REGION=us-east-1                       # Nova runs here; REQUIRED
```

Default model/backend live in `config/default.yaml` (`llm.backend: nova`,
`llm.model: us.amazon.nova-pro-v1:0`).

## 3. Get a clip

Best input: **your own recording** — single batter, **side-on**, clear, framed so
the batter is reasonably large, and **with audio** (the bat-ball "knock" is what
the session mode uses to locate shots). To pull from a URL:

```bash
.venv/bin/yt-dlp -f "bv*+ba/b" --merge-output-format mp4 -o "data/raw/%(id)s.%(ext)s" "<url>"
.venv/bin/bowler-analyze info data/raw/<file>        # check fps / resolution
```

## 4a. Analyse a single shot

```bash
.venv/bin/bowler-analyze batter data/raw/<clip>.mp4 --out data/output/shot01
#   --clip 7.5:11.5   scope to the shot window (recommended for long/broadcast clips)
#   --backend / --model   override the configured backend/model
```

Outputs in `data/output/shot01/`: **`report.pdf`**, **`analysis.json`**,
`frames.png` (the sampled backlift→contact→follow-through strip).

## 4b. Analyse a whole session → one PDF

```bash
.venv/bin/bowler-analyze session data/raw/<session>.mov --out data/output/session
#   --max-shots N    cap to the N strongest shots (default: all)
```

It detects **every bat-ball contact from the audio knock**, analyses each, and
writes a single **`session_report.pdf`** (summary + per-shot breakdown) plus
`session.json` and per-shot frame strips. One Nova call per shot.

## 5. Footage tips

- **Side-on, single batter, clean audio.** The session locator depends on the
  knock — clean net audio (no commentary) works best.
- **Untrimmed / broadcast clips:** scope with `--clip START:END`; motion-based
  shot finding is unreliable when the batter walks past the camera.
- Portrait phone videos (iPhone `.MOV`) work — rotation is handled.

## 6. Verify it ran on AWS / check cost

Everything runs in **us-east-1**. By default Bedrock does **not** log request
bodies; enable *Bedrock → Settings → Model invocation logging* if you want them.
Invocation **counts** are always in CloudWatch:

```bash
aws cloudwatch get-metric-statistics --namespace AWS/Bedrock --metric-name Invocations \
  --dimensions Name=ModelId,Value=us.amazon.nova-pro-v1:0 \
  --start-time "$(date -u -v-12H +%Y-%m-%dT%H:%M:%SZ)" --end-time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --period 86400 --statistics Sum --region us-east-1
```

Cost shows in Cost Explorer with a few hours' lag.

## 7. Switching to Claude (optional, better quality)

Claude backends need the Anthropic **Marketplace subscription** (a valid AWS
payment instrument) — see the README backend table.

```bash
# Claude on Bedrock (once the Anthropic Marketplace subscription is active):
bowler-analyze batter <clip> --out <dir> --backend bedrock --model us.anthropic.claude-opus-4-8
# Direct Anthropic API (bypasses AWS; needs ANTHROPIC_API_KEY):
bowler-analyze batter <clip> --out <dir> --backend anthropic --model claude-opus-4-8
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `No AWS region was provided` | `export AWS_REGION=us-east-1`. |
| `AccessDenied … marked as Legacy` | That Nova tier is legacy; use `us.amazon.nova-pro-v1:0`. |
| `INVALID_PAYMENT_INSTRUMENT` | A **Claude/Anthropic** Marketplace model — add a valid AWS payment method, or use `--backend nova`. |
| Session: "No bat-ball contacts found" | The clip has no audio / no clear knock. Use `batter --clip` on the shot instead. |
| Don't see requests in console | Wrong region, or model-invocation logging is off (see §6). |

"""Detect the bat-ball contact moment from the audio 'knock'.

The contact between bat and ball is a sharp, broadband transient — much more
precise in time than the visual motion, which in broadcast footage is dominated
by run-ups, camera pans and follow-through. We high-pass the audio (the crack has
strong high-frequency content; commentary sits lower) and pick the most impulsive
onset. Used to anchor frame sampling on the real contact, and surfaced in the
report as a timing/presence cue.

Best-effort: on clean net audio with an isolated knock this is sharp; on a noisy
broadcast (commentary, crowd, soft defensive contact) it can be ambiguous, so the
caller treats it as a cue and still samples a wide window around it.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, find_peaks, sosfilt


def detect_contact_time(clip_path: str, hp_hz: float = 1500.0) -> tuple[float, float] | None:
    """Return ``(contact_time_s, strength)`` of the sharpest knock, or ``None``.

    ``strength`` is the onset peak relative to the mean onset — higher = more
    clearly impulsive (a confidence proxy). ``None`` if the clip has no usable
    audio track.
    """
    with tempfile.TemporaryDirectory() as td:
        wav = str(Path(td) / "a.wav")
        try:
            subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-y", "-i", str(clip_path),
                 "-ac", "1", "-ar", "16000", "-vn", wav],
                check=True,
            )
            sr, x = wavfile.read(wav)
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            return None

    if x.ndim > 1:
        x = x[:, 0]
    x = x.astype(np.float64)
    peak = np.max(np.abs(x)) if x.size else 0.0
    if peak == 0:
        return None
    x /= peak

    sos = butter(4, hp_hz / (sr / 2), btype="high", output="sos")
    env = np.abs(sosfilt(sos, x))
    win = max(1, int(sr * 0.005))
    smooth = np.convolve(env, np.ones(win) / win, mode="same")
    onset = np.clip(np.diff(smooth, prepend=smooth[0]), 0, None)

    peaks, _ = find_peaks(onset, distance=int(sr * 0.15),
                          height=float(onset.max()) * 0.5)
    if len(peaks) == 0:
        return None
    best = int(peaks[np.argmax(onset[peaks])])
    strength = float(onset[best] / (onset.mean() + 1e-12))
    return best / sr, strength

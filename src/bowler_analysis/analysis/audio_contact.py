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


def _onset_signal(clip_path: str, hp_hz: float) -> tuple[np.ndarray, int] | None:
    """High-passed amplitude-onset signal + sample rate, or None if no audio."""
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
    return onset, sr


def detect_contacts(clip_path: str, hp_hz: float = 2000.0, min_gap_s: float = 1.0,
                    rel_height: float = 0.25, max_n: int = 60) -> list[tuple[float, float]]:
    """All bat-ball knocks as ``[(time_s, strength), ...]`` sorted by time.

    Each shot in a session is a sharp transient; this returns up to ``max_n`` of
    the strongest, separated by at least ``min_gap_s``. ``strength`` is the onset
    peak relative to the mean (impulsiveness / confidence proxy).
    """
    res = _onset_signal(clip_path, hp_hz)
    if res is None:
        return []
    onset, sr = res
    peaks, props = find_peaks(onset, distance=int(sr * min_gap_s),
                              height=float(onset.max()) * rel_height)
    if len(peaks) == 0:
        return []
    mean = float(onset.mean()) + 1e-12
    cands = [(int(p) / sr, float(props["peak_heights"][i] / mean))
             for i, p in enumerate(peaks)]
    cands.sort(key=lambda c: c[1], reverse=True)
    cands = cands[:max_n]
    cands.sort(key=lambda c: c[0])
    return cands


def detect_contact_time(clip_path: str, hp_hz: float = 2000.0) -> tuple[float, float] | None:
    """The single sharpest knock ``(time_s, strength)`` for anchoring one shot."""
    cands = detect_contacts(clip_path, hp_hz=hp_hz, min_gap_s=0.15, rel_height=0.5)
    if not cands:
        return None
    return max(cands, key=lambda c: c[1])

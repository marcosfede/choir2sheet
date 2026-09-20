"""Tempo estimation from audio."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ANALYSIS_SR = 22050


def fold_bpm(bpm: float, low: float = 60.0, high: float = 180.0) -> float:
    """Halve/double ``bpm`` until it lies in ``[low, high)``."""
    while bpm >= high:
        bpm /= 2
    while bpm < low:
        bpm *= 2
    return bpm


def estimate_tempo(audio_path: Path, *, sr: int = ANALYSIS_SR) -> float:
    """Estimate a global tempo (BPM) from an audio file with librosa's beat tracker."""
    import librosa

    y, sr = librosa.load(str(audio_path), sr=sr, mono=True)
    if len(y) == 0:
        return 120.0
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    if not np.isfinite(bpm) or bpm <= 0:
        return 120.0
    bpm = round(fold_bpm(bpm), 1)
    logger.info("Estimated tempo from %s: %.1f BPM", Path(audio_path).name, bpm)
    return bpm

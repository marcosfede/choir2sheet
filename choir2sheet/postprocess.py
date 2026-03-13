"""Optional post-processing: key detection and rhythmic quantization."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from music21 import converter, key, meter, stream, tempo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Key detection
# ---------------------------------------------------------------------------

@dataclass
class KeyResult:
    """Result of key detection analysis."""
    key: str              # e.g. "G major"
    tonic: str            # e.g. "G"
    mode: str             # "major" or "minor"
    correlation: float    # confidence (0–1)
    alternates: list[dict]  # other likely keys

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "tonic": self.tonic,
            "mode": self.mode,
            "correlation": round(self.correlation, 4),
            "alternates": self.alternates,
        }


def detect_key(score: stream.Score, algorithm: str = "krumhansl") -> KeyResult:
    """Detect the musical key of a score.

    Parameters
    ----------
    score : music21.stream.Score
        The score to analyze.
    algorithm : str
        Algorithm to use: "krumhansl" (default), "aarden", "bellman",
        "simple", or "temperley".

    Returns
    -------
    KeyResult
    """
    algorithms = {
        "krumhansl": "Krumhansl",
        "aarden": "AardenEssen",
        "bellman": "BellmanBudge",
        "simple": "Simple",
        "temperley": "Temperley",
    }
    algo_class = algorithms.get(algorithm.lower())
    if algo_class is None:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Options: {list(algorithms)}")

    detected: key.Key = score.analyze("key", algorithm=algo_class)

    # Get alternate keys
    alternates = []
    try:
        alt_analysis = detected.alternateInterpretations
        for alt in alt_analysis[:4]:
            alternates.append({
                "key": f"{alt.tonic.name} {alt.mode}",
                "correlation": round(alt.correlationCoefficient, 4),
            })
    except Exception:
        pass

    result = KeyResult(
        key=f"{detected.tonic.name} {detected.mode}",
        tonic=detected.tonic.name,
        mode=detected.mode,
        correlation=round(detected.correlationCoefficient, 4),
        alternates=alternates,
    )
    logger.info("Detected key: %s (confidence: %.3f)", result.key, result.correlation)
    return result


def apply_key(score: stream.Score, key_name: str) -> stream.Score:
    """Set the key signature on all parts.

    Parameters
    ----------
    key_name : str
        Key name like "G major", "F# minor", "Bb major".
    """
    parsed_key = key.Key(key_name)
    for part in score.parts:
        part.insert(0, parsed_key)
    logger.info("Applied key signature: %s", key_name)
    return score


def detect_key_from_midi(midi_path: Path, algorithm: str = "krumhansl") -> KeyResult:
    """Detect key directly from a MIDI file."""
    score = converter.parse(str(midi_path))
    return detect_key(score, algorithm=algorithm)


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

@dataclass
class QuantizeResult:
    """Result of quantization."""
    tempo_bpm: float
    time_signature: str
    notes_before: int
    notes_after: int

    def to_dict(self) -> dict:
        return {
            "tempo_bpm": round(self.tempo_bpm, 1),
            "time_signature": self.time_signature,
            "notes_before": self.notes_before,
            "notes_after": self.notes_after,
        }


def quantize_score(
    score: stream.Score,
    *,
    tempo_bpm: float | None = None,
    time_sig: str = "4/4",
    quantize_durations: bool = True,
) -> tuple[stream.Score, QuantizeResult]:
    """Quantize a score to a beat grid.

    Snaps note onsets and optionally durations to the nearest subdivision
    of the detected or specified tempo.

    Parameters
    ----------
    score : music21.stream.Score
        The score to quantize.
    tempo_bpm : float | None
        Tempo in BPM. If None, estimates from the score.
    time_sig : str
        Time signature string (e.g. "4/4", "3/4", "6/8").
    quantize_durations : bool
        Also quantize note durations to standard rhythmic values.

    Returns
    -------
    tuple[stream.Score, QuantizeResult]
    """
    notes_before = len(score.flatten().notes)

    # Set time signature on all parts
    ts = meter.TimeSignature(time_sig)
    for part in score.parts:
        existing_ts = part.getElementsByClass(meter.TimeSignature)
        if not existing_ts:
            part.insert(0, ts)

    # Estimate tempo if not provided
    if tempo_bpm is None:
        tempo_bpm = _estimate_tempo(score)
    logger.info("Using tempo: %.1f BPM", tempo_bpm)

    # Set tempo marking
    mm = tempo.MetronomeMark(number=tempo_bpm)
    for part in score.parts:
        existing_tempo = part.getElementsByClass(tempo.MetronomeMark)
        if not existing_tempo:
            part.insert(0, mm)

    # music21's makeNotation handles quantization:
    # - Ties notes across barlines
    # - Fills rests
    # - Applies beaming
    quantized = score.makeNotation()

    if quantize_durations:
        # Additional pass: snap durations to nearest standard value
        _snap_durations(quantized)

    notes_after = len(quantized.flatten().notes)

    result = QuantizeResult(
        tempo_bpm=tempo_bpm,
        time_signature=time_sig,
        notes_before=notes_before,
        notes_after=notes_after,
    )
    logger.info("Quantized: %d → %d notes, %s at %.0f BPM",
                notes_before, notes_after, time_sig, tempo_bpm)
    return quantized, result


def _estimate_tempo(score: stream.Score) -> float:
    """Estimate tempo from inter-onset intervals."""
    import numpy as np

    onsets = []
    for note in score.flatten().notes:
        onsets.append(float(note.offset))

    if len(onsets) < 2:
        return 120.0  # default

    onsets = sorted(set(onsets))
    iois = np.diff(onsets)
    iois = iois[iois > 0.05]  # filter out very short intervals

    if len(iois) == 0:
        return 120.0

    # Median IOI → likely beat duration in quarter-note lengths
    median_ioi = float(np.median(iois))
    # Convert quarter-note length to BPM (1 quarter = median_ioi in score time)
    # In music21, offset is in quarter-note units
    bpm = 60.0 / median_ioi if median_ioi > 0 else 120.0

    # Clamp to reasonable range
    while bpm > 200:
        bpm /= 2
    while bpm < 40:
        bpm *= 2

    return round(bpm, 1)


# Standard durations in quarter-note lengths
_STANDARD_DURATIONS = [
    0.125,   # 32nd note
    0.25,    # 16th note
    0.375,   # dotted 16th
    0.5,     # 8th note
    0.75,    # dotted 8th
    1.0,     # quarter
    1.5,     # dotted quarter
    2.0,     # half
    3.0,     # dotted half
    4.0,     # whole
]


def _snap_durations(score: stream.Score) -> None:
    """Snap note durations to nearest standard rhythmic value."""
    for note in score.flatten().notes:
        ql = note.quarterLength
        closest = min(_STANDARD_DURATIONS, key=lambda d: abs(d - ql))
        if closest != ql:
            note.quarterLength = closest


def quantize_midi(
    midi_path: Path,
    output_path: Path,
    *,
    tempo_bpm: float | None = None,
    time_sig: str = "4/4",
    output_format: str | None = None,
) -> tuple[Path, QuantizeResult]:
    """Quantize a MIDI file and export.

    Parameters
    ----------
    midi_path : Path
        Input MIDI file.
    output_path : Path
        Output file path.
    tempo_bpm : float | None
        Tempo override. Auto-detected if None.
    time_sig : str
        Time signature.
    output_format : str | None
        Export format. Inferred from extension if None.
    """
    from .score import export_score, EXPORT_FORMATS

    score = converter.parse(str(midi_path))
    quantized, result = quantize_score(
        score, tempo_bpm=tempo_bpm, time_sig=time_sig,
    )

    output_path = Path(output_path)
    if output_format is None:
        ext = output_path.suffix.lstrip(".").lower()
        output_format = EXPORT_FORMATS.get(ext, "musicxml")

    export_score(quantized, output_path, fmt=output_format)
    return output_path, result

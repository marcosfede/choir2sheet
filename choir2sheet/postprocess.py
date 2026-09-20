"""Optional post-processing: key detection and rhythmic quantization."""

from __future__ import annotations

import copy
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
    # key.Key() expects separate tonic and mode args
    parts_of_name = key_name.strip().split()
    if len(parts_of_name) >= 2:
        tonic = parts_of_name[0]
        mode = parts_of_name[-1]
        parsed_key = key.Key(tonic, mode)
    else:
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
    grid: float           # grid step in quarter lengths (0.25 = 16th note)
    notes_before: int
    notes_after: int

    def to_dict(self) -> dict:
        return {
            "tempo_bpm": round(self.tempo_bpm, 1),
            "time_signature": self.time_signature,
            "grid": self.grid,
            "notes_before": self.notes_before,
            "notes_after": self.notes_after,
        }


def _source_tempo(score: stream.Score) -> float:
    """Tempo at which the score's offsets were written (MIDI import), default 120."""
    marks = score.flatten().getElementsByClass(tempo.MetronomeMark)
    for mm in marks:
        if mm.number:
            return float(mm.number)
    return 120.0


def _snap(value: float, grid: float) -> float:
    return round(value / grid) * grid


def quantize_score(
    score: stream.Score,
    *,
    tempo_bpm: float | None = None,
    time_sig: str = "4/4",
    grid: float = 0.25,
) -> tuple[stream.Score, QuantizeResult]:
    """Quantize a score to a beat grid.

    Offsets are converted to seconds using the score's own tempo (MIDI
    imports carry one; Basic Pitch writes 120 BPM), re-expressed in quarter
    notes at ``tempo_bpm``, and both onsets and offsets are snapped to the
    nearest ``grid`` step. Notes shorter than one grid step are lengthened to
    a single step so nothing disappears. Finally ``makeNotation`` produces
    measures, ties across barlines and rests.

    Parameters
    ----------
    tempo_bpm : float | None
        Target tempo. Estimated from inter-onset intervals if omitted.
    time_sig : str
        Time signature string (e.g. "4/4", "3/4", "6/8").
    grid : float
        Smallest rhythmic unit in quarter lengths (0.25 = 16th, 0.5 = 8th).
    """
    src_bpm = _source_tempo(score)
    sec_per_ql = 60.0 / src_bpm

    notes_before = len(score.flatten().notes)

    if tempo_bpm is None:
        onsets = sorted({float(n.offset) * sec_per_ql for n in score.flatten().notes})
        tempo_bpm = _estimate_tempo(onsets)
    logger.info("Using tempo: %.1f BPM (source MIDI tempo %.1f)", tempo_bpm, src_bpm)

    scale = sec_per_ql * tempo_bpm / 60.0  # source ql → target ql

    quantized = stream.Score()
    if score.metadata is not None:
        quantized.metadata = score.metadata

    for part in score.parts:
        new_part = stream.Part()
        new_part.id = part.id
        instr = part.getInstrument(returnDefault=False)
        if instr is not None:
            new_part.insert(0, instr)
        keys = part.flatten().getElementsByClass(key.KeySignature)
        if keys:
            new_part.insert(0, keys[0])
        new_part.insert(0, meter.TimeSignature(time_sig))
        new_part.insert(0, tempo.MetronomeMark(number=tempo_bpm))

        for n in part.flatten().notes:
            start = _snap(float(n.offset) * scale, grid)
            end = _snap((float(n.offset) + float(n.quarterLength)) * scale, grid)
            if end <= start:
                end = start + grid
            new_note = copy.deepcopy(n)
            new_note.quarterLength = end - start
            new_part.insert(start, new_note)

        quantized.append(new_part)

    quantized = quantized.makeNotation()
    notes_after = len(quantized.flatten().notes)

    result = QuantizeResult(
        tempo_bpm=tempo_bpm,
        time_signature=time_sig,
        grid=grid,
        notes_before=notes_before,
        notes_after=notes_after,
    )
    logger.info("Quantized: %d → %d notes, %s at %.0f BPM",
                notes_before, notes_after, time_sig, tempo_bpm)
    return quantized, result


def _estimate_tempo(onsets_sec: list[float]) -> float:
    """Estimate tempo from inter-onset intervals (in seconds).

    Assumes the most common inter-onset interval is a beat or a simple
    subdivision of one, and folds the result into 60–180 BPM.
    """
    import numpy as np

    if len(onsets_sec) < 2:
        return 120.0

    iois = np.diff(np.asarray(sorted(onsets_sec)))
    iois = iois[iois > 0.05]
    if len(iois) == 0:
        return 120.0

    bpm = 60.0 / float(np.median(iois))
    while bpm > 180:
        bpm /= 2
    while bpm < 60:
        bpm *= 2
    return round(bpm, 1)


def quantize_midi(
    midi_path: Path,
    output_path: Path,
    *,
    tempo_bpm: float | None = None,
    time_sig: str = "4/4",
    grid: float = 0.25,
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
    from .score import export_score

    score = converter.parse(str(midi_path))
    quantized, result = quantize_score(
        score, tempo_bpm=tempo_bpm, time_sig=time_sig, grid=grid,
    )

    output_path = export_score(quantized, Path(output_path), fmt=output_format)
    return output_path, result

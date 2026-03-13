"""Note-level accuracy metrics for transcription evaluation.

Compares a transcribed MIDI against a ground-truth MIDI using standard
mir_eval note metrics (precision, recall, F1) and a custom "note loss"
that penalises pitch errors, onset/offset timing errors, and missing/extra
notes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pretty_midi

logger = logging.getLogger(__name__)


@dataclass
class NoteEvent:
    """A single note event."""
    pitch: int        # MIDI pitch (0–127)
    onset: float      # seconds
    offset: float     # seconds
    velocity: int = 64

    @property
    def duration(self) -> float:
        return self.offset - self.onset


@dataclass
class TranscriptionMetrics:
    """Container for evaluation metrics."""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    # Detailed breakdown
    total_ref_notes: int = 0
    total_est_notes: int = 0
    matched_notes: int = 0
    pitch_error_hz_mean: float = 0.0
    onset_error_ms_mean: float = 0.0
    offset_error_ms_mean: float = 0.0
    # Combined loss (lower is better, 0 = perfect)
    loss: float = 1.0

    def summary(self) -> str:
        return (
            f"F1={self.f1:.3f} (P={self.precision:.3f}, R={self.recall:.3f}) | "
            f"Matched={self.matched_notes}/{self.total_ref_notes} ref, "
            f"{self.total_est_notes} est | "
            f"Loss={self.loss:.4f} | "
            f"Onset err={self.onset_error_ms_mean:.1f}ms, "
            f"Offset err={self.offset_error_ms_mean:.1f}ms"
        )


def extract_notes(midi_path: Path) -> list[NoteEvent]:
    """Extract all note events from a MIDI file."""
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    notes = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            notes.append(NoteEvent(
                pitch=note.pitch,
                onset=note.start,
                offset=note.end,
                velocity=note.velocity,
            ))
    notes.sort(key=lambda n: (n.onset, n.pitch))
    return notes


def notes_to_mir_eval(
    notes: list[NoteEvent],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert notes to mir_eval format: (intervals, pitches_hz).

    Returns
    -------
    intervals : ndarray of shape (N, 2)
        Onset/offset times in seconds.
    pitches_hz : ndarray of shape (N,)
        Pitches in Hz.
    """
    if not notes:
        return np.zeros((0, 2)), np.zeros(0)
    intervals = np.array([[n.onset, n.offset] for n in notes])
    pitches_hz = np.array([pretty_midi.note_number_to_hz(n.pitch) for n in notes])
    return intervals, pitches_hz


def evaluate(
    reference_midi: Path,
    estimated_midi: Path,
    *,
    onset_tolerance: float = 0.05,   # 50 ms
    pitch_tolerance: float = 50.0,   # 50 cents
    offset_ratio: float | None = 0.2,
) -> TranscriptionMetrics:
    """Evaluate a transcribed MIDI against a ground-truth MIDI.

    Uses mir_eval for standard note-level metrics, plus computes a
    combined loss score.

    Parameters
    ----------
    reference_midi : Path
        Ground-truth MIDI file.
    estimated_midi : Path
        Transcribed MIDI file to evaluate.
    onset_tolerance : float
        Maximum onset deviation in seconds for a match.
    pitch_tolerance : float
        Maximum pitch deviation in cents for a match.
    offset_ratio : float | None
        Offset tolerance as ratio of note duration (None = don't check offsets).

    Returns
    -------
    TranscriptionMetrics
    """
    import mir_eval

    ref_notes = extract_notes(Path(reference_midi))
    est_notes = extract_notes(Path(estimated_midi))

    ref_intervals, ref_pitches = notes_to_mir_eval(ref_notes)
    est_intervals, est_pitches = notes_to_mir_eval(est_notes)

    metrics = TranscriptionMetrics(
        total_ref_notes=len(ref_notes),
        total_est_notes=len(est_notes),
    )

    if len(ref_notes) == 0 and len(est_notes) == 0:
        metrics.precision = metrics.recall = metrics.f1 = 1.0
        metrics.loss = 0.0
        return metrics

    if len(ref_notes) == 0 or len(est_notes) == 0:
        metrics.loss = 1.0
        return metrics

    # mir_eval note-level evaluation
    prec, rec, f1, avg_overlap = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals,
        ref_pitches,
        est_intervals,
        est_pitches,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=pitch_tolerance,
        offset_ratio=offset_ratio,
    )

    metrics.precision = float(prec)
    metrics.recall = float(rec)
    metrics.f1 = float(f1)

    # Count matched notes using mir_eval matching
    matching = mir_eval.transcription.match_notes(
        ref_intervals,
        ref_pitches,
        est_intervals,
        est_pitches,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=pitch_tolerance,
        offset_ratio=offset_ratio,
    )
    metrics.matched_notes = len(matching)

    # Compute detailed errors on matched pairs
    if matching:
        onset_errors = []
        offset_errors = []
        pitch_errors = []
        for ref_idx, est_idx in matching:
            rn = ref_notes[ref_idx]
            en = est_notes[est_idx]
            onset_errors.append(abs(rn.onset - en.onset) * 1000)  # ms
            offset_errors.append(abs(rn.offset - en.offset) * 1000)  # ms
            ref_hz = pretty_midi.note_number_to_hz(rn.pitch)
            est_hz = pretty_midi.note_number_to_hz(en.pitch)
            pitch_errors.append(abs(ref_hz - est_hz))

        metrics.onset_error_ms_mean = float(np.mean(onset_errors))
        metrics.offset_error_ms_mean = float(np.mean(offset_errors))
        metrics.pitch_error_hz_mean = float(np.mean(pitch_errors))

    # Combined loss: 1 - F1, so 0 = perfect, 1 = nothing matched
    metrics.loss = 1.0 - metrics.f1

    return metrics


def evaluate_stems(
    reference_dir: Path,
    estimated_dir: Path,
    **kwargs,
) -> dict[str, TranscriptionMetrics]:
    """Evaluate multiple stems by matching filenames."""
    ref_dir = Path(reference_dir)
    est_dir = Path(estimated_dir)
    results = {}
    for ref_file in sorted(ref_dir.glob("*.mid")):
        est_file = est_dir / ref_file.name
        if est_file.is_file():
            results[ref_file.stem] = evaluate(ref_file, est_file, **kwargs)
        else:
            logger.warning("No estimated MIDI for %s", ref_file.name)
    return results

"""Note-level cleanup applied between transcription and score assembly."""

from __future__ import annotations

import logging
from pathlib import Path

import pretty_midi

from .metrics import NoteEvent, extract_notes

logger = logging.getLogger(__name__)

# Comfortable SATB range with a little headroom: C2 (65 Hz) .. C6 (1047 Hz).
VOCAL_PITCH_RANGE = (36, 84)

# Share of a note that must be covered by an overlapping note to call it a ghost.
GHOST_OVERLAP = 0.75


def clean_notes(
    notes: list[NoteEvent],
    *,
    monophonic: bool = False,
    min_duration: float = 0.0,
    pitch_range: tuple[int, int] | None = None,
    merge_gap: float = 0.0,
) -> list[NoteEvent]:
    """Filter and tidy a list of note events.

    Parameters
    ----------
    monophonic : bool
        Enforce one note at a time. When two notes overlap, the one that
        covers the smaller share of the overlap is dropped if it is mostly
        swallowed by the other; otherwise the earlier note is truncated at
        the later onset.
    min_duration : float
        Drop notes shorter than this many seconds.
    pitch_range : (low, high) | None
        Inclusive MIDI pitch bounds; notes outside are dropped.
    merge_gap : float
        Join consecutive notes of the same pitch whose gap is at most this
        many seconds (undoes spurious re-articulations).
    """
    out = sorted(notes, key=lambda n: (n.onset, -n.velocity))

    if pitch_range is not None:
        lo, hi = pitch_range
        out = [n for n in out if lo <= n.pitch <= hi]

    if merge_gap > 0:
        out = _merge_repeats(out, merge_gap)

    if monophonic:
        out = _enforce_monophony(out)

    if min_duration > 0:
        out = [n for n in out if n.duration >= min_duration]

    return out


def _merge_repeats(notes: list[NoteEvent], gap: float) -> list[NoteEvent]:
    merged: list[NoteEvent] = []
    for n in notes:
        prev = merged[-1] if merged else None
        if prev is not None and prev.pitch == n.pitch and 0 <= n.onset - prev.offset <= gap:
            prev.offset = max(prev.offset, n.offset)
            prev.velocity = max(prev.velocity, n.velocity)
        else:
            merged.append(NoteEvent(n.pitch, n.onset, n.offset, n.velocity))
    return merged


def _enforce_monophony(notes: list[NoteEvent]) -> list[NoteEvent]:
    kept: list[NoteEvent] = []
    for n in notes:
        n = NoteEvent(n.pitch, n.onset, n.offset, n.velocity)
        while kept:
            prev = kept[-1]
            if n.onset >= prev.offset:
                break
            overlap = min(prev.offset, n.offset) - n.onset
            # A note mostly buried inside a stronger one is a ghost.
            if overlap >= GHOST_OVERLAP * n.duration and prev.velocity >= n.velocity:
                n = None
                break
            if overlap >= GHOST_OVERLAP * prev.duration and n.velocity > prev.velocity:
                kept.pop()
                continue
            if n.onset <= prev.onset:
                n = None
                break
            prev.offset = n.onset
            break
        if n is not None and n.duration > 0:
            kept.append(n)
    return kept


def clean_midi(
    midi_path: Path,
    output_path: Path | None = None,
    **kwargs,
) -> tuple[Path, int, int]:
    """Apply :func:`clean_notes` to every non-drum instrument in a MIDI file.

    Returns ``(output_path, notes_before, notes_after)``.
    """
    midi_path = Path(midi_path)
    output_path = Path(output_path) if output_path else midi_path

    src = pretty_midi.PrettyMIDI(str(midi_path))
    before = sum(len(i.notes) for i in src.instruments if not i.is_drum)

    for inst in src.instruments:
        if inst.is_drum:
            continue
        notes = [
            NoteEvent(n.pitch, n.start, n.end, n.velocity) for n in inst.notes
        ]
        cleaned = clean_notes(notes, **kwargs)
        inst.notes = [
            pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.onset, end=n.offset)
            for n in cleaned
        ]
        inst.pitch_bends = []

    after = sum(len(i.notes) for i in src.instruments if not i.is_drum)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    src.write(str(output_path))
    logger.info("Cleaned %s: %d → %d notes", midi_path.name, before, after)
    return output_path, before, after


__all__ = ["VOCAL_PITCH_RANGE", "clean_notes", "clean_midi", "extract_notes"]

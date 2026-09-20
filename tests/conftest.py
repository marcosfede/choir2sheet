from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

# C major scale, one note per second, at 120 BPM MIDI resolution.
SCALE = [60, 62, 64, 65, 67, 69, 71, 72]


def write_midi(path: Path, notes: list[tuple[int, float, float]], tempo: float = 120.0) -> Path:
    """Write ``(pitch, onset_s, offset_s)`` tuples to a single-instrument MIDI file."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=0)
    for pitch, onset, offset in notes:
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=pitch, start=onset, end=offset))
    midi.instruments.append(inst)
    midi.write(str(path))
    return path


@pytest.fixture
def scale_notes() -> list[tuple[int, float, float]]:
    return [(p, float(i), i + 0.9) for i, p in enumerate(SCALE)]


@pytest.fixture
def scale_midi(tmp_path: Path, scale_notes) -> Path:
    return write_midi(tmp_path / "scale.mid", scale_notes)


@pytest.fixture
def scale_wav(tmp_path: Path, scale_notes) -> Path:
    sr = 22050
    t = np.arange(0, 8.0, 1 / sr)
    y = np.zeros_like(t)
    for pitch, onset, offset in scale_notes:
        f = 440.0 * 2 ** ((pitch - 69) / 12)
        seg = (t >= onset) & (t < offset)
        y[seg] = 0.5 * np.sin(2 * np.pi * f * t[seg]) * np.exp(-2 * (t[seg] - onset))
    path = tmp_path / "scale.wav"
    sf.write(path, y, sr)
    return path

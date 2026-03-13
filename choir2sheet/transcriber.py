"""Audio-to-MIDI transcription using Basic Pitch."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def transcribe_to_midi(
    audio_path: Path,
    output_path: Path,
    *,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length: float = 58.0,
    minimum_frequency: float | None = None,
    maximum_frequency: float | None = None,
) -> Path:
    """Transcribe an audio file to MIDI using Basic Pitch.

    Parameters
    ----------
    audio_path : Path
        Input audio file (WAV, MP3, FLAC, OGG, etc.).
    output_path : Path
        Where to write the resulting MIDI file.
    onset_threshold : float
        Note onset confidence threshold (0–1). Higher = fewer ghost notes.
    frame_threshold : float
        Frame activation threshold (0–1). Higher = stricter detection.
    minimum_note_length : float
        Minimum note duration in milliseconds.
    minimum_frequency, maximum_frequency : float | None
        Optional frequency bounds (Hz) to filter detected notes.

    Returns
    -------
    Path
        The path to the written MIDI file.
    """
    from basic_pitch.inference import predict

    audio_path = Path(audio_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info("Transcribing %s with Basic Pitch…", audio_path.name)

    model_output, midi_data, note_events = predict(
        str(audio_path),
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
    )

    midi_data.write(str(output_path))
    logger.info("MIDI written to %s (%d notes)", output_path, len(note_events))
    return output_path


def transcribe_stems(
    stems: dict[str, Path],
    output_dir: Path,
    **kwargs,
) -> dict[str, Path]:
    """Transcribe multiple audio stems to MIDI files.

    Parameters
    ----------
    stems : dict[str, Path]
        Mapping of stem name → audio file path.
    output_dir : Path
        Directory to write MIDI files into.

    Returns
    -------
    dict[str, Path]
        Mapping of stem name → MIDI file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    midi_files: dict[str, Path] = {}
    for name, audio in stems.items():
        midi_path = output_dir / f"{name}.mid"
        try:
            transcribe_to_midi(audio, midi_path, **kwargs)
            midi_files[name] = midi_path
        except Exception:
            logger.exception("Failed to transcribe stem '%s'", name)

    return midi_files

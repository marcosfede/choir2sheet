"""End-to-end pipeline: audio file → sheet music."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from music21 import stream

from .score import build_score, export_score
from .separator import run_demucs, separate_choir, DEMUCS_MODEL_6S
from .transcriber import transcribe_stems, transcribe_to_midi

logger = logging.getLogger(__name__)


def full_pipeline(
    audio_path: Path,
    output_path: Path,
    *,
    output_format: str | None = None,
    title: str = "Transcription",
    composer: str = "",
    # Separation options
    skip_separation: bool = False,
    skip_satb: bool = False,
    mvsep_api_key: Optional[str] = None,
    device: str = "cpu",
    # Transcription options
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length: float = 58.0,
) -> Path:
    """Run the full choir2sheet pipeline.

    1. Separate audio into stems (Demucs, optionally MVSEP SATB)
    2. Transcribe each stem to MIDI (Basic Pitch)
    3. Assemble into a single score (music21)
    4. Export to the requested format

    Parameters
    ----------
    audio_path : Path
        Input audio file.
    output_path : Path
        Where to write the final score.
    output_format : str | None
        Export format (musicxml, midi, lilypond, abc, pdf). Inferred from
        extension if not specified.
    skip_separation : bool
        If True, skip Demucs and transcribe the raw audio directly.
    skip_satb : bool
        If True, skip MVSEP SATB splitting (keep vocals as one stem).
    """
    audio_path = Path(audio_path)
    output_path = Path(output_path)
    work_dir = output_path.parent / ".choir2sheet_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Separation ──────────────────────────────────────────
    if skip_separation:
        logger.info("Skipping separation; transcribing raw audio directly")
        stems = {"audio": audio_path}
    else:
        logger.info("Stage 1: Source separation")
        if skip_satb:
            stems = {}
            demucs_stems = run_demucs(
                audio_path, work_dir / "demucs", device=device
            )
            stems.update(demucs_stems)
        else:
            stems = separate_choir(
                audio_path,
                work_dir,
                mvsep_api_key=mvsep_api_key,
                device=device,
                skip_satb=skip_satb,
            )
    logger.info("Stems: %s", list(stems.keys()))

    # ── Stage 2: Transcription ───────────────────────────────────────
    logger.info("Stage 2: MIDI transcription")
    midi_files = transcribe_stems(
        stems,
        work_dir / "midi",
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length,
    )
    if not midi_files:
        raise RuntimeError("No stems were successfully transcribed to MIDI")
    logger.info("Transcribed %d stem(s) to MIDI", len(midi_files))

    # ── Stage 3: Score assembly & export ─────────────────────────────
    logger.info("Stage 3: Score assembly and export")
    score = build_score(midi_files, title=title, composer=composer)
    result = export_score(score, output_path, fmt=output_format)
    logger.info("Done! Output: %s", result)
    return result


def simple_transcribe(
    audio_path: Path,
    output_path: Path,
    *,
    output_format: str | None = None,
    title: str = "Transcription",
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
) -> Path:
    """Simplified pipeline: transcribe a single audio file directly (no separation).

    Good for single-instrument or single-voice recordings.
    """
    return full_pipeline(
        audio_path,
        output_path,
        output_format=output_format,
        title=title,
        skip_separation=True,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
    )

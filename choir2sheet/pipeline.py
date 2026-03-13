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
    # Post-processing options
    quantize: bool = False,
    tempo_bpm: float | None = None,
    time_sig: str = "4/4",
    detect_key: bool = False,
    set_key: str | None = None,
) -> dict:
    """Run the full choir2sheet pipeline.

    Returns a dict with structured results (file paths, analysis, etc.).
    """
    audio_path = Path(audio_path)
    output_path = Path(output_path)
    work_dir = output_path.parent / ".choir2sheet_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    result: dict = {"ok": True, "stages": {}}

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

    result["stages"]["separation"] = {
        "stems": {name: str(path) for name, path in stems.items()},
    }
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

    result["stages"]["transcription"] = {
        "midi_files": {name: str(path) for name, path in midi_files.items()},
    }
    logger.info("Transcribed %d stem(s) to MIDI", len(midi_files))

    # ── Stage 3: Score assembly ──────────────────────────────────────
    logger.info("Stage 3: Score assembly")
    score = build_score(midi_files, title=title, composer=composer)

    # ── Stage 4 (optional): Post-processing ──────────────────────────
    postprocess_results: dict = {}

    # Key detection
    if detect_key or set_key:
        from .postprocess import detect_key as _detect_key, apply_key

        if set_key:
            score = apply_key(score, set_key)
            postprocess_results["key"] = {"key": set_key, "source": "manual"}
        else:
            key_result = _detect_key(score)
            postprocess_results["key"] = {**key_result.to_dict(), "source": "detected"}
            score = apply_key(score, key_result.key)

    # Quantization
    if quantize:
        from .postprocess import quantize_score

        score, quant_result = quantize_score(
            score, tempo_bpm=tempo_bpm, time_sig=time_sig,
        )
        postprocess_results["quantization"] = quant_result.to_dict()

    if postprocess_results:
        result["stages"]["postprocess"] = postprocess_results

    # ── Stage 5: Export ──────────────────────────────────────────────
    logger.info("Stage 5: Export")
    exported = export_score(score, output_path, fmt=output_format)
    result["output"] = str(exported)
    logger.info("Done! Output: %s", exported)

    return result


def simple_transcribe(
    audio_path: Path,
    output_path: Path,
    *,
    output_format: str | None = None,
    title: str = "Transcription",
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
) -> dict:
    """Simplified pipeline: transcribe a single audio file directly (no separation)."""
    return full_pipeline(
        audio_path,
        output_path,
        output_format=output_format,
        title=title,
        skip_separation=True,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
    )

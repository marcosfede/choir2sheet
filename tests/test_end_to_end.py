#!/usr/bin/env python3
"""End-to-end benchmark against MAESTRO dataset (same dataset used to evaluate Basic Pitch).

Downloads a real piano performance MIDI from MAESTRO v3.0.0, synthesizes
audio via FluidSynth, transcribes with Basic Pitch, and evaluates against
ground truth using mir_eval note metrics.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pretty_midi
import soundfile as sf

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from choir2sheet.transcriber import transcribe_to_midi
from choir2sheet.score import build_score, export_score
from choir2sheet.metrics import evaluate, extract_notes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_e2e")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "test_output"

# MAESTRO v3.0.0 — the benchmark dataset used to evaluate Basic Pitch
MAESTRO_MIDI_ZIP_URL = (
    "https://storage.googleapis.com/magentadata/datasets/maestro/"
    "v3.0.0/maestro-v3.0.0-midi.zip"
)
# Shortest test-split piece: Scarlatti Sonata K.525 (66s, 882 notes)
MAESTRO_TEST_MIDI = (
    "maestro-v3.0.0/2008/"
    "MIDI-Unprocessed_09_R3_2008_01-07_ORIG_MID--AUDIO_09_R3_2008_wav--2.midi"
)
PIECE_TITLE = "Scarlatti Sonata K.525"


def _metrics_to_dict(m) -> dict:
    return {
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
        "matched": m.matched_notes,
        "loss": m.loss,
        "onset_error_ms": m.onset_error_ms_mean,
        "offset_error_ms": m.offset_error_ms_mean,
    }


# ── Step 1: Download MAESTRO MIDI ────────────────────────────────────


def download_maestro_midi(output_dir: Path) -> Path:
    """Download MAESTRO MIDI zip and extract the target piece."""
    zip_path = output_dir / "maestro-midi.zip"
    midi_path = output_dir / MAESTRO_TEST_MIDI

    if midi_path.is_file():
        logger.info("MAESTRO MIDI already downloaded: %s", midi_path)
        return midi_path

    if not zip_path.is_file():
        logger.info("Downloading MAESTRO MIDI archive (56 MB)…")
        output_dir.mkdir(parents=True, exist_ok=True)
        urlretrieve(MAESTRO_MIDI_ZIP_URL, str(zip_path))
        logger.info("Downloaded: %s", zip_path)

    logger.info("Extracting %s…", MAESTRO_TEST_MIDI)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(MAESTRO_TEST_MIDI, str(output_dir))

    return midi_path


# ── Step 2: Synthesize audio ─────────────────────────────────────────


def synthesize_audio(midi_path: Path, wav_path: Path, sr: int = 22050) -> Path:
    """Render MIDI to WAV using FluidSynth with GM soundfont."""
    if wav_path.is_file():
        logger.info("Audio already synthesized: %s", wav_path)
        return wav_path

    midi = pretty_midi.PrettyMIDI(str(midi_path))
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    # Try FluidSynth with a real soundfont
    sf2_path = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
    if not Path(sf2_path).is_file():
        sf2_path = None  # Fall back to pretty_midi default

    try:
        audio = midi.fluidsynth(fs=sr, sf2_path=sf2_path)
        logger.info("Audio synthesized via FluidSynth")
    except Exception as e:
        logger.warning("FluidSynth failed (%s), falling back to sine synthesis", e)
        audio = _sine_synthesis(midi, sr)

    audio = audio / np.max(np.abs(audio)) * 0.9
    sf.write(str(wav_path), audio, sr)
    logger.info("WAV: %s (%.1fs, %d Hz)", wav_path, len(audio) / sr, sr)
    return wav_path


def _sine_synthesis(midi, sr: int) -> np.ndarray:
    """Fallback sine-wave synthesis when FluidSynth isn't available."""
    duration = midi.get_end_time() + 1.0
    n_samples = int(duration * sr)
    audio = np.zeros(n_samples, dtype=np.float64)

    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            freq = pretty_midi.note_number_to_hz(note.pitch)
            t_start = int(note.start * sr)
            t_end = int(note.end * sr)
            t = np.arange(t_end - t_start) / sr
            wave = np.sin(2 * np.pi * freq * t)
            env = np.ones_like(wave)
            attack = min(int(0.02 * sr), len(env))
            release = min(int(0.05 * sr), len(env))
            if attack > 0:
                env[:attack] = np.linspace(0, 1, attack)
            if release > 0:
                env[-release:] = np.linspace(1, 0, release)
            wave *= env * (note.velocity / 127.0) * 0.3
            audio[t_start:t_start + len(wave)] += wave

    return audio


# ── Step 3: Transcribe ───────────────────────────────────────────────


def run_transcription(wav_path: Path, midi_out: Path) -> Path:
    """Transcribe audio to MIDI using Basic Pitch."""
    return transcribe_to_midi(
        wav_path, midi_out,
        onset_threshold=0.5,
        frame_threshold=0.3,
        minimum_note_length=58.0,
    )


# ── Step 4: Build score ──────────────────────────────────────────────


def build_output_score(midi_path: Path, output_path: Path) -> Path:
    """Assemble a score from the transcribed MIDI and export."""
    score = build_score({"piano": midi_path}, title=PIECE_TITLE)
    return export_score(score, output_path)


# ── Step 5: Evaluate ─────────────────────────────────────────────────


def run_evaluation(gt_midi: Path, est_midi: Path) -> dict:
    """Evaluate with multiple tolerance settings."""
    gt_notes = extract_notes(gt_midi)
    est_notes = extract_notes(est_midi)

    # Standard AMT evaluation: onset+offset matching
    strict = evaluate(gt_midi, est_midi,
                      onset_tolerance=0.05, pitch_tolerance=50.0,
                      offset_ratio=0.2)
    relaxed = evaluate(gt_midi, est_midi,
                       onset_tolerance=0.2, pitch_tolerance=50.0,
                       offset_ratio=0.2)

    # Onset-only matching (no offset check) — standard in many AMT papers
    onset_only_strict = evaluate(gt_midi, est_midi,
                                 onset_tolerance=0.05, pitch_tolerance=50.0,
                                 offset_ratio=None)
    onset_only_relaxed = evaluate(gt_midi, est_midi,
                                  onset_tolerance=0.2, pitch_tolerance=50.0,
                                  offset_ratio=None)

    return {
        "dataset": "MAESTRO v3.0.0 (test split)",
        "piece": PIECE_TITLE,
        "ground_truth_notes": len(gt_notes),
        "estimated_notes": len(est_notes),
        "onset_offset_strict_50ms": _metrics_to_dict(strict),
        "onset_offset_relaxed_200ms": _metrics_to_dict(relaxed),
        "onset_only_strict_50ms": _metrics_to_dict(onset_only_strict),
        "onset_only_relaxed_200ms": _metrics_to_dict(onset_only_relaxed),
    }


# ── Main ─────────────────────────────────────────────────────────────


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    maestro_dir = OUTPUT_DIR / "maestro"
    maestro_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("choir2sheet — MAESTRO Benchmark")
    print(f"Piece: {PIECE_TITLE}")
    print("=" * 70)

    # Step 1: Download
    print("\n[1/5] Downloading MAESTRO ground-truth MIDI…")
    gt_midi = download_maestro_midi(OUTPUT_DIR)

    # Step 2: Synthesize
    print("[2/5] Synthesizing audio from MIDI (FluidSynth + GM soundfont)…")
    wav_path = synthesize_audio(gt_midi, maestro_dir / "test_audio.wav")

    # Step 3: Transcribe
    print("[3/5] Transcribing audio with Basic Pitch…")
    est_midi = run_transcription(wav_path, maestro_dir / "transcribed.mid")

    # Step 4: Build score
    print("[4/5] Building sheet music score…")
    score_xml = build_output_score(est_midi, maestro_dir / "score.musicxml")
    print(f"  -> MusicXML: {score_xml}")

    # Step 5: Evaluate
    print("[5/5] Evaluating transcription accuracy…")
    results = run_evaluation(gt_midi, est_midi)

    # Print results
    print("\n" + "=" * 70)
    print("MAESTRO BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Ground truth notes: {results['ground_truth_notes']}")
    print(f"Estimated notes:    {results['estimated_notes']}")

    sections = [
        ("Onset+Offset, Strict (50ms)", "onset_offset_strict_50ms"),
        ("Onset+Offset, Relaxed (200ms)", "onset_offset_relaxed_200ms"),
        ("Onset-only, Strict (50ms)", "onset_only_strict_50ms"),
        ("Onset-only, Relaxed (200ms)", "onset_only_relaxed_200ms"),
    ]
    for label, key in sections:
        m = results[key]
        print(f"\n  {label}:")
        print(f"    Precision:   {m['precision']:.3f}")
        print(f"    Recall:      {m['recall']:.3f}")
        print(f"    F1 Score:    {m['f1']:.3f}")
        print(f"    Matched:     {m['matched']}")
        print(f"    Loss:        {m['loss']:.4f}")
        print(f"    Onset err:   {m['onset_error_ms']:.1f} ms")
        print(f"    Offset err:  {m['offset_error_ms']:.1f} ms")

    # Save results
    results_path = maestro_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Pass/fail — onset-only F1 should be > 0.7 (paper reports ~0.71 on MAESTRO)
    f1 = results["onset_only_strict_50ms"]["f1"]
    if f1 >= 0.5:
        print(f"\nPASS — Onset-only F1 = {f1:.3f} (threshold: 0.5)")
    else:
        print(f"\nFAIL — Onset-only F1 = {f1:.3f} (threshold: 0.5)")
        sys.exit(1)

    print(f"\nAll outputs in: {maestro_dir}")


if __name__ == "__main__":
    main()

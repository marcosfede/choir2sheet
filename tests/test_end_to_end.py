#!/usr/bin/env python3
"""End-to-end test: generate a known MIDI → synthesize audio → transcribe → validate.

This test creates a simple 4-part SATB chorale as MIDI, renders it to audio
using pretty_midi's built-in FluidSynth or a sine-wave fallback, then runs
the Basic Pitch transcription and compares the output against the known
ground truth using our metrics module.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

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


# ── Step 1: Create a ground-truth MIDI chorale ───────────────────────


def create_test_chorale(output_path: Path) -> Path:
    """Create a simple SATB chorale MIDI file with known notes.

    Uses the opening of "Amazing Grace" harmonized for SATB as a test case.
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=80)

    # Quarter note duration at 80 BPM = 0.75s
    q = 0.75

    # Soprano line (melody) - simple stepwise melody in C major
    soprano_notes = [
        # (pitch, start_beat, duration_beats)
        (60, 0, 2),   # C4 - half note
        (64, 2, 1),   # E4
        (67, 3, 2),   # G4 - half note
        (65, 5, 1),   # F4
        (64, 6, 1),   # E4
        (60, 7, 2),   # C4 - half note
        (67, 9, 2),   # G4 - half note
        (72, 11, 1),  # C5
        (72, 12, 2),  # C5 - half note
        (67, 14, 1),  # G4
        (64, 15, 2),  # E4 - half note
    ]

    # Alto line - harmony
    alto_notes = [
        (55, 0, 2),   # G3
        (60, 2, 1),   # C4
        (64, 3, 2),   # E4
        (62, 5, 1),   # D4
        (60, 6, 1),   # C4
        (55, 7, 2),   # G3
        (64, 9, 2),   # E4
        (67, 11, 1),  # G4
        (67, 12, 2),  # G4
        (64, 14, 1),  # E4
        (60, 15, 2),  # C4
    ]

    # Tenor line
    tenor_notes = [
        (48, 0, 2),   # C3
        (52, 2, 1),   # E3
        (55, 3, 2),   # G3
        (53, 5, 1),   # F3
        (52, 6, 1),   # E3
        (48, 7, 2),   # C3
        (55, 9, 2),   # G3
        (60, 11, 1),  # C4
        (60, 12, 2),  # C4
        (55, 14, 1),  # G3
        (52, 15, 2),  # E3
    ]

    # Bass line
    bass_notes = [
        (36, 0, 2),   # C2
        (40, 2, 1),   # E2
        (43, 3, 2),   # G2
        (41, 5, 1),   # F2
        (40, 6, 1),   # E2
        (36, 7, 2),   # C2
        (43, 9, 2),   # G2
        (48, 11, 1),  # C3
        (48, 12, 2),  # C3
        (43, 14, 1),  # G2
        (40, 15, 2),  # E2
    ]

    parts = {
        "Soprano": soprano_notes,
        "Alto": alto_notes,
        "Tenor": tenor_notes,
        "Bass": bass_notes,
    }

    for part_name, notes in parts.items():
        inst = pretty_midi.Instrument(
            program=52,  # Choir Aahs
            name=part_name,
        )
        for pitch, start_beat, dur_beats in notes:
            inst.notes.append(pretty_midi.Note(
                velocity=80,
                pitch=pitch,
                start=start_beat * q,
                end=(start_beat + dur_beats) * q,
            ))
        midi.instruments.append(inst)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(output_path))
    logger.info("Ground-truth MIDI written: %s (%d notes total)",
                output_path, sum(len(n) for n in parts.values()))
    return output_path


# ── Step 2: Synthesize audio from MIDI ───────────────────────────────


def synthesize_audio(midi_path: Path, wav_path: Path, sr: int = 22050) -> Path:
    """Render MIDI to WAV using sine-wave synthesis (no FluidSynth needed)."""
    midi = pretty_midi.PrettyMIDI(str(midi_path))

    # Try FluidSynth first, fall back to sine synthesis
    audio = None
    try:
        audio = midi.fluidsynth(fs=sr)
        logger.info("Audio synthesized via FluidSynth")
    except Exception as e:
        logger.info("FluidSynth not available (%s), using sine synthesis", e)

    if audio is None:
        # Manual sine-wave synthesis
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

                # Sine wave with simple ADSR envelope
                wave = np.sin(2 * np.pi * freq * t)

                # Simple envelope: attack 20ms, release 50ms
                env = np.ones_like(wave)
                attack_samples = min(int(0.02 * sr), len(env))
                release_samples = min(int(0.05 * sr), len(env))
                if attack_samples > 0:
                    env[:attack_samples] = np.linspace(0, 1, attack_samples)
                if release_samples > 0:
                    env[-release_samples:] = np.linspace(1, 0, release_samples)

                wave *= env * (note.velocity / 127.0) * 0.3
                audio[t_start:t_start + len(wave)] += wave

        # Normalize
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.8

        logger.info("Audio synthesized via sine waves")

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav_path), audio, sr)
    logger.info("WAV written: %s (%.1fs, %d Hz)", wav_path, len(audio) / sr, sr)
    return wav_path


# ── Step 3: Run transcription ────────────────────────────────────────


def run_transcription(wav_path: Path, midi_out: Path) -> Path:
    """Transcribe audio to MIDI using Basic Pitch."""
    logger.info("Transcribing %s…", wav_path.name)
    return transcribe_to_midi(
        wav_path,
        midi_out,
        onset_threshold=0.5,
        frame_threshold=0.3,
        minimum_note_length=58.0,
    )


# ── Step 4: Build score ──────────────────────────────────────────────


def build_output_score(midi_path: Path, output_path: Path) -> Path:
    """Assemble a score from the transcribed MIDI and export."""
    score = build_score(
        {"transcription": midi_path},
        title="E2E Test Transcription",
    )
    return export_score(score, output_path)


# ── Step 5: Evaluate ─────────────────────────────────────────────────


def run_evaluation(
    gt_midi: Path,
    est_midi: Path,
) -> dict:
    """Run evaluation and return metrics dict."""
    # Evaluate with standard tolerance (50ms onset, 50 cents pitch)
    strict = evaluate(gt_midi, est_midi, onset_tolerance=0.05, pitch_tolerance=50.0)
    # Evaluate with relaxed tolerance (200ms onset, 50 cents pitch)
    relaxed = evaluate(gt_midi, est_midi, onset_tolerance=0.2, pitch_tolerance=50.0)

    gt_notes = extract_notes(gt_midi)
    est_notes = extract_notes(est_midi)

    return {
        "ground_truth_notes": len(gt_notes),
        "estimated_notes": len(est_notes),
        "strict_50ms": {
            "precision": strict.precision,
            "recall": strict.recall,
            "f1": strict.f1,
            "matched": strict.matched_notes,
            "loss": strict.loss,
            "onset_error_ms": strict.onset_error_ms_mean,
            "offset_error_ms": strict.offset_error_ms_mean,
        },
        "relaxed_200ms": {
            "precision": relaxed.precision,
            "recall": relaxed.recall,
            "f1": relaxed.f1,
            "matched": relaxed.matched_notes,
            "loss": relaxed.loss,
            "onset_error_ms": relaxed.onset_error_ms_mean,
            "offset_error_ms": relaxed.offset_error_ms_mean,
        },
    }


# ── Main ─────────────────────────────────────────────────────────────


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("choir2sheet — End-to-End Test")
    print("=" * 70)

    # Step 1: Create ground-truth MIDI
    print("\n[1/5] Creating ground-truth MIDI chorale…")
    gt_midi = create_test_chorale(OUTPUT_DIR / "ground_truth.mid")

    # Step 2: Synthesize audio
    print("[2/5] Synthesizing audio from MIDI…")
    wav_path = synthesize_audio(gt_midi, OUTPUT_DIR / "test_audio.wav")

    # Step 3: Transcribe
    print("[3/5] Transcribing audio with Basic Pitch…")
    est_midi = run_transcription(wav_path, OUTPUT_DIR / "transcribed.mid")

    # Step 4: Build score
    print("[4/5] Building sheet music score…")
    score_xml = build_output_score(est_midi, OUTPUT_DIR / "score.musicxml")
    print(f"  → MusicXML: {score_xml}")

    # Step 5: Evaluate
    print("[5/5] Evaluating transcription accuracy…")
    results = run_evaluation(gt_midi, est_midi)

    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Ground truth notes: {results['ground_truth_notes']}")
    print(f"Estimated notes:    {results['estimated_notes']}")

    for label, key in [("Strict (50ms onset)", "strict_50ms"),
                       ("Relaxed (200ms onset)", "relaxed_200ms")]:
        m = results[key]
        print(f"\n  {label}:")
        print(f"    Precision:   {m['precision']:.3f}")
        print(f"    Recall:      {m['recall']:.3f}")
        print(f"    F1 Score:    {m['f1']:.3f}")
        print(f"    Matched:     {m['matched']}")
        print(f"    Loss:        {m['loss']:.4f}")
        print(f"    Onset err:   {m['onset_error_ms']:.1f} ms")
        print(f"    Offset err:  {m['offset_error_ms']:.1f} ms")

    # Save results to JSON
    results_path = OUTPUT_DIR / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Pass/fail based on relaxed F1
    relaxed_f1 = results["relaxed_200ms"]["f1"]
    if relaxed_f1 >= 0.3:
        print(f"\n✓ PASS — Relaxed F1 = {relaxed_f1:.3f} (threshold: 0.3)")
    else:
        print(f"\n✗ FAIL — Relaxed F1 = {relaxed_f1:.3f} (threshold: 0.3)")
        sys.exit(1)

    print(f"\nAll outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

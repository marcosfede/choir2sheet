#!/usr/bin/env python3
"""Vocal benchmark against VOCADITO dataset.

VOCADITO contains 40 solo vocal excerpts with note-level ground truth,
created by the same team behind Basic Pitch. This tests our pipeline's
ability to transcribe singing voice.

Dataset: https://zenodo.org/records/5578807
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from choir2sheet.transcriber import transcribe_to_midi
from choir2sheet.metrics import NoteEvent, TranscriptionMetrics, notes_to_mir_eval

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_vocal")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "test_output" / "vocadito"
VOCADITO_URL = "https://zenodo.org/api/records/5578807/files/vocadito.zip/content"


def download_vocadito(output_dir: Path) -> Path:
    """Download and extract VOCADITO dataset."""
    zip_path = output_dir / "vocadito.zip"
    audio_dir = output_dir / "Audio"

    if audio_dir.is_dir() and list(audio_dir.glob("*.wav")):
        logger.info("VOCADITO already downloaded")
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    if not zip_path.is_file():
        logger.info("Downloading VOCADITO (56 MB)…")
        urlretrieve(VOCADITO_URL, str(zip_path))

    logger.info("Extracting…")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(str(output_dir))

    return output_dir


def load_vocadito_notes(csv_path: Path) -> list[NoteEvent]:
    """Load VOCADITO note annotations (start_time, pitch_hz, duration)."""
    notes = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            onset = float(row[0])
            pitch_hz = float(row[1])
            duration = float(row[2])
            # Convert Hz to MIDI note number
            if pitch_hz <= 0:
                continue
            midi_note = int(round(12 * np.log2(pitch_hz / 440.0) + 69))
            midi_note = max(0, min(127, midi_note))
            notes.append(NoteEvent(
                pitch=midi_note,
                onset=onset,
                offset=onset + duration,
            ))
    notes.sort(key=lambda n: (n.onset, n.pitch))
    return notes


def vocadito_notes_to_midi(notes: list[NoteEvent], output_path: Path) -> Path:
    """Write VOCADITO note annotations to a MIDI file for evaluation."""
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, name="vocals")
    for n in notes:
        inst.notes.append(pretty_midi.Note(
            velocity=80,
            pitch=n.pitch,
            start=n.onset,
            end=n.offset,
        ))
    midi.instruments.append(inst)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(output_path))
    return output_path


def evaluate_vocal(
    ref_notes: list[NoteEvent],
    est_notes: list[NoteEvent],
    onset_tolerance: float = 0.05,
    pitch_tolerance: float = 50.0,
    offset_ratio: float | None = None,
) -> TranscriptionMetrics:
    """Evaluate transcription using mir_eval."""
    import mir_eval

    ref_intervals, ref_pitches = notes_to_mir_eval(ref_notes)
    est_intervals, est_pitches = notes_to_mir_eval(est_notes)

    metrics = TranscriptionMetrics(
        total_ref_notes=len(ref_notes),
        total_est_notes=len(est_notes),
    )

    if len(ref_notes) == 0 or len(est_notes) == 0:
        metrics.loss = 1.0
        return metrics

    prec, rec, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches,
        est_intervals, est_pitches,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=pitch_tolerance,
        offset_ratio=offset_ratio,
    )

    matching = mir_eval.transcription.match_notes(
        ref_intervals, ref_pitches,
        est_intervals, est_pitches,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=pitch_tolerance,
        offset_ratio=offset_ratio,
    )

    metrics.precision = float(prec)
    metrics.recall = float(rec)
    metrics.f1 = float(f1)
    metrics.matched_notes = len(matching)
    metrics.loss = 1.0 - float(f1)

    if matching:
        onset_errors = []
        for ref_idx, est_idx in matching:
            rn = ref_notes[ref_idx]
            en = est_notes[est_idx]
            onset_errors.append(abs(rn.onset - en.onset) * 1000)
        metrics.onset_error_ms_mean = float(np.mean(onset_errors))

    return metrics


def extract_est_notes(midi_path: Path) -> list[NoteEvent]:
    """Extract notes from a transcribed MIDI file."""
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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    midi_dir = OUTPUT_DIR / "transcribed_midi"
    midi_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("choir2sheet — VOCADITO Vocal Benchmark")
    print("40 solo vocal excerpts with note-level ground truth")
    print("=" * 70)

    # Step 1: Download
    print("\n[1/3] Downloading VOCADITO dataset…")
    data_dir = download_vocadito(OUTPUT_DIR)

    # Find all audio + annotation pairs
    audio_dir = data_dir / "Audio"
    notes_dir = data_dir / "Annotations" / "Notes"
    audio_files = sorted(audio_dir.glob("vocadito_*.wav"))

    print(f"Found {len(audio_files)} audio files")

    # Step 2: Transcribe and evaluate each
    print("[2/3] Transcribing and evaluating each excerpt…\n")

    all_results = []
    aggregate_ref = 0
    aggregate_est = 0
    aggregate_matched = 0
    f1_scores = []

    for i, wav_path in enumerate(audio_files):
        track_id = wav_path.stem  # e.g. "vocadito_10"

        # Load ground truth (use annotator 1)
        # Filename pattern: vocadito_10_notesA1.csv
        notes_file = notes_dir / f"{track_id}_notesA1.csv"
        if not notes_file.is_file():
            logger.warning("No annotations for %s, skipping", track_id)
            continue

        ref_notes = load_vocadito_notes(notes_file)
        if not ref_notes:
            continue

        # Transcribe
        est_midi_path = midi_dir / f"{track_id}.mid"
        try:
            transcribe_to_midi(
                wav_path, est_midi_path,
                onset_threshold=0.5,
                frame_threshold=0.3,
                minimum_note_length=58.0,
            )
        except Exception as e:
            logger.error("Failed to transcribe %s: %s", track_id, e)
            continue

        est_notes = extract_est_notes(est_midi_path)

        # Evaluate (onset-only, 50ms)
        m = evaluate_vocal(ref_notes, est_notes,
                           onset_tolerance=0.05,
                           pitch_tolerance=50.0,
                           offset_ratio=None)

        all_results.append({
            "track": track_id,
            "ref_notes": len(ref_notes),
            "est_notes": len(est_notes),
            "matched": m.matched_notes,
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1,
            "onset_err_ms": m.onset_error_ms_mean,
        })

        aggregate_ref += len(ref_notes)
        aggregate_est += len(est_notes)
        aggregate_matched += m.matched_notes
        f1_scores.append(m.f1)

        status = "OK" if m.f1 >= 0.3 else "LOW"
        print(f"  [{i+1:2d}/{len(audio_files)}] {track_id}: "
              f"F1={m.f1:.3f} P={m.precision:.3f} R={m.recall:.3f} "
              f"({m.matched_notes}/{len(ref_notes)} matched) [{status}]")

    # Step 3: Aggregate results
    print("\n[3/3] Computing aggregate metrics…")

    if not f1_scores:
        print("No tracks evaluated!")
        sys.exit(1)

    mean_f1 = np.mean(f1_scores)
    median_f1 = np.median(f1_scores)
    macro_precision = aggregate_matched / aggregate_est if aggregate_est else 0
    macro_recall = aggregate_matched / aggregate_ref if aggregate_ref else 0

    print("\n" + "=" * 70)
    print("VOCADITO VOCAL BENCHMARK RESULTS (onset-only, 50ms, 50 cents)")
    print("=" * 70)
    print(f"Tracks evaluated:     {len(f1_scores)}")
    print(f"Total ref notes:      {aggregate_ref}")
    print(f"Total est notes:      {aggregate_est}")
    print(f"Total matched:        {aggregate_matched}")
    print(f"Macro Precision:      {macro_precision:.3f}")
    print(f"Macro Recall:         {macro_recall:.3f}")
    print(f"Mean F1 (per-track):  {mean_f1:.3f}")
    print(f"Median F1:            {median_f1:.3f}")
    print(f"Min F1:               {min(f1_scores):.3f}")
    print(f"Max F1:               {max(f1_scores):.3f}")

    # Save results
    summary = {
        "dataset": "VOCADITO (solo vocals)",
        "metric": "onset-only, 50ms tolerance, 50 cents pitch",
        "tracks_evaluated": len(f1_scores),
        "total_ref_notes": aggregate_ref,
        "total_est_notes": aggregate_est,
        "total_matched": aggregate_matched,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "mean_f1": mean_f1,
        "median_f1": median_f1,
        "min_f1": min(f1_scores),
        "max_f1": max(f1_scores),
        "per_track": all_results,
    }

    results_path = OUTPUT_DIR / "vocal_benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    if mean_f1 >= 0.3:
        print(f"\nPASS — Mean F1 = {mean_f1:.3f} (threshold: 0.3)")
    else:
        print(f"\nFAIL — Mean F1 = {mean_f1:.3f} (threshold: 0.3)")
        sys.exit(1)


if __name__ == "__main__":
    main()

from __future__ import annotations

import pytest

from choir2sheet.metrics import NoteEvent, evaluate, evaluate_stems, extract_notes, notes_to_mir_eval
from tests.conftest import write_midi


def test_extract_notes_sorted(scale_midi, scale_notes):
    notes = extract_notes(scale_midi)
    assert [n.pitch for n in notes] == [p for p, _, _ in scale_notes]
    assert notes[0].duration == pytest.approx(0.9)


def test_notes_to_mir_eval_shapes():
    intervals, hz = notes_to_mir_eval([NoteEvent(69, 0.0, 1.0)])
    assert intervals.shape == (1, 2)
    assert hz[0] == pytest.approx(440.0)
    empty_i, empty_p = notes_to_mir_eval([])
    assert empty_i.shape == (0, 2) and empty_p.shape == (0,)


def test_evaluate_identical_is_perfect(scale_midi):
    m = evaluate(scale_midi, scale_midi)
    assert m.f1 == 1.0 and m.precision == 1.0 and m.recall == 1.0
    assert m.matched_notes == 8
    assert m.loss == 0.0
    assert m.onset_error_ms_mean == 0.0


def test_evaluate_partial_match(tmp_path, scale_midi, scale_notes):
    # Drop last two notes and shift one onset by 30 ms (inside 50 ms tolerance)
    est = list(scale_notes[:6])
    p, on, off = est[0]
    est[0] = (p, on + 0.03, off)
    est_path = write_midi(tmp_path / "est.mid", est)
    m = evaluate(scale_midi, est_path, offset_ratio=None)
    assert m.matched_notes == 6
    assert m.recall == pytest.approx(6 / 8)
    assert m.precision == pytest.approx(1.0)
    assert m.onset_error_ms_mean == pytest.approx(5.0, abs=1.0)


def test_evaluate_wrong_pitch_is_miss(tmp_path, scale_midi, scale_notes):
    est_path = write_midi(tmp_path / "est.mid", [(p + 1, on, off) for p, on, off in scale_notes])
    m = evaluate(scale_midi, est_path)
    assert m.f1 == 0.0 and m.loss == 1.0


def test_evaluate_empty_cases(tmp_path, scale_midi):
    empty = write_midi(tmp_path / "empty.mid", [])
    assert evaluate(empty, empty).f1 == 1.0
    assert evaluate(scale_midi, empty).loss == 1.0
    assert evaluate(empty, scale_midi).total_est_notes == 8


def test_evaluate_stems_matches_by_filename(tmp_path, scale_notes):
    ref = tmp_path / "ref"
    est = tmp_path / "est"
    ref.mkdir()
    est.mkdir()
    write_midi(ref / "soprano.mid", scale_notes)
    write_midi(est / "soprano.mid", scale_notes)
    write_midi(ref / "alto.mid", scale_notes)  # no estimate → skipped
    results = evaluate_stems(ref, est)
    assert list(results) == ["soprano"]
    assert results["soprano"].f1 == 1.0

from __future__ import annotations

from music21 import converter, key, meter, tempo

from choir2sheet.postprocess import (
    apply_key,
    detect_key,
    detect_key_from_midi,
    quantize_midi,
    quantize_score,
)
from tests.conftest import write_midi


def test_detect_key_from_midi_reports_c_major_family(scale_midi):
    result = detect_key_from_midi(scale_midi)
    candidates = {result.key} | {alt["key"] for alt in result.alternates}
    assert "C major" in candidates
    assert 0.0 < result.correlation <= 1.0
    assert result.to_dict()["tonic"] == result.tonic


def test_detect_key_unknown_algorithm(scale_midi):
    score = converter.parse(str(scale_midi))
    try:
        detect_key(score, algorithm="nope")
    except ValueError as e:
        assert "nope" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_detect_key_g_major(tmp_path):
    g_major = [67, 69, 71, 72, 74, 76, 78, 79]
    path = write_midi(tmp_path / "g.mid", [(p, float(i), i + 0.9) for i, p in enumerate(g_major)])
    result = detect_key_from_midi(path)
    assert result.tonic in {"G", "E"}  # G major or its relative minor


def test_apply_key_sets_signature_on_every_part(scale_midi):
    score = converter.parse(str(scale_midi))
    apply_key(score, "F# minor")
    for part in score.parts:
        ks = part.getElementsByClass(key.Key)
        assert len(ks) == 1
        assert ks[0].tonic.name == "F#" and ks[0].mode == "minor"


def test_quantize_score_sets_meter_and_tempo(scale_midi):
    score = converter.parse(str(scale_midi))
    quantized, result = quantize_score(score, tempo_bpm=120, time_sig="4/4")
    assert result.time_signature == "4/4"
    assert result.tempo_bpm == 120
    assert result.notes_before == 8
    assert result.notes_after == 8
    part = quantized.parts[0]
    assert part.flatten().getElementsByClass(meter.TimeSignature)[0].ratioString == "4/4"
    assert part.flatten().getElementsByClass(tempo.MetronomeMark)[0].number == 120


def test_quantize_score_ties_across_barlines(scale_midi):
    # Half notes in 3/4: every other note straddles a barline and is split into a tie.
    score = converter.parse(str(scale_midi))
    _, result = quantize_score(score, tempo_bpm=120, time_sig="3/4")
    assert result.notes_after == 12


def test_quantize_midi_exports_musicxml(scale_midi, tmp_path):
    out, result = quantize_midi(scale_midi, tmp_path / "q.musicxml", tempo_bpm=120)
    assert out.is_file()
    reparsed = converter.parse(str(out))
    pitches = [n.pitch.midi for n in reparsed.flatten().notes]
    assert pitches == [60, 62, 64, 65, 67, 69, 71, 72]
    onsets = [float(n.offset) for n in reparsed.flatten().notes]
    assert onsets == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]

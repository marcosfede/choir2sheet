from __future__ import annotations

import numpy as np
import soundfile as sf
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


def test_quantize_score_rescales_to_target_tempo(scale_midi):
    # Source MIDI is 120 BPM with notes 1 s apart. At 60 BPM that is one
    # quarter note apart; at 120 BPM two quarter notes apart.
    slow, res = quantize_score(converter.parse(str(scale_midi)), tempo_bpm=60)
    assert res.grid == 0.25
    assert [float(n.offset) for n in slow.flatten().notes] == [float(i) for i in range(8)]
    fast, _ = quantize_score(converter.parse(str(scale_midi)), tempo_bpm=120)
    assert [float(n.offset) for n in fast.flatten().notes] == [2.0 * i for i in range(8)]


def test_quantize_score_snaps_to_grid(scale_midi):
    quantized, _ = quantize_score(
        converter.parse(str(scale_midi)), tempo_bpm=120, grid=0.5
    )
    for n in quantized.flatten().notes:
        assert float(n.offset) % 0.5 == 0
        assert float(n.quarterLength) % 0.5 == 0
        assert n.quarterLength > 0


def test_quantize_score_ties_across_barlines(scale_midi):
    # Notes every 2 beats in 3/4: those starting on beat 2 (ql 2, 8, 14)
    # straddle a barline and are split into ties.
    score = converter.parse(str(scale_midi))
    _, result = quantize_score(score, tempo_bpm=120, time_sig="3/4")
    assert result.notes_after == 11


def test_quantize_midi_exports_musicxml(scale_midi, tmp_path):
    out, result = quantize_midi(scale_midi, tmp_path / "q.musicxml", tempo_bpm=120)
    assert out.is_file()
    reparsed = converter.parse(str(out))
    pitches = [n.pitch.midi for n in reparsed.flatten().notes]
    assert pitches == [60, 62, 64, 65, 67, 69, 71, 72]
    onsets = [float(n.offset) for n in reparsed.flatten().notes]
    assert onsets == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]


def test_estimate_tempo_from_click_track(tmp_path):
    from choir2sheet.tempo import estimate_tempo, fold_bpm

    sr = 22050
    bpm = 100.0
    t = np.arange(0, 12.0, 1 / sr)
    y = np.zeros_like(t)
    for beat in np.arange(0, 12.0, 60.0 / bpm):
        seg = (t >= beat) & (t < beat + 0.03)
        y[seg] = np.sin(2 * np.pi * 1000 * t[seg])
    path = tmp_path / "click.wav"
    sf.write(path, y, sr)
    assert abs(estimate_tempo(path) - bpm) < 3
    assert fold_bpm(400) == 100
    assert fold_bpm(25) == 100

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from choir2sheet.cli import main
from tests.conftest import write_midi


def run(*args: str) -> dict:
    result = CliRunner().invoke(main, ["-q", *args])
    assert result.stdout, result.exception
    return json.loads(result.stdout)


def test_formats():
    data = run("formats")
    assert data["ok"] is True
    assert "musicxml" in data["formats"]


def test_detect_key(scale_midi):
    data = run("detect-key", str(scale_midi))
    assert data["ok"] is True
    assert {"key", "tonic", "mode", "correlation", "alternates"} <= data.keys()


def test_quantize(scale_midi, tmp_path):
    out = tmp_path / "q.musicxml"
    data = run("quantize", str(scale_midi), str(out), "--tempo", "120", "--time-sig", "4/4")
    assert data["ok"] is True
    assert data["output"] == str(out)
    assert data["time_signature"] == "4/4"
    assert data["notes_after"] == 8
    assert out.is_file()


def test_evaluate(scale_midi):
    data = run("evaluate", str(scale_midi), str(scale_midi))
    assert data["ok"] is True
    assert data["f1"] == 1.0
    assert data["matched_notes"] == 8


def test_error_is_json(scale_midi, tmp_path):
    result = CliRunner().invoke(main, ["-q", "quantize", str(scale_midi), str(tmp_path / "x.docx")])
    assert result.exit_code == 1
    data = json.loads(result.stdout)
    assert data["ok"] is False
    assert "Unknown output format" in data["error"]


def test_missing_input_path(tmp_path):
    result = CliRunner().invoke(main, ["detect-key", str(tmp_path / "missing.mid")])
    assert result.exit_code == 2


@pytest.mark.slow
def test_midi_transcribe_and_full_pipeline(scale_wav, scale_notes, tmp_path):
    midi_out = tmp_path / "scale.mid"
    data = run("midi-transcribe", str(scale_wav), str(midi_out))
    assert data["ok"] is True
    assert midi_out.is_file()
    assert 6 <= data["notes"] <= 12

    ref = write_midi(tmp_path / "ref.mid", scale_notes)
    ev = run("evaluate", str(ref), str(midi_out))
    assert ev["f1"] >= 0.8

    score_out = tmp_path / "scale.musicxml"
    data = run(
        "transcribe", str(scale_wav), str(score_out),
        "--skip-separation", "--detect-key", "--quantize", "--tempo", "120",
    )
    assert data["ok"] is True
    assert score_out.is_file()
    assert data["stages"]["postprocess"]["key"]["source"] == "detected"
    assert data["stages"]["postprocess"]["quantization"]["tempo_bpm"] == 120.0

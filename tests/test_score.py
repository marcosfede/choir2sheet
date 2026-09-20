from __future__ import annotations

import shutil

import pytest
from music21 import converter, instrument

from choir2sheet.score import EXPORT_FORMATS, build_score, export_score, list_formats
from tests.conftest import write_midi


def test_list_formats_matches_export_table():
    assert list_formats() == sorted(EXPORT_FORMATS)
    assert {"musicxml", "mid", "ly", "abc", "pdf"} <= set(list_formats())


def test_build_score_assigns_instruments_and_ids(tmp_path, scale_notes):
    sop = write_midi(tmp_path / "s.mid", scale_notes)
    piano = write_midi(tmp_path / "p.mid", scale_notes)
    score = build_score({"soprano": sop, "piano": piano}, title="T", composer="C")

    assert score.metadata.title == "T"
    assert score.metadata.composer == "C"
    ids = [p.id for p in score.parts]
    assert ids == ["soprano", "piano"]
    assert isinstance(score.parts[0].getInstrument(), instrument.Soprano)
    assert isinstance(score.parts[1].getInstrument(), instrument.Piano)
    assert len(score.parts[0].flatten().notes) == 8


def test_build_score_skips_missing_files(tmp_path, scale_midi):
    score = build_score({"vocals": scale_midi, "ghost": tmp_path / "nope.mid"})
    assert [p.id for p in score.parts] == ["vocals"]


@pytest.mark.parametrize("ext", ["musicxml", "xml", "mid", "midi", "ly", "abc"])
def test_export_roundtrip_formats(tmp_path, scale_midi, ext):
    if ext == "ly" and shutil.which("lilypond") is None:
        pytest.skip("lilypond not installed")
    score = build_score({"vocals": scale_midi})
    out = export_score(score, tmp_path / f"out.{ext}")
    assert out.is_file() and out.stat().st_size > 0
    if ext in {"musicxml", "xml", "mid", "midi"}:
        reparsed = converter.parse(str(out))
        assert [n.pitch.midi for n in reparsed.flatten().notes] == [
            60, 62, 64, 65, 67, 69, 71, 72,
        ]


def test_export_unknown_extension(tmp_path, scale_midi):
    score = build_score({"vocals": scale_midi})
    with pytest.raises(ValueError, match="Unknown output format"):
        export_score(score, tmp_path / "out.docx")


def test_export_explicit_format_overrides_extension(tmp_path, scale_midi):
    score = build_score({"vocals": scale_midi})
    out = export_score(score, tmp_path / "weird.dat", fmt="mid")
    reparsed = converter.parse(str(out), format="midi")
    assert len(reparsed.flatten().notes) == 8

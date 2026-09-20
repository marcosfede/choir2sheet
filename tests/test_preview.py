import json
import urllib.request

import pytest
from music21 import metadata, note, stream

from choir2sheet.preview import make_server, prepare_preview, serve_in_thread


@pytest.fixture
def score_file(tmp_path):
    s = stream.Score()
    s.metadata = metadata.Metadata(title="Test Piece")
    for name, pitch in (("Soprano", "C5"), ("Alto", "E4")):
        p = stream.Part(id=name)
        p.partName = name
        for _ in range(4):
            p.append(note.Note(pitch, quarterLength=1))
        s.append(p)
    path = tmp_path / "test.musicxml"
    s.write("musicxml", fp=str(path))
    return path


def test_prepare_preview_exports_and_manifest(score_file, tmp_path):
    audio_dir = tmp_path / "stems"
    audio_dir.mkdir()
    (audio_dir / "soprano.wav").write_bytes(b"RIFF")
    (audio_dir / "notes.txt").write_text("ignored")

    prepared = prepare_preview(score_file, tmp_path / "work", audio_dir=audio_dir)
    files, manifest = prepared["files"], prepared["manifest"]

    assert files["score.musicxml"].is_file()
    assert files["score.mid"].is_file()
    assert manifest["title"] == "Test Piece"
    assert manifest["audio"] == {"soprano": "/files/audio/soprano.wav"}
    assert "audio/notes.txt" not in files


def test_prepare_preview_accepts_midi_input(score_file, tmp_path):
    prepared = prepare_preview(score_file, tmp_path / "w1")
    midi_in = prepared["files"]["score.mid"]
    prepared2 = prepare_preview(midi_in, tmp_path / "w2")
    assert prepared2["files"]["score.musicxml"].is_file()


def test_prepare_preview_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare_preview(tmp_path / "nope.musicxml", tmp_path / "w")


def test_server_routes(score_file, tmp_path):
    prepared = prepare_preview(score_file, tmp_path / "work")
    server = make_server(prepared["files"], prepared["manifest"], port=0)
    serve_in_thread(server)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        html = urllib.request.urlopen(f"{base}/").read().decode()
        assert "opensheetmusicdisplay" in html

        manifest = json.loads(urllib.request.urlopen(f"{base}/manifest.json").read())
        assert manifest["score"] == "/files/score.musicxml"

        mid = urllib.request.urlopen(f"{base}{manifest['midi']}").read()
        assert mid[:4] == b"MThd"

        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(f"{base}/files/../pyproject.toml")
        assert exc.value.code == 404
    finally:
        server.shutdown()
        server.server_close()

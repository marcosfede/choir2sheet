from __future__ import annotations

from pathlib import Path

import pytest

from choir2sheet import separator


class _Resp:
    def __init__(self, payload=None, content=b""):
        self._payload = payload
        self.content = content

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_run_demucs_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        separator.run_demucs(tmp_path / "nope.wav", tmp_path)


def test_run_demucs_collects_stems(tmp_path, monkeypatch, scale_wav):
    out = tmp_path / "out"

    def fake_run(cmd, check):
        stem_dir = out / separator.DEMUCS_MODEL_6S / scale_wav.stem
        stem_dir.mkdir(parents=True)
        for name in ("vocals", "piano", "drums"):
            (stem_dir / f"{name}.wav").write_bytes(b"")

    monkeypatch.setattr(separator.subprocess, "run", fake_run)
    stems = separator.run_demucs(scale_wav, out)
    assert set(stems) == {"vocals", "piano", "drums"}
    assert all(isinstance(p, Path) for p in stems.values())


def test_mvsep_separate_polls_and_downloads(tmp_path, monkeypatch, scale_wav):
    calls = {"status": 0}

    def fake_post(url, data, files, timeout):
        assert url.endswith("/separation/create")
        assert data["sep_type"] == str(separator.MVSEP_SATB_MODEL)
        assert data["api_key"] == "k"
        return _Resp({"data": {"hash": "abc"}})

    def fake_get(url, params=None, timeout=None):
        if url.endswith("/separation/result"):
            calls["status"] += 1
            if calls["status"] < 2:
                return _Resp({"status": "waiting"})
            return _Resp({
                "status": "done",
                "data": {"files": [
                    {"url": "https://x/soprano.wav", "name": "soprano.wav"},
                    {"url": "https://x/Alto.wav", "name": "Alto.wav"},
                ]},
            })
        return _Resp(content=b"RIFF")

    monkeypatch.setattr(separator.requests, "post", fake_post)
    monkeypatch.setattr(separator.requests, "get", fake_get)
    monkeypatch.setattr(separator.time, "sleep", lambda s: None)

    stems = separator.mvsep_separate(scale_wav, tmp_path / "satb", api_key="k", poll_interval=1)
    assert set(stems) == {"soprano", "alto"}
    assert stems["alto"].read_bytes() == b"RIFF"


def test_mvsep_separate_times_out(tmp_path, monkeypatch, scale_wav):
    monkeypatch.setattr(separator.requests, "post",
                        lambda *a, **k: _Resp({"data": {"hash": "abc"}}))
    monkeypatch.setattr(separator.requests, "get",
                        lambda *a, **k: _Resp({"status": "waiting"}))
    monkeypatch.setattr(separator.time, "sleep", lambda s: None)
    with pytest.raises(TimeoutError):
        separator.mvsep_separate(scale_wav, tmp_path, poll_interval=1, timeout=3)


def test_separate_choir_falls_back_to_vocals(tmp_path, monkeypatch, scale_wav):
    monkeypatch.setattr(
        separator, "run_demucs",
        lambda *a, **k: {"vocals": tmp_path / "vocals.wav", "piano": tmp_path / "piano.wav"},
    )

    def boom(*a, **k):
        raise RuntimeError("mvsep down")

    monkeypatch.setattr(separator, "mvsep_separate", boom)
    stems = separator.separate_choir(scale_wav, tmp_path / "w", satb_backend="mvsep")
    assert set(stems) == {"vocals", "piano"}


def test_separate_choir_sepacap_renames_colliding_instrument(tmp_path, monkeypatch, scale_wav):
    monkeypatch.setattr(
        separator, "run_demucs",
        lambda *a, **k: {"vocals": tmp_path / "vocals.wav", "bass": tmp_path / "bass.wav"},
    )
    monkeypatch.setattr(
        separator, "sepacap_separate",
        lambda *a, **k: {"soprano": tmp_path / "s.wav", "bass": tmp_path / "vb.wav"},
    )
    stems = separator.separate_choir(scale_wav, tmp_path / "w")
    assert stems == {
        "bass_inst": tmp_path / "bass.wav",
        "soprano": tmp_path / "s.wav",
        "bass": tmp_path / "vb.wav",
    }


def test_separate_choir_empty_split_falls_back(tmp_path, monkeypatch, scale_wav):
    monkeypatch.setattr(separator, "run_demucs", lambda *a, **k: {"vocals": tmp_path / "v.wav"})
    monkeypatch.setattr(separator, "sepacap_separate", lambda *a, **k: {})
    stems = separator.separate_choir(scale_wav, tmp_path / "w")
    assert set(stems) == {"vocals"}


def test_separate_choir_rejects_unknown_backend(tmp_path, scale_wav):
    with pytest.raises(ValueError):
        separator.separate_choir(scale_wav, tmp_path / "w", satb_backend="magic")


def test_separate_choir_skip_satb(tmp_path, monkeypatch, scale_wav):
    monkeypatch.setattr(separator, "run_demucs", lambda *a, **k: {"vocals": tmp_path / "v.wav"})
    stems = separator.separate_choir(scale_wav, tmp_path / "w", skip_satb=True)
    assert set(stems) == {"vocals"}

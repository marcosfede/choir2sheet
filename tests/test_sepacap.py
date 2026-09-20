from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from choir2sheet import sepacap

torch = pytest.importorskip("torch")


class _PassThrough:
    """Fake SepACap: puts the input in the soprano slot, silence elsewhere."""

    def __call__(self, x):
        idx = sepacap.SEPACAP_STEMS.index("soprano")
        return [x if i == idx else torch.zeros_like(x) for i in range(len(sepacap.SEPACAP_STEMS))], None


def test_separate_array_overlap_add_reconstructs_input():
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(int(2.5 * sepacap.SEPACAP_SR)).astype(np.float32)
    out = sepacap.separate_array(_PassThrough(), audio, chunk_seconds=1.0)
    assert out.shape == (len(sepacap.SEPACAP_STEMS), len(audio))
    sop = out[sepacap.SEPACAP_STEMS.index("soprano")]
    # Interior is exactly reconstructed; the first half-window has only one
    # (Hann-weighted) contribution and is attenuated.
    hop = sepacap.SEPACAP_SR // 2
    np.testing.assert_allclose(sop[hop:-hop], audio[hop:-hop], atol=1e-4)
    assert np.abs(out[sepacap.SEPACAP_STEMS.index("alto")]).max() == 0


def test_sepacap_separate_drops_silent_and_percussion(tmp_path, monkeypatch):
    sr = sepacap.SEPACAP_SR
    t = np.arange(0, 3.0, 1 / sr)
    sf.write(tmp_path / "v.wav", 0.3 * np.sin(2 * np.pi * 440 * t), sr)
    monkeypatch.setattr(sepacap, "load_sepacap", lambda device: _PassThrough())

    stems = sepacap.sepacap_separate(tmp_path / "v.wav", tmp_path / "out")
    assert set(stems) == {"soprano"}
    y, out_sr = sf.read(stems["soprano"])
    assert out_sr == sr and len(y) == len(t)

    stems = sepacap.sepacap_separate(tmp_path / "v.wav", tmp_path / "out2", keep_silent=True)
    assert set(stems) == set(sepacap.VOICE_STEMS)

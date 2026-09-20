"""Local multi-singer separation with SepACap (a cappella → voice parts).

SepACap (ETH-DISCO, ICASSP 2026) splits an a cappella mixture into
lead / soprano / alto / tenor / bass (+ vocal-percussion stems we discard).
It runs on CPU: roughly 4x real time on a laptop-class core count.
Weights are fetched once from Hugging Face into the HF cache.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

SEPACAP_REPO = "Tino3141/sepacap"
SEPACAP_CHECKPOINT = "SepACap.pth"
SEPACAP_CONFIG = "modelMusicSep.yaml"
SEPACAP_SR = 24000

# Output order of the JaCappella-trained model.
SEPACAP_STEMS = (
    "alto",
    "bass",
    "finger_snap",
    "lead",
    "soprano",
    "tenor",
    "vocal_percussion",
)
VOICE_STEMS = ("lead", "soprano", "alto", "tenor", "bass")

# Trained on 4 s excerpts (``dataset.max_len`` = 96000 @ 24 kHz).
CHUNK_SECONDS = 4.0
# Stems quieter than this fraction of the input RMS are treated as absent.
SILENT_STEM_RATIO = 0.02


def download_sepacap() -> tuple[Path, Path]:
    """Fetch (and cache) the SepACap checkpoint and config; returns their paths."""
    from huggingface_hub import hf_hub_download

    ckpt = Path(hf_hub_download(SEPACAP_REPO, SEPACAP_CHECKPOINT))
    cfg = Path(hf_hub_download(SEPACAP_REPO, SEPACAP_CONFIG))
    return ckpt, cfg


def load_sepacap(device: str = "cpu"):
    """Build the SepACap network and load the pretrained weights."""
    import torch
    import yaml

    from .sepacap_model import Model

    ckpt_path, cfg_path = download_sepacap()
    with open(cfg_path) as f:
        model_cfg = yaml.safe_load(f)["config"]["model"]
    model = Model(**model_cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state"]
    # Checkpoint was saved from a DataParallel wrapper.
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model.to(device)


def separate_array(
    model,
    audio: np.ndarray,
    *,
    device: str = "cpu",
    chunk_seconds: float = CHUNK_SECONDS,
) -> np.ndarray:
    """Separate a mono 24 kHz signal into ``len(SEPACAP_STEMS)`` stems.

    Runs the model on Hann-windowed chunks with 50% overlap and overlap-adds
    the results (the model is only trained on ``chunk_seconds`` excerpts).
    Returns an array of shape ``(n_stems, n_samples)``.
    """
    import torch

    n = len(audio)
    chunk = int(chunk_seconds * SEPACAP_SR)
    hop = chunk // 2
    window = np.hanning(chunk).astype(np.float32)
    out = np.zeros((len(SEPACAP_STEMS), n), dtype=np.float32)
    norm = np.zeros(n, dtype=np.float32)

    x = torch.from_numpy(audio.astype(np.float32))
    with torch.no_grad():
        for start in range(0, max(n, 1), hop):
            seg = x[start:start + chunk]
            seg_len = len(seg)
            if seg_len == 0:
                break
            if seg_len < chunk:
                seg = torch.nn.functional.pad(seg, (0, chunk - seg_len))
            sources, _ = model(seg.unsqueeze(0).to(device))
            w = window[:seg_len]
            for i, src in enumerate(sources):
                out[i, start:start + seg_len] += src.squeeze().cpu().numpy()[:seg_len] * w
            norm[start:start + seg_len] += w
            if start % (hop * 10) == 0:
                logger.debug("SepACap %.0f/%.0f s", start / SEPACAP_SR, n / SEPACAP_SR)
    return out / np.maximum(norm, 1e-3)


def sepacap_separate(
    audio_path: Path,
    output_dir: Path,
    *,
    device: str = "cpu",
    keep_silent: bool = False,
) -> dict[str, Path]:
    """Split a vocal recording into voice parts with SepACap.

    Writes one WAV per voice stem into ``output_dir`` and returns
    ``{"soprano": Path(...), "alto": Path(...), ...}``. Percussion stems are
    dropped, as are voices the model left (near-)silent unless ``keep_silent``.
    """
    import librosa

    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading SepACap model…")
    model = load_sepacap(device)
    audio, _ = librosa.load(str(audio_path), sr=SEPACAP_SR, mono=True)
    logger.info("Running SepACap on %s (%.1f s)…", audio_path.name, len(audio) / SEPACAP_SR)
    stems = separate_array(model, audio, device=device)

    input_rms = float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0
    result: dict[str, Path] = {}
    for name, signal in zip(SEPACAP_STEMS, stems):
        if name not in VOICE_STEMS:
            continue
        rms = float(np.sqrt(np.mean(signal**2))) if len(signal) else 0.0
        if not keep_silent and rms < SILENT_STEM_RATIO * input_rms:
            logger.info("SepACap stem %s is silent (rms %.4f); dropping", name, rms)
            continue
        dest = output_dir / f"{name}.wav"
        sf.write(dest, signal, SEPACAP_SR)
        result[name] = dest
    logger.info("SepACap produced stems: %s", list(result.keys()))
    return result

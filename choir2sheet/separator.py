"""Audio source separation using Demucs and MVSEP."""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Demucs – local vocal/instrument separation
# ---------------------------------------------------------------------------

DEMUCS_MODEL_4S = "htdemucs"        # vocals, drums, bass, other
DEMUCS_MODEL_6S = "htdemucs_6s"     # + guitar, piano


def run_demucs(
    audio_path: Path,
    output_dir: Path,
    model: str = DEMUCS_MODEL_6S,
    device: str = "cpu",
) -> dict[str, Path]:
    """Run Demucs source separation and return a mapping of stem name → path.

    Returns dict like ``{"vocals": Path(...), "piano": Path(...), ...}``.
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    cmd = [
        "python", "-m", "demucs",
        "-n", model,
        "-d", device,
        "-o", str(output_dir),
        str(audio_path),
    ]
    logger.info("Running Demucs: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Demucs writes to <output_dir>/<model>/<filename_without_ext>/
    stem_dir = output_dir / model / audio_path.stem
    if not stem_dir.is_dir():
        raise RuntimeError(f"Expected Demucs output directory not found: {stem_dir}")

    stems: dict[str, Path] = {}
    for wav in sorted(stem_dir.glob("*.wav")):
        stems[wav.stem] = wav
    logger.info("Demucs produced stems: %s", list(stems.keys()))
    return stems


# ---------------------------------------------------------------------------
# MVSEP – cloud SATB vocal splitting
# ---------------------------------------------------------------------------

MVSEP_API_URL = "https://mvsep.com/api"
MVSEP_SATB_MODEL = 17  # SATB vocal separation model


def mvsep_separate(
    audio_path: Path,
    output_dir: Path,
    api_key: Optional[str] = None,
    model_type: int = MVSEP_SATB_MODEL,
    poll_interval: float = 10.0,
    timeout: float = 600.0,
) -> dict[str, Path]:
    """Upload a vocal stem to MVSEP for SATB separation.

    Returns dict like ``{"soprano": Path(...), "alto": Path(...), ...}``.
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Upload
    params: dict = {"sep_type": str(model_type)}
    if api_key:
        params["api_key"] = api_key

    with open(audio_path, "rb") as f:
        logger.info("Uploading %s to MVSEP (model %d)…", audio_path.name, model_type)
        resp = requests.post(
            f"{MVSEP_API_URL}/separation/create",
            data=params,
            files={"audiofile": (audio_path.name, f)},
            timeout=120,
        )
    resp.raise_for_status()
    job = resp.json()
    job_hash = job.get("data", {}).get("hash") or job.get("hash")
    if not job_hash:
        raise RuntimeError(f"MVSEP did not return a job hash: {job}")
    logger.info("MVSEP job submitted: %s", job_hash)

    # 2. Poll for completion
    elapsed = 0.0
    while elapsed < timeout:
        time.sleep(poll_interval)
        elapsed += poll_interval
        status_resp = requests.get(
            f"{MVSEP_API_URL}/separation/result",
            params={"hash": job_hash},
            timeout=30,
        )
        status_resp.raise_for_status()
        result = status_resp.json()
        if result.get("status") == "done":
            break
        logger.debug("MVSEP status: %s (%.0fs elapsed)", result.get("status"), elapsed)
    else:
        raise TimeoutError(f"MVSEP job {job_hash} timed out after {timeout}s")

    # 3. Download results
    files = result.get("data", {}).get("files", [])
    stems: dict[str, Path] = {}
    for file_info in files:
        url = file_info["url"]
        name = file_info.get("name", url.split("/")[-1])
        dest = output_dir / name
        logger.info("Downloading %s → %s", name, dest)
        dl = requests.get(url, timeout=120)
        dl.raise_for_status()
        dest.write_bytes(dl.content)
        # Derive part name from filename (e.g. "soprano.wav" → "soprano")
        part = dest.stem.lower()
        stems[part] = dest

    logger.info("MVSEP produced stems: %s", list(stems.keys()))
    return stems


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------


def separate_choir(
    audio_path: Path,
    output_dir: Path,
    *,
    mvsep_api_key: Optional[str] = None,
    device: str = "cpu",
    skip_satb: bool = False,
) -> dict[str, Path]:
    """Full separation pipeline: Demucs → MVSEP SATB.

    Returns a dict of all final stems (e.g. soprano, alto, tenor, bass, piano).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Demucs – isolate vocals and instruments
    demucs_stems = run_demucs(audio_path, output_dir / "demucs", device=device)

    all_stems: dict[str, Path] = {}

    # Keep non-vocal stems (piano, bass, drums, etc.)
    for name, path in demucs_stems.items():
        if name != "vocals":
            all_stems[name] = path

    vocals_path = demucs_stems.get("vocals")
    if vocals_path is None:
        logger.warning("No vocals stem from Demucs; skipping SATB split")
        return all_stems

    if skip_satb:
        all_stems["vocals"] = vocals_path
        return all_stems

    # Step 2: MVSEP – split vocals into SATB
    try:
        satb_stems = mvsep_separate(
            vocals_path,
            output_dir / "satb",
            api_key=mvsep_api_key,
        )
        all_stems.update(satb_stems)
    except Exception:
        logger.exception(
            "MVSEP SATB separation failed; falling back to unsplit vocals"
        )
        all_stems["vocals"] = vocals_path

    return all_stems

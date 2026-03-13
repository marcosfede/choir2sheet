#!/usr/bin/env python3
"""Download and cache all models needed by choir2sheet.

Run this once on a new box to pre-download everything:

    uv run python -m choir2sheet.setup_models

Or selectively:

    uv run python -m choir2sheet.setup_models --basic-pitch
    uv run python -m choir2sheet.setup_models --demucs
    uv run python -m choir2sheet.setup_models --all
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("setup_models")


def setup_basic_pitch() -> bool:
    """Pre-download the Basic Pitch TensorFlow model."""
    logger.info("Setting up Basic Pitch model...")
    try:
        from basic_pitch.inference import predict
        import tempfile, numpy as np, soundfile as sf

        # Generate a tiny dummy audio to trigger model download
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            dummy = np.zeros(22050, dtype=np.float32)  # 1 second of silence
            sf.write(f.name, dummy, 22050)
            predict(f.name)

        logger.info("Basic Pitch model ready.")
        return True
    except Exception as e:
        logger.error("Failed to set up Basic Pitch: %s", e)
        return False


def setup_demucs() -> bool:
    """Pre-download Demucs models (htdemucs and htdemucs_6s)."""
    logger.info("Setting up Demucs models...")
    try:
        import demucs.pretrained
        for model_name in ["htdemucs", "htdemucs_6s"]:
            logger.info("  Downloading %s...", model_name)
            demucs.pretrained.get_model(model_name)
            logger.info("  %s ready.", model_name)
        return True
    except ImportError:
        logger.warning(
            "Demucs not installed. Install with: uv sync --extra separation"
        )
        return False
    except Exception as e:
        logger.error("Failed to set up Demucs: %s", e)
        return False


def setup_fluidsynth() -> bool:
    """Check FluidSynth and soundfont availability."""
    logger.info("Checking FluidSynth...")

    # Check binary
    try:
        result = subprocess.run(
            ["fluidsynth", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        logger.info("  FluidSynth binary: OK")
    except FileNotFoundError:
        logger.warning(
            "  FluidSynth binary not found. Install with:\n"
            "    Ubuntu/Debian: sudo apt-get install fluidsynth\n"
            "    macOS: brew install fluid-synth\n"
            "    Windows: choco install fluidsynth"
        )
        return False

    # Check soundfont
    sf2_paths = [
        Path("/usr/share/sounds/sf2/FluidR3_GM.sf2"),       # Debian/Ubuntu
        Path("/usr/share/soundfonts/FluidR3_GM.sf2"),        # Fedora
        Path("/usr/local/share/fluidsynth/FluidR3_GM.sf2"),  # Homebrew
    ]
    found = False
    for p in sf2_paths:
        if p.is_file():
            logger.info("  Soundfont: %s", p)
            found = True
            break

    if not found:
        # Check if pretty_midi has a bundled one
        try:
            import pretty_midi
            bundled = Path(pretty_midi.__file__).parent / "TimGM6mb.sf2"
            if bundled.is_file():
                logger.info("  Soundfont (bundled): %s", bundled)
                found = True
        except Exception:
            pass

    if not found:
        logger.warning(
            "  No GM soundfont found. Install with:\n"
            "    Ubuntu/Debian: sudo apt-get install fluid-soundfont-gm\n"
            "    macOS: brew install fluid-synth (includes soundfont)"
        )

    # Check Python binding
    try:
        import fluidsynth
        logger.info("  pyfluidsynth: OK")
    except ImportError:
        logger.warning(
            "  pyfluidsynth not installed. Install with: uv sync --extra synth"
        )

    return found


def check_system_deps() -> None:
    """Report on system-level dependencies."""
    logger.info("Checking system dependencies...")

    # LilyPond (optional, for PDF export)
    try:
        result = subprocess.run(
            ["lilypond", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        version = result.stdout.split("\n")[0] if result.stdout else "unknown"
        logger.info("  LilyPond: %s", version)
    except FileNotFoundError:
        logger.info(
            "  LilyPond: not found (optional, needed for PDF export)\n"
            "    Install: sudo apt-get install lilypond  |  brew install lilypond"
        )

    # MuseScore (optional, for rendering)
    try:
        result = subprocess.run(
            ["mscore", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        logger.info("  MuseScore: found")
    except FileNotFoundError:
        logger.info("  MuseScore: not found (optional, for score rendering)")


@click.command()
@click.option("--basic-pitch", "do_basic_pitch", is_flag=True,
              help="Download Basic Pitch model.")
@click.option("--demucs", "do_demucs", is_flag=True,
              help="Download Demucs separation models.")
@click.option("--all", "do_all", is_flag=True,
              help="Download all models and check all deps.")
def main(do_basic_pitch, do_demucs, do_all):
    """Download and cache models for choir2sheet."""
    if not any([do_basic_pitch, do_demucs, do_all]):
        do_all = True

    results = {}

    print("=" * 60)
    print("choir2sheet — Model Setup")
    print("=" * 60)

    if do_all or do_basic_pitch:
        results["basic_pitch"] = setup_basic_pitch()

    if do_all or do_demucs:
        results["demucs"] = setup_demucs()

    if do_all:
        results["fluidsynth"] = setup_fluidsynth()
        check_system_deps()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, ok in results.items():
        status = "OK" if ok else "MISSING/FAILED"
        print(f"  {name:20s} {status}")

    if all(results.values()):
        print("\nAll models ready!")
    else:
        print("\nSome components need attention (see warnings above).")
        sys.exit(1)


if __name__ == "__main__":
    main()

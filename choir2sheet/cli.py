"""Command-line interface for choir2sheet."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from . import __version__
from .score import list_formats


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.version_option(__version__)
def main():
    """choir2sheet — Audio to sheet music transcription."""


@main.command()
@click.argument("audio", type=click.Path(exists=True, path_type=Path))
@click.argument("output", type=click.Path(path_type=Path))
@click.option(
    "-f", "--format",
    "output_format",
    type=click.Choice(list_formats(), case_sensitive=False),
    default=None,
    help="Output format (inferred from extension if omitted).",
)
@click.option("--title", default="Transcription", help="Score title.")
@click.option("--composer", default="", help="Composer name.")
@click.option("--skip-separation", is_flag=True, help="Skip source separation.")
@click.option("--skip-satb", is_flag=True, help="Skip MVSEP SATB splitting.")
@click.option("--mvsep-api-key", envvar="MVSEP_API_KEY", default=None)
@click.option("--device", default="cpu", help="Torch device (cpu/cuda).")
@click.option("--onset-threshold", default=0.5, type=float)
@click.option("--frame-threshold", default=0.3, type=float)
@click.option("-v", "--verbose", is_flag=True)
def transcribe(
    audio: Path,
    output: Path,
    output_format: str | None,
    title: str,
    composer: str,
    skip_separation: bool,
    skip_satb: bool,
    mvsep_api_key: str | None,
    device: str,
    onset_threshold: float,
    frame_threshold: float,
    verbose: bool,
):
    """Transcribe an audio file to sheet music.

    AUDIO is the input audio file (WAV, MP3, FLAC, etc.).
    OUTPUT is the destination file (e.g. score.musicxml, score.mid, score.pdf).
    """
    _setup_logging(verbose)

    from .pipeline import full_pipeline

    try:
        result = full_pipeline(
            audio,
            output,
            output_format=output_format,
            title=title,
            composer=composer,
            skip_separation=skip_separation,
            skip_satb=skip_satb,
            mvsep_api_key=mvsep_api_key,
            device=device,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
        )
        click.echo(f"Done! Output: {result}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command("formats")
def show_formats():
    """List supported output formats."""
    for fmt in list_formats():
        click.echo(f"  {fmt}")


if __name__ == "__main__":
    main()

"""Command-line interface for choir2sheet.

Designed for agent consumption: each subcommand is independently callable,
outputs structured JSON to stdout, and logs to stderr.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from . import __version__
from .score import list_formats


def _setup_logging(verbose: bool, quiet: bool) -> None:
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,  # logs to stderr, JSON to stdout
    )


def _output(data: dict) -> None:
    """Write structured JSON to stdout."""
    click.echo(json.dumps(data, indent=2, default=str))


def _error(msg: str, code: int = 1) -> None:
    _output({"ok": False, "error": msg})
    sys.exit(code)


@click.group()
@click.version_option(__version__)
@click.option("-v", "--verbose", is_flag=True, help="Debug logging.")
@click.option("-q", "--quiet", is_flag=True, help="Suppress info logging.")
@click.pass_context
def main(ctx, verbose, quiet):
    """choir2sheet — Audio to sheet music transcription.

    Each subcommand outputs structured JSON to stdout.
    Logs go to stderr.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet
    _setup_logging(verbose, quiet)


# ── transcribe ───────────────────────────────────────────────────────


@main.command()
@click.argument("audio", type=click.Path(exists=True, path_type=Path))
@click.argument("output", type=click.Path(path_type=Path))
@click.option("-f", "--format", "output_format",
              type=click.Choice(list_formats(), case_sensitive=False),
              default=None, help="Output format (inferred from extension).")
@click.option("--title", default="Transcription", help="Score title.")
@click.option("--composer", default="", help="Composer name.")
@click.option("--skip-separation", is_flag=True, help="Skip source separation.")
@click.option("--skip-satb", is_flag=True, help="Skip MVSEP SATB splitting.")
@click.option("--mvsep-api-key", envvar="MVSEP_API_KEY", default=None)
@click.option("--device", default="cpu", help="Torch device (cpu/cuda).")
@click.option("--onset-threshold", default=0.5, type=float)
@click.option("--frame-threshold", default=0.3, type=float)
@click.option("--quantize/--no-quantize", default=False,
              help="Apply rhythmic quantization.")
@click.option("--tempo", default=None, type=float,
              help="Tempo in BPM (auto-detected if omitted).")
@click.option("--time-sig", default="4/4", help="Time signature for quantization.")
@click.option("--detect-key/--no-detect-key", default=False,
              help="Run key detection.")
@click.option("--key", "set_key", default=None,
              help="Override key signature (e.g. 'G major', 'F# minor').")
@click.pass_context
def transcribe(ctx, audio, output, output_format, title, composer,
               skip_separation, skip_satb, mvsep_api_key, device,
               onset_threshold, frame_threshold,
               quantize, tempo, time_sig, detect_key, set_key):
    """Full pipeline: audio → sheet music.

    Outputs JSON with file paths and optional analysis results.
    """
    from .pipeline import full_pipeline

    try:
        result = full_pipeline(
            audio, output,
            output_format=output_format,
            title=title, composer=composer,
            skip_separation=skip_separation,
            skip_satb=skip_satb,
            mvsep_api_key=mvsep_api_key,
            device=device,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            quantize=quantize,
            tempo_bpm=tempo,
            time_sig=time_sig,
            detect_key=detect_key,
            set_key=set_key,
        )
        _output(result)
    except Exception as e:
        _error(str(e))


# ── separate ─────────────────────────────────────────────────────────


@main.command()
@click.argument("audio", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output-dir", type=click.Path(path_type=Path),
              default=None, help="Output directory (default: alongside input).")
@click.option("--device", default="cpu")
@click.option("--skip-satb", is_flag=True)
@click.option("--mvsep-api-key", envvar="MVSEP_API_KEY", default=None)
@click.pass_context
def separate(ctx, audio, output_dir, device, skip_satb, mvsep_api_key):
    """Stage 1 only: source separation.

    Splits audio into stems (vocals, piano, etc.) using Demucs,
    optionally followed by MVSEP SATB vocal splitting.
    """
    from .separator import separate_choir

    if output_dir is None:
        output_dir = audio.parent / f"{audio.stem}_stems"

    try:
        stems = separate_choir(
            audio, output_dir,
            mvsep_api_key=mvsep_api_key,
            device=device,
            skip_satb=skip_satb,
        )
        _output({
            "ok": True,
            "stems": {name: str(path) for name, path in stems.items()},
        })
    except Exception as e:
        _error(str(e))


# ── midi-transcribe ──────────────────────────────────────────────────


@main.command("midi-transcribe")
@click.argument("audio", type=click.Path(exists=True, path_type=Path))
@click.argument("output", type=click.Path(path_type=Path))
@click.option("--onset-threshold", default=0.5, type=float)
@click.option("--frame-threshold", default=0.3, type=float)
@click.option("--minimum-note-length", default=58.0, type=float,
              help="Minimum note duration in ms.")
@click.option("--min-freq", default=None, type=float,
              help="Minimum frequency in Hz.")
@click.option("--max-freq", default=None, type=float,
              help="Maximum frequency in Hz.")
@click.pass_context
def midi_transcribe(ctx, audio, output, onset_threshold, frame_threshold,
                    minimum_note_length, min_freq, max_freq):
    """Stage 2 only: audio → MIDI transcription (Basic Pitch).

    Transcribes a single audio file to MIDI.
    """
    from .transcriber import transcribe_to_midi

    try:
        result = transcribe_to_midi(
            audio, output,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=minimum_note_length,
            minimum_frequency=min_freq,
            maximum_frequency=max_freq,
        )
        # Count notes in output
        import pretty_midi
        m = pretty_midi.PrettyMIDI(str(result))
        note_count = sum(len(i.notes) for i in m.instruments)

        _output({
            "ok": True,
            "midi_path": str(result),
            "notes": note_count,
            "duration": round(m.get_end_time(), 2),
        })
    except Exception as e:
        _error(str(e))


# ── detect-key ───────────────────────────────────────────────────────


@main.command("detect-key")
@click.argument("midi", type=click.Path(exists=True, path_type=Path))
@click.option("--algorithm", default="krumhansl",
              type=click.Choice(["krumhansl", "aarden", "bellman",
                                 "simple", "temperley"]),
              help="Key detection algorithm.")
@click.pass_context
def detect_key_cmd(ctx, midi, algorithm):
    """Detect the musical key of a MIDI file."""
    from .postprocess import detect_key_from_midi

    try:
        result = detect_key_from_midi(midi, algorithm=algorithm)
        _output({"ok": True, **result.to_dict()})
    except Exception as e:
        _error(str(e))


# ── quantize ─────────────────────────────────────────────────────────


@main.command()
@click.argument("midi", type=click.Path(exists=True, path_type=Path))
@click.argument("output", type=click.Path(path_type=Path))
@click.option("--tempo", default=None, type=float,
              help="Tempo in BPM (auto-detected if omitted).")
@click.option("--time-sig", default="4/4", help="Time signature.")
@click.option("-f", "--format", "output_format",
              type=click.Choice(list_formats(), case_sensitive=False),
              default=None)
@click.pass_context
def quantize(ctx, midi, output, tempo, time_sig, output_format):
    """Quantize a MIDI file to a beat grid and export.

    Snaps note onsets/durations to standard rhythmic values.
    """
    from .postprocess import quantize_midi

    try:
        result_path, qr = quantize_midi(
            midi, output,
            tempo_bpm=tempo,
            time_sig=time_sig,
            output_format=output_format,
        )
        _output({"ok": True, "output": str(result_path), **qr.to_dict()})
    except Exception as e:
        _error(str(e))


# ── evaluate ─────────────────────────────────────────────────────────


@main.command()
@click.argument("reference", type=click.Path(exists=True, path_type=Path))
@click.argument("estimated", type=click.Path(exists=True, path_type=Path))
@click.option("--onset-tolerance", default=0.05, type=float,
              help="Onset matching tolerance in seconds.")
@click.option("--pitch-tolerance", default=50.0, type=float,
              help="Pitch matching tolerance in cents.")
@click.option("--offset-ratio", default=None, type=float,
              help="Offset tolerance as ratio of note duration (omit to skip).")
@click.pass_context
def evaluate(ctx, reference, estimated, onset_tolerance, pitch_tolerance,
             offset_ratio):
    """Evaluate transcription accuracy (reference vs estimated MIDI)."""
    from .metrics import evaluate as eval_fn

    try:
        m = eval_fn(reference, estimated,
                    onset_tolerance=onset_tolerance,
                    pitch_tolerance=pitch_tolerance,
                    offset_ratio=offset_ratio)
        _output({
            "ok": True,
            "precision": round(m.precision, 4),
            "recall": round(m.recall, 4),
            "f1": round(m.f1, 4),
            "loss": round(m.loss, 4),
            "matched_notes": m.matched_notes,
            "total_ref_notes": m.total_ref_notes,
            "total_est_notes": m.total_est_notes,
            "onset_error_ms": round(m.onset_error_ms_mean, 1),
            "offset_error_ms": round(m.offset_error_ms_mean, 1),
        })
    except Exception as e:
        _error(str(e))


# ── formats ──────────────────────────────────────────────────────────


@main.command("formats")
def show_formats():
    """List supported output formats."""
    _output({"ok": True, "formats": list_formats()})


if __name__ == "__main__":
    main()

"""Score assembly and export using music21."""

from __future__ import annotations

import logging
from pathlib import Path

import music21
from music21 import converter, instrument, metadata, stream

logger = logging.getLogger(__name__)

# Map stem names to music21 instrument classes and clefs.
VOICE_MAP: dict[str, tuple[type, str]] = {
    "soprano": (instrument.Soprano, "treble"),
    "alto": (instrument.Alto, "treble"),
    "tenor": (instrument.Tenor, "treble8vb"),
    "bass": (instrument.Bass, "bass"),
    "vocals": (instrument.Vocalist, "treble"),
}

INSTRUMENT_MAP: dict[str, type] = {
    "piano": instrument.Piano,
    "guitar": instrument.AcousticGuitar,
    "bass": instrument.AcousticBass,  # instrument bass (not voice)
    "drums": instrument.UnpitchedPercussion,
    "other": instrument.Instrument,
}

# Supported export formats and their music21 format strings.
EXPORT_FORMATS: dict[str, str] = {
    "musicxml": "musicxml",
    "xml": "musicxml",
    "mxl": "musicxml",
    "midi": "midi",
    "mid": "midi",
    "lilypond": "lilypond",
    "ly": "lilypond",
    "abc": "abc",
    "pdf": "lilypond.pdf",
}


def build_score(
    midi_files: dict[str, Path],
    *,
    title: str = "Transcription",
    composer: str = "",
) -> stream.Score:
    """Combine per-stem MIDI files into a single music21 Score.

    Parameters
    ----------
    midi_files : dict[str, Path]
        Mapping of stem name → MIDI path (e.g. {"soprano": ..., "piano": ...}).
    title : str
        Score title.
    composer : str
        Composer / arranger name.

    Returns
    -------
    music21.stream.Score
    """
    score = stream.Score()
    md = metadata.Metadata()
    md.title = title
    if composer:
        md.composer = composer
    score.metadata = md

    for stem_name, midi_path in midi_files.items():
        midi_path = Path(midi_path)
        if not midi_path.is_file():
            logger.warning("MIDI file not found, skipping: %s", midi_path)
            continue

        logger.info("Adding stem '%s' from %s", stem_name, midi_path)
        parsed = converter.parse(str(midi_path))

        # Extract all parts from the parsed MIDI
        parts = parsed.parts if hasattr(parsed, "parts") else [parsed]

        for i, src_part in enumerate(parts):
            part = stream.Part()
            part.id = stem_name if i == 0 else f"{stem_name}_{i}"

            # Assign instrument
            stem_lower = stem_name.lower()
            if stem_lower in VOICE_MAP:
                instr_cls, clef_name = VOICE_MAP[stem_lower]
                part.insert(0, instr_cls())
            elif stem_lower in INSTRUMENT_MAP:
                part.insert(0, INSTRUMENT_MAP[stem_lower]())
            else:
                part.insert(0, instrument.Instrument())

            # Copy measures/notes
            for element in src_part:
                part.append(element)

            score.append(part)

    return score


def export_score(
    score: stream.Score,
    output_path: Path,
    fmt: str | None = None,
) -> Path:
    """Export a music21 Score to a file.

    Parameters
    ----------
    score : music21.stream.Score
        The assembled score.
    output_path : Path
        Destination file path.
    fmt : str | None
        Format override. If ``None``, inferred from the file extension.

    Returns
    -------
    Path
        The written file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt is None:
        ext = output_path.suffix.lstrip(".").lower()
        fmt = EXPORT_FORMATS.get(ext)
        if fmt is None:
            raise ValueError(
                f"Unknown output format '.{ext}'. "
                f"Supported: {', '.join(EXPORT_FORMATS.keys())}"
            )

    logger.info("Exporting score as '%s' → %s", fmt, output_path)

    if fmt == "lilypond.pdf":
        # music21 can render to PDF via LilyPond
        lily_conv = music21.converter.subConverters.ConverterLilypond()
        lily_conv.write(score, fmt="lilypond", fp=str(output_path.with_suffix(".ly")))
        # Then invoke lilypond to produce PDF
        import subprocess
        ly_file = output_path.with_suffix(".ly")
        subprocess.run(
            ["lilypond", "-o", str(output_path.with_suffix("")), str(ly_file)],
            check=True,
        )
    else:
        score.write(fmt, fp=str(output_path))

    logger.info("Score exported to %s", output_path)
    return output_path


def list_formats() -> list[str]:
    """Return a list of supported export format names."""
    return sorted(set(EXPORT_FORMATS.keys()))

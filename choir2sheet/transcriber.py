"""Audio-to-MIDI transcription using Basic Pitch."""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from .notes import VOCAL_PITCH_RANGE, clean_midi

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscriptionProfile:
    """Basic Pitch settings plus note cleanup for one kind of stem."""

    onset_threshold: float = 0.5
    frame_threshold: float = 0.3
    minimum_note_length: float = 58.0
    minimum_frequency: float | None = None
    maximum_frequency: float | None = None
    monophonic: bool = False
    pitch_range: tuple[int, int] | None = None
    merge_gap: float = 0.0

    def with_overrides(self, **kwargs) -> "TranscriptionProfile":
        """Return a copy with the non-``None`` kwargs applied."""
        return dataclasses.replace(
            self, **{k: v for k, v in kwargs.items() if v is not None}
        )


DEFAULT_PROFILE = TranscriptionProfile()

# Tuned on VOCADITO (solo voice): mean F1 0.495 -> 0.564, precision 0.40 -> 0.68 vs. DEFAULT_PROFILE.
# A single voice is monophonic, so overlapping detections are collapsed.
VOCAL_PROFILE = TranscriptionProfile(
    onset_threshold=0.8,
    frame_threshold=0.3,
    minimum_note_length=90.0,
    minimum_frequency=60.0,
    maximum_frequency=1100.0,
    monophonic=True,
    pitch_range=VOCAL_PITCH_RANGE,
    merge_gap=0.03,
)

VOCAL_STEMS = frozenset({"vocals", "lead", "soprano", "alto", "tenor", "bass"})
SATB_STEMS = frozenset({"lead", "soprano", "alto", "tenor"})


def profile_for_stem(
    name: str, stems: set[str] | frozenset[str] = frozenset(), *, vocal_profile: bool = True
) -> TranscriptionProfile:
    """Pick the profile for a stem.

    ``bass`` is ambiguous (Demucs bass guitar vs. SATB bass voice); it only
    counts as a voice when other SATB stems are present.
    """
    if not vocal_profile:
        return DEFAULT_PROFILE
    name = name.lower()
    if name == "bass":
        return VOCAL_PROFILE if SATB_STEMS & {s.lower() for s in stems} else DEFAULT_PROFILE
    return VOCAL_PROFILE if name in VOCAL_STEMS else DEFAULT_PROFILE


def transcribe_to_midi(
    audio_path: Path,
    output_path: Path,
    *,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length: float = 58.0,
    minimum_frequency: float | None = None,
    maximum_frequency: float | None = None,
    monophonic: bool = False,
    pitch_range: tuple[int, int] | None = None,
    merge_gap: float = 0.0,
) -> Path:
    """Transcribe an audio file to MIDI using Basic Pitch.

    Parameters
    ----------
    audio_path : Path
        Input audio file (WAV, MP3, FLAC, OGG, etc.).
    output_path : Path
        Where to write the resulting MIDI file.
    onset_threshold : float
        Note onset confidence threshold (0–1). Higher = fewer ghost notes.
    frame_threshold : float
        Frame activation threshold (0–1). Higher = stricter detection.
    minimum_note_length : float
        Minimum note duration in milliseconds.
    minimum_frequency, maximum_frequency : float | None
        Optional frequency bounds (Hz) to filter detected notes.
    monophonic, pitch_range, merge_gap
        Post-transcription cleanup; see :func:`choir2sheet.notes.clean_notes`.

    Returns
    -------
    Path
        The path to the written MIDI file.
    """
    from basic_pitch.inference import predict

    audio_path = Path(audio_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info("Transcribing %s with Basic Pitch…", audio_path.name)

    # Basic Pitch prints progress to stdout, which must stay JSON-only.
    with contextlib.redirect_stdout(sys.stderr):
        model_output, midi_data, note_events = predict(
            str(audio_path),
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=minimum_note_length,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
        )

    midi_data.write(str(output_path))
    logger.info("MIDI written to %s (%d notes)", output_path, len(note_events))

    if monophonic or pitch_range is not None or merge_gap > 0:
        clean_midi(
            output_path,
            monophonic=monophonic,
            pitch_range=pitch_range,
            merge_gap=merge_gap,
        )
    return output_path


def transcribe_with_profile(
    audio_path: Path, output_path: Path, profile: TranscriptionProfile
) -> Path:
    return transcribe_to_midi(audio_path, output_path, **dataclasses.asdict(profile))


def transcribe_stems(
    stems: dict[str, Path],
    output_dir: Path,
    *,
    vocal_profile: bool = True,
    **overrides,
) -> dict[str, Path]:
    """Transcribe multiple audio stems to MIDI files.

    Parameters
    ----------
    stems : dict[str, Path]
        Mapping of stem name → audio file path.
    output_dir : Path
        Directory to write MIDI files into.
    vocal_profile : bool
        Use :data:`VOCAL_PROFILE` for stems named like voices.
    **overrides
        :class:`TranscriptionProfile` fields applied to every stem when not
        ``None`` (e.g. an explicit ``onset_threshold`` from the CLI).

    Returns
    -------
    dict[str, Path]
        Mapping of stem name → MIDI file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    midi_files: dict[str, Path] = {}
    for name, audio in stems.items():
        midi_path = output_dir / f"{name}.mid"
        profile = profile_for_stem(name, stems.keys(), vocal_profile=vocal_profile)
        profile = profile.with_overrides(**overrides)
        try:
            transcribe_with_profile(audio, midi_path, profile)
            midi_files[name] = midi_path
        except Exception:
            logger.exception("Failed to transcribe stem '%s'", name)

    return midi_files

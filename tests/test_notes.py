import pretty_midi

from choir2sheet.metrics import NoteEvent
from choir2sheet.notes import clean_midi, clean_notes
from choir2sheet.transcriber import (
    DEFAULT_PROFILE,
    VOCAL_PROFILE,
    profile_for_stem,
)


def _n(pitch, onset, offset, velocity=80):
    return NoteEvent(pitch, onset, offset, velocity)


def test_pitch_range_filters_out_of_range():
    notes = [_n(30, 0, 1), _n(60, 1, 2), _n(90, 2, 3)]
    assert [n.pitch for n in clean_notes(notes, pitch_range=(36, 84))] == [60]


def test_merge_gap_joins_same_pitch_rearticulations():
    notes = [_n(60, 0.0, 0.5), _n(60, 0.52, 1.0), _n(62, 1.0, 1.5)]
    out = clean_notes(notes, merge_gap=0.03)
    assert [(n.pitch, n.onset, n.offset) for n in out] == [(60, 0.0, 1.0), (62, 1.0, 1.5)]


def test_merge_gap_respects_gap_limit():
    notes = [_n(60, 0.0, 0.5), _n(60, 0.6, 1.0)]
    assert len(clean_notes(notes, merge_gap=0.03)) == 2


def test_monophonic_truncates_earlier_note_at_later_onset():
    notes = [_n(60, 0.0, 1.0), _n(64, 0.6, 1.6)]
    out = clean_notes(notes, monophonic=True)
    assert [(n.pitch, n.onset, n.offset) for n in out] == [(60, 0.0, 0.6), (64, 0.6, 1.6)]


def test_monophonic_drops_ghost_buried_in_louder_note():
    notes = [_n(60, 0.0, 1.0, 100), _n(72, 0.3, 0.6, 40)]
    out = clean_notes(notes, monophonic=True)
    assert [n.pitch for n in out] == [60]


def test_monophonic_louder_note_replaces_buried_quiet_one():
    notes = [_n(60, 0.0, 0.5, 40), _n(72, 0.1, 1.0, 100)]
    out = clean_notes(notes, monophonic=True)
    assert [n.pitch for n in out] == [72]


def test_monophonic_never_emits_zero_duration_notes():
    notes = [_n(60, 0.0, 0.4, 90), _n(62, 0.0, 0.5, 80), _n(64, 0.0, 0.2, 80)]
    out = clean_notes(notes, monophonic=True)
    assert out and all(n.duration > 0 for n in out)
    for a, b in zip(out, out[1:]):
        assert b.onset >= a.offset


def test_min_duration_drops_short_notes():
    notes = [_n(60, 0.0, 0.05), _n(62, 1.0, 2.0)]
    assert [n.pitch for n in clean_notes(notes, min_duration=0.1)] == [62]


def test_clean_midi_rewrites_file(tmp_path):
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    inst.notes += [
        pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=1.0),
        pretty_midi.Note(velocity=80, pitch=64, start=0.5, end=1.5),
        pretty_midi.Note(velocity=80, pitch=20, start=2.0, end=3.0),
    ]
    midi.instruments.append(inst)
    path = tmp_path / "in.mid"
    midi.write(str(path))

    out, before, after = clean_midi(path, monophonic=True, pitch_range=(36, 84))
    assert (before, after) == (3, 2)
    reparsed = pretty_midi.PrettyMIDI(str(out))
    assert [(n.pitch, n.start, n.end) for n in reparsed.instruments[0].notes] == [
        (60, 0.0, 0.5),
        (64, 0.5, 1.5),
    ]


def test_profile_for_stem_picks_vocal_profile_for_voices():
    assert profile_for_stem("vocals") is VOCAL_PROFILE
    assert profile_for_stem("Soprano") is VOCAL_PROFILE
    assert profile_for_stem("piano") is DEFAULT_PROFILE
    assert profile_for_stem("audio") is DEFAULT_PROFILE
    assert profile_for_stem("vocals", vocal_profile=False) is DEFAULT_PROFILE


def test_profile_for_stem_bass_depends_on_satb_context():
    demucs = {"vocals", "drums", "bass", "other"}
    satb = {"soprano", "alto", "tenor", "bass"}
    assert profile_for_stem("bass", demucs) is DEFAULT_PROFILE
    assert profile_for_stem("bass", satb) is VOCAL_PROFILE


def test_with_overrides_ignores_none():
    p = VOCAL_PROFILE.with_overrides(onset_threshold=None, frame_threshold=0.5)
    assert p.onset_threshold == VOCAL_PROFILE.onset_threshold
    assert p.frame_threshold == 0.5
    assert p.monophonic is True

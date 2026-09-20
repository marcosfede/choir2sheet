"""Grid-search note-creation params + monophonic post-filter on cached Basic Pitch output."""
from __future__ import annotations

import contextlib
import itertools
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks.vocadito_vocals import OUTPUT_DIR, evaluate_vocal, load_vocadito_notes  # noqa: E402
from choir2sheet.metrics import NoteEvent  # noqa: E402

CACHE = OUTPUT_DIR / "model_output_cache.pkl"


def get_outputs():
    if CACHE.is_file():
        return pickle.load(open(CACHE, "rb"))
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model, predict

    model = Model(ICASSP_2022_MODEL_PATH)
    outs = {}
    for wav in sorted((OUTPUT_DIR / "Audio").glob("*.wav")):
        with contextlib.redirect_stdout(sys.stderr):
            out, _, _ = predict(str(wav), model)
        outs[wav.stem] = out
    pickle.dump(outs, open(CACHE, "wb"))
    return outs


def events_to_notes(events) -> list[NoteEvent]:
    return sorted(
        (NoteEvent(pitch=p, onset=s, offset=e, velocity=int(a * 127)) for s, e, p, a, _ in events),
        key=lambda n: (n.onset, n.pitch),
    )


def main():
    from choir2sheet.notes import clean_notes

    outs = get_outputs()
    refs = {k: load_vocadito_notes(OUTPUT_DIR / "Annotations" / "Notes" / f"{k}_notesA1.csv")
            for k in outs}
    from basic_pitch.note_creation import model_output_to_notes

    grid = itertools.product(
        [0.5, 0.6, 0.7, 0.8],      # onset
        [0.2, 0.3, 0.4],           # frame
        [5, 8, 11],                # min_note_len (frames, ~11.6ms each)
        [False, True],             # monophonic clean
    )
    for onset, frame, mnl, mono in grid:
        f1s, ps, rs = [], [], []
        for k, out in outs.items():
            _, ev = model_output_to_notes(out, onset, frame, min_note_len=mnl,
                                          min_freq=60, max_freq=1500, include_pitch_bends=False)
            notes = events_to_notes(ev)
            if mono:
                notes = clean_notes(notes, monophonic=True, pitch_range=(36, 84), merge_gap=0.03)
            m = evaluate_vocal(refs[k], notes)
            f1s.append(m.f1)
            ps.append(m.precision)
            rs.append(m.recall)
        print(f"onset={onset} frame={frame} mnl={mnl} mono={mono}: "
              f"F1={np.mean(f1s):.3f} P={np.mean(ps):.3f} R={np.mean(rs):.3f}", flush=True)


if __name__ == "__main__":
    main()

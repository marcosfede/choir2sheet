# choir2sheet

Audio to sheet music transcription pipeline. Give it an audio file, get back sheet music in any format.

## Pipeline

1. **Source Separation** — [Demucs](https://github.com/facebookresearch/demucs) (htdemucs_6s) isolates vocals, piano, bass, drums, guitar
2. **SATB Splitting** — [MVSEP](https://mvsep.com/) API splits vocals into Soprano, Alto, Tenor, Bass (optional)
3. **Transcription** — [Basic Pitch](https://github.com/spotify/basic-pitch) converts each audio stem to MIDI
4. **Score Assembly** — [music21](https://web.mit.edu/music21/) merges MIDI parts into a single score
5. **Export** — Output as MusicXML, MIDI, LilyPond, ABC, or PDF

## Install

```bash
pip install -e .
```

## Usage

### CLI

```bash
# Full pipeline with source separation
choir2sheet transcribe input.wav output.musicxml

# Skip separation (single instrument/voice)
choir2sheet transcribe input.wav output.mid --skip-separation

# Skip SATB (keep vocals as one stem)
choir2sheet transcribe input.wav output.musicxml --skip-satb

# With MVSEP SATB splitting
export MVSEP_API_KEY=your_key
choir2sheet transcribe choir.wav score.musicxml

# List supported formats
choir2sheet formats
```

### Python API

```python
from pathlib import Path
from choir2sheet.pipeline import full_pipeline, simple_transcribe

# Full pipeline (separation + transcription)
full_pipeline(Path("choir.wav"), Path("score.musicxml"))

# Simple transcription (no separation)
simple_transcribe(Path("piano.wav"), Path("piano.mid"))
```

### Evaluation

```python
from choir2sheet.metrics import evaluate

metrics = evaluate(Path("ground_truth.mid"), Path("transcribed.mid"))
print(metrics.summary())
# F1=0.694 (P=0.630, R=0.773) | Matched=34/44 ref, 54 est | Loss=0.3061
```

## Supported Output Formats

| Extension    | Format     |
|-------------|------------|
| `.musicxml` | MusicXML   |
| `.mid`      | MIDI       |
| `.ly`       | LilyPond   |
| `.abc`      | ABC        |
| `.pdf`      | PDF (via LilyPond) |

## End-to-End Test

```bash
python tests/test_end_to_end.py
```

Creates a synthetic SATB chorale, renders to audio, transcribes it back, and validates against ground truth using mir_eval note-level metrics.

## Architecture

```
choir2sheet/
├── __init__.py      # Package version
├── separator.py     # Demucs + MVSEP source separation
├── transcriber.py   # Basic Pitch audio → MIDI
├── score.py         # music21 score assembly + export
├── pipeline.py      # End-to-end orchestration
├── metrics.py       # Note-level evaluation (F1, loss)
└── cli.py           # Click CLI entry point
```

## License

MIT

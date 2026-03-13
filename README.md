# choir2sheet

Audio to sheet music transcription pipeline. Give it an audio file, get back sheet music in any format.

## Quick Start

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone <repo-url> && cd choir2sheet
uv sync

# Download models (run once)
uv run python -m choir2sheet.setup_models

# Transcribe
uv run choir2sheet midi-transcribe song.wav output.mid
```

## Setup on a New Box

```bash
# 1. Install system deps (Ubuntu/Debian)
sudo apt-get install fluidsynth fluid-soundfont-gm  # optional: audio synthesis
sudo apt-get install lilypond                         # optional: PDF export

# 2. Install project
uv sync

# 3. With source separation support (adds ~1GB of Demucs models)
uv sync --extra separation

# 4. With everything
uv sync --extra all

# 5. Download and cache ML models
uv run python -m choir2sheet.setup_models
uv run python -m choir2sheet.setup_models --demucs      # just Demucs
uv run python -m choir2sheet.setup_models --basic-pitch  # just Basic Pitch
```

## CLI Reference

All commands output **structured JSON to stdout** (logs go to stderr).
Use `-q` for clean JSON-only output. Designed for agent and script consumption.

### Full Pipeline

```bash
# Simple: audio → sheet music (no separation)
uv run choir2sheet transcribe song.wav score.musicxml --skip-separation

# With key detection and quantization
uv run choir2sheet transcribe song.wav score.musicxml \
    --skip-separation --detect-key --quantize --tempo 120 --time-sig 4/4

# Full choir pipeline with SATB separation
uv run choir2sheet transcribe choir.wav score.musicxml --mvsep-api-key $KEY
```

### Individual Stages

Each stage is independently callable — compose them as needed:

```bash
# Stage 1: Source separation
uv run choir2sheet separate choir.wav -o stems/
# → {"ok": true, "stems": {"vocals": "stems/vocals.wav", "piano": "..."}}

# Stage 2: Audio → MIDI
uv run choir2sheet midi-transcribe vocals.wav vocals.mid
# → {"ok": true, "midi_path": "vocals.mid", "notes": 47, "duration": 16.59}

# Key detection
uv run choir2sheet detect-key vocals.mid
# → {"ok": true, "key": "F major", "correlation": 0.883, "alternates": [...]}

# Quantization
uv run choir2sheet quantize vocals.mid quantized.musicxml --tempo 120
# → {"ok": true, "tempo_bpm": 120.0, "time_signature": "4/4", ...}

# Evaluate against ground truth
uv run choir2sheet evaluate ground_truth.mid transcribed.mid
# → {"ok": true, "f1": 0.864, "precision": 0.851, "recall": 0.876, ...}

# List formats
uv run choir2sheet formats
# → {"ok": true, "formats": ["abc", "lilypond", "mid", "musicxml", ...]}
```

### Agent Workflow Example

```bash
# An agent can chain these with jq or parse the JSON directly:

# 1. Transcribe
RESULT=$(uv run choir2sheet -q midi-transcribe audio.wav raw.mid 2>/dev/null)
echo "$RESULT" | jq .notes  # check note count

# 2. Detect key — show to human for confirmation
uv run choir2sheet -q detect-key raw.mid 2>/dev/null | jq .

# 3. Quantize with human-approved settings
uv run choir2sheet -q quantize raw.mid final.musicxml \
    --tempo 96 --time-sig 3/4 2>/dev/null
```

## Pipeline

```
Audio
  │
  ├─ [separate]      Demucs → vocals, piano, bass, drums, guitar
  │     │
  │     └─ [MVSEP]   SATB split → soprano, alto, tenor, bass (optional)
  │
  ├─ [midi-transcribe]  Basic Pitch → MIDI per stem
  │
  ├─ [detect-key]       Krumhansl-Schmuckler → key signature (optional)
  │
  ├─ [quantize]         Beat grid snapping → rhythmic notation (optional)
  │
  └─ [export]           music21 → MusicXML / MIDI / LilyPond / ABC / PDF
```

## Supported Output Formats

| Extension    | Format     |
|-------------|------------|
| `.musicxml` | MusicXML   |
| `.mid`      | MIDI       |
| `.ly`       | LilyPond   |
| `.abc`      | ABC        |
| `.pdf`      | PDF (via LilyPond) |

## Benchmarks

Tested against standard AMT evaluation datasets:

| Benchmark | Dataset | F1 (onset-only, 50ms) | Notes |
|-----------|---------|----------------------|-------|
| Piano | MAESTRO v3.0.0 | **0.864** | Scarlatti Sonata K.525, 882 notes |
| Voice | VOCADITO (40 tracks) | **0.495** mean | Real solo singing, 7 languages |

```bash
# Run benchmarks
uv run python tests/test_end_to_end.py    # MAESTRO piano
uv run python tests/test_vocal_benchmark.py  # VOCADITO vocals
```

## Architecture

```
choir2sheet/
├── __init__.py        # Package version
├── cli.py             # Click CLI (JSON output, composable subcommands)
├── separator.py       # Demucs + MVSEP source separation
├── transcriber.py     # Basic Pitch audio → MIDI
├── score.py           # music21 score assembly + export
├── postprocess.py     # Key detection + rhythmic quantization
├── metrics.py         # Note-level evaluation (mir_eval F1, loss)
├── pipeline.py        # End-to-end orchestration
└── setup_models.py    # Model download/cache script
```

## License

MIT

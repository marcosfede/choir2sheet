# choir2sheet — Agent Reference

Audio-to-sheet-music transcription pipeline. Every command outputs **structured JSON to stdout** (logs to stderr). Use `-q` for clean JSON-only output.

## Install

```bash
uv sync                          # core deps
uv sync --extra separation       # + Demucs source separation
uv run python -m choir2sheet.setup_models  # download ML models
```

## Commands

### `transcribe` — Full pipeline: audio → sheet music

```bash
choir2sheet transcribe <audio> <output> [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-f, --format` | (from extension) | `musicxml`, `midi`, `lilypond`, `abc`, `pdf` |
| `--skip-separation` | false | Skip Demucs; transcribe raw audio directly |
| `--skip-satb` | false | Skip MVSEP vocal splitting |
| `--mvsep-api-key` | `$MVSEP_API_KEY` | MVSEP cloud API key |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--onset-threshold` | 0.5 | Note onset confidence (0–1) |
| `--frame-threshold` | 0.3 | Frame activation threshold (0–1) |
| `--detect-key` | false | Run key detection |
| `--key` | none | Override key (e.g. `"G major"`, `"F# minor"`) |
| `--quantize` | false | Snap to beat grid |
| `--tempo` | auto | BPM override |
| `--time-sig` | `4/4` | Time signature |
| `--title` | `"Transcription"` | Score title |
| `--composer` | `""` | Composer name |

```json
{
  "ok": true,
  "output": "score.musicxml",
  "stages": {
    "separation": { "stems": { "soprano": "...", "alto": "...", "piano": "..." } },
    "transcription": { "midi_files": { "soprano": "...", "alto": "..." } },
    "postprocess": {
      "key": { "key": "G major", "tonic": "G", "mode": "major", "correlation": 0.883, "alternates": [...] },
      "quantization": { "tempo_bpm": 120.0, "time_signature": "4/4", "notes_before": 247, "notes_after": 247 }
    }
  }
}
```

### `separate` — Source separation only

```bash
choir2sheet separate <audio> [-o stems/] [--device cpu] [--skip-satb] [--mvsep-api-key KEY]
```

```json
{ "ok": true, "stems": { "soprano": "...", "alto": "...", "piano": "...", "drums": "..." } }
```

Without SATB: stems are `vocals`, `piano`, `bass`, `drums`, `guitar`, `other`.
With SATB: vocal stems become `soprano`, `alto`, `tenor`, `bass`.

### `midi-transcribe` — Audio → MIDI

```bash
choir2sheet midi-transcribe <audio> <output.mid> [--onset-threshold 0.5] [--frame-threshold 0.3] [--minimum-note-length 58] [--min-freq HZ] [--max-freq HZ]
```

```json
{ "ok": true, "midi_path": "output.mid", "notes": 47, "duration": 16.59 }
```

### `detect-key` — Key signature detection

```bash
choir2sheet detect-key <midi> [--algorithm krumhansl|aarden|bellman|simple|temperley]
```

```json
{
  "ok": true,
  "key": "G major", "tonic": "G", "mode": "major", "correlation": 0.883,
  "alternates": [{ "key": "E minor", "correlation": 0.820 }]
}
```

### `quantize` — Rhythmic quantization + export

```bash
choir2sheet quantize <midi> <output> [--tempo BPM] [--time-sig 4/4] [-f FORMAT]
```

```json
{ "ok": true, "output": "out.musicxml", "tempo_bpm": 120.0, "time_signature": "4/4", "notes_before": 247, "notes_after": 247 }
```

### `evaluate` — Compare transcription to ground truth

```bash
choir2sheet evaluate <reference.mid> <estimated.mid> [--onset-tolerance 0.05] [--pitch-tolerance 50] [--offset-ratio RATIO]
```

```json
{
  "ok": true,
  "precision": 0.851, "recall": 0.876, "f1": 0.863, "loss": 0.137,
  "matched_notes": 41, "total_ref_notes": 47, "total_est_notes": 48,
  "onset_error_ms": 28.3, "offset_error_ms": 45.1
}
```

### `formats` — List export formats

```bash
choir2sheet formats
```

```json
{ "ok": true, "formats": ["abc", "lilypond", "ly", "mid", "midi", "musicxml", "mxl", "pdf", "xml"] }
```

## Global flags

| Flag | Effect |
|------|--------|
| `-q, --quiet` | Suppress logs, JSON-only stdout |
| `-v, --verbose` | DEBUG-level logs to stderr |
| `--version` | Print version |

## Error format

All commands return this on failure (exit code 1):

```json
{ "ok": false, "error": "description of what went wrong" }
```

## Workflows

### Simple transcription (no separation)

```bash
RESULT=$(choir2sheet -q transcribe voice.wav score.musicxml --skip-separation --detect-key --quantize 2>/dev/null)
echo "$RESULT" | jq .stages.postprocess.key.key  # "D major"
```

### Staged pipeline with inspection

```bash
# 1. Transcribe to MIDI
choir2sheet -q midi-transcribe audio.wav raw.mid 2>/dev/null | jq .notes

# 2. Check detected key — present to human for confirmation
choir2sheet -q detect-key raw.mid 2>/dev/null | jq .

# 3. Quantize with confirmed settings
choir2sheet -q quantize raw.mid final.musicxml --tempo 96 --time-sig 3/4 2>/dev/null
```

### Choir with SATB separation

```bash
choir2sheet -q transcribe choir.wav score.musicxml \
  --mvsep-api-key "$KEY" --detect-key --quantize 2>/dev/null
```

### Batch evaluation

```bash
for f in estimated/*.mid; do
  ref="reference/$(basename "$f")"
  choir2sheet -q evaluate "$ref" "$f" 2>/dev/null | jq "{file: \"$(basename "$f")\", f1: .f1}"
done
```

## Pipeline stages

```
Audio → [separate] Demucs → stems (vocals, piano, bass, drums, guitar)
                     ↓
              [MVSEP] → SATB split (soprano, alto, tenor, bass)  [optional]
                     ↓
       [midi-transcribe] Basic Pitch → MIDI per stem
                     ↓
         [detect-key] Krumhansl-Schmuckler → key signature  [optional]
                     ↓
           [quantize] Beat grid snap → clean rhythms  [optional]
                     ↓
            [export] music21 → MusicXML / MIDI / LilyPond / ABC / PDF
```

## Tuning thresholds

| Use case | onset-threshold | frame-threshold | Notes |
|----------|----------------|-----------------|-------|
| Conservative (fewer false positives) | 0.5 | 0.3 | Default — good starting point |
| Aggressive (catch quiet notes) | 0.3 | 0.1 | More artifacts |
| Polyphonic / choir | 0.4 | 0.25 | Balance for dense textures |

## Work directory

The `transcribe` command creates `.choir2sheet_work/` alongside the output:

```
.choir2sheet_work/
├── demucs/htdemucs_6s/{stem}/  — separated audio stems
├── satb/                        — SATB vocal splits
└── midi/                        — per-stem MIDI files
```

## Dependencies

- **Required**: basic-pitch, music21, click, numpy, pretty-midi, mir-eval, soundfile
- **Optional**: `demucs` (separation), `pyfluidsynth` (synthesis), `lilypond` (PDF export)
- **Env vars**: `MVSEP_API_KEY`, `TORCH_HOME`

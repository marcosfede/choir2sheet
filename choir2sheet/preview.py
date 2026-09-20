"""Local web preview of a score: notation + per-track MIDI playback.

Standalone: takes any music21-readable score (MusicXML, MIDI, ABC, ...),
exports MusicXML + MIDI into a work dir, and serves a static single-page
app that renders the notation (OpenSheetMusicDisplay) and plays the parts
as MIDI tracks with per-track solo/mute. Optionally serves audio files
(e.g. separated stems) next to the tracks.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from music21 import converter

from .score import export_score

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "preview_static"
AUDIO_EXTS = (".wav", ".mp3", ".ogg", ".flac", ".m4a")


def prepare_preview(
    score_path: Path,
    work_dir: Path,
    *,
    audio_dir: Optional[Path] = None,
) -> dict:
    """Export the score to MusicXML + MIDI and build the preview manifest.

    Returns ``{"files": {key: Path}, "manifest": {...}}`` where manifest
    references files by ``/files/<key>`` URLs.
    """
    score_path = Path(score_path)
    if not score_path.is_file():
        raise FileNotFoundError(score_path)
    work_dir.mkdir(parents=True, exist_ok=True)

    score = converter.parse(str(score_path))
    xml_path = export_score(score, work_dir / f"{score_path.stem}.musicxml", fmt="musicxml")
    midi_path = export_score(score, work_dir / f"{score_path.stem}.mid", fmt="midi")

    files: dict[str, Path] = {"score.musicxml": xml_path, "score.mid": midi_path}
    audio: dict[str, str] = {}
    if audio_dir is not None:
        for p in sorted(Path(audio_dir).iterdir()):
            if p.suffix.lower() in AUDIO_EXTS:
                key = f"audio/{p.name}"
                files[key] = p
                audio[p.stem] = f"/files/{key}"

    md = score.metadata
    title = (md.title or md.movementName if md else None) or score_path.stem
    parts = [
        p.partName or (str(p.id) if not isinstance(p.id, int) else f"Part {i + 1}")
        for i, p in enumerate(score.parts)
    ]
    manifest = {
        "title": title,
        "parts": parts,
        "source": str(score_path),
        "score": "/files/score.musicxml",
        "midi": "/files/score.mid",
        "audio": audio,
    }
    return {"files": files, "manifest": manifest}


class _Handler(SimpleHTTPRequestHandler):
    """Serves the static app, the manifest, and an allow-list of files."""

    def __init__(self, *args, files: dict[str, Path], manifest: dict, **kwargs):
        self._files = files
        self._manifest = manifest
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self):  # noqa: N802 (http.server API)
        path = self.path.split("?", 1)[0]
        if path == "/manifest.json":
            self._send_bytes(json.dumps(self._manifest).encode(), "application/json")
        elif path.startswith("/files/"):
            key = path[len("/files/"):]
            target = self._files.get(key)
            if target is None or not target.is_file():
                self.send_error(404)
                return
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self._send_bytes(target.read_bytes(), ctype)
        else:
            super().do_GET()

    def _send_bytes(self, data: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        logger.debug("http: " + fmt, *args)


def make_server(
    files: dict[str, Path],
    manifest: dict,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
) -> ThreadingHTTPServer:
    """Create (but don't start) the preview HTTP server. ``port=0`` picks a free port."""
    handler = partial(_Handler, files=files, manifest=manifest)
    return ThreadingHTTPServer((host, port), handler)


def serve_in_thread(server: ThreadingHTTPServer) -> threading.Thread:
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return t

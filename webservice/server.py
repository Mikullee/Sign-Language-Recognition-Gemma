"""Browser test service for the Knee42 Transformer recognizer.

Serves a single page that can recognize signing three ways:

  camera   MediaPipe runs in the browser; only landmark coordinates are POSTed
           to ``/predict``.  The video never leaves the viewer's machine.
  upload   a video file is tracked server-side, split on wrist motion, and each
           segment scored.  Queued, because one MediaPipe tracker cannot be shared.
  link     the same, after fetching the URL with yt-dlp (optional dependency).

Standard library only, apart from the recognition package itself.  Run with:

    python -m webservice.server --port 8642

TLS is required: browsers only expose ``getUserMedia`` on a secure origin.  A
self-signed certificate is generated on first start if ``openssl`` is available,
otherwise pass ``--certfile``/``--keyfile``.

There is no authentication.  Anyone who can reach the port can use it, so bind
it to a trusted network or put it behind a reverse proxy that authenticates.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import ssl
import subprocess
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from recognition.config import preview_paths
from recognition.transformer.landmarks import HAND_LANDMARKS, POSE_LANDMARKS, TrackedFrame
from recognition.transformer.recognizer import Knee42TransformerRecognizer
from recognition.transformer.segmentation import analyze_frames, analyze_video


HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"

MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_VIDEO_SECONDS = 180.0
JOB_TTL_SECONDS = 3600
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
DOWNLOAD_TIMEOUT = 150


class ServiceConfig:
    """Everything the handler needs that is not global state."""

    def __init__(self, args: argparse.Namespace) -> None:
        paths = preview_paths()
        self.bundle_dir = args.bundle or paths.runtime_bundle_dir
        self.hand_model = args.hand_model or paths.hand_model
        self.pose_model = args.pose_model or paths.pose_model
        self.vendor_dir = Path(
            args.vendor_dir or os.environ.get("SLR_WEB_VENDOR_DIR", HERE / "vendor" / "mediapipe")
        )
        self.allow_url_fetch = bool(args.allow_url_fetch)
        self.recognizer = Knee42TransformerRecognizer(self.bundle_dir)


# ── job queue: video analysis is slow and the tracker is not reentrant ────────

_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()
_QUEUE: list[tuple[str, Path, bool]] = []
_QUEUE_EVENT = threading.Event()


def create_job(title: str) -> str:
    job_id = uuid.uuid4().hex
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "state": "queued",
            "phase": "waiting for a free tracker",
            "title": title,
            "done": 0,
            "total": 0,
            "result": None,
            "error": None,
            "created": time.time(),
        }
    return job_id


def update_job(job_id: str, **fields: Any) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is not None:
            job.update(fields)


def read_job(job_id: str) -> dict | None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None


def sweep_jobs() -> None:
    cutoff = time.time() - JOB_TTL_SECONDS
    with _JOBS_LOCK:
        for job_id in [key for key, job in _JOBS.items() if job["created"] < cutoff]:
            _JOBS.pop(job_id, None)


def worker_loop(config: ServiceConfig) -> None:
    while True:
        _QUEUE_EVENT.wait()
        with _JOBS_LOCK:
            item = _QUEUE.pop(0) if _QUEUE else None
            if not _QUEUE:
                _QUEUE_EVENT.clear()
        if item is None:
            continue
        job_id, path, cleanup = item
        try:
            run_video_job(config, job_id, path)
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller as job error
            update_job(job_id, state="error", error=str(exc))
        finally:
            if cleanup:
                path.unlink(missing_ok=True)
            sweep_jobs()


def run_video_job(config: ServiceConfig, job_id: str, path: Path) -> None:
    update_job(job_id, state="running", phase="tracking with MediaPipe")

    def progress(done: int, total: int) -> None:
        update_job(job_id, done=done, total=total)

    result = analyze_video(
        path,
        config.recognizer,
        hand_model=config.hand_model,
        pose_model=config.pose_model,
        max_seconds=MAX_VIDEO_SECONDS,
        progress=progress,
    )
    update_job(job_id, state="done", phase="finished", result=result)


def enqueue(job_id: str, path: Path, cleanup: bool = True) -> None:
    with _JOBS_LOCK:
        _QUEUE.append((job_id, path, cleanup))
    _QUEUE_EVENT.set()


# ── /predict: landmarks captured in the browser ──────────────────────────────


def _points(raw: Any, expected: int) -> np.ndarray | None:
    if raw is None:
        return None
    array = np.asarray(raw, dtype=np.float32)
    if array.ndim == 2 and array.shape == (expected, 3):
        return array
    return None


def frames_from_payload(raw_frames: list[dict]) -> list[TrackedFrame]:
    """Rebuild tracked frames from the browser's MediaPipe output.

    The page sends raw, un-flipped Tasks API output with handedness untouched --
    the same convention the training cache was built under.  Flipping or swapping
    sides in the page would put inference on a different distribution.
    """
    frames: list[TrackedFrame] = []
    for index, entry in enumerate(raw_frames):
        if not isinstance(entry, dict):
            continue
        hands: dict[str, np.ndarray] = {}
        for hand in entry.get("hands") or []:
            if not isinstance(hand, dict):
                continue
            side = str(hand.get("handedness", "")).strip().capitalize()
            landmarks = _points(hand.get("landmarks"), HAND_LANDMARKS)
            if side in ("Left", "Right") and landmarks is not None:
                hands[side] = landmarks
        frames.append(
            TrackedFrame(
                index=index,
                timestamp=float(entry.get("timestamp", index / 30.0)),
                pose=_points(entry.get("pose"), POSE_LANDMARKS),
                hands=hands,
            )
        )
    return frames


def predict_payload(config: ServiceConfig, payload: dict) -> dict:
    raw_frames = payload.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ValueError("payload needs a non-empty 'frames' list")
    frames = frames_from_payload(raw_frames)
    if not frames:
        raise ValueError("no usable frames in the payload")
    return analyze_frames(frames, config.recognizer, topk=5)


# ── multipart: stream the upload to disk instead of buffering it ─────────────


def _boundary_of(content_type: str) -> bytes:
    for part in content_type.split(";"):
        key, _, value = part.strip().partition("=")
        if key.lower() == "boundary":
            return value.strip('"').encode("ascii")
    raise ValueError("multipart upload has no boundary")


def read_multipart_file(rfile, length: int, content_type: str, destination) -> str:
    """Copy the first file part of a multipart body to ``destination``.

    Written by hand rather than with ``cgi.FieldStorage``, which was removed in
    Python 3.13, and streamed rather than buffered so a 200 MB upload does not
    become 200 MB of resident memory.  Returns the client-supplied filename.
    """
    boundary = b"--" + _boundary_of(content_type)
    remaining = length
    filename = ""

    def read_line() -> bytes:
        nonlocal remaining
        line = rfile.readline(8192)
        remaining -= len(line)
        return line

    while remaining > 0:
        line = read_line()
        if not line:
            raise ValueError("upload ended before any file part")
        if not line.startswith(boundary):
            continue
        headers: list[str] = []
        while remaining > 0:
            header = read_line().decode("utf-8", "replace").strip()
            if not header:
                break
            headers.append(header)
        disposition = next(
            (h for h in headers if h.lower().startswith("content-disposition")), ""
        )
        if "filename=" not in disposition:
            continue
        filename = disposition.split("filename=", 1)[1].strip().strip('"')
        break

    if not filename:
        raise ValueError("no file field in the upload")

    terminator = b"\r\n" + boundary
    window = b""
    while remaining > 0:
        chunk = rfile.read(min(1024 * 1024, remaining))
        if not chunk:
            break
        remaining -= len(chunk)
        window += chunk
        cut = window.find(terminator)
        if cut >= 0:
            destination.write(window[:cut])
            return filename
        keep = len(terminator)
        if len(window) > keep:
            destination.write(window[:-keep])
            window = window[-keep:]
    destination.write(window.split(terminator)[0])
    return filename


# ── HTTP ─────────────────────────────────────────────────────────────────────


class Handler(BaseHTTPRequestHandler):
    server_version = "Knee42Webservice"
    config: ServiceConfig

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}", flush=True)

    # -- helpers ------------------------------------------------------------

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path) -> None:
        if not path.is_file():
            self._json({"ok": False, "message": f"not found: {path.name}"}, 404)
            return
        suffix = path.suffix.lower()
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
            ".mjs": "text/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".wasm": "application/wasm",
            ".task": "application/octet-stream",
        }.get(suffix, "application/octet-stream")
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _safe(self, root: Path, relative: str) -> Path | None:
        """Resolve inside ``root`` only; anything escaping it returns None."""
        root = root.resolve()
        try:
            candidate = (root / relative.lstrip("/")).resolve()
        except (OSError, ValueError):
            return None
        return candidate if root in candidate.parents or candidate == root else None

    # -- routes -------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        route = self.path.split("?", 1)[0]
        if route in ("/", "/index.html"):
            self._file(STATIC / "index.html")
            return
        if route == "/health":
            self._json(
                {
                    "ok": True,
                    "model": self.config.recognizer.bundle.model_card.get("model_id"),
                    "classes": len(self.config.recognizer.labels),
                    "url_fetch": self.config.allow_url_fetch and bool(shutil.which("yt-dlp")),
                    "max_upload_bytes": MAX_UPLOAD_BYTES,
                    "max_video_seconds": MAX_VIDEO_SECONDS,
                }
            )
            return
        if route.startswith("/labels"):
            recognizer = self.config.recognizer
            self._json(
                {
                    "ok": True,
                    "labels": [
                        {"label": label, "text": recognizer.display_text(label)}
                        for label in recognizer.labels
                    ],
                }
            )
            return
        if route.startswith("/job/"):
            job = read_job(route[len("/job/") :])
            self._json(job or {"ok": False, "message": "unknown job"}, 200 if job else 404)
            return
        if route.startswith("/vendor/mediapipe/"):
            path = self._safe(self.config.vendor_dir, route[len("/vendor/mediapipe/") :])
            self._file(path) if path else self._json({"ok": False, "message": "bad path"}, 400)
            return
        path = self._safe(STATIC, route)
        self._file(path) if path else self._json({"ok": False, "message": "bad path"}, 400)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        route = self.path.split("?", 1)[0]
        try:
            if route == "/predict":
                self._handle_predict()
            elif route == "/analyze_upload":
                self._handle_upload()
            elif route == "/analyze_url":
                self._handle_url()
            else:
                self._json({"ok": False, "message": f"no such route: {route}"}, 404)
        except ValueError as exc:
            self._json({"ok": False, "message": str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001 - never leak a traceback to the page
            self._json({"ok": False, "message": f"server error: {exc}"}, 500)

    def _handle_predict(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_JSON_BYTES:
            raise ValueError("payload is empty or too large")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        result = predict_payload(self.config, payload)
        result["ok"] = True
        self._json(result)

    def _handle_upload(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            raise ValueError(f"upload must be between 1 byte and {MAX_UPLOAD_BYTES} bytes")
        handle, temp_path = tempfile.mkstemp()
        with os.fdopen(handle, "wb") as target:
            filename = read_multipart_file(
                self.rfile, length, self.headers.get("Content-Type", ""), target
            )
        suffix = Path(filename).suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            Path(temp_path).unlink(missing_ok=True)
            raise ValueError(f"unsupported video type: {suffix or '(none)'}")
        job_id = create_job(Path(filename).name)
        enqueue(job_id, Path(temp_path))
        self._json({"ok": True, "job_id": job_id})

    def _handle_url(self) -> None:
        if not self.config.allow_url_fetch:
            raise ValueError("URL fetching is disabled; start the server with --allow-url-fetch")
        if not shutil.which("yt-dlp"):
            raise ValueError("yt-dlp is not installed")
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(max(length, 0)).decode("utf-8") or "{}")
        url = str(payload.get("url", "")).strip()
        if not url.lower().startswith(("http://", "https://")):
            raise ValueError("url must start with http:// or https://")

        job_id = create_job(url)
        target_dir = Path(tempfile.mkdtemp())
        update_job(job_id, state="running", phase="downloading")
        try:
            subprocess.run(
                ["yt-dlp", "--no-playlist", "-o", str(target_dir / "video.%(ext)s"), url],
                check=True,
                capture_output=True,
                timeout=DOWNLOAD_TIMEOUT,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            shutil.rmtree(target_dir, ignore_errors=True)
            raise ValueError(f"download failed: {exc}") from exc
        downloaded = next((p for p in target_dir.iterdir() if p.is_file()), None)
        if downloaded is None:
            shutil.rmtree(target_dir, ignore_errors=True)
            raise ValueError("download produced no file")
        enqueue(job_id, downloaded)
        self._json({"ok": True, "job_id": job_id})


# ── TLS ──────────────────────────────────────────────────────────────────────


def ensure_certificate(cert_dir: Path) -> tuple[Path, Path]:
    """Return an existing certificate pair, generating a self-signed one if needed."""
    cert_dir.mkdir(parents=True, exist_ok=True)
    certfile, keyfile = cert_dir / "server.crt", cert_dir / "server.key"
    if certfile.is_file() and keyfile.is_file():
        return certfile, keyfile
    if not shutil.which("openssl"):
        raise SystemExit(
            "no certificate found and openssl is unavailable; "
            "pass --certfile and --keyfile explicitly"
        )
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "365",
            "-subj", "/CN=localhost",
            "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1",
            "-keyout", str(keyfile), "-out", str(certfile),
        ],
        check=True,
        capture_output=True,
    )
    return certfile, keyfile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8642)
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--hand-model", type=Path, default=None)
    parser.add_argument("--pose-model", type=Path, default=None)
    parser.add_argument(
        "--vendor-dir",
        type=Path,
        default=None,
        help="directory served at /vendor/mediapipe/ (MediaPipe web assets and .task files)",
    )
    parser.add_argument("--certfile", type=Path, default=None)
    parser.add_argument("--keyfile", type=Path, default=None)
    parser.add_argument(
        "--allow-url-fetch",
        action="store_true",
        help="enable the paste-a-link mode, which shells out to yt-dlp",
    )
    args = parser.parse_args(argv)

    config = ServiceConfig(args)
    Handler.config = config

    if args.certfile and args.keyfile:
        certfile, keyfile = args.certfile, args.keyfile
    else:
        certfile, keyfile = ensure_certificate(HERE / "certs")

    threading.Thread(target=worker_loop, args=(config,), daemon=True).start()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=str(certfile), keyfile=str(keyfile))
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

    card = config.recognizer.bundle.model_card
    print(f"[init] model  {card.get('model_id')} ({len(config.recognizer.labels)} classes)")
    print(f"[init] bundle {config.bundle_dir}")
    print(f"[init] vendor {config.vendor_dir}")
    print(f"[init] listening on https://{args.host}:{args.port}  (TLS: {certfile.name})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[exit] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

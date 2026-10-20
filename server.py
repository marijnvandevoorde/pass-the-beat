"""pass-the-beat — sidecar service that runs the beat-this model.

A small FastAPI service that runs the CPJKU `beat-this` beat tracker over
audio and returns a beat grid. The audio is either uploaded as the request
body (`POST /analyze-audio`, what pass-the-aux uses: the service needs no
storage of its own) or read from a read-only `/music` mount
(`POST /analyze`, for the compose setup that still shares a folder).
pass-the-aux calls it through its `AudioAnalyzer` port (the
`BeatThisAnalyzer` adapter) when `BEAT_ANALYZER=pass-the-beat`.

Why a sidecar: beat-this is a PyTorch model — it has no place in
pass-the-aux's near-zero-dependency, type-stripped TypeScript runtime. So
we wrap it in a self-contained Python service, reached over a plain HTTP
contract on the shared `pta` docker network.

The model is loaded ONCE at startup (it is the expensive part) and reused
for every request; one analysis runs at a time. The raw beats are cleaned
up into a DJ-usable grid by `grid.py`.

Status codes, as pass-the-aux reads them: 200 grid (possibly empty = "no
pulse"); 422 the file itself can't be decoded (permanent for that file);
500 the model failed on this input (permanent); 503 the service can't do
the work right now (model loading, busy, out of memory, ffmpeg missing,
internal error): try again later.
"""

import logging
import os
import subprocess
import tempfile
import threading
import time
import traceback
from contextlib import asynccontextmanager
from urllib.parse import unquote

import numpy as np
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import grid

# Shared, read-only mount of pass-the-aux's music/ dir. Track paths arrive
# relative to this root (e.g. "<uid>/song.mp3").
MUSIC_ROOT = os.environ.get("MUSIC_ROOT", "/music")
# beat-this checkpoint name; "final0" is the default released model.
CHECKPOINT = os.environ.get("BEATTHIS_CHECKPOINT", "final0")
DEVICE = os.environ.get("BEATTHIS_DEVICE", "cpu")
# Mono PCM sample rate we decode to before analysis. beat-this resamples
# internally; 22.05 kHz is its native rate, so no resample happens.
DECODE_SR = 22_050
# Largest upload `/analyze-audio` accepts. A 10-minute 24-bit FLAC is
# well under this; anything bigger is not a song.
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))
# Requests allowed to wait for the model while one runs. More than that
# answer 503 at once instead of queueing past the caller's timeout.
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", "4"))
# ffmpeg decode limit. A song decodes in a second or two; a decode that
# takes this long is a broken file.
DECODE_TIMEOUT_S = int(os.environ.get("DECODE_TIMEOUT_S", "180"))
# Shorter audio has no pulse to track (and crashes the model below ~50 ms).
MIN_AUDIO_SEC = 1.0

log = logging.getLogger("pass-the-beat")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(_h)
    log.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    log.propagate = False

_model = {"audio2beats": None}
# One analysis at a time: parallel torch inference only multiplies memory.
_model_lock = threading.Lock()
_stats_lock = threading.Lock()
_stats = {"waiting": 0, "running": 0, "analysed": 0, "failed": 0, "last_error": None}


def _load_model():
    # Importing here keeps `uvicorn --reload` and test imports cheap; the
    # heavy torch import only happens on real boot. Audio2Beats (not
    # File2Beats) takes a decoded waveform — we feed it ffmpeg-decoded PCM
    # so beat-this never touches the file itself (its torchaudio/soundfile
    # loader can't decode mp3 in this image).
    from beat_this.inference import Audio2Beats

    return Audio2Beats(checkpoint_path=CHECKPOINT, device=DEVICE, dbn=False)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    t0 = time.monotonic()
    _model["audio2beats"] = _load_model()
    log.info(
        "model %s loaded on %s in %.1fs (analyzer v%d)",
        CHECKPOINT, DEVICE, time.monotonic() - t0, grid.ANALYZER_VERSION,
    )
    yield
    _model["audio2beats"] = None


app = FastAPI(title="pass-the-beat sidecar", lifespan=lifespan)


class AnalyzeRequest(BaseModel):
    """`path` is relative to MUSIC_ROOT (never absolute, never `..`)."""

    path: str


class AnalysisError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _err(message: str, status: int, retry_after: int | None = None) -> JSONResponse:
    headers = {"Retry-After": str(retry_after)} if retry_after else None
    return JSONResponse(status_code=status, content={"error": message}, headers=headers)


@app.exception_handler(Exception)
async def _unhandled(_request: Request, exc: Exception) -> JSONResponse:
    # A bug here must not look like "the model failed on this file" (500 is
    # stored as final by pass-the-aux), so answer 503: try again later.
    log.error("unhandled error: %s\n%s", exc, traceback.format_exc())
    return _err(f"internal error: {exc}", 503, retry_after=30)


def _safe_path(rel: str) -> str | None:
    """Resolve `rel` under MUSIC_ROOT, rejecting path traversal."""
    root = os.path.realpath(MUSIC_ROOT)
    full = os.path.realpath(os.path.join(root, rel))
    if full == root or full.startswith(root + os.sep):
        return full
    return None


def _decode_mono(path: str) -> np.ndarray:
    """Decode to mono float32 PCM at DECODE_SR via the ffmpeg binary —
    robust across mp3/m4a/flac, unlike beat-this's own audio loader.
    Raises AnalysisError: 422 for a file ffmpeg can't decode, 503 when
    ffmpeg itself is missing."""
    try:
        out = subprocess.run(
            [
                "ffmpeg", "-nostdin", "-v", "error", "-i", path,
                "-ac", "1", "-ar", str(DECODE_SR), "-f", "f32le", "-",
            ],
            capture_output=True, timeout=DECODE_TIMEOUT_S,
        )
    except FileNotFoundError:
        raise AnalysisError(503, "ffmpeg is not installed in this container")
    except subprocess.TimeoutExpired:
        raise AnalysisError(422, f"decoding took longer than {DECODE_TIMEOUT_S}s")
    pcm = np.frombuffer(out.stdout, dtype=np.float32)
    if pcm.size == 0:
        detail = out.stderr.decode("utf-8", "replace").strip().splitlines()
        reason = detail[-1] if detail else f"ffmpeg exit code {out.returncode}"
        raise AnalysisError(422, f"could not decode audio: {reason}")
    return pcm


def _is_out_of_memory(e: BaseException) -> bool:
    text = str(e).lower()
    return isinstance(e, MemoryError) or "out of memory" in text or "allocat" in text


def _empty(duration: float) -> dict:
    """No trackable pulse — mirrors the local analyzer's null contract."""
    return {
        "beats": [], "downbeatIndices": [], "downbeatPhase": 0, "bpm": None,
        "confidence": 0.0, "duration": round(duration, 3), "firstSolidBeat": -1,
        "analyzerVersion": grid.ANALYZER_VERSION, "stats": {"raw_beats": 0},
    }


def _raw_result(beats: list[float], downbeats: list[float], duration: float) -> dict:
    """Raw beats, used only if grid post-processing itself fails."""
    b = np.asarray(beats)
    period = grid.beat_period(b)
    idx = sorted({i for i in grid.nearest(b, np.asarray(downbeats), 0.1) if i >= 0})
    return {
        "beats": beats, "downbeatIndices": idx,
        "downbeatPhase": (idx[0] % 4) if idx else 0,
        "bpm": round(60.0 / period, 2) if period else None,
        "confidence": 0.1, "duration": round(duration, 3), "firstSolidBeat": -1,
        "analyzerVersion": 1, "stats": {"raw_beats": len(beats), "fallback": True},
    }


def _run_model(pcm: np.ndarray) -> tuple[list[float], list[float]]:
    audio2beats = _model["audio2beats"]
    if audio2beats is None:
        raise AnalysisError(503, "model not loaded yet")
    try:
        beat_arr, downbeat_arr = audio2beats(pcm, DECODE_SR)
    except Exception as e:  # noqa: BLE001 — any model failure
        if _is_out_of_memory(e):
            raise AnalysisError(503, f"out of memory: {e}")
        raise AnalysisError(500, f"analysis failed: {e}")
    return [float(t) for t in beat_arr], [float(t) for t in downbeat_arr]


def _analyze_file(full: str, label: str) -> dict:
    """Decode and analyse one local audio file, one at a time. Returns the
    wire dict or raises AnalysisError."""
    with _stats_lock:
        _stats["waiting"] += 1
    with _model_lock:
        with _stats_lock:
            _stats["waiting"] -= 1
            _stats["running"] += 1
        try:
            return _analyze_locked(full, label)
        finally:
            with _stats_lock:
                _stats["running"] -= 1


def _analyze_locked(full: str, label: str) -> dict:
    t0 = time.monotonic()
    pcm = _decode_mono(full)
    # The decoded length, not the container's claim: it is what the
    # browser decodes too, and VBR mp3 headers can be off.
    duration = pcm.size / DECODE_SR
    t1 = time.monotonic()
    if duration < MIN_AUDIO_SEC:
        log.info("%s: %.2fs of audio, too short to track", label, duration)
        return _empty(duration)
    beats, downbeats = _run_model(pcm)
    t2 = time.monotonic()
    if not beats:
        result = _empty(duration)
    else:
        try:
            a = grid.analyse(beats, downbeats, pcm, DECODE_SR, duration)
            result = {
                "beats": a.beats,
                "downbeatIndices": a.downbeat_indices,
                "downbeatPhase": a.downbeat_phase,
                "bpm": a.bpm,
                "confidence": a.confidence,
                "duration": round(duration, 3),
                "firstSolidBeat": a.first_solid_beat,
                "analyzerVersion": grid.ANALYZER_VERSION,
                "stats": a.stats,
            }
        except Exception as e:  # noqa: BLE001 — never lose a grid to a cleanup bug
            log.error("%s: grid post-processing failed, sending raw beats: %s\n%s",
                      label, e, traceback.format_exc())
            result = _raw_result(beats, downbeats, duration)
    t3 = time.monotonic()
    log.info(
        "%s: %.0fs audio, decode %.1fs, model %.1fs, grid %.2fs; %d beats, "
        "bpm %s, phase %d, mix-in beat %d, confidence %.2f, %s",
        label, duration, t1 - t0, t2 - t1, t3 - t2, len(result["beats"]),
        result["bpm"], result["downbeatPhase"], result["firstSolidBeat"],
        result["confidence"], result["stats"],
    )
    return result


def _respond(full: str, label: str):
    """Run one analysis, mapping failures to the documented status codes."""
    try:
        result = _analyze_file(full, label)
    except AnalysisError as e:
        with _stats_lock:
            _stats["failed"] += 1
            _stats["last_error"] = f"{label}: {e.message}"
        log.warning("%s: %d %s", label, e.status, e.message)
        return _err(e.message, e.status, retry_after=30 if e.status == 503 else None)
    with _stats_lock:
        _stats["analysed"] += 1
    return result


def _busy() -> JSONResponse | None:
    """503 while the model loads or when the queue is full."""
    if _model["audio2beats"] is None:
        return _err("model not loaded yet", 503, retry_after=10)
    with _stats_lock:
        waiting = _stats["waiting"]
    if waiting >= MAX_QUEUE:
        return _err(f"busy: {waiting} analyses already waiting", 503, retry_after=30)
    return None


@app.get("/healthz")
def healthz() -> dict:
    with _stats_lock:
        stats = dict(_stats)
    return {
        "status": "ok",
        "model_loaded": _model["audio2beats"] is not None,
        "checkpoint": CHECKPOINT,
        "analyzerVersion": grid.ANALYZER_VERSION,
        "busy": stats["running"] > 0,
        "waiting": stats["waiting"],
        "analysed": stats["analysed"],
        "failed": stats["failed"],
        "last_error": stats["last_error"],
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """Analyse a file on the shared `/music` mount (compose setups)."""
    busy = _busy()
    if busy is not None:
        return busy
    full = _safe_path(req.path)
    if full is None:
        return _err(f"path outside the music root: {req.path}", 400)
    if not os.path.isfile(full):
        return _err(f"file not found: {req.path}", 404)
    return _respond(full, req.path)


@app.post("/analyze-audio")
async def analyze_audio(request: Request):
    """Analyse audio uploaded as the raw request body. The caller owns the
    storage; this service only spools the bytes to a temp file (ffmpeg
    needs a seekable input for m4a with a trailing moov atom), analyses
    it and deletes it again."""
    busy = _busy()
    if busy is not None:
        return busy
    label = unquote(request.headers.get("x-track-name", "upload"))
    fd, tmp = tempfile.mkstemp(prefix="ptb-", suffix=".audio")
    try:
        size = 0
        with os.fdopen(fd, "wb") as f:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    log.warning("%s: upload over %d bytes refused", label, MAX_UPLOAD_BYTES)
                    return _err(f"upload too large (limit {MAX_UPLOAD_BYTES} bytes)", 413)
                f.write(chunk)
        if size == 0:
            return _err("empty body", 400)
        return await run_in_threadpool(_respond, tmp, label)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

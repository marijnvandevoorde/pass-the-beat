"""pass-the-beat — sidecar service that runs the beat-this model.

A small FastAPI service that runs the CPJKU `beat-this` beat tracker over
audio and returns a beat grid. The audio is either uploaded as the request
body (`POST /analyze-audio`, what pass-the-aux uses: the service needs no
storage of its own) or read from a read-only `/music` mount
(`POST /analyze`, for the compose setup that still shares a folder). pass-the-aux calls it through its `AudioAnalyzer` port (the
`BeatThisAnalyzer` adapter) when `BEAT_ANALYZER=pass-the-beat`.

Why a sidecar: beat-this is a PyTorch model — it has no place in
pass-the-aux's near-zero-dependency, type-stripped TypeScript runtime. So
we wrap it in a self-contained Python service, reached over a plain HTTP
contract on the shared `pta` docker network.

The model is loaded ONCE at startup (it is the expensive part) and reused
for every request.
"""

import os
import subprocess
import tempfile
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel

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

_model = {"audio2beats": None}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Importing inside the lifespan keeps `uvicorn --reload` and test
    # imports cheap; the heavy torch import only happens on real boot.
    # Audio2Beats (not File2Beats) takes a decoded waveform — we feed it
    # ffmpeg-decoded PCM so beat-this never touches the file itself (its
    # torchaudio/soundfile loader can't decode mp3 in this image).
    from beat_this.inference import Audio2Beats

    _model["audio2beats"] = Audio2Beats(
        checkpoint_path=CHECKPOINT, device=DEVICE, dbn=False
    )
    yield
    _model["audio2beats"] = None


app = FastAPI(title="pass-the-beat sidecar", lifespan=lifespan)


class AnalyzeRequest(BaseModel):
    """`path` is relative to MUSIC_ROOT (never absolute, never `..`)."""

    path: str


def _err(message: str, status: int) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


def _safe_path(rel: str) -> str | None:
    """Resolve `rel` under MUSIC_ROOT, rejecting path traversal."""
    root = os.path.realpath(MUSIC_ROOT)
    full = os.path.realpath(os.path.join(root, rel))
    if full == root or full.startswith(root + os.sep):
        return full
    return None


def _duration_sec(path: str) -> float:
    """Container duration via ffprobe — robust across mp3/m4a/flac/etc."""
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=nw=1:nk=1", path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        return float(out.stdout.strip())
    except (ValueError, subprocess.SubprocessError):
        return 0.0


def _decode_mono(path: str) -> np.ndarray:
    """Decode to mono float32 PCM at DECODE_SR via the ffmpeg binary —
    robust across mp3/m4a/flac, unlike beat-this's own audio loader."""
    try:
        out = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-i", path,
                "-ac", "1", "-ar", str(DECODE_SR), "-f", "f32le", "-",
            ],
            capture_output=True, timeout=180,
        )
        return np.frombuffer(out.stdout, dtype=np.float32)
    except subprocess.SubprocessError:
        return np.zeros(0, dtype=np.float32)


def _downbeat_indices(beats: list[float], downbeats: list[float]) -> list[int]:
    """Map downbeat times to indices into `beats` (nearest beat). beat-this
    emits downbeats as a subset of beats, so the match is exact-ish."""
    idxs: list[int] = []
    bi = 0
    for dt in downbeats:
        while (
            bi + 1 < len(beats)
            and abs(beats[bi + 1] - dt) <= abs(beats[bi] - dt)
        ):
            bi += 1
        if beats:
            idxs.append(bi)
    return idxs


def _bpm(intervals: np.ndarray) -> float | None:
    """Tempo from the inter-beat intervals. beat-this places beats on a
    50 fps frame grid, so a single interval (and hence the median) is
    quantised to 20 ms steps: 0.46 s / 0.48 s, i.e. 130.4 / 125 BPM, with
    nothing in between. Averaging the intervals of the steady part (within
    10 % of the median, so missed or doubled beats don't count) cancels
    the quantisation and gives a usable tempo, e.g. 128.0."""
    if intervals.size == 0:
        return None
    med = float(np.median(intervals))
    if med <= 0:
        return None
    steady = intervals[np.abs(intervals - med) <= 0.1 * med]
    return 60.0 / float(np.mean(steady if steady.size else intervals))


def _first_solid_beat(pcm: np.ndarray, beats: list[float]) -> int:
    """Index of the first beat where sustained energy begins — used as a
    visual "first solid cue" marker past sparse intros. Returns 0 when no
    clear ramp is found, -1 when there are no beats at all."""
    if not beats:
        return -1
    win = max(1, int(DECODE_SR * 0.25))
    frames = len(pcm) // win
    if frames < 2:
        return 0
    rms = np.sqrt(
        np.mean(pcm[: frames * win].reshape(frames, win) ** 2, axis=1) + 1e-12
    )
    voiced = rms[rms > 1e-5]
    if voiced.size == 0:
        return 0
    threshold = 0.5 * float(np.median(voiced))
    need = max(1, int(2.0 / 0.25))  # energy must hold for ~2 s
    solid_t: float | None = None
    for i in range(len(rms) - need):
        if bool(np.all(rms[i : i + need] >= threshold)):
            solid_t = i * 0.25
            break
    if solid_t is None:
        return 0
    for idx, t in enumerate(beats):
        if t >= solid_t:
            return idx
    return 0


@app.get("/healthz")
def healthz() -> dict:
    return {
        "status": "ok",
        "model_loaded": _model["audio2beats"] is not None,
        "checkpoint": CHECKPOINT,
    }


def _analyze_file(full: str, label: str):
    """Run the model over one local audio file. Returns the wire dict, or
    a JSONResponse error: 422 when the file cannot be decoded (permanent
    for that file), 500 when the model itself fails."""
    audio2beats = _model["audio2beats"]
    if audio2beats is None:
        return _err("model not loaded yet", 503)

    # One ffmpeg decode feeds both the model and the first-solid
    # heuristic — beat-this never opens the file itself.
    pcm = _decode_mono(full)
    if pcm.size == 0:
        return _err(f"could not decode audio: {label}", 422)

    try:
        beat_arr, downbeat_arr = audio2beats(pcm, DECODE_SR)
    except Exception as e:  # noqa: BLE001 — any model failure
        return _err(f"analysis failed: {e}", 500)

    beats = [float(t) for t in beat_arr]
    downbeats = [float(t) for t in downbeat_arr]
    duration = _duration_sec(full)

    if not beats:
        # No trackable pulse — mirror the local analyzer's null contract.
        return {
            "beats": [], "downbeatIndices": [], "bpm": None,
            "confidence": 0.0, "duration": duration, "firstSolidBeat": -1,
        }

    intervals = np.diff(beats)
    bpm = _bpm(intervals)
    # Confidence proxy: beat-this gives no score, so derive one from how
    # regular the inter-beat intervals are (low coefficient of variation
    # ⇒ steady grid ⇒ high confidence).
    if intervals.size >= 2 and float(np.mean(intervals)) > 0:
        cv = float(np.std(intervals) / np.mean(intervals))
        confidence = max(0.0, min(1.0, 1.0 - cv * 4.0))
    else:
        confidence = 0.5

    return {
        "beats": beats,
        "downbeatIndices": _downbeat_indices(beats, downbeats),
        "bpm": bpm,
        "confidence": confidence,
        "duration": duration,
        "firstSolidBeat": _first_solid_beat(pcm, beats),
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """Analyse a file on the shared `/music` mount (compose setups)."""
    if _model["audio2beats"] is None:
        return _err("model not loaded yet", 503)

    full = _safe_path(req.path)
    if full is None:
        return _err(f"path outside the music root: {req.path}", 400)
    if not os.path.isfile(full):
        return _err(f"file not found: {req.path}", 404)
    return _analyze_file(full, req.path)


@app.post("/analyze-audio")
async def analyze_audio(request: Request):
    """Analyse audio uploaded as the raw request body. The caller owns the
    storage; this service only spools the bytes to a temp file (ffmpeg
    needs a seekable input for m4a with a trailing moov atom), analyses
    it and deletes it again."""
    if _model["audio2beats"] is None:
        return _err("model not loaded yet", 503)

    fd, tmp = tempfile.mkstemp(prefix="ptb-", suffix=".audio")
    try:
        size = 0
        with os.fdopen(fd, "wb") as f:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    return _err("upload too large", 413)
                f.write(chunk)
        if size == 0:
            return _err("empty body", 400)
        label = request.headers.get("x-track-name", "upload")
        return await run_in_threadpool(_analyze_file, tmp, label)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

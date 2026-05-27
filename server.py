"""beat-this analysis sidecar.

A small FastAPI service that runs the CPJKU `beat-this` beat tracker over
audio files on a shared, read-only `/music` volume and returns a beat
grid. pass-the-aux calls it through its `AudioAnalyzer` port (the
`BeatThisAnalyzer` adapter) when `BEAT_ANALYZER=beatthis`.

Why a sidecar: beat-this is a PyTorch model — it has no place in
pass-the-aux's near-zero-dependency, type-stripped TypeScript runtime. This
mirrors the existing `sptf/` precedent: a self-contained Python service,
reached over a plain HTTP contract on the private `backend` network.

The model is loaded ONCE at startup (it is the expensive part) and reused
for every request, exactly like sptf reuses its authenticated session.
"""

import os
import subprocess
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
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


app = FastAPI(title="beat-this sidecar", lifespan=lifespan)


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


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    audio2beats = _model["audio2beats"]
    if audio2beats is None:
        return _err("model not loaded yet", 503)

    full = _safe_path(req.path)
    if full is None:
        return _err(f"path outside the music root: {req.path}", 400)
    if not os.path.isfile(full):
        return _err(f"file not found: {req.path}", 404)

    # One ffmpeg decode feeds both the model and the first-solid
    # heuristic — beat-this never opens the file itself.
    pcm = _decode_mono(full)
    if pcm.size == 0:
        return _err(f"could not decode audio: {req.path}", 500)

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
    bpm = 60.0 / float(np.median(intervals)) if intervals.size else None
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

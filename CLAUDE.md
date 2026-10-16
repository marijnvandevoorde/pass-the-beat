# pass-the-beat sidecar

Dockerized Python service that runs the CPJKU **beat-this** beat tracker
over audio files and returns a beat grid. pass-the-aux calls it through its
`AudioAnalyzer` port (`src/infrastructure/beat-this-analyzer.ts`) when
`BEAT_ANALYZER=pass-the-beat`.

## Why this exists

pass-the-aux's in-house Ellis-DP beat tracker is sub-frame accurate but has
mediocre recall — quiet intros, ballads, and sparse drums often yield no
grid, which hurts the auto-transition. beat-this (a PyTorch model) has
much better recall. PyTorch has no place in pass-the-aux's near-zero-dep,
type-stripped TypeScript runtime, so it lives here as a sidecar — a
self-contained service on the shared `pta` docker network reached over
a plain HTTP contract.

## Contract

- `POST /analyze-audio` — the audio file as the raw request body (what
  pass-the-aux sends; the service needs no storage). Spooled to a temp
  file, analysed, deleted. Same response as `/analyze`.
- `POST /analyze` — body `{ "path": "<uid>/song.mp3" }`, a path relative
  to the read-only `/music` mount. Returns
  `{ beats[], downbeatIndices[], bpm, confidence, duration, firstSolidBeat }`.
  `beats` are second offsets; `downbeatIndices` index into `beats`;
  `firstSolidBeat` is a beat index (or -1). On failure: non-2xx +
  `{ error }`: 422 undecodable audio (permanent for that file), 503
  model still loading, 500 model failure. pass-the-aux stores a 422 as
  "analysed, no grid" so it is not retried; 503 / unreachable means "try
  again later".
- `GET /healthz` — `{ status, model_loaded, checkpoint }`.

The Node adapter collapses `downbeatIndices` to the existing 4/4
`downbeatPhase` (`downbeatIndices[0] % 4`) — no DB schema change. The
explicit per-beat downbeat array is a future enhancement.

## Model lifecycle

The checkpoint (`final0`) is baked into the image at build time and the
`Audio2Beats` model is loaded once in the FastAPI lifespan — loading is
the expensive part, so it is reused for every request. Requests are
effectively serialized by the single in-process model.

The service decodes audio itself with the **ffmpeg binary** and feeds
beat-this a raw waveform (`Audio2Beats`, not `File2Beats`) — beat-this's
own torchaudio/soundfile loader cannot decode mp3 in this image.

## File access

pass-the-aux uploads the bytes (`/analyze-audio`); its library lives in
an object-storage bucket, not on a shared volume. `/analyze` by path
still works for compose setups that mount a `music/` dir read-only at
`/music`. The service never writes outside its temp dir.

## Scaling

The service is stateless, so on Kubernetes it runs at **zero replicas**
and pass-the-aux scales it to 1 only when tracks need a grid (it waits
for `/healthz` to report `model_loaded`), then back to 0 after an idle
period. Model load time is therefore paid once per batch, not per track.

## Operations

In pass-the-aux's stack the service is internal (`expose: 8000`, no
host port). For local hacking on this repo, the in-repo `compose.yml`
publishes a host port and requires `MUSIC_DIR` in `.env`:

```bash
cp .env.example .env   # set MUSIC_DIR
docker compose up -d --build
docker compose logs -f pass-the-beat
curl http://localhost:8000/healthz
```

## Files

| Path | Role |
|---|---|
| `server.py` | FastAPI app — model lifespan + `/analyze` + `/healthz` |
| `Dockerfile` | python:3.11-slim + ffmpeg + torch CPU + beat_this |
| `compose.yml` | Standalone single-service deploy (pass-the-aux's stack defines its own) |
| `.env.example` | Standalone overrides (`MUSIC_DIR` only for `/analyze`) |
| `requirements.txt` | fastapi, uvicorn, numpy, beat_this (git) |
| `.dockerignore` | keeps build context lean |

<p align="center">
  <img src="passthebeat.png" alt="pass-the-beat" width="180" />
</p>

# pass-the-beat

A small, Dockerized HTTP service that runs the **beat-this** neural beat
tracker (CPJKU — <https://github.com/CPJKU/beat_this>) over audio files
and returns a beat grid.

It is a **sidecar for [pass-the-aux](https://github.com/marijnvandevoorde/pass-the-aux)**,
our DJ app: pass-the-aux calls it through its `AudioAnalyzer` port when
`BEAT_ANALYZER=pass-the-beat` to get a high-recall beat grid for track
analysis and auto-transition. It is a plain HTTP API, so it can also be
used standalone.

## Why it exists

pass-the-aux ships an in-house, dependency-free beat tracker (Ellis
dynamic programming). It is sub-frame accurate but has mediocre recall
— quiet intros, ballads and sparse drums often yield no grid, which
hurts the auto-transition.

`beat-this` is a PyTorch model with far better recall. PyTorch has no
place in pass-the-aux's near-zero-dependency, type-stripped TypeScript
runtime, so it lives here as a separate service — a self-contained
sidecar reached over a plain HTTP contract.

## API

### `POST /analyze-audio`

The audio file itself as the raw request body (any content type ffmpeg
can decode; send `Content-Length`). An optional `X-Track-Name` header is
only used in error messages. This is what pass-the-aux uses: the service
needs no access to the caller's storage. The upload is spooled to a temp
file, analysed and deleted; bodies over `MAX_UPLOAD_BYTES` (200 MB) get
`413`.

```bash
curl -X POST http://localhost:8000/analyze-audio --data-binary @song.mp3
```

### `POST /analyze`

For setups that share a folder instead. Request body:

```json
{ "path": "path/to/song.mp3" }
```

`path` is relative to the read-only `/music` mount.

Response:

```json
{
  "beats": [0.51, 0.98, 1.45],
  "downbeatIndices": [0, 4, 8],
  "bpm": 124.0,
  "confidence": 0.9,
  "duration": 212.3,
  "firstSolidBeat": 16
}
```

- `beats` — beat times, in seconds.
- `downbeatIndices` — indices into `beats` that fall on a downbeat.
- `firstSolidBeat` — index of the first reliable beat, or `-1`.

Both endpoints return the same response. On failure: a non-2xx status
with `{ "error": "<message>" }` — `422` when the audio cannot be decoded
(permanent for that file), `503` while the model is still loading, `500`
when the model itself fails.

### `GET /healthz`

```json
{ "status": "ok", "model_loaded": true, "checkpoint": "final0" }
```

## How it works

- The **`final0` checkpoint** is baked into the Docker image at build
  time. The `Audio2Beats` model is loaded **once** in the FastAPI
  lifespan — loading is the expensive part — and reused for every
  request, so requests are effectively serialized by the single
  in-process model.
- The service decodes audio itself with the **`ffmpeg` binary** and
  feeds beat-this a raw waveform; beat-this's own torchaudio/soundfile
  loader cannot decode mp3 in this image.
- With `/analyze-audio` the service has no storage at all. With
  `/analyze`, the host `music/` directory is mounted **read-only** at
  `/music`. The service never writes outside its temp dir.
- It holds nothing between requests, so it can be scaled to zero and
  started only when there is something to analyse (pass-the-aux does this
  on Kubernetes).

## Run

### As part of pass-the-aux (normal case)

The [pass-the-aux](https://github.com/marijnvandevoorde/pass-the-aux)
`docker compose` stack already wires this service up on the shared
`pta` network; pass-the-aux uploads each track to `/analyze-audio`.
Nothing extra to do — set `BEAT_ANALYZER=pass-the-beat` on the pass-the-aux side and bring
the stack up.

### In production

Production runs on the Kubernetes cluster `sv-oke` (Deployment `pass-the-beat`, namespace
`apps`), scaled from 0 to 1 by pass-the-aux while it analyses. `scripts/deploy.sh` ships
`origin/main` there with the shared `sv-deploy` script from the Oracle repo (config in
`deploy.env`); `scripts/deploy.sh --check` shows what runs. `SV_DEPLOY` overrides the script's
path (default `../../Oracle/deploy/sv-deploy`).

### Standalone

A self-contained `compose.yml` ships in this repo. Copy
`.env.example` to `.env`; point `MUSIC_DIR` at a host directory of audio
if you want to use `/analyze` by path:

```bash
cp .env.example .env   # optionally edit MUSIC_DIR
docker compose up -d --build
curl http://localhost:8000/healthz
curl -X POST http://localhost:8000/analyze \
     -H 'content-type: application/json' \
     -d '{"path": "song.mp3"}'
```

Defaults: host port `8000`, CPU-only torch, `final0` checkpoint.
`MUSIC_DIR` defaults to `./music`; `.env.example` documents the optional
overrides.

## Files

| Path | Role |
|---|---|
| `server.py` | FastAPI app — model lifespan, `/analyze-audio`, `/analyze`, `/healthz` |
| `Dockerfile` | `python:3.11-slim` + ffmpeg + torch (CPU) + `beat_this` |
| `compose.yml` | Standalone single-service deployment |
| `.env.example` | Optional overrides for the standalone deploy |
| `requirements.txt` | `fastapi`, `uvicorn`, `numpy`, `beat_this` |
| `.dockerignore` | Keeps the Docker build context lean |
| `CLAUDE.md` | Design notes / contributor contract |

## Credits

Beat tracking by **beat-this** (Foscarin, Schlüter & Widmer — CPJKU).
This service only wraps the model in an HTTP API; all model credit is
theirs. See <https://github.com/CPJKU/beat_this>.

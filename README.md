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
  "downbeatPhase": 0,
  "bpm": 124.0,
  "confidence": 0.9,
  "duration": 212.3,
  "firstSolidBeat": 16,
  "analyzerVersion": 2,
  "stats": { "raw_beats": 431, "extra_dropped": 7, "filled": 0, "breaks": 0, "bar_consistency": 0.97 }
}
```

- `beats` — a cleaned beat grid, in seconds (see *How the grid is built*).
  Holes of a whole number of beats are filled with interpolated beats.
- `downbeatPhase` — `beats[i]` starts a bar when `i % 4 == downbeatPhase`,
  for the whole track.
- `downbeatIndices` — the detected downbeats that agree with that phase
  (indices into `beats`), so `downbeatIndices[0] % 4 == downbeatPhase`.
- `bpm` — from the steady beat intervals (not quantised to the model's
  50 fps frames).
- `firstSolidBeat` — the **mix-in point**: index of the first downbeat from
  which the kick/bass band holds at the track's main level for two bars;
  `-1` when there is no such point in the first 40 % of the track (drumless
  or slow-building tracks), when the track is too short, or when there are
  no beats.
- `confidence` — 0..1, share of detected (not interpolated) beats times how
  steady the grid is. beat-this has no score of its own.
- `duration` — decoded length in seconds.
- `analyzerVersion` — bumps whenever the grid changes meaningfully, so a
  caller can re-grid tracks analysed by an older version.
- `stats` — diagnostics: raw beat count, extra beats dropped, beats filled,
  breaks in the grid, share of detected downbeats on the chosen phase.

Both endpoints return the same response. An empty `beats` array means "no
pulse" (also for audio under one second). On failure: a non-2xx status
with `{ "error": "<message>" }`:

| Status | Meaning | Retry? |
|---|---|---|
| `400` | empty body / path outside the music root | no |
| `404` | `/analyze`: file not found | no |
| `413` | upload over `MAX_UPLOAD_BYTES` | no |
| `422` | ffmpeg can't decode the file (reason included), or the decode took over `DECODE_TIMEOUT_S` | no, permanent for that file |
| `500` | the model failed on this input | no |
| `503` | model still loading, queue full, out of memory, ffmpeg missing, or an internal error; has `Retry-After` | yes |

### `GET /healthz`

```json
{
  "status": "ok", "model_loaded": true, "checkpoint": "final0",
  "analyzerVersion": 2, "busy": false, "waiting": 0,
  "analysed": 12, "failed": 0, "last_error": null
}
```

## How it works

- The **`final0` checkpoint** is baked into the Docker image at build
  time. The `Audio2Beats` model is loaded **once** in the FastAPI
  lifespan — loading is the expensive part — and reused for every
  request. One analysis runs at a time; up to `MAX_QUEUE` (4) wait, more
  get `503` right away.
- The service decodes audio itself with the **`ffmpeg` binary** and
  feeds beat-this a raw waveform; beat-this's own torchaudio/soundfile
  loader cannot decode mp3 in this image.
- With `/analyze-audio` the service has no storage at all. With
  `/analyze`, the host `music/` directory is mounted **read-only** at
  `/music`. The service never writes outside its temp dir.
- It holds nothing between requests, so it can be scaled to zero and
  started only when there is something to analyse (pass-the-aux does this
  on Kubernetes).

## How the grid is built

beat-this without its DBN post-processor has high recall, which also means
it emits things that are not beats: half- and third-beat "extra beats"
through snare rolls and hi-hat triplets, and holes in sparse parts. A DJ
app that snaps to the nearest beat and counts bars as "every 4th beat"
falls off the grid at the first one. `grid.py` therefore:

1. keeps the chain of detected beats that best fits one steady period
   (dynamic programming; tempo drift of a few percent is fine) and fills
   holes of a whole number of beats;
2. picks one bar phase for the whole track from the majority of beat-this's
   downbeats, dropping up to 3 beats after a break in the grid so the
   phase holds across it;
3. finds the mix-in point from the 40–150 Hz band, judged per bar so
   half-time and broken beats count.

Every analysis logs one line with timings and these stats.

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
| `server.py` | FastAPI app — model lifespan, `/analyze-audio`, `/analyze`, `/healthz`, status codes |
| `grid.py` | Pure numpy post-processing: clean grid, bar phase, mix-in point |
| `tests/` | `python -m unittest discover tests` (no torch needed); `python tests/smoke.py` loads the real model |
| `Dockerfile` | `python:3.11-slim` + ffmpeg + torch (CPU) + `beat_this` |
| `compose.yml` | Standalone single-service deployment |
| `.env.example` | Optional overrides for the standalone deploy |
| `requirements.txt` | `fastapi`, `uvicorn`, `numpy`, `beat_this` (pinned commit) |
| `.dockerignore` | Keeps the Docker build context lean |
| `CLAUDE.md` | Design notes / contributor contract |

## Credits

Beat tracking by **beat-this** (Foscarin, Schlüter & Widmer — CPJKU).
This service only wraps the model in an HTTP API; all model credit is
theirs. See <https://github.com/CPJKU/beat_this>.

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
  `{ beats[], downbeatIndices[], downbeatPhase, barStarts[], bpm,
  confidence, duration, firstSolidBeat, mixOutBeat, analyzerVersion,
  stats }`. `beats` are second offsets of the cleaned grid;
  `downbeatIndices` index into `beats` and all sit on `downbeatPhase`
  (`i % 4`); `barStarts` is every bar's downbeat (odd bars move the later
  ones); `firstSolidBeat` is the mix-in beat index and `mixOutBeat` the
  mix-out beat index (both downbeats, or -1). The README documents every
  field.
- Status codes: 200 grid (empty = no pulse); 400/404/413 bad request; 422
  undecodable (permanent for that file); 500 the model failed on this
  input; 503 try again later (loading, queue full, out of memory, ffmpeg
  missing, any unhandled error). pass-the-aux stores 400–500 (except
  404/408/429) as "analysed, no grid" and never retries them, so anything
  that is not a property of the file **must** be 503.
- Additive fields only. When the grid output changes meaningfully, bump
  `grid.ANALYZER_VERSION`; pass-the-aux only re-grids tracks when its own
  `BEATTHIS_ANALYZER_VERSION` is bumped, so tell its maintainer.
- `GET /healthz` — `{ status, model_loaded, checkpoint }`.

pass-the-aux (≥ its `BEATTHIS_ANALYZER_VERSION` 5) stores `barStarts`
and `mixOutBeat` next to the grid and counts bars from `barStarts`; older
aux versions ignore them and use `downbeatPhase`, which stays valid.

## Grid post-processing (`grid.py`)

Pure numpy, unit-tested without torch. Raw beat-this output (no DBN) has
extra half/third beats in rolls and holes in sparse parts; pass-the-aux
snaps to the nearest beat and counts bars as `i % 4`, so the raw output
made fades land off-beat. `regularize` keeps the best steady chain and
fills whole-beat holes, `assign_bars` picks one bar phase (dropping up to
3 beats after a break), `bar_positions` places every bar's downbeat
(Viterbi: a 3- or 5-beat bar needs ~4 bars of downbeats on the new phase,
a half-bar shift ~8, because beat-this flips to the half bar in
breakdowns), `mix_in_index` finds the first downbeat with two bars of
kick/bass at the track's main level, `mix_out_index` the phrase boundary
(8 bars from the mix-in) where the outro starts, or the last one that
leaves 8 bars before the end. Tuning constants sit at the
top of the file; check changes against real tracks, not only the
synthetic tests.
Half/double tempo is left to beat-this; pass-the-aux octave-matches
tempos when syncing.

## Model lifecycle

The checkpoint (`final0`) is baked into the image at build time and the
`Audio2Beats` model is loaded once in the FastAPI lifespan — loading is
the expensive part, so it is reused for every request. A lock runs one
analysis at a time; `MAX_QUEUE` more may wait, the rest get 503.

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
| `server.py` | FastAPI app — model lifespan, endpoints, status codes, logging |
| `grid.py` | Grid post-processing (clean grid, bar phase, mix-in) |
| `tests/` | `python -m unittest discover tests` (fake model) + `tests/smoke.py` (real model) |
| `Dockerfile` | python:3.14-slim + ffmpeg + torch CPU + beat_this |
| `compose.yml` | Standalone single-service deploy (pass-the-aux's stack defines its own) |
| `.env.example` | Standalone overrides (`MUSIC_DIR` only for `/analyze`) |
| `requirements.txt` | fastapi, uvicorn, numpy, beat_this (git, pinned commit) |
| `.dockerignore` | keeps build context lean |

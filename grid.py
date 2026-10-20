"""Turns raw beat-this output into a grid a DJ app can mix on.

Pure numpy, no torch: every function here is unit-tested without the model.

beat-this without its DBN post-processor is a high-recall tracker, which
also means it emits things that are not beats: half- and third-beat
"extra beats" through snare rolls and hi-hat triplets, and holes in sparse
parts. pass-the-aux snaps the incoming track to the nearest beat and
models bars as "every 4th beat", so one extra beat puts the incoming track
on an off-beat and shifts every later bar. This module:

1. picks the chain of detected beats that best fits one steady period and
   fills holes of a whole number of beats (`regularize`);
2. derives one bar phase for the whole grid, dropping up to 3 beats after
   a break so "every 4th beat" stays true across it (`assign_bars`);
3. finds the mix-in point: the first downbeat from which the kick/bass
   band holds for two bars (`mix_in_index`).
"""

from dataclasses import dataclass, field

import numpy as np

# Bump when the grid this module produces changes meaningfully, so callers
# can re-grid tracks analysed by an older version (served as
# `analyzerVersion` in every response and in /healthz).
# 1: raw beat-this beats (pass-the-beat up to 2026-10).
# 2: regularised grid, robust bar phase, kick-based mix-in point.
ANALYZER_VERSION = 2

# A step between two kept beats may deviate this much from a whole number
# of periods (in beats). Extra beats sit at 0.33 / 0.5 of a period and are
# rejected; tempo drift of a few percent passes.
STEP_TOLERANCE = 0.2
# Cost of a step per beat of deviation squared, and per missing beat.
DEVIATION_COST = 25.0
MISSING_BEAT_COST = 0.05
# Longest hole (in beats) filled with interpolated beats. Longer holes,
# or holes that are not a whole number of beats, break the grid instead.
MAX_FILL_BEATS = 64
# Cost of breaking the grid (starting a new run of beats).
BREAK_COST = 3.0
# Mix-in: the kick/bass band must clear MIX_IN_FRACTION of the track's
# loud level on MIX_IN_RUN consecutive beats (two bars).
MIX_IN_RUN = 8
MIX_IN_FRACTION = 0.4
MIX_IN_PERCENTILE = 80
# A mix-in point later than this share of the track is not one.
MIX_IN_MAX_SHARE = 0.4
LOW_BAND_HZ = (40.0, 150.0)
# Below this share of the full-band level the low band is leakage, not a
# kick or bass line (real dance tracks sit at 0.75-0.9).
MIN_LOW_SHARE = 0.1


def beat_period(beats: np.ndarray) -> float | None:
    """Beat period in seconds from the inter-beat intervals.

    beat-this places beats on a 50 fps frame grid, so a single interval
    (and the median) is quantised to 20 ms steps: 0.46 s / 0.48 s, i.e.
    130.4 / 125 BPM, with nothing in between. Averaging the intervals of
    the steady part (within 10 % of the median, so missed or extra beats
    don't count) cancels the quantisation, e.g. 128.0."""
    if beats.size < 2:
        return None
    intervals = np.diff(beats)
    med = float(np.median(intervals))
    if med <= 0:
        return None
    steady = intervals[np.abs(intervals - med) <= 0.1 * med]
    return float(np.mean(steady if steady.size else intervals))


@dataclass
class Grid:
    """A regularised beat grid, split into runs at breaks."""

    beats: np.ndarray  # seconds, strictly ascending
    real: np.ndarray  # bool per beat: detected (True) or interpolated
    # Index into `beats` where each run starts (always starts with 0).
    run_starts: list[int] = field(default_factory=list)
    extra_dropped: int = 0  # detected beats left out
    filled: int = 0  # interpolated beats added


def regularize(raw: np.ndarray, period: float) -> Grid:
    """Best chain of detected beats for one steady period, holes filled.

    Dynamic programme over the detected beats: every kept beat scores 1, a
    step of k periods costs its squared deviation plus a little per missing
    beat, and a step that is not close to a whole number of periods is only
    possible as a break (fixed cost, no fill)."""
    n = int(raw.size)
    if n == 0 or period <= 0:
        return Grid(np.zeros(0), np.zeros(0, dtype=bool), [])
    t = raw.astype(np.float64)
    score = np.ones(n)
    prev = np.full(n, -1)
    kind = np.zeros(n, dtype=np.int8)  # 0 start, 1 step, 2 break
    steps = np.zeros(n, dtype=np.int64)
    # Running best score among beats at least 0.75 period back (break
    # candidates), advanced with a pointer since `t` is sorted.
    best_back = -np.inf
    best_back_i = -1
    ptr = 0
    max_span = (MAX_FILL_BEATS + 0.5) * period
    lo = 0
    for j in range(n):
        while ptr < j and t[j] - t[ptr] >= 0.75 * period:
            if score[ptr] > best_back:
                best_back, best_back_i = score[ptr], ptr
            ptr += 1
        while t[j] - t[lo] > max_span:
            lo += 1
        best, best_i, best_k, best_kind = 1.0, -1, 0, 0
        for i in range(lo, j):
            d = (t[j] - t[i]) / period
            k = round(d)
            if k < 1:
                continue
            dev = d - k
            if abs(dev) > STEP_TOLERANCE:
                continue
            s = score[i] + 1.0 - DEVIATION_COST * dev * dev - MISSING_BEAT_COST * (k - 1)
            if s > best:
                best, best_i, best_k, best_kind = s, i, k, 1
        if best_back_i >= 0:
            s = best_back + 1.0 - BREAK_COST
            if s > best:
                best, best_i, best_k, best_kind = s, best_back_i, 0, 2
        score[j], prev[j], steps[j], kind[j] = best, best_i, best_k, best_kind

    # Backtrack from the best end.
    j = int(np.argmax(score))
    chain: list[int] = []
    while j >= 0:
        chain.append(j)
        j = int(prev[j])
    chain.reverse()

    beats: list[float] = []
    real: list[bool] = []
    run_starts = [0]
    filled = 0
    for pos, j in enumerate(chain):
        if pos > 0:
            if kind[j] == 2:
                run_starts.append(len(beats))
            elif steps[j] > 1:
                i = chain[pos - 1]
                k = int(steps[j])
                for m in range(1, k):
                    beats.append(t[i] + (t[j] - t[i]) * m / k)
                    real.append(False)
                filled += k - 1
        beats.append(float(t[j]))
        real.append(True)
    return Grid(
        np.asarray(beats),
        np.asarray(real, dtype=bool),
        run_starts,
        extra_dropped=n - len(chain),
        filled=filled,
    )


def nearest(beats: np.ndarray, times: np.ndarray, max_dist: float) -> list[int]:
    """Index of the nearest beat for each time, -1 when none is close."""
    out: list[int] = []
    for ti in times:
        i = int(np.searchsorted(beats, ti))
        cands = [c for c in (i - 1, i) if 0 <= c < beats.size]
        if not cands:
            out.append(-1)
            continue
        c = min(cands, key=lambda c: abs(beats[c] - ti))
        out.append(c if abs(beats[c] - ti) <= max_dist else -1)
    return out


@dataclass
class Bars:
    """Bar phase for the whole grid (4/4)."""

    beats: np.ndarray
    real: np.ndarray
    phase: int  # beats[i] is a downbeat when i % 4 == phase
    downbeat_indices: list[int]  # detected downbeats that agree with phase
    consistency: float  # share of detected downbeats on `phase`, 0..1


def assign_bars(grid: Grid, downbeats: np.ndarray, period: float) -> Bars:
    """One bar phase for the whole grid.

    Each run between breaks gets its own majority phase from beat-this's
    downbeats; a later run that disagrees with the first loses its first
    1-3 beats, so `i % 4 == phase` holds across the break."""
    beats, real = grid.beats, grid.real
    if beats.size == 0:
        return Bars(beats, real, 0, [], 0.0)
    near = nearest(beats, downbeats, 0.3 * period)
    starts = list(grid.run_starts) + [int(beats.size)]
    keep = np.ones(beats.size, dtype=bool)
    target: int | None = None
    offset = 0  # beats dropped so far
    for r in range(len(starts) - 1):
        a, b = starts[r], starts[r + 1]
        hits = [i for i in near if a <= i < b]
        if not hits:
            continue
        counts = np.bincount([(i - offset) % 4 for i in hits], minlength=4)
        phase = int(np.argmax(counts))
        if target is None:
            target = phase
            continue
        drop = (phase - target) % 4
        if drop and b - a > drop:
            keep[a : a + drop] = False
            offset += drop
    if target is None:
        target = 0
    new_index = np.cumsum(keep) - 1
    kept_beats, kept_real = beats[keep], real[keep]
    on_phase = sorted(
        {int(new_index[i]) for i in near if i >= 0 and keep[i] and new_index[i] % 4 == target}
    )
    detected = sum(1 for i in near if i >= 0)
    consistency = len(on_phase) / detected if detected else 0.0
    return Bars(kept_beats, kept_real, target, on_phase, consistency)


def low_band_envelope(
    pcm: np.ndarray, sr: int, hop_sec: float = 0.02
) -> tuple[np.ndarray, np.ndarray, float]:
    """Kick/bass-band (40-150 Hz) and full-band magnitude per hop, computed
    in chunks so a long track never needs one huge STFT.
    Returns (low, full, frames per second)."""
    frame = 1024
    hop = max(1, int(round(sr * hop_sec)))
    if pcm.size < frame:
        return np.zeros(0), np.zeros(0), sr / hop
    n_frames = 1 + (pcm.size - frame) // hop
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    band = (freqs >= LOW_BAND_HZ[0]) & (freqs <= LOW_BAND_HZ[1])
    window = np.hanning(frame).astype(np.float32)
    env = np.empty(n_frames, dtype=np.float32)
    full = np.empty(n_frames, dtype=np.float32)
    chunk = 2048
    for c0 in range(0, n_frames, chunk):
        c1 = min(n_frames, c0 + chunk)
        idx = (np.arange(c0, c1)[:, None] * hop) + np.arange(frame)[None, :]
        spec = np.abs(np.fft.rfft(pcm[idx] * window, axis=1))
        power = spec**2
        env[c0:c1] = np.sqrt(np.sum(power[:, band], axis=1))
        full[c0:c1] = np.sqrt(np.sum(power, axis=1))
    return env, full, sr / hop


def mix_in_index(bars: Bars, pcm: np.ndarray, sr: int, duration: float) -> int:
    """Index of the mix-in beat: the first downbeat from which the kick/bass
    band holds for MIX_IN_RUN beats, most of them detected (not filled).

    -1 when there is no such point, when the track is too short, or when it
    would land past MIX_IN_MAX_SHARE of the track. A drumless intro at full
    loudness does not count, unlike the broadband RMS used before."""
    beats = bars.beats
    if beats.size < MIX_IN_RUN:
        return -1
    env, full, rate = low_band_envelope(pcm, sr)
    if env.size == 0:
        return -1
    if np.percentile(env, MIX_IN_PERCENTILE) < MIN_LOW_SHARE * np.percentile(full, MIX_IN_PERCENTILE):
        return -1  # no kick or bass anywhere
    lo = np.clip(((beats - 0.05) * rate).astype(int), 0, env.size - 1)
    hi = np.clip(((beats + 0.08) * rate).astype(int) + 1, 1, env.size)
    per_beat = np.array([float(env[a:b].max()) if b > a else 0.0 for a, b in zip(lo, hi)])
    ref = float(np.percentile(per_beat, MIX_IN_PERCENTILE))
    if ref <= 0:
        return -1
    # Judge whole bars, not single beats: half-time and broken beats have
    # no kick on some beats, and a lone hit per bar is not a groove yet. A
    # bar is strong when at least two of its beats clear the threshold; the
    # mix-in is the first downbeat that starts MIX_IN_RUN / 4 strong bars in
    # a row.
    bars_needed = MIX_IN_RUN // 4
    need_real = MIX_IN_RUN - 2
    starts = list(range(bars.phase, beats.size - 3, 4))
    strong = [
        float(np.sort(per_beat[s : s + 4])[-2]) >= MIX_IN_FRACTION * ref
        for s in starts
    ]
    for k in range(len(starts) - bars_needed + 1):
        if not all(strong[k : k + bars_needed]):
            continue
        s = starts[k]
        if int(bars.real[s : s + 4 * bars_needed].sum()) < need_real:
            continue
        if duration > 0 and beats[s] > MIX_IN_MAX_SHARE * duration:
            return -1
        return int(s)
    return -1


def confidence(bars: Bars, period: float) -> float:
    """0..1: share of detected beats in the grid times how steady the steps
    are. beat-this gives no score of its own."""
    beats = bars.beats
    if beats.size < 3 or period <= 0:
        return 0.0 if beats.size == 0 else 0.5
    iv = np.diff(beats)
    steps = iv[np.abs(iv / period - 1) <= STEP_TOLERANCE]
    if steps.size < 2:
        return 0.0
    cv = float(np.std(steps) / np.mean(steps))
    regularity = max(0.0, min(1.0, 1.0 - cv * 4.0))
    real_share = float(bars.real.mean())
    return round(regularity * real_share, 3)


@dataclass
class Analysis:
    beats: list[float]
    downbeat_indices: list[int]
    downbeat_phase: int
    bpm: float | None
    confidence: float
    first_solid_beat: int
    stats: dict


def analyse(raw_beats, raw_downbeats, pcm: np.ndarray, sr: int, duration: float) -> Analysis:
    """Full post-processing of one track's beat-this output."""
    raw = np.asarray(raw_beats, dtype=np.float64)
    downs = np.asarray(raw_downbeats, dtype=np.float64)
    period = beat_period(raw)
    if raw.size == 0 or period is None:
        return Analysis([], [], 0, None, 0.0, -1, {"raw_beats": int(raw.size)})
    grid = regularize(raw, period)
    bars = assign_bars(grid, downs, period)
    period = beat_period(bars.beats) or period
    first_solid = mix_in_index(bars, pcm, sr, duration)
    return Analysis(
        beats=[round(float(b), 4) for b in bars.beats],
        downbeat_indices=bars.downbeat_indices,
        downbeat_phase=bars.phase,
        bpm=round(60.0 / period, 2),
        confidence=confidence(bars, period),
        first_solid_beat=first_solid,
        stats={
            "raw_beats": int(raw.size),
            "extra_dropped": grid.extra_dropped,
            "filled": grid.filled,
            "breaks": len(grid.run_starts) - 1,
            "bar_consistency": round(bars.consistency, 3),
        },
    )

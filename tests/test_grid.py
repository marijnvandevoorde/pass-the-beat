"""Unit tests for grid.py on synthetic beats. No model, no ffmpeg.

Run from the repo root: `python -m unittest discover tests`."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import grid  # noqa: E402

SR = 22_050
P = 0.5  # 120 BPM


def steady(start=0.25, end=60.0, period=P):
    return np.arange(start, end, period)


def pcm_with(
    kick_from: float | None, total: float, pad: bool = True, kick_until: float | None = None
) -> np.ndarray:
    """A loud 440 Hz pad all along, plus 60 Hz kicks on every beat from
    `kick_from` seconds (None = never) until `kick_until` (None = the end)."""
    t = np.arange(int(total * SR)) / SR
    x = 0.3 * np.sin(2 * np.pi * 440 * t) if pad else np.zeros_like(t)
    if kick_from is not None:
        for b in steady(0.25, total, P):
            if b < kick_from or (kick_until is not None and b >= kick_until):
                continue
            i0 = int(b * SR)
            n = int(0.12 * SR)
            seg = np.arange(min(n, t.size - i0)) / SR
            x[i0 : i0 + seg.size] += 0.8 * np.sin(2 * np.pi * 60 * seg) * np.exp(-seg * 20)
    return x.astype(np.float32)


class BeatPeriod(unittest.TestCase):
    def test_quantised_intervals_average_out(self):
        # 128 BPM on a 50 fps grid: intervals of 0.46 and 0.48 s.
        rng = np.random.default_rng(1)
        true = np.arange(0, 120, 60 / 128)
        beats = np.round(true * 50) / 50 + rng.normal(0, 0.001, true.size)
        self.assertAlmostEqual(60 / grid.beat_period(beats), 128, delta=0.3)

    def test_too_few_beats(self):
        self.assertIsNone(grid.beat_period(np.array([1.0])))


class Regularize(unittest.TestCase):
    def test_clean_grid_unchanged(self):
        b = steady()
        g = grid.regularize(b, P)
        np.testing.assert_allclose(g.beats, b)
        self.assertTrue(g.real.all())
        self.assertEqual(g.run_starts, [0])

    def test_extra_half_and_third_beats_dropped(self):
        b = steady()
        halves = b[(b > 20) & (b < 24)] + P / 2  # snare roll
        thirds = np.concatenate([b[(b > 40) & (b < 42)] + P / 3, b[(b > 40) & (b < 42)] + 2 * P / 3])
        raw = np.sort(np.concatenate([b, halves, thirds]))
        g = grid.regularize(raw, P)
        np.testing.assert_allclose(g.beats, b)
        self.assertEqual(g.extra_dropped, halves.size + thirds.size)

    def test_hole_filled_with_interpolated_beats(self):
        b = steady()
        hole = (b > 30) & (b < 34)
        g = grid.regularize(b[~hole], P)
        np.testing.assert_allclose(g.beats, b, atol=1e-9)
        self.assertEqual(int((~g.real).sum()), int(hole.sum()))
        self.assertEqual(g.filled, int(hole.sum()))

    def test_tempo_drift_kept_as_one_run(self):
        # 3 % faster over the track (live drummer).
        periods = np.linspace(P, P * 0.97, 200)
        b = np.cumsum(periods)
        g = grid.regularize(b, float(np.mean(periods)))
        self.assertEqual(g.beats.size, b.size)
        self.assertEqual(g.run_starts, [0])

    def test_off_grid_restart_breaks_without_filling(self):
        first = steady(0.25, 30)
        second = steady(30.25 + 0.4 * P, 60)  # 0.4 beat out of phase
        g = grid.regularize(np.concatenate([first, second]), P)
        self.assertEqual(len(g.run_starts), 2)
        self.assertEqual(g.filled, 0)
        self.assertEqual(g.beats.size, first.size + second.size)

    def test_empty(self):
        g = grid.regularize(np.zeros(0), P)
        self.assertEqual(g.beats.size, 0)


class AssignBars(unittest.TestCase):
    def test_majority_phase_wins_over_a_wrong_first_downbeat(self):
        b = steady()
        g = grid.regularize(b, P)
        downs = np.concatenate([[b[0]], b[1::4]])  # first one is wrong
        bars = grid.assign_bars(g, downs, P)
        self.assertEqual(bars.phase, 1)
        self.assertEqual(bars.downbeat_indices[0] % 4, 1)
        self.assertTrue(all(i % 4 == 1 for i in bars.downbeat_indices))

    def test_phase_holds_across_a_break(self):
        first = steady(0.25, 30)  # 60 beats, downbeats on index 0 mod 4
        second = steady(30.25 + 0.4 * P, 60)
        g = grid.regularize(np.concatenate([first, second]), P)
        # In the second run the real downbeats sit on its local beat 2.
        downs = np.concatenate([first[0::4], second[2::4]])
        bars = grid.assign_bars(g, downs, P)
        self.assertEqual(bars.phase, 0)
        on_phase = set(bars.downbeat_indices)
        for t in downs:
            i = int(np.argmin(np.abs(bars.beats - t)))
            self.assertIn(i, on_phase, f"downbeat at {t:.2f}s not on phase")
            self.assertEqual(i % 4, 0)
        self.assertGreater(bars.consistency, 0.99)

    def test_no_downbeats(self):
        g = grid.regularize(steady(), P)
        bars = grid.assign_bars(g, np.zeros(0), P)
        self.assertEqual(bars.phase, 0)
        self.assertEqual(bars.downbeat_indices, [])


class BarPositions(unittest.TestCase):
    def test_regular_bars(self):
        self.assertEqual(grid.bar_positions(40, list(range(1, 40, 4))), list(range(1, 40, 4)))

    def test_no_downbeats_follow_the_phase(self):
        self.assertEqual(grid.bar_positions(12, [], 2), [2, 6, 10])

    def test_a_lone_wrong_downbeat_changes_nothing(self):
        det = list(range(0, 80, 4)) + [42]
        self.assertEqual(grid.bar_positions(80, sorted(det)), list(range(0, 80, 4)))

    def test_a_three_beat_bar_moves_the_later_bars(self):
        # Bars of 4 up to beat 40, one bar of 3 beats, then 4 again: every
        # later downbeat sits one beat earlier than "every 4th beat".
        det = list(range(0, 41, 4)) + list(range(43, 100, 4))
        starts = grid.bar_positions(100, det)
        self.assertEqual(starts, det)
        lengths = np.diff(starts)
        self.assertEqual(int((lengths == 3).sum()), 1)

    def test_a_short_half_bar_flip_is_not_followed(self):
        # beat-this jumps to the half bar for 4 bars in a breakdown, then
        # comes back: that is the tracker, not the music.
        det = list(range(0, 40, 4)) + list(range(42, 58, 4)) + list(range(60, 120, 4))
        self.assertEqual(grid.bar_positions(120, det), list(range(0, 120, 4)))

    def test_assign_bars_returns_bar_starts(self):
        b = steady()
        g = grid.regularize(b, P)
        bars = grid.assign_bars(g, b[1::4], P)
        self.assertEqual(bars.bar_starts, list(range(1, b.size, 4)))


class MixIn(unittest.TestCase):
    def bars_for(self, total, phase=0):
        b = steady(0.25, total)
        g = grid.regularize(b, P)
        return grid.assign_bars(g, b[phase::4], P)

    def test_drumless_intro_skipped_to_the_kick(self):
        bars = self.bars_for(60)
        idx = grid.mix_in_index(bars, pcm_with(16.0, 60), SR, 60)
        self.assertEqual(idx, 32)  # 16.25 s, a downbeat
        self.assertEqual(idx % 4, bars.phase)

    def test_snaps_to_downbeat(self):
        bars = self.bars_for(60, phase=2)
        idx = grid.mix_in_index(bars, pcm_with(16.0, 60), SR, 60)
        # The kick enters on beat 3 of the bar at 15.25 s: mix in on that
        # bar's downbeat.
        self.assertEqual(idx % 4, 2)
        self.assertAlmostEqual(bars.beats[idx], 15.25)

    def test_kick_from_the_start(self):
        bars = self.bars_for(60)
        self.assertEqual(grid.mix_in_index(bars, pcm_with(0.0, 60), SR, 60), 0)

    def test_no_kick_at_all(self):
        bars = self.bars_for(60)
        self.assertEqual(grid.mix_in_index(bars, pcm_with(None, 60), SR, 60), -1)

    def test_kick_too_late_is_not_a_mix_in(self):
        bars = self.bars_for(60)
        self.assertEqual(grid.mix_in_index(bars, pcm_with(40.0, 60), SR, 60), -1)

    def test_short_track(self):
        bars = self.bars_for(2)
        self.assertEqual(grid.mix_in_index(bars, pcm_with(0.0, 2), SR, 2), -1)


class MixOut(unittest.TestCase):
    TOTAL = 120.0  # 240 beats, 60 bars of 2 s

    def bars_for(self, phase=0):
        b = steady(0.25, self.TOTAL)
        g = grid.regularize(b, P)
        return grid.assign_bars(g, b[phase::4], P)

    def mix_out(self, pcm, mix_in=-1, phase=0):
        bars = self.bars_for(phase)
        return bars, grid.mix_out_index(bars, pcm, SR, self.TOTAL, mix_in)

    def test_on_the_phrase_where_the_outro_starts(self):
        # Kick until 96.25 s (bar 48 = phrase 6): a 12-bar outro.
        bars, idx = self.mix_out(pcm_with(0.0, self.TOTAL, kick_until=96.25), mix_in=0)
        self.assertEqual(idx, 48 * 4)
        self.assertAlmostEqual(bars.beats[idx], 96.25)

    def test_outro_off_the_phrase_mixes_at_the_phrase_before(self):
        # Kick stops at bar 51: mix out at bar 48, three bars before.
        bars, idx = self.mix_out(pcm_with(0.0, self.TOTAL, kick_until=102.25), mix_in=0)
        self.assertEqual(idx, 48 * 4)

    def test_phrases_count_from_the_mix_in(self):
        # Mix-in on bar 2 (4.25 s): phrases start on bars 2, 10, ..., 42, 50.
        # The kick stops at bar 49, so the mix-out is bar 42 (not 48).
        pcm = pcm_with(4.0, self.TOTAL, kick_until=98.25)
        bars, idx = self.mix_out(pcm, mix_in=8)
        self.assertEqual(idx, 42 * 4)
        self.assertEqual((idx - 8) % (4 * grid.PHRASE_BARS), 0)

    def test_groove_to_the_end_takes_the_last_phrase_with_room(self):
        # No outro: the last phrase boundary leaving 8 bars is bar 48.
        bars, idx = self.mix_out(pcm_with(0.0, self.TOTAL), mix_in=0)
        self.assertEqual(idx, 48 * 4)
        self.assertGreaterEqual(self.TOTAL - bars.beats[idx], grid.MIX_OUT_MIN_BARS * 4 * P)

    def test_short_outro_is_no_outro(self):
        # 4 bars of outro: too short to mix in, take the last phrase with room.
        bars, idx = self.mix_out(pcm_with(0.0, self.TOTAL, kick_until=112.25), mix_in=0)
        self.assertEqual(idx, 48 * 4)

    def test_on_a_downbeat(self):
        bars, idx = self.mix_out(pcm_with(0.0, self.TOTAL, kick_until=96.25), phase=2)
        self.assertIn(idx, bars.bar_starts)
        self.assertEqual(idx % 4, 2)

    def test_no_kick_at_all(self):
        _, idx = self.mix_out(pcm_with(None, self.TOTAL))
        self.assertEqual(idx, -1)

    def test_outro_in_the_first_half_is_not_a_mix_out(self):
        _, idx = self.mix_out(pcm_with(0.0, self.TOTAL, kick_until=40.25), mix_in=0)
        self.assertEqual(idx, -1)

    def test_short_track(self):
        b = steady(0.25, 10)
        bars = grid.assign_bars(grid.regularize(b, P), b[0::4], P)
        self.assertEqual(grid.mix_out_index(bars, pcm_with(0.0, 10), SR, 10), -1)


class Analyse(unittest.TestCase):
    def test_end_to_end_on_messy_beats(self):
        b = steady(0.25, 60)
        rolls = b[(b > 20) & (b < 24)] + P / 2
        raw = np.sort(np.concatenate([b[(b < 30) | (b > 33)], rolls]))
        downs = np.concatenate([[b[0]], b[1::4]])
        a = grid.analyse(raw, downs, pcm_with(16.0, 60), SR, 60)
        self.assertAlmostEqual(a.bpm, 120, delta=0.1)
        self.assertEqual(len(a.beats), b.size)
        self.assertEqual(a.downbeat_phase, 1)
        self.assertEqual(a.downbeat_indices[0] % 4, 1)
        self.assertEqual(a.first_solid_beat % 4, 1)
        self.assertGreater(a.confidence, 0.8)
        self.assertEqual(a.stats["extra_dropped"], rolls.size)
        self.assertEqual(a.bar_starts, list(range(1, b.size, 4)))
        self.assertEqual(a.stats["odd_bars"], 0)

    def test_no_beats(self):
        a = grid.analyse([], [], np.zeros(SR, dtype=np.float32), SR, 1.0)
        self.assertEqual(a.beats, [])
        self.assertIsNone(a.bpm)
        self.assertEqual(a.first_solid_beat, -1)
        self.assertEqual(a.bar_starts, [])
        self.assertEqual(a.mix_out_beat, -1)


if __name__ == "__main__":
    unittest.main()

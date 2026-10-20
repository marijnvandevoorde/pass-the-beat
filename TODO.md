# Open work

- Tune the mix-in and mix-out points on a bigger library: checked against level profiles of 7
  album tracks, not by ear (`grid.py` constants `MIX_IN_*`, `MIX_OUT_*`, `OUTRO_QUIET_FRACTION`).
  Rusty Nails gets mix-in 82.7 s (slow build), Eating Hooks -1; mix-outs leave 8–13 bars, A New
  Error 33 (55 s beatless outro). None of the 7 has a long DJ outro with the kick still running
  (only the full-band drop would catch that).
- Odd-bar costs (`ODD_BAR_COST`, `HALF_BAR_COST`) are tuned so none of the 7 local tracks gets an
  odd bar except one 3-beat bar in Wildfire's outro; Zdarlight's half-bar flips in breakdowns are
  treated as tracker noise. Check on a track with a real edit.
- Half/double tempo is left to beat-this (pass-the-aux octave-matches tempos when syncing).

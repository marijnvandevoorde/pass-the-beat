# Open work

- Deploy analyzer v2 (clean grid, bar phase, kick-based mix-in, 503 for transient failures):
  `scripts/deploy.sh` after the PR is merged. Then pass-the-aux must bump
  `BEATTHIS_ANALYZER_VERSION` to 5 so existing tracks are re-gridded.
- Tune the mix-in point on a bigger library: checked against level profiles of 7 tracks, not by ear
  (`grid.py` constants `MIX_IN_*`). Rusty Nails gets 82.7 s (slow build), Eating Hooks gets -1.
- Odd bars (a 2/4 bar inside a 4/4 track) still flip the bar phase for the rest of the track,
  because pass-the-aux models bars as `i % 4`. Real fix is aux-side: use `downbeatIndices`
  per bar instead of one phase.
- Half/double tempo is left to beat-this (pass-the-aux octave-matches tempos when syncing).

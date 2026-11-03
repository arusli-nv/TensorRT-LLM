# Rehearsal (fin6_ra/rb/rc) go/no-go: pre-registered before any rehearsal data (monitor, 2026-10-23)

Derived from what the finals must establish, not from what is easy to check. The finals claim: (1) FT costs ~0 on the
healthy path in every mode, (2) recovery works for every failure type within the user's bars, (3) the comparison is
fair (identical configs, real main as reference), (4) accuracy is unchanged. Each claim fails if one of its
preconditions is false, so each precondition is a rehearsal check. Any FAIL = fix and rerun the affected part before
finals; finals use the same build/harness/launch generator, regenerated only for wheel name and seeds.

## A. The build is the code we think it is
- A1 verify_wheel: 0 differ / 0 stale vs 7e8dfb37df (BUILD-REFREEZE2). A2 the wheel contains the _note_step_change
  guard. A3 NEGCTL: the new FT-off overlap test fails on the old wheel, passes on the new one (the test can see the bug).
- A4 every arm's log names the wheel it ran: freeze arms 7e8dfb37df, main arms 9cc37929e8 (install log / version line).

## B. Every arm actually serves (the failure that cost last night)
- B1 every STD case: server_ready, result.json DONE, every planned healthy-<load> row rc 0, no Traceback /
  "Executor loop crashed" / "Hang detected" / SMOKE_FAIL in any rank log before the last row ends.
- B2 FT off AND FT on in all three modes, the main arm in both modes, LB and ulfm arms: each passes B1.

## C. Off and on differ only in FT (fairness)
- C1 per pair, the effective server config (serve.sh's written yaml + env dump) differs only in FT keys; list any other
  difference. C2 same seed file (hash) and PIN_CHECK OK, cache_miss equal within each off/on pair; main arm PIN_CHECK
  recorded (descriptive). C3 MPIRUN_START mca= of the kill rows vs the healthy FT arms: identical, or the difference is
  listed as a disclosure. C4 burn-in and loads as planned in every arm (bench args in logs).

## D. The measurement machinery measures
- D1 finals_stats.py runs on RA/RB with --window-s set to the SMOKE window and reports every planned pair (none
  silently missing); Xid checked on all 8 nodes in every STD arm; ListenOverflows deltas recorded.
- D2 ctx_stall_cost (1.5x), trim_readout, kill verdicts (KILL_FINAL_VERDICT per row), COLD_RESTART lines,
  ACCURACY_KILL_VERDICT all produce output; every figure 1-11 has its input files (checklist per figure).
- D3 rehearsal numbers are NOT evidence (n=1, short windows) and are never reported. Sanity only: an FT cost cell
  beyond +/-3% vs CLAIMS 171/173 (no-EPLB) or ~0 (online) is investigated before finals, not explained away.

## E. Recovery meets the user's bars in every row and mode
- E1 every kill row: KILL_FINAL_VERDICT PASS; lost ranks == the expected victims (rotated: 5/22, 13/29, nodes 1/5,
  3/7; twokill v and v+4; stop rank 2); resume <= 10 s; post-loss throughput >= 0.8 x live/total; >= 20 after-kill
  requests; the trigger is the fixed --kill-after-steps in every row (incl. twokill/stop).
- E2 rank0 and deadline end the deployment on their own within 180 s; no hang to the job limit.
- E3 the frozen-rank row: confirm ~30-35 s + resume within the bar.

## F. Accuracy
- F1 GSM8K FT on (healthy and after a kill) vs FT off: within the run-to-run spread of the refs; recipe labelled
  (dep32-final.yaml, victim rank 5).

## G. The finals schedule fits (no wasted or idle GPU time)
- G1 measured case minutes (HX, MAIN, kill rows, D, A) -> recompute j1..j8: every group <= 7000 s with >= 10 min
  margin; j1 G0 / j2 G8 second parts start with >= 40 min left; j3 ROUND_MIN satisfiable; group spreads < 10 min.
- G2 no group idled > 10 min in the rehearsal (bundle task end times).

GO = A-G all PASS (or a listed, user-visible disclosure for C1/C3). Then finals j1..j8, then push + PR drafts.

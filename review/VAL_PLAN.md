# VAL plan: freeze validation VAL-H1, VAL-H2, VAL-K, VAL-P (redraft 2026-10-22 15:53 PT for the v5-zero freeze; nothing submitted)

Target: the freeze build = v5-zero rebased onto current main, no overlay. All four launch files carry placeholders.
Every group: EXCLUDED_NODE guard, then a short in-job unit gate (7 fast FT files, ~6 min; idle-reaper rule < 10 min);
the full FT suite runs in the freeze build job's extras. Hang detection on in every FT case (hang_confirm_s 30);
Open MPI's in-band detector off (option A: frozen-rank detection = the 30 s confirm, a stated limit).
Dry runs (`tools/dry_run.sh <file> 16 0 8`) on build-v5zero-f9e6bc0f8b + overlay vzero-65a39b411d (+ -tests),
VAL-P with MAIN_BUILD build-main-0829689c52: all four PASS (15:52). Rerun on the real hashes before submitting (rule 2).

## Placeholders to fill (all four files)
- `B=build-FREEZE-HASH; SRC=$R/overlays/build-src-FREEZE-HASH` (+ `# TODO hash`): the freeze wheel and its snapshot.
- VAL-P `MB=build-main-REBASE-TARGET` (+ `# TODO main`): main's wheel at the freeze build's rebase target, built with
  verify_wheel (not 0829689c52, which is the v5 merge-base; BUILD-MAIN 8168840 is for the dry run only).
- launch/val_h_cmd.txt is now a stub that exits 1 (VAL-H split into H1/H2).

## VAL-H1 (launch/val_h1_cmd.txt): healthy no-EPLB at 32/GPU, ~97 min/group
- Tuning at 128 of std-noeplb + std-noeplb-ft on this build (C14, PIN_CHECK), then 2 rotated rounds of off | ft |
  ft-noslottable (std-noeplb-ft-noslottable: route() instead of the slot table; proof line fixed 15:35); group 8 reversed.
  n=2 per arm per group, n=4 in total. Readout: trim_readout (gap ft - off vs CLAIMS 173 floor; ft - noslottable),
  stall, client tok/s; validity: off G0 vs G8 <= 0.1 ms, cache_miss equal. H5 tail dropped (no room).
## VAL-H2 (launch/val_h2_cmd.txt): online EPLB C21/C22, ~101 min/group
- std-online | std-online-ft at 32/GPU in both groups (monitor 16:45), tune on this build, 2 rotated rounds: n=4/arm (both loads in
  one group need ~115 min of cases). Seed launch/std/std-seed2.json (no room for 2 online tunes, ~26 min).
- Readout: mvp/analysis/ctx_stall_cost.py (context steps, stall band / trains, whole window) + tools/ctx_classify.py (refill/backlog/decode, CLAIMS 170 metric) + trim_readout
  --off std-online-1. FT - off within the round-to-round off spread closes C21 (DECISIONS 12:09).
## VAL-K (launch/val_k_cmd.txt): kill rows, ~76 / ~75 min (balanced by KILL-SMOKE 8167545 case times)
- G0: rank5, node1, rank0, rank5pf, deadline (C10), then rank5 at 512/GPU. G8: twokill, stop (8 waves,
  KF_STOP_ITERATIONS), SHUTDOWN-R1 (ft-clean / ft-loss / noft-clean), rank5upd on std-online.yaml.
- Doc launch (no pml/btl flags). Verdict: KILL_FINAL_VERDICT incl. H23 (resume <= 10 s, stop timed from the hang
  confirmation with h23_confirm_s apart; tput >= 0.8 x live/32 or N/A for an empty after-window; >= 20 completed after).
## VAL-P (launch/val_p_cmd.txt): ~92 / ~94 min
- G0 C30: std-noeplb-main (MAIN_BUILD) | std-noeplb, same nodes, 32+128 per case, merged pin, 2 rounds ABBA.
- G8 C17 EXACT-FINAL (ref, perturbed control, rank-5 kill; doc launch), then H26 std-noeplb | std-noeplb-ulfm at 32, ABBA.

## Run order (2 concurrent 16-node jobs)
1. Freeze build + verify_wheel, and MAIN_BUILD at the rebase target; fill placeholders; dry run all four; presubmit.
2. VAL-K + VAL-H1 together (correctness first; H1 gives the healthy verdict and its gate on the wheel).
3. VAL-P + VAL-H2 when slots free. Any code issue -> refreeze and rerun the set (code-freeze barrier).

## Harness changes for this redraft (job.sh via temp + mv, bash -n OK)
- job.sh KILL-FINAL stop row: `--iterations ${KF_STOP_ITERATIONS:-8}` (was 3): STOP-RERUN 8168466 stop-1's load ended
  ~4 s after the resume (r0 steps after = 0); 8 waves leave ~2 min of steps after it (+~2 min per stop case).
- val_verdicts.h23_checks: tok_s_after None -> h23_tput_ratio=N/A, not a fail (resume / completed still checked);
  t_h23.py updated, t_h23 + t_verdicts PASS; replay of 8168466 stop-1 -> PASS (resume 3.43 s, confirm 33.65 s, tput N/A).
C30 readout caveat (monitor 16:15): main@9cc37929e8 vs our FT-off includes the intended ADP gate fix (PX, GUARD2 CLAIMS 168: +0.3-0.4% client tok/s at 128/512) and P0c stream error events; state both in the readout. If the parity gap exceeds the threshold, attribute it with one extra arm (our FT-off with the gate fix off) if a knob exists, else skip.

# Finals schedule (FINALS_PLAN v4; drafted 2026-10-22 night; nothing submitted)

26 launch files + 1 smoke driver in `launch/`, generated from one spec (estimates are sums of measured pieces, listed
in each file's header). All carry `B=build-FREEZE-HASH; SRC=$R/overlays/build-src-FREEZE-HASH` + `# TODO hash`, and
`KS=200` + `# TODO N` (fixed kill trigger, FREEZE_CHECKLIST H4 open). P files add `MB=build-main-9cc37929e8`.
Every file: 16 nodes, two 8-node groups, segment 8, limit 7000 s per group, EXCLUDED_NODE guard, in-job unit gate
(VAL-K's 9 files) first in each group. `SMOKE=1` runs the same arms, loads, rows, knobs and readouts with short lengths;
`FIN_GROUP=0|8 [FIN_PART=n]` runs one group (part) inside a caller's 8-node group (used by `fin_smoke_cmd.txt`).

## Design under the 1.9 h cap (how v4 maps to groups)
- Main healthy set (H, figs 1-2): each mode x {32, 128} in one sweep per arm, 600 s windows. A group's own tuning pass
  (12-37 min) + one off/on pair (58-68 min) fills a group, so **8 pairs = 8 groups x 1 pair** (AB / BA alternating),
  not 4 groups x ABBA x 2 (that needs ~140-180 min per group). Every mode runs on 8 groups and in both G0 and G8.
- HX appendix (8 + 512, 300 s): 4 groups x 1 pair per mode.
- P (figs 3-4): no-EPLB groups run off + static-LB floor (+ main in 4 of 8); online groups run off + main. Same
  method as H (32 + 128, 600 s, same tuning). Floor and main read the group's std-noeplb / std-online table.
- R (figs 5-8): no separate R jobs; kill rows fill the slack of H / HX / P / A+D groups. Every row x mode = 4 cases.
- One tactic table per mode **per group** (job.sh's native tune -> merge -> pin within a run). A single table per mode
  across groups/jobs needs FT_COST_E2E_SEED (exists), i.e. a seed: not used (brief: never a pinned seed).

## Jobs (est. min per group = gate + tuning + cases + kill fills + ~2 per kill invocation + 1)
| File | Block | G0: content, est. min | G8: content, est. min | Kill rows carried (mode:row x n) |
|---|---|---|---|---|
| launch/fin_h1_cmd.txt | H (+ R fill, C) | no-EPLB 1 pair, 113 | online 1 pair, 110 | no:rank5x1, no:node1x1 |
| launch/fin_h2_cmd.txt | H (+ R fill, C) | online 1 pair, 110 | offline 1 pair, 109 | no:stopx1 |
| launch/fin_h3_cmd.txt | H (+ R fill, C) | offline 1 pair, 109 | no-EPLB 1 pair, 113 | no:stopx1, no:rank5x1, no:node1x1 |
| launch/fin_h4_cmd.txt | H (+ R fill, C) | online 1 pair, 110 | no-EPLB 1 pair, 113 | no:rank5x1, no:node1x1 |
| launch/fin_h5_cmd.txt | H (+ R fill, C) | offline 1 pair, 109 | online 1 pair, 110 | no:stopx1 |
| launch/fin_h6_cmd.txt | H (+ R fill, C) | no-EPLB 1 pair, 113 | offline 1 pair, 109 | no:rank5x1, no:node1x1, no:stopx1 |
| launch/fin_h7_cmd.txt | H (+ R fill, C) | no-EPLB 1 pair, 113 | online 1 pair, 110 | no:twokillx1, no:rank5pfx1 |
| launch/fin_h8_cmd.txt | H (+ R fill, C) | online 1 pair, 110 | offline 1 pair, 108 | no:deadlinex1 |
| launch/fin_h9_cmd.txt | H (+ R fill, C) | offline 1 pair, 108 | no-EPLB 1 pair, 113 | no:deadlinex1, no:twokillx1, no:rank5pfx1 |
| launch/fin_h10_cmd.txt | H (+ R fill, C) | online 1 pair, 110 | no-EPLB 1 pair, 113 | no:twokillx1, no:rank5pfx1 |
| launch/fin_h11_cmd.txt | H (+ R fill, C) | offline 1 pair, 108 | online 1 pair, 110 | no:deadlinex1 |
| launch/fin_h12_cmd.txt | H (+ R fill, C) | no-EPLB 1 pair, 113 | offline 1 pair, 108 | no:twokillx1, no:rank5pfx1, no:deadlinex1 |
| launch/fin_hx1_cmd.txt | HX (appendix) + R | no-EPLB 8/512 1 pair, 111 | online 8/512 1 pair, 108 | on:rank5x1, on:node1x1, on:deadlinex1, on:stopx1 |
| launch/fin_hx2_cmd.txt | HX (appendix) + R | online 8/512 1 pair, 108 | offline 8/512 1 pair, 106 | on:stopx1, on:twokillx1, on:rank5pfx1 |
| launch/fin_hx3_cmd.txt | HX (appendix) + R | offline 8/512 1 pair, 106 | no-EPLB 8/512 1 pair, 111 | on:twokillx1, on:rank5pfx1, on:rank5x1, on:node1x1, on:deadlinex1 |
| launch/fin_hx4_cmd.txt | HX (appendix) + R | online 8/512 1 pair, 108 | no-EPLB 8/512 1 pair, 111 | on:stopx1, on:rank5x1, on:node1x1, on:deadlinex1 |
| launch/fin_hx5_cmd.txt | HX (appendix) + R | offline 8/512 1 pair, 106 | online 8/512 1 pair, 108 | on:twokillx1, on:rank5pfx1, on:stopx1 |
| launch/fin_hx6_cmd.txt | HX (appendix) + R | no-EPLB 8/512 1 pair, 111 | offline 8/512 1 pair, 106 | on:rank5x1, on:node1x1, on:deadlinex1, on:twokillx1, on:rank5pfx1 |
| launch/fin_p1_cmd.txt | P (+ floor, R fill) | no-EPLB off+floor+main, 105 | online off+main, 105 | on:rank5updx1 |
| launch/fin_p2_cmd.txt | P (+ floor, R fill) | online off+main, 105 | no-EPLB off+floor, 113 | on:rank5updx1, no:rank0x1, no:rank5@32x1, on:rank0x1 |
| launch/fin_p3_cmd.txt | P (+ floor, R fill) | no-EPLB off+floor, 113 | no-EPLB off+floor+main, 105 | no:rank0x1, no:rank5@32x1, on:rank0x1 |
| launch/fin_p4_cmd.txt | P (+ floor, R fill) | online off+main, 105 | no-EPLB off+floor+main, 105 | on:rank5updx1 |
| launch/fin_p5_cmd.txt | P (+ floor, R fill) | no-EPLB off+floor, 113 | online off+main, 105 | no:rank0x1, no:rank5@32x1, on:rank0x1, on:rank5updx1 |
| launch/fin_p6_cmd.txt | P (+ floor, R fill) | no-EPLB off+floor+main, 105 | no-EPLB off+floor, 113 | no:rank0x1, no:rank5@32x1, on:rank0x1 |
| launch/fin_ad1_cmd.txt | A, D (+ R fill) | GSM8K kill x2, nokill x2, ref x2, 97 | restart online x4, 95 | on:rank5@32x2, on:rank5@32x2 |
| launch/fin_ad2_cmd.txt | D | restart no-EPLB x4, 53 | restart offline x4, 56 | - |

n delivered: H 8 pairs per mode at 32 and 128 (600 s); floor (std-noeplb-lb) 8 groups; P main vs our off 4 groups per
mode; HX 4 pairs per mode at 8 and 512 (300 s); R 4 per row per mode (17 row x mode cells, 68 cases); A 2 kill + 2
nokill + 2 ref; D 4 failures + relaunches per mode. Group pairs within 0.3-8.1 min.

## 2-slot schedule (value order; wall = longest group + ~3 min allocation; queue waits not included)
| Start (h) | Slot 1 | Slot 2 |
|---|---|---|
| 0.0 | fin_smoke (61 min) | fin_h1 |
| 1.1 / 1.9 | fin_h2 | fin_h3 |
| 3.0 / 3.9 | fin_h4 | fin_h5 |
| 4.9 / 5.8 | fin_h6 | fin_h7 |
| 6.8 / 7.7 | fin_h8 | fin_h9 |
| 8.7 / 9.6 | fin_h10 | fin_h11 |
| 10.6 / 11.5 | fin_h12 | fin_hx1 |
| 12.6 / 13.4 | fin_hx2 | fin_hx3 |
| 14.4 / 15.3 | fin_hx4 | fin_hx5 |
| 16.3 / 17.2 | fin_hx6 | fin_p1 |
| 18.2 / 19.0 | fin_p2 | fin_p3 |
| 20.2 / 20.9 | fin_p4 | fin_p5 |
| 22.0 / 22.8 | fin_p6 | fin_ad1 |
| 23.9 | fin_ad2 (ends 24.9) | |

Order: smoke, then H (headline: all modes at 32/128; the no-EPLB R rows ride along), then HX (carries most online R
rows, so R completes here, before P), then P, then A + D. **Total ~24.9 h wall** (monitor's rough estimate: ~12 h).
Where the time goes: per-group tuning ~1,240 group-min (~5.2 h wall), 600 s windows at two loads (~60-68 min of cases
per pair), one pair per group. With one cross-job table per mode (FT_COST_E2E_SEED, needs the user's OK) the tuning
goes away: ~19.5-20 h. Today's 20-wave cap would shorten every 32/GPU case by ~6 min (but then the windows miss v4).

## Readout after each job
Run on the login node once both groups end (`<job>` = Slurm id). Figure inputs in brackets.
- Every job: `grep -h "UNITS_GATE\|BUNDLE_TASK first" $R/logs/bundle/<job>-{0,8}.log`.
- H / HX / P (each STD run dir: STD-<job>, STD-G1-<job>):
  - `cat $R/mvp/runs/STD{,-G1}-<job>/tactic_check.txt` (PIN_CHECK, cache_miss equal) and `std_layout.txt`.
  - `python3 -I $R/tools/trim_readout.py $R/mvp/runs/STD-<job> $R/mvp/runs/STD-G1-<job> --off <off-arm>-1`
    [fig 1 primary: decode trim gap, stall ms/step; end-to-end client tok/s/GPU; fig 2 throughput/TPOT/TTFT;
    fig 3 off vs lb vs ft; fig 4 main vs off; HX: appendix fig 2].
  - `python3 -I $R/mvp/analysis/ctx_stall_cost.py $R/mvp/runs/STD-<job> $R/mvp/runs/STD-G1-<job>` [block C: ctx /
    stall band / whole window].
  - `python3 -I $R/tools/ctx_classify.py --load healthy-{32,128} --full-pos 5 --off <run>/<off-arm>-1 --on <run>/<ft-arm>-1`
    [block C refill / backlog / decode classes].
  - Rank-0 logs of each case: KV tokens per rank, ghost-slot memory, /dev/shm host copies, ready_s [block O].
- Kill fills: `grep -h "KILL_FINAL_VERDICT\|KILL_FINAL_SUMMARY" $R/mvp/runs/KILL-FINAL-*-<job>/kill_final.txt`
  [figs 5 timeline (per-step trace), 6 resume_s / h23_confirm_s, 7 tok/s after vs live/total, 8 request fates].
- A: `grep -h "ACCURACY_KILL_" $R/mvp/runs/ACCURACY-KILL-*-<job>/accuracy_kill.txt` [appendix to fig 8].
- D: `cat $R/mvp/runs/COLD-RESTART*-<job>/cold_restart.txt` [fig 9: down_s per mode vs FT resume_s; GPU-hours].
- Every number goes to mvp/CLAIMS.md with run, scale, n and metric before it enters the deck.

## Smoke (launch/fin_smoke_cmd.txt, one 16-node job, ~56 / ~61 min)
G0: fin_h1 G0 part 1 (no-EPLB tuning -> pin, std-noeplb | std-noeplb-ft at 32 + 128). G8: fin_h1 G0 part 2 (no-EPLB
rank5 + node1 with the fixed trigger), fin_ad1 G8 part 2 (online rank5 at 32), fin_ad2 G0 (no-EPLB warm restart).
Then each group's readouts over its run dirs (SMOKE_FAIL readout ... on a failure). Skipped (>60 min per group): online /
offline healthy and tuning, HX, P main / floor, kill rows twokill / stop / rank5pf / rank0 / deadline / rank5upd and the
online 128 rows, D online / offline, GSM8K (no subset knob). A full-path smoke = every fin file with SMOKE=1, ~1 h each.

## Dry runs (tools/dry_run.sh <copy> 16 0 8 on build-freeze-5330b899f0 + build-src-5330b899f0; copies in $TMPDIR)
- PASS with and without SMOKE=1: all fin_h1-h12, fin_hx1-hx6, fin_p1-p6, fin_ad2 (50 runs).
- FAIL fin_ad1 (both modes): all 15 / 9 cases launch; the ACCURACY-KILL summary in job.sh globs
  `$C/mvp/runs/ACCURACY-KILL-*-<job>/` instead of the dry run's scratch dir (FileNotFoundError). Dry-run artifact of
  job.sh; a real run writes that dir. Not fixed (job.sh untouched).
- FAIL fin_smoke: all 9 cases launch; its readouts run on dry-run stubs (no rank-0.log) and report SMOKE_FAIL / "No
  such file", as intended for broken inputs. The readout paths resolve (kill / restart readouts pass).

## Not covered / missing in job.sh (v4 asks, no knob today)
- Steady-window length: `COST_LOAD_ITERATIONS` (waves per load) is the knob used, but STD's sweep hardcodes
  `--iterations 20` (job.sh:2819) and the harness takes min(20, waves): 600 s at 32 (55 waves) and 128 (23), 300 s at
  8 (40) are unreachable today (32: ~218 s, 128: ~520 s, 8: ~153 s). Needs a job.sh knob; no duration knob exists.
- ABBA x 2 within a group (v4: 4 groups x 2 rounds): does not fit 1.9 h with 600 s two-load sweeps; 8 groups x 1 pair.
- One table per mode tuned once across groups: only FT_COST_E2E_SEED (a seed; not used). Within a run: supported.
- Victim rotation (ranks 5/13/22/29, nodes 1/3/5/7): missing; rows hardcode rank 5, node 1, twokill 5 then 9, stop 2.
- Fixed trigger: WIDEEP_KILL_AFTER_STEPS exists for rank5 / node1 / rank0 / deadline / rank5pf / rank5upd and
  COLD-RESTART; missing for twokill and stop. N not chosen (TODO N).
- Static-LB floor on main's kernel: missing (std-noeplb-lb runs our wheel); main parity arm exists (std-*-main).
- Main arm in the same groups as FT on: only in separate P groups (off + main), not next to std-*-ft.
- Online sweeps are separate for 32/128 (H) and 8/512 (HX), as v4's split implies; not one launch over all four loads.
- Identical burn-in: WIDEEP_BURNIN=32:6 (exists) in every healthy arm; recovery and restart cases have no burn-in knob.
- Accuracy on std-online.yaml: missing (ACCURACY-KILL is fixed to dep32-final.yaml); GSM8K subset for the smoke: missing.
- Frozen-rank row at 128: job.sh's stop row runs at 64/GPU.
- D offline has no FT-recovery counterpart (no offline R rows in v4).

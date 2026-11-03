# Red team lane "perf-method": FT healthy-cost finals (R1 NVFP4 DEP32, GB200), read 00:02 PT 2026-10-21
Paths: R = .../.wideep-ft-runs/claude, H = R/mvp/harness, RUNS = R/mvp/runs. Read-only; scratch scripts in $TMPDIR (floor.py, act.py).
Status: no finals dirs yet (no STD-*/STD-G1-* > 8114000). Broker: 8114247 (v2 build) RUNNING, 8114567 PENDING.

## A. Bugs that change the published numbers (fix before any readout)
1. **steady_state.py's "full-load" filter uses num_generation_tokens, which counts MTP draft tokens (2 per request).**
   H/steady_state.py:133 (`tokens >= 0.9*load`) and H/burst_counts.py:222, final_cost_table.py:45 do the same. The log
   shows `num_generation_tokens: 16, num_active = 8`, so the filter really means active >= 0.45 x load. step_readout.py
   and kv_bins.py already use num_active. Measured on the prelim (token floor vs `num_active >= 0.9*load`, rank-0 host step median):
   - STD-8110144 std-offline-ft-1 @512: 1867 steps, 61.89 ms vs 426 steps, 68.71 ms. The HANDOFF's "offline -9.5% at 512
     (queueing)" comes **from this filter**, not from queueing.
   - STD-G1-8110144 std-noeplb-ft-1 @128: 3851 steps, 24.225 vs 3076 steps, 24.526 ms (-1.2% bias, makes FT look faster).
   - At 8 and 32 the effect is <=0.01 ms. At 128 and 512 it lets ramp and tail steps in.
   Fix: in steady_state.steady() and burst_counts, filter on `num_active >= 0.9*load` (num_active is in the same log line;
   extend LINE). Every primary number, stall share and since-ready histogram goes through these functions.
2. **The primary should be the step median, converted after.** steady_tok_s_gpu = median(tokens)/median(step). Its median
   token count moves with the floor, so it is not the same as converting the step cost (prelim noeplb @32: 3.03% vs 3.13%).
   Compute cost on steady_step_ms (host_step_time, as the protocol says) and convert with tok/s/GPU = 2*load/step for display only.
3. **Rank 0 may not reach full load at 512 in an FT arm.** In std-offline-ft-1 @512, rank 0's num_active never reached 512
   (it sat at 400-480, kv_cache_util max 0.53). The FT-off arm sat at 496-512. So a "full-load" step comparison at 512 compares
   different batches. Add a per-row load-attainment line (share of window steps with active >= 0.9 L, per arm) and keep 512 out
   of every test (see B3).
4. **ab_stats.clean() drops the whole row** (step metrics too) when failed > 5% or when any short != failed (ab_stats.py:30-36).
   The protocol says such a row keeps its step metrics. final_cost_table.load_case does the same. std_collect.py:23 drops a row on
   **any** failed request, not >5%. Make all three follow the protocol rule.

## B. Protocol / design issues
1. **The stall-share test cannot pass or fail at n=4.** A one-sided Wilcoxon signed-rank test on 4 group pairs has a minimum
   p of 1/16 = 0.0625 > 0.05. The Hodges-Lehmann CI from 10 Walsh averages has no 95% level either. As written, the 0.5-point
   co-primary cannot be decided. Pre-register a rule that can be decided before data lands, e.g. HL point estimate <= +0.5 points
   AND no group pair > +2 points; report Wilcoxon p as descriptive and state its floor of 0.0625. Optionally pool the 3 headline
   loads per mode (12 pairs, clustered by group), labelled as clustered.
2. **The FT order is unbalanced since F4 was dropped.** FT-first counts over the 4 groups per mode: w/o EPLB 1 of 4
   (F2G1 only); offline 3 of 4 (F1G0, F3G0, F3G1); online 2 of 4. Block position (1st or 2nd pair in a group) is 2/2 for every
   mode, which is good. A second-case effect d (warmer page cache, shm left over, thermal) biases w/o EPLB by -d/2 and offline
   by +d/2. **Cheap fix, if F3 is not yet submitted:** F3 G0 -> `std-offline std-offline-ft std-noeplb-ft std-noeplb` (still
   X-off X-on Y-on Y-off). That makes FT-first 2/2 in every mode. It is a data-independent design change, so log it as an
   amendment before submitting F3. Otherwise, report an order-adjusted estimate as a sensitivity, using online's balanced
   pairs to estimate d.
3. **Cell family / headline mismatch.** Holm is "over the 12 cells", but 512 is "not in the headline". Make the family the 9
   headline cells (3 modes x 8/32/128). Show 512 descriptively with no test. Otherwise "every cell" fails on 512's batch and
   queueing artifacts (A3), which have nothing to do with FT's per-step cost.
4. **Power at G=4 (df=3).** One-sided t(.95,3) = 2.353 gives an upper bound of mean + 1.18*sd. Holm on the worst cell uses
   t(1-.05/9,3) = 5.63, so mean + 2.8*sd (12 cells: 6.23 -> 3.1*sd). With bench_b's group-level sd (0.6 at 8/GPU, 0.15-0.3 at
   128), the IUT passes at 8/GPU only if the true cost is <~0.3%. The Holm per-cell list will mostly read "not established".
   Pre-state that outcome in the write-up wording: "cost X% [one-sided 95% UB]".
5. **The groups are not fully independent.** G0 and G1 of the same job share the time slot and the job-level effect, which
   bench_b found dominant (job x arm SD 0.5-0.7 at 8). Each mode has 2 groups from one job (noeplb: F1; offline: F3; online:
   F2). Add a sensitivity row: average within a job first, then t over 3 jobs (df=2, t=2.92). If it disagrees on pass/fail,
   call the cell "not established".
6. **No slow-mode rule.** "Exclusions: none" is right. But one ~9% slow-mode launch (bench_b section 1) in one of 4 groups
   decides the cell. Report the pre-registered sensitivity without flagged launches (final_cost_table already has the
   >15%-burst flag at 8/GPU). Both must agree.
7. **32/GPU KV bins use the wrong metric.** The protocol compares 32/GPU within KV-fill bins, but kv_bins.py bins
   prev_device_step_time (device), while the primary is host loop-step. Add a `--host` option (host_step_time of the same
   line, no shift), or state that the 32/GPU binned check is device time. kv_bins also handles one pair only, so it needs a
   per-group run + t over groups (C).
8. **PIN_CHECK risk for w/o EPLB.** In the prelim, std-noeplb (FT off) was PIN_CHECK FAIL in both groups (autotune=1216, 38
   entries not in the seed). seed2 is OK on main's wheel (TAIL-CTRL-8112755), but it is unproven on v2 cef4670789. Read F1's
   tactic_check.txt right after its first std-noeplb case ends. If it FAILs, every w/o-EPLB group is invalid and seed2 must be
   extended before F2. cache_miss=192 "fallback tactic" lines appear in every harness arm (0 in stock). They are symmetric, but
   disclose them in the stock-vs-harness note.
9. **itl_stalls is coarse.** Its "ITL" is the chunk gap (stream_interval 10, about 145 ms median), so a single 2x step adds about
   10% to one chunk and never crosses the 2x threshold. It only sees multi-step episodes. Label it so; keep it descriptive.
10. Timestamps: steady_state, step_readout and kv_bins parse the log's local wall time (PT) with the reader's TZ. Run them on PT
    hosts. A UTC container shifts windows by 7 h, which shows up as 0 steps (loud, not silent).

## C. Which tool produces each table/figure (and what is missing)
Finals dirs: RUNS/STD-<F1..F3 id> (G0) and RUNS/STD-G1-<id> (G1); one case per arm per dir (`<arm>-1`). Group = dir.
| Output | Command | Status |
|---|---|---|
| Per-case steady rows | `python3 H/steady_state.py RUNS/STD-<id>/std-noeplb-ft-1` | after fix A1/A2 |
| PRIMARY: per-cell paired cost, one-sided 95% UB, IUT pass, Holm list, df=2 job sensitivity, slow-mode sensitivity | none correct | **MISSING** (see below) |
| Interim pooled table (two-sided t, group-paired) | `python3 H/draft_cases/final_cost_table.py --ladder std-noeplb,std-noeplb-ft,, --loads 8,32,128,512 RUNS/STD-<a> RUNS/STD-G1-<a> ...` (repeat with offline / online) | works: dirs passed directly pair on (dir, round) = group. Two-sided t(.975), its own "noise/COST" verdict (not the protocol), clean() bug A4, token floor A1 |
| Step mean, p99 | same tool / ab_stats fields steady_mean_step_ms, steady_p99_step_ms | per group only in ab_stats (n=1, nan CI: do not cite the in-job ab_stats-*.txt) |
| Stall share per row | `python3 H/burst_counts.py RUNS/STD-<id> ...` | A1 bug; the HL+Wilcoxon over groups is **MISSING**. burst_ranksum.py is an unpaired Mann-Whitney on burst counts, not share |
| Post-ready stall histogram | `python3 H/burst_counts.py --since-ready 30 <dirs>` (burn-in row included? it reads `healthy*` rows only, so **burn-in is NOT shown**; the 20:50 guard requires it) | fix: include `burnin-*` rows in the --since-ready output |
| 32/GPU KV-binned | `python3 H/kv_bins.py RUNS/STD-<id>/std-X-1 RUNS/STD-<id>/std-X-ft-1 --load 32` per group | device metric (B7); aggregation over groups **MISSING** |
| Device-step mechanism (all ranks) | `python3 H/step_readout.py RUNS/STD-<id> --arms std-noeplb,std-noeplb-ft` | OK (num_active floor) |
| Client tok/s/GPU, TPOT (descriptive, median + min-max) | `python3 R/mvp/analysis/std_collect.py OUT.json <finals dirs only>` then `std_figs.py OUT.json OUTDIR FONTDIR "" ` | std_figs' table_healthy shows an unpaired client-throughput "Throughput %" column that reads as the headline. Retitle it "client, descriptive", or add the primary-step column from the missing tool. Do not pass prelim dirs (8110144) or PIN-FAIL cases (std_collect has no exclusion) |
| Client stall descriptive | `python3 H/itl_stalls.py --row healthy-32 <dirs>` (repeat per row) | B9 caveat |
| Load attainment at 512 | none | **MISSING** (A3) |
| Stock vs harness | itl_stalls / std_collect on TAIL-CTRL(-B)-8112755 | stock has no rank logs: client only |
| Kills / accuracy | kill_collect.py, kf_verdict in job.sh; accuracy_diff.py | not reviewed here (redteam2/kills.md) |
Missing tool spec (`finals_readout.py`, about 120 lines, reuses steady_state.iterations). Input: finals run dirs. Exclusions:
PIN_CHECK FAIL from tactic_check.txt (reported, not used). Per (mode, load, group): host-step median with the num_active
floor, cost% = (on-off)/off. Per cell: n, mean, sd, one-sided UB with t(.95,n-1), IUT pass over the 9 headline cells, Holm
adjusted p for H0 cost >= 1%. Job-averaged df=2 row; slow-mode-dropped row; stall share per pair (HL, max pair, Wilcoxon p
with its floor); KV-binned 32 per group (host) + t; load attainment per row. Print a CLAIMS-ready line per cell.

## D. Checklist for finals as they land
- tactic_check.txt: every case PIN_CHECK OK, autotune=0 (prelim noeplb FT-off had autotune=1216).
- std_layout.txt: noeplb-ft 320 slots with 2 ghost/rank; offline-ft 320 = 9+1; online-ft the same; FT-off 256/288.
- result.json: rows burnin-32, warm-healthy-L/healthy-L for 8/32/128/512; healthy windows >= 150 s; failed/short counts.
- round-1.go = 1 (no FT_COST_E2E_STOP); 4 cases per group; group end times within minutes of each other.
- Build check: wheel = build-v2-cef4670789, no overlay (preflight SMOKE_FAIL otherwise).
- Per row: steady steps with num_active >= 0.9L, compared to the token floor (floor.py). Flag any row where they differ > 0.2%.
- Stall share > 15% at 8/GPU = slow-mode flag; quarter-window drift at 32/GPU > 2% -> use KV-binned.

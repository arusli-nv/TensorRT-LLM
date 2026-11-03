# Red team B: measuring FT's healthy-path cost (statistics and protocol)

Data: 573 healthy rows from 44 runs (GHOST-COST*, LADDER*, TAIL-CTRL*, STD-8110144, STD-G1-8110144), re-read from
rank-0 logs. Steady step means decode-only, num_active >= 0.9 x load. "dmed" = rank-0 device-step median;
"hmed" = rank-0 wall loop-step (host_step_time) median; "stall share" = time in steps > 2x median, as a % of
steady time (the burst_counts definition). Extraction script and output: /tmp/claude-161238/tmp.N7lEgvhtV6/{ext.py,ana.py,ana2.py,all.jsonl}.
Cost is always FT on vs FT off. Every number below is n=1 to n=3 per cell unless marked; none has a CLAIMS row yet.

## 1. Noise model: three nested levels, and the outer one dominates

Within-job, same-arm case-to-case CV (%, median over arm groups; clean = every rep has stall share < 5%):

| load | client tok/s | TPOT med | TPOT mean | hmed | dmed | groups |
|---|---|---|---|---|---|---|
| 8 clean | 0.29 | 0.27 | 0.19 | 0.03 | 0.01 | 18 |
| 8 with stalls | 13.6 (max 39) | 13.5 | 14.1 | 1.37 | 0.32 (max 6.9) | 26 |
| 32 clean | 0.67 | 0.87 | 0.33 | 0.22 | 0.03 | 36 |
| 32 with stalls | 3.3 | 4.3 | 4.0 | 0.53 | 0.09 | 8 |
| 128 | 0.56 (max 4.6) | 0.82 | 0.56 | 0.41 | 0.06 | 43 |
| 512 | 2.0 | 2.0 | 1.9 | 0.42 | 0.66 | 3 |

- **Between node groups** (same arm, same job, group A vs B): dmed differs by 0.1-1.3% (LADDER-8092287 noft-main
  +0.95..+1.32%, LADDER-8106373 +1.0..+1.2%). STD-8110144 vs STD-G1 std-noeplb-ft-1: dmed 0.00 / +0.08 / -0.72 / +0.07%,
  client tok/s -0.6 / +0.6 / +2.5 / +1.0% at 8/32/128/512. The node group is a fixed effect about as large as the
  effect being measured, so each arm comparison must stay inside one node group.
- **Between jobs: FT's cost itself moves (this is the key finding).** Same build and arms in GHOST-COST 8087786 / 8087790 /
  8089166 (3 reps per arm per job). dmed cost at 8/GPU: +1.95 [1.93 1.97 1.96], +0.55 [0.54 0.55 0.57],
  +1.11 [1.09 1.11 1.12]. Reps inside a job agree to about 0.02 points, but jobs differ by 1.4 points. At 32: +0.87 / +1.58 / +1.00;
  at 128: +1.05 / +1.06 / +0.99. A/A in the same jobs: |dmed| <= 0.2. The job x arm interaction has an SD
  (sigma_int) of about 0.5-0.7 points at 8, 0.3-0.4 at 32 and about 0.15 at 128, which is 10-30x the within-job rep SD. A
  candidate cause is per-job tactics or slot layout. The pinned/transplanted table may remove it; that is untested.
  **Consequence:** the existing ab_stats CIs (Student t over rounds inside one job) are too narrow and support no
  claim beyond that job. The unit of replication is the node group / job, not the case.
- **Slow-mode launches:** occasionally a whole case runs about 9% slower by dmed (LADDER-8102532 noft-2 at 8:
  15.00 vs 13.75 ms; LADDER noft-aa at 8 in 8087790/8089166: about +9.5% in both reps; LADDER-8092287 B noft-aa -8.6%). These
  launches go with heavy stall share (25-53%). They are not stall steps that hide behind the median: the whole step distribution shifts.
- **Stall episodes** (stall share per case, all 117/117/115/16 cases): 8/GPU p10/p50/p90/max = 2.7 / 3.7 / 38.8 / 55.8%
  (44% of cases > 5%); 32: 0.5 / 1.7 / 4.9 / 27.9 (10%); 128: 0.1 / 0.7 / 1.6 / 4.7; 512: 0.1 / 0.4 / 1.0 / 5.4. Mean vs median
  step: +3.4% typical at 8 (max +122%), +0.5% at 32 (max +37%), +1.4% at 128 (max +7%). TAIL-CTRL-8108645 at 32: share
  2.2-50% within one arm (ft-w 11% vs 50% across reps; noft-main-w 25% / 14.5%). A floor of about 2-5 single 330-350 ms steps
  per 2048-step window at 8/GPU appears in every arm (about 3% of time; GHOST-COST burst_counts). It belongs to the baseline, not to FT.
  One episode moves client tok/s, TPOT and any mean by 5-40% at 8/32. Medians do not move.
- **Window non-stationarity at 32/GPU (STD config):** quarter-window hmed for STD-8110144 noft goes 13.26 -> 13.58 -> 13.90 -> 13.99
  (+5.5%), and FT goes 14.30 / 14.11 / 13.91 / 14.31. These are different trajectories under the same request schedule (wave/context-length phase).
  This explains the n=1 STD 32/GPU "cost" (dmed +2.2%, hmed +3.2%), which no other job reproduces (other jobs: +0.3-1.6).
  At 8 the quarters are flat (11.88-11.91). At 512 they drift about 1.3%. Burn-in: STD-8109649 std-stats at 128 has a first quarter of 167 ms
  against 24-25 ms afterwards.
- **STD (n=1 per arm, long windows of 10.2k / 7.6k / 3.5k / 1.3k steps):** w/o EPLB dmed -0.17 / +2.18 / +0.56 / +0.23%
  and client tok/s -4.4 (FT faster) / +1.2 / -0.05 / +1.0%. Online dmed +0.62 / -0.02 / +0.27 / +0.63% and client
  -2.9 / -2.5 / +1.8 / -1.4%. Client signs flip at random; dmed agrees with the earlier jobs except at 32.

## 2. Sample size (one-sided non-inferiority = upper side of TOST, margin +1%, alpha 0.05)

Only the upper bound matters for a cost claim, so use the one-sided 95% upper bound (the upper end of a two-sided 90% CI) < +1%.
TOST with ±1% is the same test plus a lower bound that FT never fails. The estimator is the mean over G node groups of the
within-group paired difference, with r reps per arm per group. Variance = sigma_int^2/G + sigma_w^2/(G r).
Half-width h = t(0.95, G-1) x sqrt(that). The claim passes only if true cost + h < 1.

| metric (load) | sigma_int | sigma_w (rep diff) | G for h<=0.5 | G for h<=0.3 | verdict |
|---|---|---|---|---|---|
| dmed / hmed (8) | 0.6 | 0.05-0.3 | 6 | 13 | best SNR; misses stall share |
| dmed / hmed (32, STD windows) | 0.35 + phase noise ~1 | 1 (n=1, unverified) | 6-8 | >12 | fix the window first (section 3) |
| dmed / hmed (128, 512) | 0.15-0.3 | 0.1-0.7 | 3-4 | 5-6 | fine |
| client tok/s, clean (any load) | >=0.5 | 0.8-1.2 | 9-10 (r=2) | 25+ | not powerable overnight |
| client tok/s / mean TPOT (8, 32) | - | 15-40 (stalls) | hundreds | - | descriptive only |

- **Feasibility check:** the latest measured w/o EPLB dmed cost is +1.6% at 8 and +0.6-0.9% at 32-512 (#135; GHOST-COST
  8097555 / 8099447 / 8102532: +0.6-1.9%). With a true cost of 0.8%, h must be <= 0.2, which needs G of about 30. **No overnight n can show "<1%"
  for w/o EPLB if #135 is right.** Pre-register that outcome as "cost measured at X% [CI]", not "not resolved". Online
  (+0.3-0.6% dmed at 8/32/512, +1.4% at 128 on 547, about 0.1% on b183) and offline (no FT-on data yet) can pass with G >= 5 if
  their true cost is <= 0.4%.
- **Best honest primary:** wall loop-step median (hmed) -> steady decode tok/s/GPU. It is as stable as dmed at G level, and it
  includes host-side FT work (watchdog, liveness, overlap loop) that dmed hides. Use dmed only for mechanism.
  Client tok/s and TPOT are reported but are not the test metric: their within-job CV of 0.3-2% clean (and 14% at 8 with
  stalls) plus prefill mixing at 128/512 cannot reach ±1% on this budget.
- **Honesty guard:** the median hides episodes, and FT-only episodes were seen before (GLM 9.7% at 8). So add a **co-primary**:
  stall share FT on minus FT off, with a pre-registered margin (<= +0.5 points; Hodges-Lehmann + Wilcoxon over all cases
  per mode x load) **and** a slow-launch count per arm (Fisher exact). The headline is "<1% per step AND no
  increase in stall share/slow launches". Report step mean / client tok/s as the user-visible total next to it.

## 3. Design

- **Pair inside a node group, and replicate across groups.** Each group runs both arms of a mode. Because sigma_int >> sigma_w,
  **more groups with r=1 beats fewer groups with r=2.** Put 2 modes per group slot (4 cases): X-off, X-on, Y-on, Y-off, and
  mirror it in the next slot (X-on, X-off, Y-off, Y-on). That counterbalances position and drift. Rotate the mode pairs so
  every mode meets every position and appears in both concurrent jobs.
- **Same tactics in both arms:** one pinned table per mode, applied to both arms in every group (the GHOST-COST spread
  is the likely tactics lottery). The self-tuned standard recipe gets one separate descriptive pair per mode, not pooled.
  Record node lists, honor EXCLUDE_NODES, and use the same placement YAML (offline), dataset seed and row schedule in both arms.
- **Warm-up / burn-in:** keep the warm-healthy-L rows. Also drop the first 30 s of each healthy-L window from step metrics
  (pre-registered). Readout: quarter-window medians per case. If any quarter is > 1.5x the others' median, that is a burn-in fail and gets flagged.
- **Window:** STD windows are long enough for 8/128/512 (CV of dmed 0.01-0.06%). For 32, either (a) analyze step time
  matched on context: bin steady steps by kv_cache_util / cached_kv_tokens and compare arms within bins (equal-weight
  average), or (b) run the window to an integer number of request waves with randomized arrival offsets so the
  context mix desynchronizes. (a) costs no GPU time; pre-register it as the 32/GPU primary if the quarters drift by more than 2%.
- **Stalls:** never exclude steps from the primary median (it is already robust). The step mean is reported with all steps, and
  also trimmed of > 2x-median steps (both, labelled). Do not drop or rerun cases with episodes. Slow-mode launches
  (dmed > 1.05x the same-arm median across groups in that mode) stay in the primary. A sensitivity row without them is shown;
  if the two disagree on pass/fail, the cell is "not established".
- **Multiplicity:** the claim "<1% in all 12 mode x load cells" is an intersection-union test. Each cell's one-sided 5%
  test must pass, and no correction is needed. Any per-cell "passes" list or "worst cell" headline needs Holm across 12 (or 99.6%
  simultaneous bounds, Bonferroni), because picking the best or worst cell after the fact is selection.

## 4. Reporting

- Per cell: the paired mean cost in % and µs, the one-sided 95% upper bound and the two-sided 95% CI (t over G groups), G, cases per arm,
  the margin line at +1%, and the pass/fail per IUT. Show sigma_int and sigma_w estimates so the reader sees the group effect.
- Plots: a per-cell dot plot of each group's paired difference (all groups, ordered by job) with the CI bar and a ±1% band;
  a stall-share strip plot per arm; quarter-window traces per case for 32 and 512.
- Anti-cherry-picking: freeze this protocol (metric, filters, 30 s trim, flags, margin, estimator, failed-case rules)
  in a dated file **before** submission. Every case that starts is reported. A startup failure gets one makeup at the
  end of the same slot, and is logged. A row with failed/short requests > 5% is out of client metrics but kept for step
  metrics, and listed. No re-analysis with new filters after the data is seen unless it is labelled exploratory. Add a CLAIMS row per cell
  (run, scale, n, metric).
- Wording: "cost <= X% (95% upper bound)", not "~0%". If a cell's upper bound is > 1%, say so with the point estimate.

## 5. Recommended overnight protocol (2 concurrent 16-node jobs, 2 h each, 4 waves = 16 group slots)

Observed case time on STD-8110144: ready 5-11 min + about 16.5 min of rows (20+169+19+185+30+210+68+297 s) = 22-28 min. So 4 cases
fit per 2 h slot with pinned tables. Self-tuned startup (about 23 min) allows 3.

1. **Arms:** {noeplb, offline, online} x {off, on}, all on one wheel and one pinned table per mode, with the standard config
   (FT on adds only the FT keys and its ghost slots).
2. **Slots:** 16 group slots x 4 cases = 64 cases. Mode pairs rotate (NO+OF, OF+ON, ON+NO) in mirrored ABBA, which gives each mode
   about 10-11 groups with r=1 (G about 10, 10-11 cases per arm). Predicted one-sided h at G=10: hmed about 0.35 (8), 0.2-0.6 (32),
   about 0.15 (128/512). That passes only if the true cost is <= 0.6% at 8 and <= 0.8% at 128/512.
3. **Optional:** reserve the last wave (4 slots) for one descriptive self-tuned pair per mode plus makeups. That leaves 12 slots and G
   of about 8 per mode (h about 0.4 at 8).
4. **Kill rule:** never cancel a slot because of an interim read. Stop early only on startup failure or 0 throughput.
5. **Readout:** run steady_state / step_readout per case, burst_counts --since-ready 30, and itl_stalls (descriptive). The pooled
   analysis runs across all 16 slots, paired by group (not ab_stats inside one job).

Unresolved risks: if w/o EPLB really costs 0.6-1.6% (#135), the protocol will report that cost and will not show "<1%".
sigma_int with pinned tables is unknown, and slots 1-2 measure it (compare A/B-group pair spreads).
Offline FT-on has no data yet, so its first slot is the first evidence.

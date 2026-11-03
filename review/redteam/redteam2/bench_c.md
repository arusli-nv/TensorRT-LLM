# bench_c: red team of the overnight final-benchmark plan (perf view), 2026-10-20 ~21:50 PT

Sources: STD-8110144 / STD-G1-8110144 result.json (ready_s, start_wall_s, elapsed_s), KILL-FINAL-*-8102532 case
dirs (config-0.yaml mtime -> done mtime), launch/{std1,stock,kills_final8}_cmd.txt, harness job.sh/serve.sh/bench.py,
HANDOFF tail, broker job list (8110144 RUNNING, 8111350 KILLS PENDING).

## 0. Measured case durations (what the timeline uses)
| case | ready (startup) | ladder 8/32/128/512 | case wall | source |
|---|---|---|---|---|
| std-noeplb / -ft | 365 / 365 s | ~16.7 min | 23.5 / 23.0 min | STD-8110144 |
| std-offline (-ft est.) | 411 s | ~17 min | ~24.5 (~26) min | STD-8110144 (running) |
| std-online / -ft | 663 / 592 s | ~15.3 min | 27.0 / 25.5 min | STD-G1-8110144 |
| kill, static (dep32 cfg) | 285-309 s | 1 load | 7 min | KILL-FINAL-OFF-A-8102532 |
| kill, online (dep32 cfg) | 565-573 s | 1 load | 11-12 min | KILL-FINAL-ON-A-8102532 |
- 4-case group as in std1: G0 ~98 min, G1 ~101 min (projected end ~22:24). Burn-in 32/GPU >=60 s adds ~1.5 min/case
  -> ~105 min/group: fits a 2 h job with ~10 min margin, no more. A 5th case never fits.
- Kills with std configs: expect ~10 min static / ~14 min online (std ready 6/11 min + load + 45 s FT shutdown),
  not 15-20. 18 cases (3 modes x rank,node x n=3) = ~205 group-min = one 2 h job, only if balanced (below).
- Self-tuned startup (+~11 min/case, plan's 23 vs 12) makes a case 35-38 min -> 3 cases/group.

## 1. Timeline of the plan as stated (2 slots, short qos 2 h, no queue gaps assumed)
| slot | PT | job | content | est |
|---|---|---|---|---|
| A | now-22:25 | STD 8110144 (prelim, rc1) | already running | ends ~22:25 |
| B | ~22:25-23:15 | KILLS 8111350 (rc1+overlay, n=1) | G0 3 static kills / G1 2 online + 1 offline | ~45 min |
| A | 22:40-00:10 | BUILD final (1 node) | wheel + gtests + units + integration | 90 min |
| B | 23:15-00:50 | STOCK (main wheel) | per group 2 stock (~29 min) + 2 harness 32/GPU (~15 min) | ~90 min |
| A | 00:15-02:00 | HEALTHY-F1 | 8 cases (4 pairs) | 105 min |
| B | 00:55-02:40 | HEALTHY-F2 | 8 cases | 105 min |
| A | 02:05-03:50 | HEALTHY-F3 | 8 cases | 105 min |
| B | 02:45-04:45 | KILLS-F n=3 | 18 cases | 110-120 min (tail may be dropped by kf_go) |
| A | 03:55-05:25 | BUILD-ONLY intermediate trees | 1 node | 90 min |
| B | 04:50-06:00 | spare 70 min | kill spill / nothing | |
Verdict: it "fits" only with zero queue waits and zero failed jobs, and it does NOT deliver n=6: 3 jobs x 2 groups x
4 cases = 24 cases = 6 arms x n=4 (4 FT-off/FT-on pairs per mode). n=6 per arm needs 36 cases = 4.5 healthy jobs.
Any one failure (build, config reject, reaper) costs ~2 h and pushes kills or F3 past 06:00. No slack.

## 2. Cut / add / reorder (most trustworthy data by 06:00)
1. CUT the build-only intermediate-tree job from the night. It yields no data and costs a full slot (1 node holds a
   job slot). Intermediate trees: pre-commit + py_compile + CPU unit tests locally; GPU build after 06:00. If the
   final tree's C++ equals rc1's (HANDOFF 21:15: final8 = rc1 + .py/.md only), compile risk of the PR splits is low.
2. ADD HEALTHY-F4 in its place (03:55-05:40) -> 32 cases, 16 pairs, >=5 pairs per mode (n=5/5/6). State n honestly
   (n=5-6 per mode over 4 jobs), not "n=6 per arm".
3. REORDER: final kills second after the wheel (slot B 00:55), healthy F2/F3/F4 after. Kills are the riskier job
   (new std configs + graph-size path); an early failure leaves a second chance; healthy n degrades gracefully.
4. Submit the final build the minute slot A frees (~22:25), not 22:40, if the tree is frozen: 15 min = one case.
   Freeze rule: tree at 22:25 is the tree; restack/lean work that changes product files goes after the night.
5. Pair design per job (balanced groups, every mode on >=2 node groups, FT order alternated -- std1 always runs
   FT-off first within a mode, so FT-on is confounded with time-since-job-start / page-cache / shm aging):
   F1 G0 noeplb,noeplb-ft,online-ft,online | G1 offline-ft,offline,noeplb-ft,noeplb
   F2 G0 online,online-ft,offline-ft,offline | G1 noeplb-ft,noeplb,online,online-ft
   F3 G0 offline,offline-ft,noeplb-ft,noeplb | G1 online-ft,online,offline,offline-ft ; F4 = F1 with groups swapped.
   Group estimates 101-106 min, ends within ~6 min of each other (reaper).
6. Kills job balance: G0 = noeplb rank5 x3 + node1 x3 + offline rank5 x2 + offline node1 x2 (~102 min);
   G1 = online rank5 x3 + node1 x3 + offline rank5 x1 + node1 x1 (~104 min). Use the same std-offline placement
   as the healthy arms (checked: std-offline-ft-filled.yaml == STD-8110144 placement, 59 layers, identical -- keep it so).
7. Headline readout = rank-0 full-step device median per load (iteration log), paired within group. Client throughput
   cannot carry a <1% claim: n=1 pairs in STD-8110144 already swing -4.4%..+1.8% (noeplb 8/GPU FT-on +4.4% FASTER,
   online 128 FT-on -1.8%). Report client tok/s/GPU, tok/s/user, TTFT with CIs as secondary.
8. Fold the 38 shapes std-noeplb tuned in-job (TACTIC_CHECK std-noeplb-1: autotune=1216, PIN_CHECK FAIL, 38 entries
   not in the pinned cache) into launch/std/std-seed.json before F1, so every job runs identical tactics; then require
   PIN_CHECK OK on every case.
9. Drop TLLM_TEST_POPULATE=timed (std1 has it) and any other TLLM_TEST_* diagnostic from the finals env unless the
   populate fix is in the wheel and its timing is the agreed disclosure; use burst_counts --since-ready instead.
10. Disclose (or fix later): std-noeplb-ft.yaml is NOT the FT-off config plus FT -- it adds load_balancer num_slots 256,
    layer_updates_per_iter 0. "FT on vs off, identical config" is false for w/o EPLB; the cost includes the static LB
    object. A one-pair control (FT-off with the 256 static LB) would separate it; only if F4 has room.
11. STOCK job stays in the build window (main's wheel needs no final tree) -- best possible filler. Note its stock vs
    harness diff confounds launch hooks with tactic selection (self-tuned vs pinned); say so in the readout.

## 3. Should the finals run entirely through the stock path? No.
- FT cannot be enabled "via config" in a stock launch today: FT on needs ULFM mpirun, the separate authority process
  with key/endpoint (serve.sh starts it), and TLLM_FAULT_TOLERANCE_CONFIG (a file pointed to by env). A "stock FT"
  arm is a custom launcher anyway; FT-off uses plain mpirun. That launcher difference is real deployment cost and is
  already in the harness A/B.
- Lost readouts: print_iter_log (stock drops it; recipe default false) -> no step medians, no num_active filter, no
  stall histogram, no post-kill rank-0 rate; TLLM_LOG_LEVEL info -> no [startup] stage timings and likely no FT phase
  timings; testhooks -> no kill triggers at all (kills impossible stock), no backlog 16384 (512/GPU = 16384 concurrent
  connections may fail stock). What remains is client throughput, which (item 2.7) cannot resolve <1%.
- Self-tuning: +~11 min/case -> 3 cases/group -> 18 healthy cases in 3 jobs = n=3 per arm instead of 4-6, AND each
  launch picks its own tactics, adding a per-launch kernel lottery to the FT-off/FT-on difference. Strictly worse data.
- What "standard TRT-LLM" honestly means here: the wheel unchanged (FINAL_BUILD: no markers, no overlay, PKG_CHECK
  bad=0), standard recipe configs, standard env; harness adds only observation (iteration log, info log, samplers) and
  pinned tactics. Back it with: (a) STOCK job (main wheel) showing harness ~= stock at 32/GPU client-side, and (b) if any
  slot is left, one stock FT-off + stock FT-on pair on the final wheel at 32/GPU (client only) as a sanity anchor, not
  the headline. MIG-AB 8108645: the nolog arm (iteration log off) had the same medians (15.1-15.6 ms in all arms) -- cite it as the log-overhead check.

## 4. Slot-wasting risks and cheap pre-checks
| risk | seen before | pre-check (minutes, CPU) |
|---|---|---|
| Config rejected at startup (graph sizes, LB fields, new validators in final wheel) | 8110820 21:20 | In the build job after the wheel installs: render each arm's config-N.yaml with serve.sh's rewrite and construct TorchLlmArgs from it (all 6 std + kill variants incl. KF_GRAPH_SIZES). Plus check every job state 5 min after start. |
| Wheel != tree | overlays all day | dist dir name == FINAL_BUILD; wheel's embedded commit == PR stack top sha; launch lines carry no MVP_OVERLAY / KF_ALLOW_OVERLAY; first case: PKG_CHECK bad=0 extra=0, no OVERLAY/MARKERS line in timeline. |
| Build fails (compile, quota) -> whole night idle | 8108177, 8108612 | git diff --stat rc1..final -- cpp/ setup.py requirements* empty (else compile risk); home usage < 8 GB; build script exports HOME under /tmp; pre-commit + CPU units green locally before submit. |
| Group idles -> reaper cancels the job | 8107485 (19:10), 8109649 (20:40) | per-group sum of measured case times 95-108 min and groups within ~10 min; STD_PLACEMENT set (no STD-STATS); FT_COST_E2E_ROUND_MIN <= group estimate; a startup failure ends the group (not waits). |
| 2 h limit truncates the last (FT-on) case | std groups ~105 min | burn-in counted in the estimate; last case per group = the shorter-startup mode; incremental result rows are kept but a half pair is dropped in the analysis. |
| Burn-in silently absent | std1 cmd lacks WIDEEP_BURNIN | grep the launch line; first result.json has row label burnin-32. |
| Tactics differ across jobs/arms | TACTIC_CHECK std-noeplb-1 FAIL | seed merge (2.8); stop-rule: any PIN_CHECK FAIL after F1 case 1 -> fix seed before F2. |
| Normal qos starves at 16 nodes | MIG-AB 8106124 40 min pending | short qos only; next job prepared + presubmitted before the previous ends; broker, not squeue, for waits. |
| Kill bundle overruns -> tail cells n=2 | kf_go 15 min guard | balance per 2.6; kf_rows files written right after submit; spill cells first in the spare slot. |
| Stock arm never ran yet (23 min startup is a guess) | none | 8111350/STOCK first stock case: watch ready time; COST_CASE_TIMEOUT_S=3000 already; if >30 min, drop to 1 stock arm per group. |
| Main wheel's /dev/shm leftovers (390 GB) after SIGTERM'd stock cases poison the next case | serve.sh note | shm-N.log of the case after a stock case shows cleanup; keep stock and FT cases in separate groups. |
| Prelim 8111350 on rc1+overlay mistaken for final kills | -- | label it SMOKE in CLAIMS; finals only from the final-wheel kill job. |

Bottom line: plan as stated gives n=4 (not 6) healthy pairs per mode with zero slack. Drop the build-only job, add a
4th healthy job, run final kills right after the wheel, alternate FT order, merge the tuned shapes into the seed,
and headline step medians from the harness; keep stock as the "standard TRT-LLM" cross-check, not the measurement.

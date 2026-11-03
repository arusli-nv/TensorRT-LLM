# History sweep: FT-on rank-0 slow steps across all runs (log analysis, 2026-10-24 09:26 PDT)

Files:
- `HISTORY_STALL_SWEEP.csv`: one row per served arm. There are 1259 arms from `mvp/runs` and `archive`; 1232 are R1 DEP32.
- `HISTORY_STALL_SWEEP.pairs.csv`: 700 same-run-dir pairs of a FT-on arm with a FT-off arm of the same mode, kill
  cases excluded.
- `HISTORY_STALL_SWEEP.py`: the sweep, `--enrich` and `--pairs`.

The cold columns come from `tools/cold_steps.py` (`first_steps`, first 150 iterations, iteration 1 excluded). Kill arms
are read only up to the first recovery line. The steady columns come from `dist32.case_stats` (the same as
`trim_readout.py`). "Slow step" below means a rank-0 step over 200 ms in that cold window.

How each attribute was found:
- **FT state:** `env-0.txt`, else the authority's settings line, else rank 0's "registering with the WideEP recovery
  authority". The emptied-slot testhook logs "peer-loss recovery keeps N slots empty" with FT off, so that line alone
  does not count as FT on.
- **hang_detection, peer_failure_poll_s, extra_slots:** the authority log's "fault-tolerance settings" line.
- **Host copies:** the peak of the `shared` column in rank-0 node's `hostmem-0.log`.
- **Build:** `MVP_BUILD`, else the install log path, else the wheel sha16 matched against every `build-*/dist` wheel.

## Confounds (read before the table)

- **The cold window changed around 10-20.** Up to about 10-19, the first 150 iterations were the first load, with
  prefill. Every arm then had 3-76 slow steps from the workload alone. From about 10-20, a burn-in row with one
  request runs first, and a clean arm has 0 slow steps. So only same-run pairs are compared, and the pre-10-20 rows
  are coarse.
- **Before about 10-20, `hostmem` shows node `/dev/shm` occupancy, not this arm's copies.** Leftover segments put
  344 GB in eplb0 arms. The `shm` column is a reliable "host copies" proxy only from about 10-20 on.
- **hang_detection is fully confounded with the build.** It is off in every v2/v3/v5 STD arm and on only in the
  freeze builds (after the rebase onto main 9cc).
- **The G0/G1 directories of a job are different node sets.** Pairs are within one directory.
- **n is small.** Some arms are reused in several pairs; the counts below are per FT-on arm.
- **Build trust.** Clean = the CLAIMS wheel-audit list or `build-freeze-*`. Everything else is lower trust.
  `build-main-0829689c52` has no served arm.

## Decisive contrasts

Clean builds, from 10-20 on unless noted. Each row compares FT-on arms with the FT-off arms of the same directory.

| Contrast | n (on-arms) | Result |
|---|---|---|
| Offline, FT on with **hang_detection off** (v2 cef4670789, 365 GB host copies) vs off | 3 | Slow steps 11 / 0 / 60 vs 0 / 0 / 0. rc1 (unaudited): 16 vs 3. |
| Offline, FT on with hang_detection on (freeze 7e8d / 58ca) vs off | 5 | 5/5 worse. Median 21 slow steps (8.2 s) vs 0. |
| Online, FT on **hang off** (v2 / v3) vs off (both about 365 GB host copies) | 20 | 14/20 worse. Median 26 slow steps (11.7 s) vs 0. The `-ft-sm` transport variant still has 41 / 17. |
| Online, FT on hang on (freeze) vs off | 8 | 4/8 worse by more than 1 s. Median 2.9 s vs 0. |
| Offline, FT on with **peer_failure_poll_s 1.0** (KNOB-AB 8240302, in progress) vs off on the same rank-0 host | 1 | 27 slow steps (10.2 s, p90 342 ms) vs 0. |
| No-EPLB, FT on (ring copies 0.6 GiB/layer, about 35-49 GB; watch thread, per-collective checks, ghost VMM slots) vs off | 65 hang off + 14 hang on | Worse in only 1/65 and 1/14. Median 0 vs 0. |
| FT **off** with full host copies (online) vs FT off with none, same directory (v2) | 4 runs | 0 vs 0 slow steps. In the finals, every FT-off online and main arm (8/8) has 0 slow steps; p90 is up to 92 ms in 4 of 8. |
| FT off with the emptied/ghost-slot layout (host copies mapped) vs eplb0, same directory (10-18/19, lower trust, first-load windows) | 12 runs | 2-3x the slow steps in 12/12. FT-off online stalled like FT-on online in this era. |
| Builds up to 937403a6cc (10-15..10-17): offline FT **without** host copies, watch thread polling every 5 ms | 30 matched pairs | Equal counts in 23. FT on was higher in 2 and lower in 5 (workload-dominated windows, lower trust). |
| Online extra slots, hang off (all builds, lower trust): ghost / empty / borrowed | 64 / 8 / 6 | Stalls in every kind: worse in 24/64 / 6/8 / 4/6. No-EPLB ghost (VMM) is clean (row 6). |

**Steady windows.** This is the median on-minus-off stall in ms/step at 32/GPU, clean builds.

| Mode | hang off | hang on |
|---|---|---|
| No-EPLB | +0.17 (n=58) | +0.00 (n=14) |
| Offline | +0.03 (n=3) | +0.16 (n=5) |
| Online | +0.04 (n=11) | +0.28 (n=6) |

At 128/GPU every cell is within ±0.15. The online excess is visible only on the freeze builds, so it is confounded
with the build and the finals harness.

## Verdict per suspect

- **(a) Hang detection: REFUTED as the cause of the cold stalls.** Offline and online FT arms with hang_detection
  off (v2/v3) stall as the finals do, and no-EPLB FT with it on is clean. For the steady stall it is not testable:
  the build is a confound. KNOB-AB's `-nohang` arm is the direct test.
- **(b) PeerFailureWatch / per-collective checks: REFUTED as sufficient.** They are present in every FT arm,
  including the clean no-EPLB ones and the 10-15..17 arms that polled every 5 ms without host copies. A 1 s poll
  still stalls (n=1). The per-collective check alone is not testable.
- **(c) Host copies: SUPPORTED as necessary, not sufficient.** Large slow steps appear only where FT on meets about
  365 GB per node of host copies. Ring copies (≤49 GB) are clean. FT off with the same 365 GB (online, main) is
  clean in today's cold window. So this is an **FT × full-host-copies interaction**. An FT-only code path that
  touches the copies is `_start_populate` (MADV_POPULATE_READ prefault of the imported peer segments, FT only,
  `moe_load_balancer.py:411`). It is joined before ready and takes 0.0-0.1 s per layer (WIDEEP_POPULATE logs), so
  any effect comes from the page tables it leaves behind. That is a hypothesis, not tested.
- **(d) Ghost / VMM slots: REFUTED as necessary.** The empty and borrowed kinds stall too, and no-EPLB ghost is
  clean.
- **When it started.** Offline FT got host copies with `host_copies=updates_enabled or peer_loss_recovery_enabled()`.
  It is first present at 7684541f66 (10-18) and absent at 937403a6cc (10-17). The first clean-window offline-FT
  arm already stalls (rc1 10-20, v2 10-21). Online FT has stalled in every build with a usable window.

## Cheapest remaining A/B

One job on the same nodes, offline, at 32/128 plus the cold window:
1. `std-offline-ft`
2. `std-offline-ft` with `TLLM_TEST_POPULATE=off` (an existing testhook; no rebuild)
3. `std-offline` (FT off) with about 365 GB of `/dev/shm` filled and prefaulted by 4 processes
4. Optionally FT off with the copies mapped but not populated

Arm 2 isolates the FT-only prefault. Arm 3 isolates plain occupancy versus the mapping. Use
`WIDEEP_NUMA_SAMPLER=1` in diagnostic arms only. KNOB-AB's `-nohang` and `-nohang-poll1` arms already cover (a)/(b)
when they finish.

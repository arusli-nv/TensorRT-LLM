# Cold-phase slow steps: what the existing logs show (read-only analysis, 2026-10-24)

R = `.wideep-ft-runs/claude/mvp/runs`. Main pair: `R/STD-8206863/std-offline-ft-1` (FT on, `mode=ft`) vs
`R/STD-8206863/std-offline-1` (FT off, `mode=noft`). Both use build-freeze-58caa0fde6 on the same 8 nodes.
Logs available per arm: `rank-*.log` (only rank 0 has iteration lines), `vmstat-<node>.log` (1 s), `dmon-<node>.log`
(about 1-2 s per GPU: sm, power, clocks, pviol/tviol), `hostmem-<node>.log` (one `df /dev/shm` + `free` snapshot at
ready), `dropcache-*.log`, `timeline`. There is no /proc/vmstat, per-NUMA meminfo, PSI, or per-rank iteration log.

## 1. The slow steps

Rank 0's first 150 serving iterations (one request, `num_scheduled_requests=1`, generation only):
- FT on: 19:33:20-19:33:36. 39 steps took more than 50 ms, and they add up to **15.3 s of the 16.5 s window**.
  Without the stalls, a step is 10.8-11.5 ms.
- FT off: 19:51:56-19:51:58. No step took more than 50 ms, and the whole window lasted 1.4 s.
- For each stall, the next iteration's `prev_device_step_time` matches the stall's `host_step_time`. Example: iter 133
  has host 955.8 ms, and iter 134 has prev_device 955.77 ms. So the device step is as long as the host step.
- The stalls are **not periodic**. Gaps between stall onsets are 0.10-0.96 s and irregular (see the onset list below).
  They come in clusters of 1-3 slow steps, separated by runs of normal 11 ms steps.
- The stalls **die out after about 90 s**. Counting FT-on stalls of generation-only steps over 100 ms in 10 s bins from
  the first serving iteration: 8.8, 9.1, 8.5, 5.3, 4.6, 5.4, 3.4, 5.1, 2.2, 0.5, 0.3, 0.0 s stalled per bin. FT off is
  0.0-0.5 s per bin throughout. After t+100 s the per-minute counts of slow steps are similar in the two arms (ramp
  steps under load).

Stall onsets in the FT-on arm (seconds into the window, then ms): 0.59/588, 1.38/793, 1.93/549, 2.47/539, 2.98/510,
3.44/259, 3.86/416, 4.22/361, 4.68/461, 5.42/736, 6.09/620, 6.95/599, 7.49/529, 8.10/283, 8.58/301, 9.06/341,
9.72/660, 10.27/441, 11.05/749, 11.63/587, 12.77/281, 13.06/249, 14.02/956, 14.50/451, 14.91/401, 15.45/519,
16.28/603.

## 2. Cross-arm correlation (all STD arms in `review/cold_steps_finals.txt`)

`/dev/shm` usage and free memory come from `hostmem-*.log`.

| /dev/shm per node | arms | arms with cold p90 > 15 ms |
|---|---|---|
| 0 GB (FT-off offline and no-EPLB, including the `noft-ulfm` ob1/TCP arms) | 18 | **0** |
| 49 GB (`std-noeplb-ft`) | 4 | 2 (G1-8202865: 59.6 ms; G1-8208527: 16.7 ms) |
| 390 GB, FT on (`*-ft`) | 8 | 5 (461, 248, 208, 141.7, 18.7 ms) |
| 390 GB, FT off (online EPLB: `std-online-1`, `std-online-main-1`) | 8 | 4 (G1-8206860 online-1: 91.7 ms; 8206860 online-main: 64.0 ms; G2-8202865 online-main: 49.9 ms; G1-8206860 online-main: 25.4 ms) |

**FT off shows the same stalls.** For example, `STD-G1-8206860/std-online-1` (FT off, standard transport `mca=[]`)
had 45 steps over 50 ms, which add up to 3.8 s of its 5.8 s window.

Across the arms, the stalls go with **a populated /dev/shm (EPLB host copies)**, not with FT. FT off is clean in
offline only because FT-off offline has no host copies (0 GB) and FT-on offline has 390 GB.

Two caveats limit this conclusion:
- The online FT-off arms also run `layer_updates_per_iter=1`, which reads host weights every iteration. That is a
  partial confound.
- The control "FT off over ob1/TCP is clean" (`std-noeplb-ulfm-1`) was run with 0 GB of shm. No arm has ob1/TCP,
  390 GB of shm and FT off, so transport × shm has not been separated.

## 3. Hypotheses

**H2: host memory pressure. Classic pressure is refuted; a shm-linked host effect is supported by correlation only.**
- Evidence against swap or reclaim pressure. In `std-offline-ft-1/vmstat-{0..7}.log` over 19:33:20-37, the
  per-node minimum free memory is 362-424 GB (FT off: 758-832 GB). swpd, si and so are 0 on every node, there is no
  swap device, `b` ≤ 5, `wa` ≤ 1, and `bi` ≤ 1252 blocks/s.
- `hostmem-*.log` (FT on): `/dev/shm` holds 390.4 GB of 480 GB (76%) on every node. "available" is 389-407 GB.
- The page-cache drop (`dropcache-0.log`: 385 GiB, 3.1 s) ended at about 19:33:10, which is 10 s before the first
  request. FT off has the same step.
- One weak signal. During the stall period, mean `sy` over the 8 nodes is 2.3-2.5% and decays to 1.0% by t+100 s,
  the same time the stalls stop. Node 6 is highest at 2.7-3.2%. FT off is 1.0-1.3% in the same bins. However, FT off
  also reaches `sy` 2.0-2.3% in its idle gap (t+70-90 s), so this may only reflect a waiting executor.
- Compaction, THP activity and per-NUMA-node fill **cannot be tested**: there is no /proc/vmstat, per-NUMA meminfo
  or PSI. `sysinfo-0.txt` shows THP=madvise, defrag=madvise and numa_balancing=0. A 390 GB tmpfs on a node with two
  ~480 GB Grace NUMA nodes could nearly fill one NUMA node even though system-wide free memory is 400 GB. That is the
  leading untested mechanism.

**H3: FT background threads. Refuted as the primary cause.**
- The stalls are not at a 1 s period or a 20 ms period (onset gaps are 0.10-0.96 s and irregular).
- No FT event is logged on any rank between 19:33 and 19:35. `grep` of `rank-*.log` finds only the warm-up/capture
  lines, and the last line of `authority.log` is SERVING at 19:31:30.
- The same stall pattern appears in FT-off arms that have shm (section 2).
- A small FT contribution cannot be excluded. The idle p50 is 11.2 vs 9.7 ms in this pair, but no-EPLB FT on vs off
  is 9.4 vs 9.3 ms, which points to the shm/EPLB arms here too.

**Which rank is slow: cannot be tested.**
- Only rank 0 prints iterations. `rank-1..31.log` have 0 `iter =` lines.
- Indirect evidence from dmon over the stall window (252 GPU samples across all 32 GPUs): mean SM is 23%, and **64%
  of samples are 0%**. FT off: mean 95%, 0% of samples at 0%.
- A rank waiting inside the NVLinkOneSided all-to-all spins at 100% SM. So most GPUs are not even launched during a
  stall. All ranks are blocked on the host (a host-side per-iteration collective, or all hosts being slow together),
  not waiting on one slow GPU. dmon samples are not aligned across nodes, so no single node can be singled out.

**GPU side: refuted.**
- `dmon-{0..7}.log` over 19:33:20-37: pclk stays at 2062 MHz on all 32 GPUs, mclk at 3996 MHz, pviol = 0, tviol = 0,
  and gtemp ≤ 30 C.
- The GPUs are idle (low power, 224-258 W mean), not throttled.

## 4. Missing evidence and the cheapest decisive A/B

Missing evidence:
- 1 s samples of `/proc/vmstat` (compact_stall, compact_fail, pgscan_*, thp_*), `/sys/devices/system/node/node*/meminfo`
  and `/proc/pressure/{memory,cpu}`.
- Per-rank iteration logs, or one rank per node, with sub-second timestamps.
- A `py-spy dump` of all ranks on one node during t+0-30 s, to see where the hosts block.

Cheapest decisive A/B, with one variable on the same nodes and the same build: run **FT off offline** (`std-offline`,
which normally has 0 GB shm) twice back to back.
- Arm A: as is.
- Arm B: before launch, a dummy holder process on each node fallocates and populates a 390 GB file in `/dev/shm`. Use
  the same placement as the FT host copies, with no MADV mapping by the ranks.
- Read cold-step p90 and the stalled seconds.

Prediction if shm occupancy is the cause: B stalls (p90 in the hundreds of ms) and A stays at about 9.8 ms. If B is
clean, the cause sits in the FT host-copy code path (the ranks' mappings and populate) or in ob1/TCP + EPLB. The
follow-up would then be FT-on offline with and without host copies. Add the per-NUMA and /proc/vmstat sampling to
both arms so a positive result also names the mechanism.

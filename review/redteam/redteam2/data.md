# Why GHOST-COST-8102532 load-8 shows -7.4% throughput / +11.9% TPOT, while the device-step median is only +1.6%

Scope: read-only analysis of rank-0.log iteration lines, result.json/bench JSON, hostpressure/vmstat/dmon.
Runs: GHOST-COST-8102532 (static layout, no online EPLB; FT off = lay-eplb0-1/2, FT on = ft-ghost320-1/2) and
LADDER-8102532 (online EPLB; noft-1/2 vs ft-1/2). Setup: 32 GPUs (8 nodes x 4), DeepSeek-R1-0528-FP4, ISL/OSL 1024/1024,
each healthy-L row = 2 waves of 32*L requests (iterations=2), n=2 replicas per arm. GHOST has no MTP (1.0 token/iter); LADDER
has MTP (~2 tokens/step). Scripts: $TMPDIR/d5/{parse,rows2,steady,early,bins,slow,hp}.py.

## 1. Answer (load 8)

The whole arm delta comes from one replica, ft-ghost320-2, with a ~10 s stall burst inside its measured window.
ft-ghost320-1 alone is -1.0% throughput and +1.8% TPOT.

| replica (healthy-8) | out tok/s | median TPOT ms | client duration s | sum of step times s | steady dev median / mean / p99 ms | stall s (steps >1.5x median) |
|---|---|---|---|---|---|---|
| lay-eplb0-1 | 21411 | 10.93 | 24.49 | 24.19 | 10.87 / 10.90 / 12.1 | 0.00 |
| lay-eplb0-2 | 20991 | 10.94 | 24.98 | 24.69 | 10.87 / 10.96 / 10.9 | 0.18 |
| ft-ghost320-1 | 21205 | 11.13 | 24.72 | 24.43 | 11.04 / 11.08 / 11.1 | 0.08 |
| ft-ghost320-2 | 18038 | 13.35 | 29.07 | 28.79 | 11.06 / 13.13 / 56.5 | 3.56 |

("Steady" = decode steps with num_active >= 0.9*L and no context requests. The device time of step i is taken from
prev_device_step_time on line i+1.)

Breakdown of ft-ghost320-2's extra +4.3 s over the FT-off mean (24.74 s):
- about +3.6 s: device stall tail (148 steps of 16-170 ms, between 59 s and 69 s after server_ready);
- about +0.4 s: median shift (2043 steps x +0.19 ms, the same +1.6% seen on ft-ghost320-1);
- about +0.2 s: slower context steps;
- about 0 s: host overhead.

Why the metrics disagree:
- The median ignores a 7% tail.
- TPOT per request is (e2e - TTFT)/1023, i.e. the **mean** step time, and that is +10.9% (steady mean device 11.0 -> 13.1 on that replica).
- Throughput = 512*1024 / duration, and duration grew 4.3 s.

## 2. Host vs device; gaps between steps

- For decode steps, sum(host_step_time) - sum(device) is between -1.0 and +0.3 s in every row. With the overlap scheduler, the
  host step time is just the wait on the previous device step. Host overhead is hidden: the loop is device-bound or waits on a peer.
- No time is unaccounted for between log lines. Over every measured window, the log-timestamp span matches the sum of
  host_step_time within the 1 s timestamp resolution. The sum of step times also matches the client duration minus about 0.3 s
  (HTTP/streaming overhead).
- In the stall steps, host is about equal to device (for example 36/36, 58/64, 104/103 ms). Rank 0 spends that time on the device
  inside the step, waiting in the MoE all-to-all for a late peer. This is not a host gap on rank 0. Rank 0 cannot tell which
  peer: the other ranks print no iteration lines.
- Ruled out: GPU clocks (dmon pclk flat at 2062 MHz, no power or thermal violation), PCIe traffic, and MTP/non-decode steps.
  GHOST has no MTP, and its context steps are the same in both arms.

## 3. The ~16-18 s "step" (present in every row of every case)

- It is iteration "N" with num_ctx=1, 1024 context tokens, num_active=1, logged 12-15 s after the row's start_wall_s.
  - host_step_time covers the executor idling while the benchmark client starts up (tokenizer, dataset). That takes 15-24 s and
    grows with load: about 16 s at load 8, 22-24 s at load 512.
  - The request in that step is the client's **initial single test prompt**, followed by 1023 batch-1 decode steps of about 8.7 ms.
- It is **outside** the throughput window. benchmark_serving starts its timer after the test prompt, and the trailing step-sum
  equals the duration.
- It does poison two things:
  - the next line's prev_device_step_time (the logged value equals the idle time, 15-24 s);
  - any analysis that defines a row by wall clock: the 1023 test-prompt steps at 8.7 ms pull the median down and steps/s up.
    The "steps/s 90 vs 83" figures probably mix these in. Over the measured window, steps/s is about 85 for FT off.
- It is not a wave boundary. Wave boundaries inside a row show up as 2-3 context steps of about 345 ms (load 8). At 128/512 they
  show up as dozens of 0.3-1.5 s context steps.

## 4. The stall burst over time (time since server_ready, 10 s bins, stall seconds)

GHOST (static layout). Every FT-on replica checked has an early stall cluster. FT off almost never does:
- FT on, 17 replicas across jobs 8079803, 8087786, 8087790, 8089166, 8099447 and 8102532: 1.4-10 s of stall in the bins
  10-60 s after ready. The cluster ends about 50-70 s after ready, which is about 170-240 s after "moe_load_balancer finalize
  model done".
- FT off, 14 replicas: at most 0.7 s per bin in that period.
- Where the cluster lands decides the reported cost:
  - warm-8 row (0-23 s) or the healthy-8 test prompt (about 23-45 s): not measured;
  - healthy-8 measured window (about 45-70 s): hit. Examples: 8102532 ft-2 at 60 s, -15%; 8099447 ft-2, 7.2% excess;
    8079803 ft-1, 7.9% excess.
- Historical load-8 excess-time (burst_counts.txt, all GHOST jobs): ft-ghost320 mean 4.3%, 3 of 22 replicas above 5%
  (16.4, 7.9, 7.2). lay-eplb0 mean 3.0%, 0 of 22 above 5%; lay-eplb0-aa 1 of 16 (6.9%).
- At load 32/128 there is no burst (FT on vs off within +-1%).

LADDER (online EPLB). **Both** arms have heavy early stalls:
- noft-2: 7.0 s in the 40 s bin, 5.2 s in the 50 s bin;
- ft-1: 9.7 s in the 20 s bin, 8.2 s in the 60 s bin;
- ft-2: 6.9 s in the 20 s bin;
- noft-1: clean.
So the load-8 sign flips (+9.0% thr for FT on), because noft-2 was the replica that got hit.

**Correlation with host state.**
- /dev/shm EPLB host copies per node:
  - GHOST FT on: 46 GB;
  - GHOST FT off: 0;
  - LADDER, both arms: 390 GB (online EPLB keeps every expert).
- So early stalls appear exactly where host copies exist.
- In LADDER the stall bins line up with D-state bursts summed over all nodes (vmstat "b"):
  - ft-2: 124 D-samples in the 20 s bin;
  - ft-1: 77/51 in the 20/30 s bins;
  - noft-2: 48/40 in the 50/60 s bins;
  - noft-1: 0-4 per bin.
  io-PSI reaches 5-8% in the same windows. All of this is gone after 80 s.
- In GHOST FT on, the signal is weak: 17-20 D-samples vs 7 in the first 80 s, and io-PSI up to 4.4% on node 5 at 60-72 s,
  aligned with the ft-2 burst. CPU us/sy and iowait are flat.
- hostpressure has no migration fields, so migration cannot be checked.
- Prime suspect: the `eplb_host_copy_populate` daemon thread (`moe_load_balancer.py:_populate_in_background`, madvise
  POPULATE_READ over the imported shm segments, about 3/4 of the node's copies per rank). It starts at finalize and runs
  through the first minutes of serving. It does not log when it finishes, so this is circumstantial: it matches the
  timing, the arm split (shm present or absent) and the D-state counts.
- Not supported:
  - ULFM ob1/tcp host collectives: the detector runs all the time, but stalls are only early.
  - Host time between steps: none (see section 2).
  - MTP: GHOST has none.

## 5. Other loads (arm means, FT on vs off; steady = act >= 0.9L, no context steps)

| load | out thr | median TPOT | steady dev median | steady dev mean | steady dev p99 | steady tok/s/rank |
|---|---|---|---|---|---|---|
| GHOST 8 | -7.4% | +11.9% | +1.6% | +10.9% (ft-1 alone +1.6%) | burst | -9.2% (ft-1 alone -1.6%) |
| GHOST 32 | +0.7% | -1.0% | +0.6% | +0.1% | +1.9% | -0.2% |
| GHOST 128 | -1.0% | +1.5% | +0.7% | +0.4% | -3.5% | -0.9% |
| GHOST 512 | -0.6% | -8.2% (*) | +0.9% | +3.5% (ft-1 has a 2.1 s burst, p99 206 ms) | - | -6.2% (ft-1) / -1.0% (ft-2) |
| LADDER 8 | +9.0% | -12.4% | -3.6% (noft-2 burst) | - | - | - |
| LADDER 32 | +3.1% | -4.5% | +0.3% (noft-1 1.45 s burst) | - | - | - |
| LADDER 128 | -2.1% | +1.5% | +1.4% | +1.4% | - | -1.2% |
| LADDER 512 | -3.9% | +5.2% | +1.1% | +2.2% | - | -2.1% |

(*) The 512 TPOT is driven by row composition: ft-ghost320-1 had a TPOT of 59 vs 73-77 for the others at the same throughput,
because of a different prefill/decode interleave across the 2 waves.
- At 128/512, context steps are 37-45% of the row (128: 22-24 s of 63 s; 512: 77-99 s of 205 s). So output throughput and
  TPOT there mostly measure the two prefill waves and the ramp, not decode cost.
- At 8 and 32, context steps are 5% and 19% of the row.

Note: the FT-on arm also differs in MoE layout: num_slots=320 (10 per rank) with a static load balancer, vs no load balancer.
The consistent +0.6-1.6% steady median likely includes that, not only the FT runtime.

## 6. Recommendations for finals

Headline metric set: per load, per replica, then arm mean with the range.
1. **Steady decode device step**: median, mean and p99 over steps with num_active >= 0.9*L and num_ctx=0, using the next line's
   prev_device_step_time. Drop any step that follows an idle gap (host > 2 s). This is the "healthy cost".
2. **Stall time**: sum of (dev - median) over steps > 1.5x median, as % of the window, plus a time-since-ready histogram.
   Report it next to the healthy cost, never folded in silently. Flag any replica above 2%.
3. **Steady tok/s/GPU**: sum(num_active * tokens/step) / sum(host_step_time) over the steady steps. Tokens/step = 1, or the
   client's avg_decoded_tokens_per_iter with MTP.
4. **Client metrics** as end-to-end checks:
   - output tok/s/GPU;
   - tok/s/user (user_throughput);
   - TPOT median **and mean** (TPOT tracks the mean step);
   - TTFT.
   Note that at 128/512 they are dominated by prefill and the ramp.
5. **Wave boundaries/ramp**: keep them out of the steady metrics through the act >= 0.9L filter. Leave them in the client
   throughput, and say so. Do not use wall-clock row windows for step statistics: they include the idle step and the
   1023-step test prompt.

Method:
- **Burn-in until the host-copy populate finishes.** Log the populate thread's completion, and have the harness wait for it,
  or wait at least 120 s after server_ready. Then run an unmeasured burn-in row (for example healthy-32 for about 60 s) before the
  first measured row. Never let load 8 be the first measured row after ready.
- **Row length:** at load 8 one row is about 2 x 12 s, so a 10 s stall costs 15%. Use iterations >= 4, or at least 60 s of steady
  decode per row. Keep 2 waves at 128/512, but take the cost from the steady window.
- **Replicas/order:** at least 3 replicas per arm, with ABAB interleave as now. Report every replica plus the median. A
  burst-hit replica is shown and attributed, not averaged in.
- **Instrumentation to make this decisive:**
  - a log line when `eplb_host_copy_populate` finishes, per rank;
  - per-rank iteration timing (or at least the slowest-rank id per step) to find the late peer;
  - D-state process names (ps -eo stat,comm,wchan) in hostpressure during the first 120 s.

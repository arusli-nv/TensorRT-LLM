# WideEP Fault Tolerance MVP Design Doc

Andrew Rusli | Created: Oct 19, 2026 | Updated: Oct 25, 2026 | Code: v5-final `27442bd91d`

## Overview & Motivation

Today, losing one rank of a wide expert-parallel (WideEP) deployment ends the whole job. Serving resumes only after a
restart (minutes).

With fault tolerance (FT), the surviving ranks resume serving in 0.8-2.5 s on the CUDA graphs captured at startup.
Next phase: bring lost ranks back (N-k to N ranks) while the survivors serve.

Why it matters:

- Failures are more common at large scale, and restarts are costly [1, 2].
- NVIDIA's Dynamo cluster, 18-week trace (~150 replicas, mostly <= 8 GPUs): 46 restarts a day, median 3.5 min each;
  77% were process deaths or hangs with the GPU intact [1].
- WideEP deployments span 32 to 64+ GPUs, so they are even more exposed.

| | Without FT | With FT |
|---|---|---|
| One rank lost | job ends; restart (5-10 min) | serving resumes in 0.8-1.6 s (up to 2.4 s if killed in prefill) |
| One node's ranks lost | job ends; restart | serving resumes in 1.4-2.8 s |
| One rank frozen | job ends after the 300 s hang detector | fenced 30-34 s after the freeze, then recovered |
| Healthy throughput | baseline | output tok/s within noise of FT off; at 32/GPU pooled n=8: -0.7% to +0.3% by mode |

## Goals & Scope

Goals:

- **Performance.** FT costs as little as possible while healthy (target < 1%). Throughput after a loss only needs to be
  reasonable; recovery back to N comes later.
- **Recovery time.** Seconds, on the same CUDA graphs: no recapture, reload or restart.
- **Correctness.** Survivors' requests continue from exactly their tokens. A lost rank's requests run again or end
  with a retryable error.

Supported:

- MoE models (tested with DeepSeek-R1); aggregated serving.
- Attention DP + EP over all ranks (DEP), expert TP 1.
- NVLinkOneSided all-to-all.
- No EPLB, offline EPLB and online EPLB.
- Overlap scheduler; MTP (other speculative decoding is untested).
- KV Cache Manager V2.
- Open MPI with ULFM (`mpirun --with-ft ulfm`), mpi4py >= 4.

Failures covered:

- One rank process lost (SIGKILL, crash, OOM kill).
- One node's rank processes lost, host up.
- Further losses after a recovery, while the survivors' slots still hold every expert.
- One frozen rank (SIGSTOP), with hang detection on.

Anything else fails closed: rank 0, rank 0's node, host death, NVLink fabric faults, a loss the slots cannot cover.

Startup checks every requirement and refuses with its name. The full list is in Appendix A.

## Design Overview

FT keeps the same CUDA graphs after a loss. Rank count, slots per rank, buffer addresses and kernels stay fixed. A
recovery changes only what the graphs read on every launch:

- the live-rank mask;
- the abort flag;
- EPLB's expert-to-slot tables;
- the contents of expert slots.

Three ideas make this work:

- **Ghost slots.** Spare expert slots at addresses the graphs already know, with no memory behind them while healthy.
  On a failure they get memory, and lost experts are copied in from host copies.
- **Same-address remap.** Survivors move the all-to-all buffers to fresh memory at the same addresses. A dead rank's
  late writes land in old memory no survivor reads.
- **Recovery authority.** One process outside MPI that admits the loss, fences the failed ranks, and makes every
  survivor apply the same plan, or none.

![Figure 1](../review/figures/design/fig10a_architecture.png)
*Figure 1. Each rank's watch reports to the authority; each rank's recovery hook edits the tables in place.*

While healthy, FT adds only:

- **Rank mask.** Kernels skip ranks that are not live.
- **Abort polling.** A wait on a peer checks the abort flag only after ~100k cycles, so normal waits never poll.
- **Step commit.** A step applies only if no peer failed. This is one host check per step.
- **Extra slots.** A few spare slots per rank. Kernel tuning uses only the loaded slots.
- **Transport.** ULFM needs Open MPI's ob1 layer, so host collectives run over TCP.

## Ghost Slots

A layer's experts are stacked in one tensor per GPU; each position is a slot. FT adds spare slots at the end: the ghost
slots.

- **Layout.** The tensor is one virtual address range. Resident slots get GPU memory at startup.
- **Healthy.** Ghost slots map a small shared scratch area (a few MB per process). Routing never points at them, so
  they add no GEMM work.
- **On a failure.** Each survivor that receives a lost expert maps GPU memory behind a ghost slot and copies the expert
  in. Memory comes from free GPU memory first, then from the KV cache.
- **Abandoned plan.** The memory goes back to scratch.

Example, DeepSeek-R1 on 32 GPUs: 8 + 2 slots per rank without EPLB; 9 + 1 with online EPLB (288 + 32 slots).

![Figure 2](../review/figures/design/fig10c_ghost_slots.png)
*Figure 2. Ghost slots: scratch while healthy, GPU memory after a restore.*

A restore never reads from the lost GPU. It copies from host copies in `/dev/shm`:

| Placement | Host copies | Restore |
|---|---|---|
| Online EPLB, or replicas | Every node keeps every expert | From the node's own copies, at Install |
| No replicas | Each node keeps the next node's experts (a ring) | A copy-node rank stages each expert to the target, then the target loads it |

- Startup reserves all host-copy pages, so a full `/dev/shm` fails at startup, not later.
- FT keeps no host-copy pages mapped while healthy (a restore faults in what it copies; online EPLB's own per-step
  reads still touch them, as on main). As far as measured, compaction then stalls serving no more than with FT off
  (Appendix E).
- Ring copies cost about 1 / (number of nodes) of the expert weights per node.

## Expert Placement

Every survivor computes the same plan from the same inputs. Each lost expert:

- goes to an empty slot on a live rank first;
- else takes the slot of an expert with at least two copies;
- prefers the rank that has restored the fewest experts, then the lowest rank.

This spreads the extra work evenly. An expert with no copy left makes the loss unrecoverable. Online EPLB rebalances at
its next update.

## Late Writes

A dead rank may still have NVLink writes in flight. Fencing ends the process, but not the writes. So FT never reads the
old all-to-all memory again:

1. Each survivor clears the failed ranks in the mask, unmaps the buffers, and keeps the addresses.
2. Each survivor maps fresh memory at the same addresses. Failed ranks' rows stay unmapped.
3. Survivors confirm that all of them mapped, or all of them stop.
4. The graphs replay on the fresh memory.

- Late writes land in released memory that no survivor maps.
- The old memory frees only when the failed processes end, so the remap may borrow KV cache for a moment. It is
  returned at the commit, or soon after.

![Figure 3](../review/figures/design/fig10d_late_write_remap.png)
*Figure 3. Late writes land in released memory; survivors map fresh memory at the same addresses.*

## Recovery Flow

| Failure | Detected by | Time |
|---|---|---|
| Process ends | ULFM, polled by a native watch every 20 ms | ~0.02 s (median, rank and node kills) |
| Fatal CUDA error | The rank kills itself | as above |
| Frozen rank | Hang detection: no heartbeat for 5 s, held 30 s | 30-34 s, then fenced |
| All ranks stuck in GPU all-to-all waits, or rank 0 hung | Hang detection | job ends after 30 s |
| All ranks stuck in a host collective | Not detected by FT; main's 300 s detector | job ends (600 s seen once) |

- **Detect.** The watch sets the abort flag. Kernels waiting on the dead rank return.
- **Agree.** Survivors run the ULFM shrink and agree on who failed. They stop, discard uncommitted steps (reverting KV
  and requests), and tell the authority they stopped.
- **Plan.** Each survivor computes the new placement and maps memory for its ghost slots. The authority fences the
  failed ranks and proposes a new generation. Survivors remap the all-to-all buffers, stage ring copies, and reply
  with a hash of the plan.
- **Install.** Once all hashes match, survivors load the lost experts, switch routing and install the new mask. Then
  the authority commits.
- **Resume.** Discarded steps replay on the new placement. Serving continues on the same CUDA graphs.

![Figure 4](../review/figures/design/fig10b_recovery_sequence.png)
*Figure 4. Recovery sequence on the survivors and the authority.*

A recovery is bounded: about twice `recovery_timeout_s` (30 s) in the worst case. The authority aborts first if it
stalls.

## Other Notes

- **A loss before Install** restarts the round with every lost rank. A loss after Install ends the job.
- **Requests.**
  - Surviving requests continue from exactly the tokens they had.
  - A lost rank's request reruns if it returned nothing yet; otherwise it ends with a retryable error.
  - A request interrupted twice without progress is quarantined (retryable error).
  - Details are in Appendix B.
- **`/health`** reports `healthy`, `degraded` or `unprotected` (200), and `recovering` or `terminal` (503). So
  orchestrators don't restart a job that is only recovering.
- **Security.** The authority's HMAC key leaves the environment once read; no child process inherits it.
- **Authority.** It runs as its own process outside the MPI job, so it outlives any worker; if it is lost,
  serving continues but a further loss would end the job, and `/health` reports `unprotected` (tested on EP4 only,
  CLAIMS 88, 101).
- **Shutdown.** Stop `trtllm-serve`. A SIGTERM to `mpirun` looks like a loss that includes rank 0, so the job ends
  through the fail-closed path.

## Results

Setup: DeepSeek-R1 NVFP4 on 32 GB200 GPUs (8 nodes x 4), DEP32 (attention DP + expert parallel
over all 32 GPUs), overlap scheduler, MTP with 1 draft token. Load N/GPU = N x 32 concurrent requests.
Benchmark settings (all arms): random 1k/1k tokens exactly, `--ignore-eos`, all requests released at once, MTP
acceptance forced to 1 (`TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS`, as in TRT-LLM's perf recipes), stream_interval 10,
16 GiB KV host tier, pinned kernel tactics, warm-up + burn-in before measuring. FT arms use Open MPI ob1/TCP (ULFM
needs it); FT-off arms use the default launcher. Absolute tok/s are not comparable to published numbers.
Builds: **v5-final** `27442bd91d` (default; a Python overlay on the `58caa0fde6` C++ build). **Frozen build**
`58caa0fde6`: differs only in how FT maps host copies. **Fix-1** `d71f169e4b`: only the first of the two host-copy
fixes (Appendix E). Rows not rerun on v5-final come from the frozen build: pass/fail should carry over; times can
differ.

### Recovery

Each cell: kill -> rank 0's first step after the commit (server timestamps). In parentheses: kill -> the test
script seeing rank 0's "resumed" log line, polled every 0.5 s. That lags the first step by up to ~0.45 s, but comes
before it when the step is slow (prefill rows). Load 128/GPU (frozen rank: 64/GPU). The table shows the original
v5-final runs (mean of 2 or 1 run, as marked); the 4-run means are in the replication line below. All rows pass
(CLAIMS 193).

| Failure | No EPLB | Online EPLB |
|---|---|---|
| One rank killed | 1.5 s (1.7 s), n=2 | 0.8 s (1.2 s), n=2 |
| One rank killed in prefill * | 2.4 s (1.7 s), n=1 | 1.9 s (1.1 s), n=1 |
| One rank killed in an EPLB update | - | 0.8 s (1.2 s), n=1 |
| One node killed (4 ranks) | 2.5 s (2.7 s), n=2 | 1.6 s (1.8 s), n=2 |
| Two losses, one after the other | first 1.7 s (1.7 s), second: 2.1 s (script time only), n=1 | first 0.8 s (1.2 s), second: 1.6 s (script time only), n=1 |
| Frozen rank (SIGSTOP), 64/GPU | fenced 34.1 s and first step 37.2 s after the freeze, n=1 | fenced 30.0 s and first step 33.1 s after the freeze, n=1 |
| Restart instead, FT off (frozen build) ** | 318 s (n=1) | 605 s (584-625, n=2) |

- \* In the prefill rows the first step after the recovery is a full prefill, so it ends ~1 s after the resume line.
- \*\* Restart = kill -> the new server's first token: old server's exit, then weight loading, autotuning and graph
  capture; page cache kept (job 8206870, CLAIMS 198). Weight loading alone varied 80-362 s. Offline EPLB restart:
  334-541 s.

Recovery phases (time since kill):

| Phase | Rank kill | Node kill |
|---|---|---|
| Detected | 0.02 s | 0.02 s |
| Plan | 0.2-0.3 s | 0.7-1.0 s |
| Experts loaded, online EPLB | 0.7-0.8 s | 1.3-1.7 s |
| Experts loaded, no EPLB (ring copies staged from another node first) | 1.4-1.5 s | 2.4-2.5 s |

- The frozen build ran every row 2-3 times: script-observed resume 1.0-2.2 s for rank kills, 1.7-3.2 s for node kills
  (CLAIMS 176-178). Its no-EPLB node kill read 3.2 s vs 2.7 s on fix-1 and v5-final (CLAIMS 185, 193). A v5-final
  repeat read 3.22 s by the script but 2.81 s by server timestamps. By server timestamps the frozen build's node
  kill took 2.94 / 2.95 s (job 8211797) vs 2.42-2.81 s on v5-final: ~0.3 s of the gap is the script's 0.5 s polling,
  ~0.1-0.5 s is server-side; cause not isolated.
- Replication on v5-final (job 8285728, CLAIMS 203; same code and knobs, decided after the first results were seen):
  rank and node kills, 2 more each, all 8 pass. First step: no EPLB rank 1.53 / 1.45 s, node 2.42 / 2.81 s; online
  rank 0.89 / 0.83 s, node 1.42 / 1.53 s. Means over all 4 runs per row: no EPLB rank 1.5 s, node 2.6 s; online rank
  0.85 s, node 1.55 s.
- Offline EPLB (v5-final, n=1 each, CLAIMS 202; its offline placement differs slightly from the healthy-cost runs'):
  rank kill 1.4 s (1.7 s), node kill 1.7 s (1.7 s); both pass.
- Rank 0 lost, or recovery past its deadline: the job ends cleanly, as designed (frozen build, 4/4 each).

![Figure 6](../review/figures/v5final/fig5_recovery_timeline.png)
*Figure 6. Serving through a loss: rank-0 tokens per second and step time around each kill (red line). Grey band:
detect to first step.*

![Figure 7](../review/figures/v5final/fig6_resume_time.png)
*Figure 7. Resume time per kill, split into phases.*

![Figure 8](../review/figures/finals/fig9_downtime.png)
*Figure 8. Downtime: FT resume (seconds) vs a restart with FT off (minutes). Frozen build.*

### After the loss

Client-side throughput after / before a loss: the first full decode wave after the resume vs the waves before the
kill, at fixed concurrency (CLAIMS 209). Median over the v5-final kills; ideal = share of GPUs left.

| Failure | Measured | Ideal |
|---|---|---|
| One rank (31/32 live), 128/GPU | 0.955x (9 cases) | 0.969x |
| One node (28/32 live), 128/GPU | 0.858x (9 cases) | 0.875x |

- Both land at or just below the share of GPUs left. An earlier estimate (rank-0 tok/s x live ranks: 0.95x / 0.89x)
  read high for node kills; it is replaced by the client measurement.
- Two cases dip more right after recovery: a frozen rank (0.82-0.88x for ~5 waves) and a rank killed during prefill
  with online EPLB (0.73x on the first wave, 0.82x over two). Not explained yet.
- One post-kill wave per case (12-14 s), and healthy waves vary 0-5%, so each case is noisy.

![Figure 9](../review/figures/v5final/fig7_post_loss_tput.png)
*Figure 9. Throughput after / before each loss.*

![Figure 10](../review/figures/v5final/fig8_request_fate.png)
*Figure 10. What happened to each in-flight request. Survivors' requests all completed. Requests held by the lost
rank(s) had already streamed tokens, so each ended with a retryable error: 128 per rank kill, 512 per node kill (3% /
12.5% of the 4096 in flight). No other request failed. A restart instead drops all 4096.*

### Accuracy

GSM8K strict match, 1319 questions, served by the 31 survivors after rank 5 was killed and recovered (online EPLB,
MTP acceptance not forced). This checks the recovered model's answers, not interrupted requests.

| Build | FT on, kill | FT off, no kill | Delta (+-2 SE) |
|---|---|---|---|
| v5-final | 0.9477 | 0.9469 | +0.001 (+-0.017) |
| Frozen, run 1 | 0.9469 | 0.9416 | +0.005 (+-0.018) |
| Frozen, run 2 | 0.9431 | 0.9409 | +0.002 (+-0.018) |
| Fix-1 build | 0.9401 | 0.9484 | -0.008 (+-0.018) |

Every delta is within 2 SE. Of questions answered differently on v5-final, 21 went right-to-wrong and 22
wrong-to-right (CLAIMS 185, 193).

![Figure 11](../review/figures/v5final/fig8b_gsm8k.png)
*Figure 11. GSM8K with a kill vs FT off.*

### Healthy cost

Method: `benchmark_serving` against `trtllm-serve` at fixed concurrency (load = concurrent requests per GPU, x32 in
total). Each pair runs FT on and FT off back to back on the same 8 nodes; each cell (mode x load) has 4 pairs on 4
node sets, order alternated. Values: (FT on - FT off) / FT off, mean over the 4 pairs, with a 95% CI (t, 3 df)
(CLAIMS 194). Negative tok/s or positive TPOT (time per output token) / TTFT (time to first token) = FT slower.

| Mode | Load | Output tok/s/GPU | Median TPOT | p99 TPOT | Median TTFT |
|---|---|---|---|---|---|
| No EPLB | 32 | +0.50% (-1.40, +2.41) | -2.41% | -1.99% | +8.5% (-8.2, +25.3) |
| No EPLB | 128 | -1.15% (-3.30, +1.01) | -0.55% | +0.75% | +6.6% (-14.2, +27.4) |
| Offline EPLB | 32 | -2.03% (-3.56, -0.49) | +2.21% | +3.55% | +0.5% (-8.3, +9.3) |
| Offline EPLB | 128 | -0.35% (-4.13, +3.43) | -0.27% | +0.00% | +3.0% (-10.7, +16.7) |
| Online EPLB | 32 | -0.18% (-1.94, +1.57) | -0.29% | -0.31% | +6.7% (-21.9, +35.3) |
| Online EPLB | 128 | -0.05% (-1.67, +1.57) | -0.11% | +0.41% | -0.2% (-6.9, +6.6) |

- In the original runs only offline at 32/GPU was significantly below FT off (-2.0%, p99 TPOT +3.6%); it did not
  replicate (table below). In the other cells
  the CIs include 0 but are 3-8 points wide, so these runs cannot detect a ~1% cost. With 95% confidence
  (one-sided), FT's throughput loss is at most 1.5% with online EPLB, 2.7% without EPLB and 3.2% with offline EPLB.
- 24 tests (6 cells x 4 metrics) should give about one 'significant' result by chance; the offline cell is still
  reported as a cost.
- TTFT: no significant change in any cell (every CI spans 0; TTFT is noisy at these loads).
- No stall burst (a 30-80 s slowdown from kernel memory compaction; Known costs) in any of the 8 online measurement
  windows (4 FT, 4 FT off).

Replication at 32/GPU (jobs 8285727, 8287475, 8289412, 8289662; CLAIMS 206). Same code, knobs and method; 4 new pairs
per cell, order balanced 2/2. Decided on 10-25 after the original n=4 was seen; the reporting rule was set before any
replication data. One online FT run hung (Known Limits) and was replaced like for like.

| Mode (32/GPU) | Original n=4 | Replication n=4 | Pooled n=8 |
|---|---|---|---|
| No EPLB | +0.50% (-1.40, +2.41) | -0.01% (-2.22, +2.20) | +0.25% (-0.78, +1.28) |
| Offline EPLB | -2.03% (-3.56, -0.49) | +0.65% (-4.16, +5.46) | -0.69% (-2.80, +1.42) |
| Online EPLB | -0.18% (-1.94, +1.57) | +0.29% (-1.37, +1.95) | +0.05% (-0.80, +0.91) |

- Output tok/s/GPU, FT vs off. The original offline -2.0% did not replicate: pooled -0.7% [-2.8, +1.4].
- Pooled n=8 one-sided 95% bounds at 32/GPU: FT's throughput loss is at most 0.58% without EPLB, 2.38% offline and
  0.63% online. The trimmed per-step bounds are +0.99% / +0.63% / +0.67%, all under 1% at 32/GPU; '< 1% per step' is
  still not claimed (that rule was added after the rehearsal, and 128/GPU was not replicated).
- At these sample sizes, FT's throughput cost at 32/GPU cannot be separated from run-to-run noise (pairs range
  -3.8% to +2.8%). The steady per-step cost below is the measured FT cost.
- The per-step cost replicates: pooled n=8 trimmed step +0.87% [+0.72, +1.02] no EPLB, +0.50% [+0.34, +0.66]
  offline, +0.35% [-0.04, +0.75] online at 32/GPU (CLAIMS 206).
- Latency, pooled n=8 at 32/GPU, every CI includes 0: median TPOT -1.8% to +0.1%; p99 TPOT -3.9% to +1.0% (the
  original offline +3.6% did not replicate: pooled +0.97% [-3.49, +5.42]); median TTFT +3.6% to +12.2%, with the
  largest point estimate online (+12.2% [-5.5, +29.8]). TTFT is noisy here; it is not shown to be unchanged.
- Pooling all 24 pairs at 32/GPU across modes (descriptive, not pre-registered), median TTFT is +7.3% [+1.3, +13.3]
  with FT; p99 TTFT is unchanged (+0.2% [-0.7, +1.0]); no shift at 128/GPU. Mechanism: FT runs ~5% fewer prefill
  steps with ~6% more requests each, at the same prefill time per token, so in this closed-loop benchmark the median
  first token waits a little longer. Not the TCP transport (FT off over the same transport shows no rise). Why FT
  batches prefill more is not known (CLAIMS 208).

![Figure 12](../review/figures/v5final/fig1_ft_cost.png)
*Figure 12. FT cost per cell: client throughput, and rank 0's per-step time (full and trimmed mean), with 95% CIs.*

![Figure 13](../review/figures/v5final/fig2_modes_abs.png)
*Figure 13. Absolute output tok/s/GPU, median TPOT and TTFT, FT on vs off, per mode and load.*

Per-step cost: rank 0's decode step time from the server log. Full mean = all steady steps. Trimmed mean = steps
up to 1.5x the median (drops stall steps). Positive = FT slower.

| Mode | Load | Full mean | Trimmed mean |
|---|---|---|---|
| No EPLB | 32 | +0.18% (-2.08, +2.43) | +0.96% (+0.68, +1.25) |
| No EPLB | 128 | +0.93% (-1.48, +3.34) | +0.53% (+0.06, +1.01) |
| Offline EPLB | 32 | +1.73% (-1.00, +4.47) | +0.59% (+0.28, +0.89) |
| Offline EPLB | 128 | +0.60% (-1.90, +3.09) | +0.34% (-0.02, +0.69) |
| Online EPLB | 32 | +0.63% (-0.95, +2.22) | +0.48% (-0.35, +1.30) |
| Online EPLB | 128 | +0.52% (+0.12, +0.92) | +0.39% (+0.10, +0.68) |

- "< 1% per step" is not claimed. The pre-registered rule (every cell's one-sided 95% upper bound < 1% for both the
  trimmed and the full mean) fails: full-mean bounds reach +3.76%. The trimmed-only rule added later also fails:
  +1.18% (no EPLB 32), +1.09% (online 32).

![Figure 14](../review/figures/v5final/fig3_noeplb_decomposition.png)
*Figure 14. No-EPLB FT cost per pair: client throughput and trimmed step time.*

### Baseline

Our FT off matches main's decode step time: trimmed step -0.26% to +0.37% at 32 and 128/GPU (online and no EPLB,
n=4, frozen build, descriptive). Context scheduling differs by design (Appendix C).

![Figure 15](../review/figures/finals/fig4_main_parity.png)
*Figure 15. Our FT off and FT on vs main, per pair. Frozen build.*

### Memory

- GPU, after a loss: restored experts take up to 0.44 GB per survivor after a rank loss, 1.75 GB after a node loss.
- GPU, healthy: ghost slots map a few MB of scratch per process.
- Host: FT keeps expert copies in /dev/shm, reserved at startup. Offline or online EPLB: every node keeps every expert
  (~390 GB per node for R1). No EPLB: a ring of copies (~49 GB per node). FT off: none for no or offline EPLB;
  main's online EPLB keeps its own host copies.

### Known costs

- **One healthy FT run hung (1 of 36 FT-on runs, 0 of 31 FT-off).** All ranks stalled in a host collective during a
  replication run; FT did not detect it, and the job ended after 600 s. Cause not isolated; see Known Limits,
  "Host-collective stall". No presented number comes from that run.

- **Steady per-step cost:** +0.3% to +1.0% (trimmed mean).
  - A roughly fixed time per step: ~0.6% at 32/GPU (14 ms steps), ~0.3% at 128/GPU (26 ms steps). It does not grow
    with load (n=4 each):

    | Mode | 32/GPU | 128/GPU |
    |---|---|---|
    | Online EPLB | 69 us | 97 us |
    | Offline EPLB | 83 us | 89 us |
    | No EPLB (incl. the static load balancer) | 133 us | 142 us |

  - Without EPLB, FT still routes through EPLB's slot tables with a fixed placement (the static load balancer).
  - Most of it shows in rank 0's GPU forward time (medium confidence; review/diag/steady_cost_analysis.md).
  - All-to-all kernels (dispatch + combine, 59 calls per step: 58 MoE layers + MTP). FT adds per call ~0.6-0.8 us on
    one node (4 ranks, 8-32 tokens/rank; ~40 us/step; CLAIMS 191). A kernel-only benchmark across 4-32 GPUs, at 32
    GPUs and 64 tokens/rank (job 8282103, CLAIMS 200, diagnostic) gives +1.0 us without EPLB statistics (~59 us/step)
    and +3.0 us with online EPLB's statistics (~178 us/step if back to back). Only the statistics path grows with GPU
    count. In serving, online's cost (69 / 97 us) is no higher than offline's (83 / 89 us), so overlap likely hides
    most of it (inference, not measured).
  - Mask-in-local-memory hypothesis tested and refuted for the statistics path (job 8289096, 32 GPUs, 64 tokens/rank,
    n=3, same nodes; CLAIMS 205). A kernel variant keeping the live-rank mask in shared memory (branch `mask-spill`,
    stack 24 B vs 64 B) left the statistics-path cost unchanged (+3.21 vs +2.96 us per round trip) and trimmed the
    no-statistics cost slightly (+0.69 vs +0.85 us, ~10 us/step). What grows with GPU count is likely the serial
    per-peer statistics loop itself (inference); the rest of the per-step cost is not yet attributed.
  - Single-run checks found no cost from hang detection, the failure poll, spare-slot memory or the extra slots
    (CLAIMS 173, 181, 182).
- **Offline at 32/GPU: -2.0% in the original runs, not replicated** (replication +0.65%; pooled n=8 -0.69% [-2.80,
  +1.42]; CLAIMS 206). Analysis of the original logs (review/diag/offline32_analysis.md; medium confidence):
  - FT on ran at the same speed as on earlier builds (3349 vs 3364 and 3322 tok/s/GPU); this batch's FT-off arms were
    fast (3419 vs 3369 and 3355). Over all 12 pairs across three builds the cell averages about -1.0% (post hoc,
    descriptive).
  - That ~1.0% splits into ~0.45% steady per-step cost and ~0.6% from occasional slow decode steps not caused by rank
    0's own prefill (FT had more in 9 of 12 pairs; see next bullet).
  - A follow-up with per-step logs from every rank (job 8282497, 4 cases per arm, diagnostic; CLAIMS 201) traced these
    stalls to other attention-DP ranks prefilling out of phase, spread over many ranks (inference). They hit FT off as
    often as FT on (43 vs 33 stalls; 0-25 per case). Prefill on another rank covered 42 of 43 FT-off stalls and
    26 of 33 FT stalls; 7 FT stalls are unexplained. In that run they looked like run-to-run phasing, not an FT cost.
  - Across all 16 healthy offline-32 pairs (original, earlier builds, replication), FT arms had more rank-0 far stalls
    in 12 (sign test p=0.035; p=0.17 with the 4 STALL-ATTR pairs added). In the replication the extra was +1.3 s,
    ~0.1% of the window (CLAIMS 207). So FT may make these stalls slightly more frequent; the effect is small and its
    cause is not attributed.
  - Host memory does not explain the build-to-build spread (every FT-on arm holds ~390 GB of /dev/shm per node). Its
    share of the FT gap is untested.
- **Kernel memory compaction** (stalls on host-copy pages; reduced to the FT-off level as far as measured). Fix 1:
  no mapping of peers' copies. Fix 2: a rank unmaps its own copies after writing them (Appendix E).
  - Startup stall on a used node (CLAIMS 187, 10-90 s window): FT with both fixes 2.7 / 5.7 s vs FT off 3.2 s
    (n=1; 2.2 s on a fresh node); fix 1 alone 15-17 s. Before the fixes 21-46 s (n=3; CLAIMS 184, a different metric).
  - Mid-run stall bursts: 3 of 12 online FT windows before the fixes; now 0/8 FT and 0/9 FT off in the healthy-32
    windows (CLAIMS 194, 196, 197; one FT healthy-8 window in job 8273195 had 3 short 3-5 s slow buckets). At the old rate, 0 of 8 would still happen ~10% of the time, so this is not proof of removal.
  - Online EPLB reads host copies every step, so main keeps a small exposure of its own. Details: Appendix E.

## Known Limits

- **Frozen rank.** Fenced only with hang detection, 30-34 s after the freeze. Two ranks frozen at once are not confirmed.
- **Hang threshold.** A hang is confirmed only after 30 s: at 5 s a healthy rank holding the GIL was falsely fenced,
  while at 30 s no 45 s GIL hog or all-rank or node host stall was (older builds, n=1 each; CLAIMS 77, 97, 103).
- **Host-collective stall.** If every rank stalls inside a host collective, FT's hang detection blames no rank (each
  is alive and waiting), so main's 300 s detector ends the job. Seen once: 1 of 36 healthy FT-on runs, 0 of 31 FT-off
  (replication job 8285727; CLAIMS 204). The stall was in main's attention-DP state allgather over FT's Open MPI TCP
  transport; the cause is not isolated. FT's recovery then made the job end at 600 s instead of 300 s.
  Follow-ups: a long healthy soak (FT transport vs default) and three detector fixes (review/POST_FINAL_ANALYSIS_PLAN.md).
- **Second loss.** After Install, or a lost commit message, the job ends.
- **Not recoverable.** Rank 0 or its node; a node plus its ring-copy node; a loss the slots cannot cover.
- **Out of reach.** Host death, GPU loss and NVLink fabric faults end the job, as without FT.
- **CFT.** CFT (the all-to-all's counted-write path, driver >= 615) is untested; every run used the fence path
  (driver 580).
- **No rejoin** until restart.
- **KV cache.** Memory given to restored experts stays until restart.
- **Online EPLB on GPUs whose memory is a NUMA node** uses backed spare slots.
- **Validated.** v5-final: 15/15 recovery rows and GSM8K with a kill pass (CLAIMS 193). Frozen build: rank kill 9/9,
  in prefill 5/5, in an EPLB update 2/2, node kill 4/4, two losses 4/4, frozen rank 4/4, rank 0 ends the job 4/4,
  deadline ends the job 4/4, GSM8K with a kill 2/2 (source: review/DESIGN_DOC_v5final_filled.md;
  invalid runs excluded, CLAIMS 177). Not tested on v5-final: two frozen ranks at once, CFT, host death,
  NVLink fabric faults.

![Figure 16](../review/figures/design/fig11_limits.png)
*Figure 16. Known limits and requirements.*

## Alternatives Considered

- **A second CUDA graph set for the degraded layout.** Doubles graph memory and capture time.
- **Restore only into replica slots.** No new memory, but drops replicas and raises the peak expert load after a node
  loss. Used only when ghost slots run out.
- **Backed spare slots.** No mapping at recovery, but keeps memory from the KV cache for good.
- **Reuse the all-to-all memory after a wait.** Needs a bound on a dead GPU's in-flight writes. The remap needs none.

---

## Appendix A. Requirements

Startup refuses a missing requirement and names it.

| Area | Required |
|---|---|
| Layout | TP size = EP size = world size; no PP or CP; expert TP 1; no LM-head TP; default attention-DP router; MPI executor |
| MoE | NVLinkOneSided fence path; an EPLB load balancer |
| KV cache | V2; no transceiver, connector, SWA scratch reuse, KV pool rebalance, KV cache events, conversation manager, cross KV |
| Executor | A generation sampler; no guided decoding, beam search, PEFT or sleep; hang timeout above `shrink_timeout_s + 2 x recovery_timeout_s` |
| MPI | ULFM (`--with-ft ulfm`, shrink, revoke); `MPI_THREAD_MULTIPLE`; ob1; hcoll off (the launcher defaults `coll ^hcoll,ucc`) |
| Coverage | The survivors of every failure domain hold every expert; `/dev/shm` fits the host copies |
| Authority | Reachable; same FT config; HMAC key in hex, >= 16 bytes |

Rejected per request: repetition, presence and frequency penalties; multimodal input with the overlap scheduler;
logits post-processors.

## Appendix B. Request Fate

| Request | Outcome |
|---|---|
| Survivor, in a discarded step | Replayed from its exact tokens |
| Interrupted again before any progress | Quarantined: retryable error |
| Lost rank, nothing returned yet | Re-run on a survivor |
| Lost rank, tokens already streamed | Retryable: HTTP 503, or an SSE error then `[DONE]` |
| Lost rank, cancel pending | Cancelled |
| Not yet received by every rank | Re-admitted |
| New, pinned to a lost DP rank | HTTP 400 |

## Appendix C. FT-off differences vs main

| Change | Effect |
|---|---|
| Attention-DP burst-release rule (not FT) | Scheduling and TTFT with `attention_dp_enable_balance`; its own PR |
| mpi4py 3.1.5 -> 4.1.2 (not FT) | Same in every measured arm |
| All-to-all flag word 4 -> 8 bytes | Layout only |
| FT kernel branches | Compiled out when FT is off |
| Failed streams end with an SSE error and `[DONE]` (not FT) | Client-visible; its own PR |
| Recovery hook call sites | A few no-op calls per step |

## Appendix D. Code map

Code refs at v5-final `27442bd91d`, for reviewers.

| Topic | Code |
|---|---|
| Recovery orchestration | `_torch/pyexecutor/peer_loss_recovery.py` (requirements 416-463; recovery 950-1111; remap 1379-1458; quarantine 1570-1607) |
| Authority, config, client | `executor/recovery/{authority,config,client}.py` |
| Ghost slot memory | `cpp/tensorrt_llm/runtime/moeLoadBalancer/ghostSlotMemory.{h,cpp}` |
| Load balancer, placement, host copies | `_torch/moe/fused_moe/moe_load_balancer.py` (986-1032, 1197-1212, 1451-1544) |
| Mask, abort, commit | `_torch/moe/fused_moe/communication/execution_state.py`; `cpp/.../moeAlltoAllKernels.cu` (223-311) |
| Remap | `_torch/distributed/mnnvl_memory.py` (943-1021) |
| ULFM, watch, hcoll check | `_torch/distributed/communicator.py` (850-1091) |
| Shutdown | `llmapi/mpi_session.py` (141-200) |

## Appendix E. Host copies and kernel memory compaction

The issue. Kernel memory compaction (kcompactd, proactive by default) moves movable pages to build huge pages.
The /dev/shm EPLB host copies are movable. While a page moves, any thread that touches it waits
(migration_entry_wait_on_locked, D state). FT made this worse in two ways: every rank mapped every local
rank's copy (prefault), and each rank kept its own copy mapped.

When it hits:
- Startup on a used node. A large free (page-cache drop, previous server's teardown) starts a 45-90 s
  compaction pass that moves ~100 GB per memory node. With the prefault, every rank's steps stall. CLAIMS 183.
- Mid-window, online EPLB. Online EPLB reads the host copies every step. Before the fixes, 3 of 12 online FT
  windows had a 30-80 s burst. FT off showed one partial burst in a diagnostic run (review/diag/midburst2_readout.md; 0/10 FT-off
  cost windows). CLAIMS 192.

The fixes (in the PRs):
1. No peer prefault: a rank maps a peer's copy only when a restore reads it. Startup stall 21-46 s ->
   6-11 s (n=3 each). CLAIMS 184.
2. The owner drops its own mapping after writing its copy (MADV_DONTNEED; the shm pages stay). Startup stall
   16.8 / 15.0 s -> 2.7 / 5.7 s, FT off 2.2 s (fresh node) / 3.2 s. CLAIMS 187 (its 10-90 s window; on it fix 1
   alone reads 15-17 s, not 6-11 s).

On the final code: 0 mid-window bursts in every window checked (cost runs 0/4 FT + 0/4 off; long online
windows 0/2 FT + 0/3 off; B-AB2 0/2 FT + 0/2 off; CLAIMS 194, 196, 197). Restore and accuracy unchanged (CLAIMS 193).

What did not work, and why:
- Pinning (cudaHostRegister) cuts compaction stalls ~5x but does not remove them: kernel 6.8 still isolates and
  unmaps pinned shmem pages before it gives up on them (CLAIMS 192). Pinning every copy starved nodes in an earlier
  run (job 8121589; HANDOFF 10-21).

Next step: driver-allocated host memory (option B).
- Host copies in cuMemCreate host-NUMA pinned memory, owned by the driver, which compaction should not move.
- 1-node probe: 15 s of slow copies vs 71 s (prefaulted /dev/shm) and 44 s (lazy /dev/shm), with residual
  waits (22 vs 109 / 86 samples): inconclusive by the rule set before the probe. Slow-copy time tracks how many
  pages were migrated, and fewer were migrated during the driver-memory blocks; in its one block with heavy
  migration, slow copies (8.9 s) were within the /dev/shm range. CLAIMS 195.
- At full scale it needs reserving before weight loading. Then main's loader can skip read-ahead; startup was
  128-170 s slower than v5-final on the same nodes (CLAIMS 196). A partial read-ahead in the loader fixes that; it changes main's loader.
- Full-scale A/B (2 groups of 8 nodes, one run per arm): no steady gain over the final code (output tok/s/GPU
  -1.1% to +0.3% at 32/GPU, -2.5% to -0.8% at 128/GPU), higher TTFT than FT off, 0 bursts in either. Not adopted; it is
  the next step if bursts reappear. CLAIMS 196.

## References

1. Dynamo production fault-tolerance study, arXiv 2609.25451.
2. TENT (Mooncake team), LLM-serving fault data, arXiv 2604.00368, Table 1.

# WideEP Fault Tolerance MVP Design Doc

Andrew Rusli | Created: Oct 19, 2026 | Updated: Oct 23, 2026 | Code: v5-final `58caa0fde6`

## Overview & Motivation

Today, losing one rank of a wide expert-parallel (WideEP) deployment ends the whole job. Serving resumes only after a
restart (minutes).

With fault tolerance (FT), the surviving ranks regroup in about a second and keep serving on the same CUDA graphs
captured at startup. The next phase is recovery from N-k back to N ranks while the survivors serve.

Why it matters:

- Failures are more common at large scale, and restarts are costly.
- An 18-week trace of NVIDIA's Dynamo serving cluster (~150 replicas, mostly 8 GPUs or fewer) saw 46 restarts a day,
  with a median recovery of 3.5 min. 77% came from engine process deaths or hangs that left the GPU intact (Dynamo
  paper).
- WideEP deployments span 32 to 64+ GPUs, so they are even more exposed.

| | Without FT | With FT |
|---|---|---|
| One rank lost | job ends; restart | serving resumes in median 1.21 s (min-max 1.01-2.21 s, n=9); 9/9 PASS; 32/128 req/GPU |
| One node's ranks lost | job ends; restart | serving resumes in median 2.46 s (min-max 1.72-3.21 s, n=4); 4/4 PASS; 128 req/GPU |
| One rank frozen | job ends after the 300 s hang detector | fenced after ~35 s, then recovered |
| Healthy throughput | baseline | client output tok/s/GPU, FT on vs off, % = (on - off) / off (negative = FT slower): no EPLB 32/GPU -0.61% (95% CI [-4.51, +3.28], n=4); no EPLB 128/GPU -0.49% (95% CI [-1.08, +0.10], n=4); offline EPLB 32/GPU -0.97% (95% CI [-1.37, -0.57], n=4); offline EPLB 128/GPU -0.38% (95% CI [-1.24, +0.47], n=4); online EPLB 32/GPU -3.50% (95% CI [-8.49, +1.50], n=4); online EPLB 128/GPU +0.40% (95% CI [-2.65, +3.46], n=4) |

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

- **Rank mask.** Kernels skip ranks that are not live. no EPLB trim +0.57% / full +0.45% (one-sided 95% UB +0.68 / +1.15, n=4); offline EPLB trim +0.45% / full +1.15% (one-sided 95% UB +0.67 / +2.03, n=4); online EPLB trim +0.65% / full +2.79% (one-sided 95% UB +0.97 / +4.83, n=4)
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

Example, DeepSeek-R1 on 32 GPUs: 8 + 2 slots per rank without EPLB, 9 + 1 with online EPLB (288 slots).

![Figure 2](../review/figures/design/fig10c_ghost_slots.png)
*Figure 2. Ghost slots: scratch while healthy, GPU memory after a restore.*

A restore never reads from the lost GPU. It copies from host copies in `/dev/shm`:

| Placement | Host copies | Restore |
|---|---|---|
| Online EPLB, or replicas | Every node keeps every expert | From the node's own copies, at Install |
| No replicas | Each node keeps the next node's experts (a ring) | A copy-node rank stages each expert to the target, then the target loads it |

- Startup reserves all host-copy pages, so a full `/dev/shm` fails at startup, not later.
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
| Process ends | ULFM, polled by a native watch every 20 ms | median 0.02 s (min-max 0.01-0.03 s, n=24) from the victim's kill to rank 0's phase=detect (rank, node, two-loss rows) |
| Fatal CUDA error | The rank kills itself | as above |
| Frozen rank | Hang detection: no heartbeat for 5 s, held 30 s | ~35 s, then fenced |
| All ranks stuck, or rank 0 hung | Hang detection | job ends after 30 s |

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
- **Shutdown.** Stop `trtllm-serve`. A SIGTERM to `mpirun` looks like a loss that includes rank 0, so the job ends
  through the fail-closed path.

## Results

DeepSeek-R1 NVFP4 on 32 GB200 GPUs (8 nodes x 4); overlap scheduler, MTP, 1k/1k tokens; no, offline and online EPLB.
FT on vs off on the same build and nodes.

- **Healthy cost.** FT on vs off, % = (on - off) / off: no EPLB 8/GPU (appendix): client tok/s +3.12% (CI [+0.88, +5.37]), trim step +0.59% (CI [+0.45, +0.72]), full step -4.22% (CI [-5.97, -2.46]), n=2; no EPLB 32/GPU: client tok/s -0.61% (CI [-4.51, +3.28]), trim step +0.80% (CI [+0.62, +0.97]), full step +0.33% (CI [-2.69, +3.36]), n=4; no EPLB 128/GPU: client tok/s -0.49% (CI [-1.08, +0.10]), trim step +0.35% (CI [-0.07, +0.77]), full step +0.57% (CI [-0.88, +2.02]), n=4; offline EPLB 8/GPU (appendix): client tok/s +4.15% (CI [-29.64, +37.95]), trim step +0.62% (CI [-0.11, +1.36]), full step -4.21% (CI [-46.07, +37.64]), n=2; offline EPLB 32/GPU: client tok/s -0.97% (CI [-1.37, -0.57]), trim step +0.57% (CI [+0.32, +0.81]), full step +1.50% (CI [-0.33, +3.33]), n=4; offline EPLB 128/GPU: client tok/s -0.38% (CI [-1.24, +0.47]), trim step +0.33% (CI [-0.03, +0.68]), full step +0.80% (CI [+0.22, +1.38]), n=4; online EPLB 8/GPU (appendix): client tok/s -1.69% (CI [-31.35, +27.97]), trim step +1.15% (CI [+0.05, +2.24]), full step +0.57% (CI [-39.70, +40.84]), n=2; online EPLB 32/GPU: client tok/s -3.50% (CI [-8.49, +1.50]), trim step +0.96% (CI [+0.18, +1.75]), full step +4.48% (CI [-1.23, +10.19]), n=4; online EPLB 128/GPU: client tok/s +0.40% (CI [-2.65, +3.46]), trim step +0.33% (CI [+0.04, +0.63]), full step +1.11% (CI [+0.73, +1.48]), n=4; per-step 'FT cost < 1%': rule A NOT claimed, rule B NOT claimed; [FINALS: TPOT and TTFT per mode at 8/32/128 req/GPU]
- **Online EPLB at low load.** [FINALS: online decomposition]
- **Recovery.** kill -> all ranks resumed: rank kill: no EPLB 32/GPU median 1.725 s (min-max 1.24-2.21 s, n=2), no EPLB 128/GPU median 1.71 s (min-max 1.19-2.21 s, n=3), online EPLB 32/GPU median 1.11 s (min-max 1.01-1.21 s, n=2), online EPLB 128/GPU median 1.115 s (min-max 1.02-1.21 s, n=2); node kill: no EPLB 128/GPU median 3.205 s (min-max 3.2-3.21 s, n=2), online EPLB 128/GPU median 1.72 s (min-max 1.72-1.72 s, n=2); two sequential losses: no EPLB 128/GPU median 1.21 s (min-max 1.2-1.22 s, n=2), online EPLB 128/GPU median 1.21 s (min-max 1.2-1.22 s, n=2); frozen rank: no EPLB 64/GPU median 34.35 s (min-max 34.1-34.6 s, n=2), online EPLB 64/GPU median 36.595 s (min-max 36.09-37.1 s, n=2); rank kill in prefill: no EPLB 128/GPU median 1.58 s (min-max 1.09-2.09 s, n=3), online EPLB 128/GPU median 1.18 s (min-max 1.18-1.18 s, n=2); rank kill in EPLB update: online EPLB 128/GPU median 1.195 s (min-max 1.19-1.2 s, n=2); rank-0 phases from the kill (median, rank and node kills): detect 0.02 s, proposed 0.28 s, ready 1.3 s, committed 1.35 s, first_step 1.43 s; restart in the same allocation, FT off, kill -> serving again (down_s): no EPLB 318.1 s (n=1), offline EPLB median 437.4 s (min-max 333.5-541.3 s, n=2), online EPLB median 604.75 s (min-max 584.4-625.1 s, n=2); [FINALS: the authority's stage durations (AUT:585-597)]
- **After the loss.** rank-0 decode tok/s x live ranks, after / before the loss: rank kill no EPLB 32/GPU median 1.088x (min-max 1.081-1.095x, n=2) (live 31/32); rank kill no EPLB 128/GPU median 0.946x (min-max 0.941-0.96x, n=3) (live 31/32); rank kill online EPLB 32/GPU median 1.139x (min-max 1.133-1.146x, n=2) (live 31/32); rank kill online EPLB 128/GPU median 0.956x (min-max 0.934-0.978x, n=2) (live 31/32); node kill no EPLB 128/GPU median 0.878x (min-max 0.874-0.882x, n=2) (live 28/32); node kill online EPLB 128/GPU median 0.875x (min-max 0.873-0.878x, n=2) (live 28/32)
- **Memory.** [FINALS: ghost slot healthy bytes per GPU]; memory behind restored expert slots per survivor (restored_mb): rank kill: busiest survivor median 444 MB (min-max 396-444 MB, n=9), median survivor median 360 MB (min-max 336-360 MB, n=9); node kill: busiest survivor median 1614 MB (min-max 1476-1752 MB, n=4), median survivor median 1572 MB (min-max 1476-1668 MB, n=4); [FINALS: KV lent to the remap and when returned]
- **Accuracy.** GSM8K strict-match 0.9469±0.0062 with a rank kill vs 0.9416±0.0065 FT off without (delta +0.0053, 2 SE 0.0180, PASS; n=1319/1319 questions); strict-match 0.9431±0.0064 with a rank kill vs 0.9409±0.0065 FT off without (delta +0.0022, 2 SE 0.0182, PASS; n=1319/1319 questions)
- **Baseline.** our FT off vs main, % = (ours - main) / main: no EPLB 32/GPU trim step -0.05% (95% CI [-0.13, +0.04], n=4); no EPLB 32/GPU client tok/s +1.39% (95% CI [-1.47, +4.25], n=4); no EPLB 128/GPU trim step -0.01% (95% CI [-0.21, +0.20], n=4); no EPLB 128/GPU client tok/s -0.11% (95% CI [-1.07, +0.85], n=4); online EPLB 32/GPU trim step -0.26% (95% CI [-0.69, +0.18], n=4); online EPLB 32/GPU client tok/s +2.04% (95% CI [-1.05, +5.14], n=4); online EPLB 128/GPU trim step +0.37% (95% CI [+0.21, +0.53], n=4); online EPLB 128/GPU client tok/s -1.09% (95% CI [-2.58, +0.39], n=4); [FINALS: FT off on the ULFM launcher vs FT off on UCX, to
  separate the transport from FT]
- **Note.** Main's warmup leaves two token sizes to compile on first use, with or without FT (upstream). Kill runs warm
  them first.

| Figure | Shows |
|---|---|
| ![](../review/figures/data/fig1_ft_cost.png) | 1. FT cost per mode and load |
| ![](../review/figures/data/fig2_modes_abs.png) | 2. Throughput per mode at 32 and 128 req/GPU (2x: 8 and 512) |
| ![](../review/figures/data/fig3_noeplb_decomposition.png) | 3. No-EPLB cost: load balancer vs FT |
| ![](../review/figures/data/fig5_recovery_timeline.png) | 5. Throughput around a kill |
| ![](../review/figures/data/fig6_resume_time.png) | 6. Kill to resume, by phase |
| ![](../review/figures/data/fig7_post_loss_tput.png) | 7. Throughput after vs before a loss |
| ![](../review/figures/data/fig8_request_fate.png) | 8. Fate of in-flight requests (8b: GSM8K) |
| ![](../review/figures/data/fig9_downtime.png) | 9. Downtime: FT vs restart |

## Known Limits

- **Frozen rank.** Fenced only with hang detection, after ~35 s. Two ranks frozen at once are not confirmed.
- **Second loss.** After Install, or a lost commit message, the job ends.
- **Not recoverable.** Rank 0 or its node; a node plus its ring-copy node; a loss the slots cannot cover.
- **Out of reach.** Host death, GPU loss and NVLink fabric faults end the job, as without FT.
- **No rejoin** until restart.
- **KV cache.** Memory given to restored experts stays until restart.
- **Online EPLB on GPUs whose memory is a NUMA node** uses backed spare slots.
- Validated: kill runs: rank kill 9/9 PASS (no EPLB 5/5, online EPLB 4/4); rank kill in prefill 5/5 PASS (no EPLB 3/3, online EPLB 2/2); rank kill in EPLB update 2/2 PASS (online EPLB 2/2); node kill 4/4 PASS (no EPLB 2/2, online EPLB 2/2); two sequential losses 4/4 PASS (no EPLB 2/2, online EPLB 2/2); frozen rank 4/4 PASS (no EPLB 2/2, online EPLB 2/2); rank 0 kill (must end) 4/4 PASS (no EPLB 2/2, online EPLB 2/2); deadline (must end) 4/4 PASS (no EPLB 2/2, online EPLB 2/2); GSM8K with a rank kill 2/2 PASS; restarts (FT off) n=5; jobs 8206860, 8206861, 8206863, 8206864, 8206870, 8211740, 8211797, 8215035; [FINALS: integration cases on the frozen build]

![Figure 5](../review/figures/design/fig11_limits.png)
*Figure 5. Known limits and requirements.*

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

Code refs at `58caa0fde6`, for reviewers.

| Topic | Code |
|---|---|
| Recovery orchestration | `_torch/pyexecutor/peer_loss_recovery.py` (requirements 416-463; recovery 950-1111; remap 1379-1458; quarantine 1570-1607) |
| Authority, config, client | `executor/recovery/{authority,config,client}.py` |
| Ghost slot memory | `cpp/tensorrt_llm/runtime/moeLoadBalancer/ghostSlotMemory.{h,cpp}` |
| Load balancer, placement, host copies | `_torch/moe/fused_moe/moe_load_balancer.py` (997-1043, 1208-1223, 1467-1560) |
| Mask, abort, commit | `_torch/moe/fused_moe/communication/execution_state.py`; `cpp/.../moeAlltoAllKernels.cu` (223-311) |
| Remap | `_torch/distributed/mnnvl_memory.py` (943-1021) |
| ULFM, watch, hcoll check | `_torch/distributed/communicator.py` (850-1091) |
| Shutdown | `llmapi/mpi_session.py` (141-200) |

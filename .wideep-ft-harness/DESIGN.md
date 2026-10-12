# WideEP fault tolerance: surviving a worker loss

## Summary

- Today, losing one worker process (rank) of a WideEP job kills the whole job.
- With fault tolerance (FT), the surviving ranks keep serving **on the CUDA graphs captured at
  startup**: no reload, recompile or recapture.
- Survivors' requests continue with the same tokens. Only the lost rank's requests fail, with a
  retryable error.

![Recovery time and throughput](evidence/results.svg)

DeepSeek-R1 NVFP4, 32 GB200 GPUs (EP32, attention DP). Kimi-K2 and 16 and 64 GPUs show similar
results. FT currently needs the overlap scheduler and MTP off, which halves throughput versus
the production config (section 7).

## 1. Today's failover (without FT)

- A rank dies → its peers' MoE all-to-all kernels spin on flags it never sets, until a device
  timeout traps and kills their CUDA contexts. Host collectives that include it hang or abort.
- The whole job exits. All in-flight requests are lost.
- The job must be torn down, rescheduled and started again: model load and warmup alone take
  81–173 s for R1 on 32 GPUs.

## 2. Failover with FT

```text
rank dies
  → detect      a host collective fails with a ULFM error, or a watcher thread sees it
  → unblock     revoke host communicators; set the abort flag that spinning MoE kernels poll
  → discard     the interrupted step reaches no request; its KV reservations are undone
  → agree       ULFM Shrink gives every survivor the same membership; each reports to the
                external recovery authority
  → install     route the lost rank's experts to their copies; mask the rank in the kernels
  → commit      the authority commits once every survivor has installed; serving resumes
```

**ULFM** (MPI fault tolerance, Open MPI 5 + mpi4py 4):
- calls involving a dead process return an error instead of hanging or aborting the job;
- `Revoke` makes a communicator fail on every member, which wakes blocked ranks;
- `Shrink` returns a communicator without the dead ranks, identical on every survivor.

## 3. How long until serving resumes

- **New requests:** served 0.16–0.34 s after the kill (vs. a restart of minutes), per the chart.
- **Recovery itself:** kill → all survivors resumed in 0.14 s.
- **Requests already running when the rank is killed:**

| Killed while… | Result |
|---|---|
| requests generate tokens | all finish within 1.4–3.1 s |
| GPUs run the MoE all-to-all (7 runs) | all finish within 2.6–3.2 s |
| new prompts are processed, full load | 248 of 248 finish within 4.6 s |

- Times include generating each request's remaining tokens.
- Tokens match a reference run that never used the lost rank (also with prefix caching on). At
  64 requests per GPU, greedy tokens are not reproducible even between two runs without a kill,
  so there only completion and coherence are checked.
- Throughput with FT on: −2.4% at 8 requests per GPU, −3.7% at 64 (chart). After a loss: 98% per
  remaining GPU. FT built in but off: no change (kernels instruction-identical to main).
- At 64 requests per GPU, the same kills pass: new requests served after 0.31 s, in-flight
  requests done within 4.6 s (5.5 s at full load).

## 4. When we keep the captured CUDA graphs, and how

**When:**
- a process is lost (not a GPU, link or driver failure);
- every expert of the lost rank has a copy on another rank (static EPLB with spare slots; at
  EP32, R1 has 8 experts + 1 spare slot per GPU);
- it is the first loss, and not rank 0;
- the configuration is supported (section 7).

Otherwise the job ends as today, with an explicit error on every request.

**How:** a graph bakes in kernel arguments and addresses, not memory contents.
- **Fixed:** EP size, rank ids, graph shapes, buffer layout, all addresses. The lost rank keeps its
  seat; its slots just stay empty.
- **Rewritten in place, between steps:**
  - EPLB routing tables → the lost rank's experts point at their copies;
  - the live-rank mask, read by dispatch and combine;
  - the abort flag and each rank's workspace counters, reset.
- **Host side** (outside graphs): collectives move to the survivor communicator; the lost rank's
  entry in each gather is filled with a neutral value.
- nsys confirms every decode all-to-all still runs inside the original graphs.

## 5. What makes it safe

- **Late writes.** A rank can die with NVLink writes still in flight into peers' buffers. Main's
  buffer layout moves every step, so such a write could land on a live rank's rows. FT gives each
  source rank a fixed slot (section 6).
- **Spinning kernels.** The abort flag lives in pinned host memory and is set by a plain CPU
  write, so it reaches kernels that occupy the whole GPU, graph-launched or not. A real timeout
  still traps.
- **Garbage data.** An aborted combine never marks its output ready. Rows from masked ranks, or of
  an aborted step, reach the expert GEMMs as expert id −1 with zero data.
- **No partial output.** A commit gate checks the abort flag before a step updates any request;
  an aborted step is discarded and run again.
- **KV cache and prefix caching.** The discarded step's KV reservations are undone after its GPU
  work finishes (KV cache managers V1 and V2). Its blocks are never offered for reuse.
- **No lost or duplicated responses.** An interrupted response gather is resent.
- **All-or-nothing switch.** Only the authority's Commit changes membership, after every survivor
  reports the same expert placement.
- **Bounded.** Every wait has a deadline; any failure ends the job as today. A rank whose own loop
  crashes while the others can recover exits alone.

## 6. Symmetric memory and kernel changes

The all-to-all workspace is symmetric memory: each GPU allocates a buffer that every peer maps
over NVLink. **Dispatch** pushes tokens into the receiving GPU's buffer; **combine** pulls expert
outputs from peers' buffers. Completion flags live in the same buffers.

**Who owns which rows of one GPU's dispatch buffer** (one payload shown; each digit is the
source rank that writes that row). Main sizes each source's rows by
`runtime_max_tokens_per_rank`, the largest local batch across ranks this step; FT sizes them by
the fixed `max_num_tokens_per_rank`.

```text
row                0 1 2 3 4 5 6 7 8 9 10 11

Main, step n       0 0 0 0 1 1 1 1 2 2 2  2     runtime_max_tokens_per_rank = 4
Main, step n+1     0 0 1 1 2 2 3 3 4 4 5  5     runtime_max_tokens_per_rank = 2
                                   ^
                                   late write from dead rank 2, aimed at row 8 in step n,
                                   lands in rank 4's rows in step n+1

FT, every step     0 0 0 0 1 1 1 1 2 2 2  2     max_num_tokens_per_rank = 4 (fixed):
                                                rank 2's late writes stay in rank 2's rows
```

- FT also stores each source's payloads together: `[source][payload][max_num_tokens_per_rank]`
  versus main's `[payload][source][runtime_max_tokens_per_rank]`.
- Same workspace size, plus a small alignment pad per source slot: main already reserves room for
  the largest possible `runtime_max_tokens_per_rank`.
- **New gather kernel (FT only):** the fixed slots are not contiguous per payload, so it packs the
  valid rows into the dense `[source][runtime_max_tokens_per_rank]` input the expert GEMMs
  expect. It overlaps with dispatch (programmatic dependent launch). Combine needs no copy
  because it pulls.
- **Dispatch and combine (FT path):** read the descriptor (live-rank mask + abort-flag pointer)
  at a fixed address; skip masked ranks in every wait and load; exit spin loops on abort.
- **FT off:** the same kernels without the descriptor, instruction-identical to main.

**Alternatives considered:**
- pull-based dispatch: a round trip per MoE layer and a rewrite of main's hot kernel;
- strided expert inputs: would change every MoE backend;
- recapturing graphs after a loss: slow, and graphs captured with fewer ranks would not match.

## 7. Limits

- **Biggest gap: FT runs at about half of production throughput.** It does not support the
  overlap scheduler or MTP yet; with both, the same 32 GPUs reach 157k tokens/s versus 80k
  (chart).
- **Not supported yet** (rejected at startup): overlap scheduler, MTP and guided decoding, online
  EPLB, disaggregated serving, LM-head TP in ADP, beam search, PEFT, PP/CP, encoder-decoder,
  hybrid/Mamba, the CFT all-to-all mode, Ray and RPC executors.
- **Other limits:** one loss per job; rank 0 must survive; the lost rank's requests fail rather than
  move; the server keeps reporting "healthy".

Launch and configuration: `docs/source/features/wide-ep-fault-tolerance.md`.

## 8. PRs

One commit per PR, in merge order. The first four are standalone fixes.

| # | Commit | PR |
|---|---|---|
| 1 | [bf3791cb9c](https://github.com/arusli-nv/TensorRT-LLM/commit/bf3791cb9c) | `[fix]` propagate fatal RPC worker errors |
| 2 | [b0bb2786ad](https://github.com/arusli-nv/TensorRT-LLM/commit/b0bb2786ad) | `[fix]` bound RPC socket teardown with unreachable peers |
| 3 | [6f7331dc5e](https://github.com/arusli-nv/TensorRT-LLM/commit/6f7331dc5e) | `[fix]` skip request updates without a sampling result |
| 4 | [82e568a236](https://github.com/arusli-nv/TensorRT-LLM/commit/82e568a236) | `[fix]` keep EPGroupHealth identity across ConfigurableMoE deepcopy |
| 5 | [200602863d](https://github.com/arusli-nv/TensorRT-LLM/commit/200602863d) | `[infra]` upgrade mpi4py to 4.1.2 |
| 6 | [c989b2bc8a](https://github.com/arusli-nv/TensorRT-LLM/commit/c989b2bc8a) | `[feat]` WideEP recovery authority |
| 7 | [7de2a38e53](https://github.com/arusli-nv/TensorRT-LLM/commit/7de2a38e53) | `[feat]` ULFM survivor communicators for host collectives |
| 8 | [bf6ff93509](https://github.com/arusli-nv/TensorRT-LLM/commit/bf6ff93509) | `[feat]` fault-tolerant execution descriptor for the NVLinkOneSided all-to-all |
| 9 | [948efcc0e9](https://github.com/arusli-nv/TensorRT-LLM/commit/948efcc0e9) | `[feat]` MoE execution state for WideEP recovery |
| 10 | [c4354a62f5](https://github.com/arusli-nv/TensorRT-LLM/commit/c4354a62f5) | `[feat]` resident-replica EPLB placement after a rank loss |
| 11 | [7abbf48a36](https://github.com/arusli-nv/TensorRT-LLM/commit/7abbf48a36) | `[feat]` let attention-DP routing skip absent ranks |
| 12 | [68524ed0c1](https://github.com/arusli-nv/TensorRT-LLM/commit/68524ed0c1) | `[feat]` keep the MPI session serving after a peer loss |
| 13 | [80f188e1c7](https://github.com/arusli-nv/TensorRT-LLM/commit/80f188e1c7) | `[feat]` recover the executor from a WideEP peer loss, with the docs page |

## 9. Status against the Phase 1 work items

### Reused merged foundations

| ID | Already on main | Used for |
|---|---|---|
| 1a.1 (#13302) | process-local rank-health mask and generation | source of the live-rank mask |
| 1a.2 (#13404) | rank masking in the all-to-all kernels, at launch time | extended to run-time masking inside CUDA graphs |
| 1a.3, 1a.4 (#15524) | Python binding, detection-only watchdog | kept |
| 1b.1, 1b.2 (#15525) | mask-only EPLB API; rejects zero-replica outcomes | routing around the lost rank |
| 1c.1 (#15677) | EP error classification | kept |
| 1d.0 (#14160) | no abort or parent kill from signal handlers | survivors stay alive |

### New in this work

| ID | Delivered | Commits |
|---|---|---|
| 1a.8 | recoverable abort: run-time descriptor, host-memory abort flag | [8][c8], [4][c4] |
| 1a.8a | late-write safety (non-CFT): fixed per-source slots, gather kernel | [8][c8], [9][c9] |
| 1a.8b | abort, rearm and mask install for the bound workspaces | [9][c9] |
| 1a.11 | rank-mask execution inside CUDA graphs (main rejected it) | [8][c8] |
| 1b.2a | coverage check at startup and at loss time; authority admission | [10][c10], [6][c6] |
| 1b.4a | fixed-N logical/physical mapping; fixed-size gathers | [10][c10], [7][c7], [11][c11] |
| 1c.3 | MPI failure transport via ULFM (mpi4py 4.1.2) | [5][c5], [7][c7] |
| 1c.3a | ULFM runtime adapter; survivor–authority protocol | [7][c7], [6][c6] |
| 1c.4, 1c.4c | engine hook: commit gate, KV rollback, response resend, retryable errors | [13][c13], [3][c3] |
| 1c.4a | survivor communicator; attention-DP routing over survivors | [7][c7], [11][c11] |
| 1c.4b | external recovery authority as sole coordinator | [6][c6] |
| 1d.1 | degraded MPI session, bounded teardown, startup config checks | [12][c12], [1][c1], [2][c2], [13][c13] |
| 1d.4a | multi-node kill qualification; 4-GPU CI test | [13][c13] |
| 1d.5 | overhead and degraded throughput measured (section 3) | — |

### Validated configurations

- **Models and scale:** DeepSeek-R1 NVFP4 on 16, 32 and 64 GB200 GPUs; Kimi-K2 NVFP4 on 32.
- **Serving config:** attention DP, expert TP 1, static EPLB with spare slots, NVLinkOneSided
  (non-CFT), CUTEDSL MoE backend, decode CUDA graphs, non-overlap scheduler, KV cache managers V1
  and V2, chunked prefill, prefix caching on and off, fp8 KV cache and low-precision combine at 64
  requests per GPU.
- **Failures:** SIGKILL of one rank (1, 2, 30 or 62, including other nodes) while idle, while
  generating, during the MoE all-to-all, during prefill, and under full load. An uncovered loss
  fails closed.
- **Tests:** unit tests in the CPU and GPU pre-merge stages; a 4-GPU GB200 integration test
  post-merge.

### Remaining Phase 1 qualification

- **CI image:** needs Open MPI with ULFM and mpi4py 4.1.2; until then the ULFM tests skip.
- **Production config:** overlap scheduler and MTP (FT halves throughput without them), online
  EPLB (1b.3).
- **1d.3:** health reporting (the server reports "healthy" while degraded).
- **1d.7:** v1 qualification, including other MoE and attention backends and long soak runs.
- **Not started:** 1a.6 (TwoSided), 1a.7 (NCCL), 1b.7 (N−K placement with missing experts),
  1c.2 (per-rank evidence budgets).

### Follow-ups beyond Phase 1

CFT, Ray, disaggregated serving, repeated failures (1c.6), restore to N ranks.

[c1]: https://github.com/arusli-nv/TensorRT-LLM/commit/bf3791cb9c
[c2]: https://github.com/arusli-nv/TensorRT-LLM/commit/b0bb2786ad
[c3]: https://github.com/arusli-nv/TensorRT-LLM/commit/6f7331dc5e
[c4]: https://github.com/arusli-nv/TensorRT-LLM/commit/82e568a236
[c5]: https://github.com/arusli-nv/TensorRT-LLM/commit/200602863d
[c6]: https://github.com/arusli-nv/TensorRT-LLM/commit/c989b2bc8a
[c7]: https://github.com/arusli-nv/TensorRT-LLM/commit/7de2a38e53
[c8]: https://github.com/arusli-nv/TensorRT-LLM/commit/bf6ff93509
[c9]: https://github.com/arusli-nv/TensorRT-LLM/commit/948efcc0e9
[c10]: https://github.com/arusli-nv/TensorRT-LLM/commit/c4354a62f5
[c11]: https://github.com/arusli-nv/TensorRT-LLM/commit/7abbf48a36
[c12]: https://github.com/arusli-nv/TensorRT-LLM/commit/68524ed0c1
[c13]: https://github.com/arusli-nv/TensorRT-LLM/commit/80f188e1c7

[authority]: https://github.com/arusli-nv/TensorRT-LLM/blob/80f188e1c7/tensorrt_llm/executor/recovery/authority.py
[communicator]: https://github.com/arusli-nv/TensorRT-LLM/blob/80f188e1c7/tensorrt_llm/_torch/distributed/communicator.py
[kernels]: https://github.com/arusli-nv/TensorRT-LLM/blob/80f188e1c7/cpp/tensorrt_llm/kernels/moe/communication/moeAlltoAllKernels.cu
[op]: https://github.com/arusli-nv/TensorRT-LLM/blob/80f188e1c7/cpp/tensorrt_llm/thop/moe/communication/moeAlltoAllOp.cpp
[state]: https://github.com/arusli-nv/TensorRT-LLM/blob/80f188e1c7/tensorrt_llm/_torch/moe/fused_moe/communication/execution_state.py
[nvlink]: https://github.com/arusli-nv/TensorRT-LLM/blob/80f188e1c7/tensorrt_llm/_torch/moe/fused_moe/communication/nvlink_one_sided.py
[eplb]: https://github.com/arusli-nv/TensorRT-LLM/blob/80f188e1c7/tensorrt_llm/_torch/moe/fused_moe/moe_load_balancer.py
[adp]: https://github.com/arusli-nv/TensorRT-LLM/blob/80f188e1c7/tensorrt_llm/_torch/pyexecutor/scheduler/adp_router.py
[session]: https://github.com/arusli-nv/TensorRT-LLM/blob/80f188e1c7/tensorrt_llm/llmapi/mpi_session.py
[recovery]: https://github.com/arusli-nv/TensorRT-LLM/blob/80f188e1c7/tensorrt_llm/_torch/pyexecutor/peer_loss_recovery.py
[executor]: https://github.com/arusli-nv/TensorRT-LLM/blob/80f188e1c7/tensorrt_llm/_torch/pyexecutor/py_executor.py
[kv]: https://github.com/arusli-nv/TensorRT-LLM/blob/80f188e1c7/tensorrt_llm/_torch/pyexecutor/resource_manager.py
[docs]: https://github.com/arusli-nv/TensorRT-LLM/blob/80f188e1c7/docs/source/features/wide-ep-fault-tolerance.md
[ci]: https://github.com/arusli-nv/TensorRT-LLM/blob/80f188e1c7/tests/integration/defs/wide_ep_ft/test_peer_loss_recovery.py

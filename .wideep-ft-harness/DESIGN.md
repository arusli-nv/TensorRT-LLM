# [WIP] WideEP Fault Tolerance

## Overview

- Today, losing one worker process (rank) of a WideEP job kills the whole job, and a restart takes
  minutes.
- This MVP (MPI, non-CFT), an initial proof of concept, survives the loss of one rank other than
  rank 0, provided that rank's experts have replicas on other GPUs. The surviving ranks keep
  serving **on the CUDA graphs captured at startup** and route the lost rank's experts to their
  replicas.
- Survivors' in-flight requests continue from their last token, with correct output; only the
  interrupted step is redone. Only the lost rank's own requests fail, because their KV cache was
  lost; they end with an explicit error asking the client to retry.

### Reused merged foundations

| ID | Already on main | Used for |
|---|---|---|
| 1a.1 (#13302) | `EPGroupHealth`: process-local rank-health mask and generation | the committed live-rank mask |
| 1a.2 (#13404) | rank masking in the all-to-all kernels, at launch time | extended to run-time masking inside graphs |
| 1a.3, 1a.4 (#15524) | Python binding, detection-only watchdog | kept |
| 1b.1, 1b.2 (#15525) | mask-only EPLB API; rejects zero-replica outcomes | routing around the lost rank |
| 1c.1 (#15677) | EP error classification | kept |
| 1d.0 (#14160) | no abort or parent kill from signal handlers | survivors stay alive |

### v7 draft PRs

None is built on; each is either outside the MVP path or replaced by a graph-compatible design.

| ID | Draft | Used? | Why |
|---|---|---|---|
| 1c.3 (#15785) | MPI failure-evidence transport (gossip on a dedicated communicator) | no; plugs in later | one admitted loss needs only the ULFM watcher; see the boundary below |
| 1a.8 (#15895) | cooperative all-to-all abort: abort epoch as a kernel argument | no; commit 5 replaces it | a per-call argument is frozen by CUDA graphs, so it keeps the capture guard; ours latches a descriptor address. Worth reusing: its abort diagnostics and C++ kernel test |
| 1a.7 (#15789) | NCCL abort and survivor re-init | no | the MVP path uses MPI and the NVLink all-to-all, no NCCL group with the lost rank |
| 1d.3 (#15788) | EP rank-health metrics | no | observability is outside the MVP; commit 1 is the same `EPGroupHealth` deepcopy fix |

### New contributions

One commit per PR, in merge order; each commit link shows all its files and tests. 1 is a
standalone fix; 10 depends only on main. Whole series as one diff:
[main...wideep-ft](https://github.com/NVIDIA/TensorRT-LLM/compare/main...arusli-nv:TensorRT-LLM:wideep-ft).

| # | Commit | PR | Work items | Main code |
|---|---|---|---|---|
| 1 | [84fc6524bd](https://github.com/arusli-nv/TensorRT-LLM/commit/84fc6524bd) | `[fix]` keep EPGroupHealth identity across ConfigurableMoE deepcopy | prerequisite (1a.8) | [ep_group_health.py](https://github.com/arusli-nv/TensorRT-LLM/blob/94e41375b0/tensorrt_llm/_torch/moe/fused_moe/ep_group_health.py) |
| 2 | [efc083e9d3](https://github.com/arusli-nv/TensorRT-LLM/commit/efc083e9d3) | `[infra]` upgrade mpi4py to 4.1.2 for ULFM fault tolerance | 1c.3 | [install_mpi4py.sh](https://github.com/arusli-nv/TensorRT-LLM/blob/94e41375b0/docker/common/install_mpi4py.sh) |
| 3 | [7954af7253](https://github.com/arusli-nv/TensorRT-LLM/commit/7954af7253) | `[feat]` add an external recovery authority for WideEP peer loss | 1c.4b, 1c.3a, 1b.2a | [authority.py](https://github.com/arusli-nv/TensorRT-LLM/blob/94e41375b0/tensorrt_llm/executor/recovery/authority.py), [client.py](https://github.com/arusli-nv/TensorRT-LLM/blob/94e41375b0/tensorrt_llm/executor/recovery/client.py), [protocol.py](https://github.com/arusli-nv/TensorRT-LLM/blob/94e41375b0/tensorrt_llm/executor/recovery/protocol.py) |
| 4 | [1fd4629c5c](https://github.com/arusli-nv/TensorRT-LLM/commit/1fd4629c5c) | `[feat]` ULFM survivor communicators for host collectives | 1c.3, 1c.3a, 1c.4a | [communicator.py](https://github.com/arusli-nv/TensorRT-LLM/blob/94e41375b0/tensorrt_llm/_torch/distributed/communicator.py), [_utils.py](https://github.com/arusli-nv/TensorRT-LLM/blob/94e41375b0/tensorrt_llm/_utils.py) |
| 5 | [d6f44ab1d7](https://github.com/arusli-nv/TensorRT-LLM/commit/d6f44ab1d7) | `[feat]` add a run-time rank mask and abort to the NVLinkOneSided all-to-all | 1a.8, 1a.8a, 1a.11 | [moeAlltoAllKernels.cu](https://github.com/arusli-nv/TensorRT-LLM/blob/94e41375b0/cpp/tensorrt_llm/kernels/moe/communication/moeAlltoAllKernels.cu), [moeAlltoAllOp.cpp](https://github.com/arusli-nv/TensorRT-LLM/blob/94e41375b0/cpp/tensorrt_llm/thop/moe/communication/moeAlltoAllOp.cpp), [moeAlltoAllKernels.h](https://github.com/arusli-nv/TensorRT-LLM/blob/94e41375b0/cpp/tensorrt_llm/kernels/moe/communication/moeAlltoAllKernels.h) |
| 6 | [300b5218ef](https://github.com/arusli-nv/TensorRT-LLM/commit/300b5218ef) | `[feat]` bind MoE layers to the all-to-all execution descriptor for recovery | 1a.8a, 1a.8b | [execution_state.py](https://github.com/arusli-nv/TensorRT-LLM/blob/94e41375b0/tensorrt_llm/_torch/moe/fused_moe/communication/execution_state.py), [nvlink_one_sided.py](https://github.com/arusli-nv/TensorRT-LLM/blob/94e41375b0/tensorrt_llm/_torch/moe/fused_moe/communication/nvlink_one_sided.py), [_mnnvl_utils.py](https://github.com/arusli-nv/TensorRT-LLM/blob/94e41375b0/tensorrt_llm/_mnnvl_utils.py) |
| 7 | [f6c4a211da](https://github.com/arusli-nv/TensorRT-LLM/commit/f6c4a211da) | `[feat]` route a lost rank's experts to their replicas after a peer loss | 1b.2a, 1b.4a | [moe_load_balancer.py](https://github.com/arusli-nv/TensorRT-LLM/blob/94e41375b0/tensorrt_llm/_torch/moe/fused_moe/moe_load_balancer.py), [model_engine.py](https://github.com/arusli-nv/TensorRT-LLM/blob/94e41375b0/tensorrt_llm/_torch/pyexecutor/model_engine.py) |
| 8 | [29e3553ea6](https://github.com/arusli-nv/TensorRT-LLM/commit/29e3553ea6) | `[feat]` let attention-DP routing skip absent ranks | 1b.4a, 1c.4a | [adp_router.py](https://github.com/arusli-nv/TensorRT-LLM/blob/94e41375b0/tensorrt_llm/_torch/pyexecutor/scheduler/adp_router.py), [py_executor.py](https://github.com/arusli-nv/TensorRT-LLM/blob/94e41375b0/tensorrt_llm/_torch/pyexecutor/py_executor.py), [request_utils.py](https://github.com/arusli-nv/TensorRT-LLM/blob/94e41375b0/tensorrt_llm/_torch/pyexecutor/request_utils.py) |
| 9 | [701ee950a1](https://github.com/arusli-nv/TensorRT-LLM/commit/701ee950a1) | `[feat]` keep the MPI session serving after a peer loss | 1d.1 | [mpi_session.py](https://github.com/arusli-nv/TensorRT-LLM/blob/94e41375b0/tensorrt_llm/llmapi/mpi_session.py), [moe_load_balancer.py](https://github.com/arusli-nv/TensorRT-LLM/blob/94e41375b0/tensorrt_llm/_torch/moe/fused_moe/moe_load_balancer.py) |
| 10 | [ddd5a7f568](https://github.com/arusli-nv/TensorRT-LLM/commit/ddd5a7f568) | `[feat]` roll back KV Cache Manager V2 allocations of a discarded step | 1c.4c | [kv_cache_manager_v2.py](https://github.com/arusli-nv/TensorRT-LLM/blob/94e41375b0/tensorrt_llm/_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py) |
| 11 | [94e41375b0](https://github.com/arusli-nv/TensorRT-LLM/commit/94e41375b0) | `[feat]` recover the executor from a WideEP peer loss | 1c.4, 1c.4c, 1d.1, 1d.4a | [peer_loss_recovery.py](https://github.com/arusli-nv/TensorRT-LLM/blob/94e41375b0/tensorrt_llm/_torch/pyexecutor/peer_loss_recovery.py), [py_executor.py](https://github.com/arusli-nv/TensorRT-LLM/blob/94e41375b0/tensorrt_llm/_torch/pyexecutor/py_executor.py), [wide-ep-fault-tolerance.md](https://github.com/arusli-nv/TensorRT-LLM/blob/94e41375b0/docs/source/features/wide-ep-fault-tolerance.md) |

### Validated configurations

- **Models and scale:** DeepSeek-R1 NVFP4 on 16, 32 and 64 GB200 GPUs; Kimi-K2 NVFP4 on 32.
- **Serving config:** attention DP, expert TP 1, static EPLB with redundant experts (8 experts + 1
  redundant slot per GPU for R1 at EP32), NVLinkOneSided fence path, CUTEDSL MoE backend, decode CUDA graphs,
  non-overlap scheduler, KV Cache Manager V2, chunked prefill, KV cache reuse on and off, fp8 KV
  cache and low-precision combine at 64 requests per GPU.
- **Failures:** SIGKILL of rank 1, 2, 30 or 62 (including other nodes) while idle, while generating,
  during the MoE all-to-all, during prefill and under full load. An uncovered loss fails closed.
- **Tests:** unit tests in the CPU and GPU pre-merge stages; a 4-GPU GB200 integration test
  (DeepSeek-V3-Lite, kill while idle and while generating) runs post-merge.

![Recovery time and throughput](evidence/results.svg)

| R1, 32 GPUs, rank killed while… | Result |
|---|---|
| no requests run | new requests served after 0.16–0.34 s (restart without FT: 81–173 s) |
| requests generate tokens | all finish within 1.4–3.1 s |
| GPUs run the MoE all-to-all (7 runs) | all finish within 2.6–3.2 s |
| new prompts are processed, full load | 248 of 248 finish within 4.6 s |
| its experts have no copy elsewhere | every request fails with an explicit error within 0.04 s |

- Recovery itself (kill → all survivors resumed): 0.14 s. Tokens match a reference run that never
  used the lost rank. nsys shows every decode all-to-all inside the original graphs.
- Throughput with FT on: −2.4% at 8 requests per GPU, −3.7% at 64. After a loss: 98% per remaining
  GPU. FT built in but off: kernels instruction-identical to main.

### Remaining qualification

- **Scope:** KV Cache Manager V2 only; for disaggregated serving, the Python transceiver only.
- **Kernel cost:** the gather adds 4–14 µs per MoE layer (section below); fusing or removing it is
  open.
- **Production config:** overlap scheduler and MTP are not supported yet; with them the same 32
  GPUs reach 157k tokens/s versus 80k with FT. Online EPLB (1b.3) is not supported.
- **CI:** an image with ULFM Open MPI and mpi4py 4.1.2; until then the ULFM tests skip.
- **1c.3a fencing of a live rank:** the MVP fences only by process death; a hung or unreachable
  rank ends the job through the hang detector (needs live-rank fencing, plus 1c.2 and 1c.3 for
  suspicion).
- **1d.3** health reporting; **1d.7** v1 qualification (other backends, soak runs).
- **Not started:** 1a.6 TwoSided, 1a.7 NCCL, 1b.7 N−K placement, 1c.2 per-rank evidence budgets.
- **Follow-ups:** CFT, Ray, disaggregated serving, repeated failures (1c.6), restore to N ranks.

### Integration boundary with #15785 (1c.3, evidence transport)

Evidence means "rank R looks dead"; only the authority turns it into a membership change.

| Concern | Owner | Never |
|---|---|---|
| detect and spread failure evidence | #15785 `FailureEvidenceTransport`; our ULFM watcher becomes one implementation | changes membership, the rank mask or communicators |
| decide and commit membership | recovery authority (1c.4b) | detects failures itself |
| apply a commit: shrink the host communicator, install the mask and placement | workers (`MPIDist`, MoE layers) | act before the authority's `Install` |

- One handoff: the transport's `FailureDetectedCallback(failed_rank, ...)` makes the worker send
  `Stopped`. Our watcher calls `on_failure()` with no arguments, so a small adapter joins the two.
- Separate communicators: #15785 revokes only its own detection subcommunicator, for terminal
  aborts; recovery revokes only the host-collective communicator, to wake survivors.
- No dependency today. Once #15785 merges, the authority and protocol stay unchanged.

## MVP Architecture

### Integration diagram

```text
[R] reused from main   [N] new in this work

Phase 1: cancel the attempt (each survivor, on its own)
  detect    [N] ULFM watcher, or a host collective failing with a ULFM error
  cancel    [N] abort flag in pinned host memory, read through the graph-visible descriptor
            [N] ULFM Revoke wakes ranks blocked in host collectives
  discard   [N] commit gate drops the step; KV Cache Manager V2 rollback
  report    [N] Stopped → recovery authority (sole CPU coordinator)

Phase 2: install routing and mask (graph replay paused; authority-driven)
  agree     [N] ULFM Shrink → survivor communicator; Propose → Ready with placement digest
  route     [R] mask-only EPLB API: the lost rank's experts → their copies
  mask      [R] EPGroupHealth commits the live mask → [N] written into the descriptor in place
  rearm     [N] reset workspaces and abort flag → Installed

Phase 3: permission to resume
  commit    [N] authority sends Commit only after matching Installed from every survivor
  resume        survivors replay the same CUDA graphs on N−1 ranks
```

- **ULFM** (Open MPI 5, mpi4py 4): calls involving a dead process return an error instead of
  aborting; `Revoke` makes a communicator fail on every member; `Shrink` returns the same
  survivor-only communicator on every survivor.
- **Graph-visible descriptor:** graphs bake in addresses, not contents. Dispatch and combine read
  one small tensor at a fixed address (live-rank mask + abort-flag pointer); after a loss only it
  and the routing tables are rewritten in place. Shapes, rank ids, layout and addresses never
  change.

### Safety features

- **Resume only on permission:** survivors resume only after a Commit for their generation, which
  the authority sends after every survivor reports Installed with the same placement digest.
- **Cancellation reaches spinning kernels:** a CPU write to mapped host memory needs no GPU work,
  so it reaches kernels that occupy the whole GPU. A real timeout still traps.
- **No garbage data:** an aborted combine never marks its output ready; rows of masked ranks or of
  an aborted step reach the expert GEMMs as expert id −1 with zero data.
- **No partial output:** the commit gate discards an aborted step before it updates any request.
- **KV cache and KV cache reuse:** the step's KV reservations are undone after its GPU work finishes;
  its blocks are never offered for reuse.
- **No lost or duplicated responses:** an interrupted response gather is resent.
- **Own crash:** a rank whose own event loop fails (for example a CUDA error on that GPU) while
  the others can recover exits alone, so the survivors recover instead of main's crash handler
  killing every rank.
- **Bounded and fail-closed:** every wait has a deadline; an uncovered loss, a second loss, rank 0's
  loss or any recovery failure ends the job with an explicit error.
- **Late writes:** below.

### Late writes: symmetric memory and kernel design

Each GPU allocates one all-to-all workspace in symmetric memory, mapped by every peer over NVLink.
Dispatch pushes tokens into the receiving GPU's workspace; combine pulls outputs from peers'. A
rank can die with writes still in flight. The hazard is main's **runtime-dependent layout**, not
rank renumbering (rank ids never change): main sizes each source's rows by
`runtime_max_tokens_per_rank`, so rows move between steps.

```text
row                0 1 2 3 4 5 6 7 8 9 10 11      digit = source rank that owns the row

Main, step n       0 0 0 0 1 1 1 1 2 2 2  2     runtime_max_tokens_per_rank = 4
Main, step n+1     0 0 1 1 2 2 3 3 4 4 5  5     runtime_max_tokens_per_rank = 2
                                   ^
                                   late write from dead rank 2, aimed at row 8 in step n,
                                   lands in rank 4's rows in step n+1

FT, every step     0 0 0 0 1 1 1 1 2 2 2  2     max_num_tokens_per_rank = 4 (fixed):
                                                rank 2's late writes stay in rank 2's rows
```

**Everything a source writes into a peer's workspace is confined to storage indexed by that
source**, which later execution neither consumes nor repurposes:

| Written by source k into a peer | Location | After rank k is lost |
|---|---|---|
| dispatch payloads | fixed region k: `[source][payload][max_num_tokens_per_rank]` | gather skips masked sources; region k is never reassigned |
| dispatch and combine completion flags | word k of each flag array | masked sources are never waited on |
| receive counters (two parities) | word k of each parity | not read for masked sources |
| EPLB gathered statistics | row k | ignored for masked sources |
| combine outputs | none (combine pulls) | — |

- **Backing memory stays valid:** bound workspaces are pinned until model teardown, so late writes
  never hit freed or reused memory. Survivors keep the lost rank's mapping but never read it.
- **Verified:** a 2-process test SIGKILLs a rank mid-dispatch at three timings; every byte outside
  rank k's regions above is unchanged, and masked execution then continues correctly.
- **Workspace size:** unchanged except a small alignment pad per source region.
- **Gather kernel (new, FT only):** packs the valid rows of the fixed regions into the dense input
  the expert GEMMs expect; overlapped with dispatch (programmatic dependent launch).
- **Dispatch and combine (FT path):** read the descriptor, skip masked ranks in every wait and load,
  exit spin loops on abort. FT off: instruction-identical to main.
- **Alternatives rejected:** a pull-based dispatch (a round trip per MoE layer, rewrite of main's
  hot kernel); strided expert inputs (changes every MoE backend).

**Benchmarks** (R1, 32 GPUs, nsys on one rank, medians over ~139k MoE-layer executions from CUDA
graphs):

| | Dispatch | Combine | Expert-id step | Dispatch start → combine end |
|---|---|---|---|---|
| FT off, 8 requests/GPU | 13.4 µs | 34.4 µs | sanitize 9.4 µs | 81.7 µs |
| FT on, 8 requests/GPU | 14.6 µs | 36.1 µs | gather 13.8 µs | 88.4 µs (+8%) |
| FT off, 64 requests/GPU | 15.3 µs | 41.7 µs | sanitize 10.6 µs | 125.3 µs |
| FT on, 64 requests/GPU | 16.4 µs | 43.1 µs | gather 24.6 µs | 144.1 µs (+15%) |

- Dispatch and combine changes cost 1–2 µs each (descriptor read, mask checks).
- The gather is the main cost and grows with batch: it replaces main's sanitize kernel but copies
  every row, where main hands the GEMMs a zero-copy view.
- End to end: −2.4% throughput at 8 requests per GPU, −3.7% at 64 (about 19 µs × 58 MoE layers
  per step, plus the commit-gate sync). Fusing or removing the gather is the main optimization.

Launch and configuration: `docs/source/features/wide-ep-fault-tolerance.md`.

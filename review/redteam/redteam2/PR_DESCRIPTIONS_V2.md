# WideEP fault tolerance: PR descriptions (v2, 11 PRs)

**Stack order (each PR stacks on the one before it):**
1/10 → 2/10 → 3/10 → 4/10 → 5/10 → 6/10 → 7a/10 → 7b/10 → 10/10. Wave 2 adds 8/10 and then 9/10, both stacked on 10/10.

- **Core:** 1/10 to 10/10 stand alone. Fault tolerance (FT) recovers a lost process using *empty* extra EPLB slots.
- **Wave 2:** 8/10 adds hung-rank detection and fencing, and 9/10 adds ghost slots and ring host copies. Wave 2 is opened only after the core lands.

**Wave-1 prerequisites** (separate PRs, merged first; numbers TBD):
- SSE stream error events
- MoE fake-op arguments and shared EP health
- mpi4py 4.1.2
- KV cache manager V2 step rollback
- EPLB planner unusable slots
- EPLB load publish
- EPLB host sharer expert lists

**Placeholders:** `#TBD` stands for PR numbers. Build/test status: the final tree v2 cef4670789 was built (build 8114247) and tested in the release container: load-balancer gtest 41/41, unit groups gpu_moe 128, cpu 963, executor 291, eplb_gpu 15, every runnable FT unit file, and the WideEP integration test 12/12 passed. The FT all-to-all files that start ranks with MPI spawn (test_nvlink_one_sided.py, test_moe_comm.py) run under an mpirun launch (job 8120397). Per-PR intermediate trees are not built yet (a build-only job is planned).

**Stack performance (measured on the full stack, v2 cef4670789; each PR alone is not benchmarked).** FT off and FT on run the same wheel and configs (perf-sanity R1 recipe at DEP32, MTP-1), paired by node group; mean cost with its one-sided 95% upper bound (CLAIMS 140). The bar is <1%: every cell meets it except w/o EPLB at 32 req/GPU, where FT costs +2.68% (an FT-specific device cost; a static load balancer without FT costs +0.20%, CLAIMS 141), and offline EPLB at 32 req/GPU, whose bound is +1.27%. Online EPLB's client TPOT is +1.7-1.9% at 32-128 req/GPU with no decode-step cost: its prefill cost is not established and is a next step.

| Config (DeepSeek-R1 NVFP4, DEP32 GB200, ISL/OSL 1k/1k, MTP-1) | Metric | FT off | FT on | Delta [one-sided 95% UB] | n groups |
|---|---|---|---|---|---|
| w/o EPLB, 8 req/GPU | rank-0 decode step ms | 11.904 | 11.894 | -0.08% [+0.04%] | 4 |
| same | client TPOT median ms | 6.36 | 6.50 | +2.08% (unpaired: medians over groups) | 4 |
| w/o EPLB, 32 req/GPU | rank-0 decode step ms | 13.812 | 14.181 | +2.68% [+3.86%] | 4 |
| same | client TPOT median ms | 8.52 | 8.57 | +0.54% (unpaired: medians over groups) | 4 |
| w/o EPLB, 128 req/GPU | rank-0 decode step ms | 24.633 | 24.746 | +0.46% [+0.67%] | 4 |
| same | client TPOT median ms | 19.23 | 19.43 | +1.07% (unpaired: medians over groups) | 4 |
| offline EPLB, 8 req/GPU | rank-0 decode step ms | 12.473 | 12.441 | -0.26% [-0.13%] | 3 |
| same | client TPOT median ms | 6.49 | 6.50 | +0.07% (unpaired: medians over groups) | 3 |
| offline EPLB, 32 req/GPU | rank-0 decode step ms | 14.163 | 14.230 | +0.47% [+1.27%] | 3 |
| same | client TPOT median ms | 8.10 | 8.24 | +1.66% (unpaired: medians over groups) | 3 |
| offline EPLB, 128 req/GPU | rank-0 decode step ms | 24.413 | 24.343 | -0.28% [-0.07%] | 3 |
| same | client TPOT median ms | 16.49 | 16.61 | +0.71% (unpaired: medians over groups) | 3 |
| online EPLB, 8 req/GPU | rank-0 decode step ms | 13.442 | 13.517 | +0.56% [+0.74%] | 4 |
| same | client TPOT median ms | 7.33 | 7.32 | -0.18% (unpaired: medians over groups) | 4 |
| online EPLB, 32 req/GPU | rank-0 decode step ms | 14.619 | 14.624 | +0.03% [+0.78%] | 4 |
| same | client TPOT median ms | 8.84 | 9.01 | +1.93% (unpaired: medians over groups) | 4 |
| online EPLB, 128 req/GPU | rank-0 decode step ms | 22.140 | 22.111 | -0.13% [+0.48%] | 4 |
| same | client TPOT median ms | 17.20 | 17.50 | +1.71% (unpaired: medians over groups) | 4 |

Recovery (CLAIMS 142, 15/15 kills PASS, 128 req/GPU): rank-0 interrupt to resume, rank kill 261-451 ms (one 2785 ms Lustre log-write outlier), node kill 832-1429 ms. Accuracy after a kill during decode: GSM8K 0.9469 +- 0.0062 vs 0.9454 +- 0.0063 (CLAIMS 143).

---

## PR 1/10

**Title:** `[None][feat] WideEP FT 1/10: Add the fault-tolerance config and mode switch`

```markdown
@coderabbitai summary

## Description

**Summary.** Part 1 of 10 of WideEP fault tolerance. It depends on no other PR in the stack and is followed by #TBD (2/10).
- `TLLM_FAULT_TOLERANCE_MODE=1` turns on fault tolerance (detection only). Adding `TLLM_FAULT_TOLERANCE_AUTHORITY` also enables peer-loss recovery.
  - New helpers `fault_tolerance_mode()` and `peer_loss_recovery_enabled()`. The second is cached and raises if the authority address is set without the mode.
- New `tensorrt_llm.executor.recovery` package with a validated `FaultToleranceConfig`, read from the YAML file named by `TLLM_FAULT_TOLERANCE_CONFIG`. Core fields:
  - `extra_slots`: only `empty`, which is also the default
  - `failure_domain`: `rank` or `node`
  - `recovery_timeout_s`
  - plus `failure_domains(hosts, ...)`
- `wide_ep_ft.py` reads the mode through the new helper.
- `trtllm-llmapi-launch` ends with SIGKILL when a worker ends abnormally under recovery, so the survivors see the loss instead of an `MPI_Abort`.

**Design.** The config is environment-driven and is not part of `LlmArgs`. Moving it into `LlmArgs`, with the golden manifest and serve schemas, is a separate follow-up.

**FT-off delta:** a helper call replaces `wide_ep_ft`'s private env read (same result) and the launcher exits through a function (same status); details:
- `wide_ep_ft` calls `fault_tolerance_mode()` instead of its private environment read. The result is the same.
- The launcher exits through `exit_with_worker_status`, which runs `exit $status` exactly as before.
- The cleanup trap's workspace removal moved into a function. Its behaviour is unchanged.

**Performance.** No runtime path changes.

**Limitations / known gaps.**
- On its own, this PR only adds configuration; nothing in product code reads `fault_tolerance_config()`, `extra_slots`, `failure_domain`, `recovery_timeout_s` or `peer_loss_recovery_enabled()` yet. First readers: 3/10 (`peer_loss_recovery_enabled()` in the MoE scheduler and decoder runner), 6/10 (the config fields, for the extra slots and failure domains) and 7a/10 (the executor recovery). Recovery arrives with 4/10–7b/10.

## Test Coverage

- `tests/unittest/executor/test_recovery_config.py` (l0_cpu, pre-merge)
- `tests/unittest/llmapi/test_trtllm_llmapi_launch_exit.py` (l0_cpu, pre-merge): exit status, with FT on and off
- Build/test status: the final stack tree (v2 cef4670789; the stack tip matches it in product code, its test files are trimmed) was built (build 8114247, BUILD_SUCCESS). Unit groups on that tree: gpu_moe 128 passed, cpu 963 passed, executor 291 passed, eplb_gpu 15 passed. The remaining FT unit files, the load-balancer gtest and the integration test are running (results pending: job 8115508). This PR's own intermediate tree is not built yet (a build-only job is planned).

## PR Checklist

Please review the following before submitting your PR:
- PR description clearly explains what and why. If using CodeRabbit's summary, please make sure it makes sense.
- PR Follows [TRT-LLM CODING GUIDELINES](https://github.com/NVIDIA/TensorRT-LLM/blob/main/CODING_GUIDELINES.md) to the best of your knowledge.
- Test cases are provided for new code paths (see [test instructions](https://github.com/NVIDIA/TensorRT-LLM/tree/main/tests#1-how-does-the-ci-work))
- If PR introduces API changes, an appropriate PR label is added - either `api-compatible` or `api-breaking`. For `api-breaking`, include `BREAKING` in the PR title.
- Any new dependencies have been scanned for license and vulnerabilities
- [CODEOWNERS](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.github/CODEOWNERS) updated if ownership changes
- Documentation updated as needed
- Update [tava architecture diagram](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.github/tava_architecture_diagram.md) if there is a significant design change in PR.
- The reviewers assigned automatically/manually are appropriate for the PR.

- [x] Please check this after reviewing the above items as appropriate for this PR.

## GitHub Bot Help

To see a list of available CI bot commands, please comment `/bot help`.

Owners: @NVIDIA/trt-llm-devs @NVIDIA/trt-llm-infra-devs @NVIDIA/trt-llm-moe-devs @NVIDIA/trt-llm-runtime-devs. Size: product +117/−10, tests +142.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

---

## PR 2/10

**Title:** `[None][feat] WideEP FT 2/10: Add rank mask, abort record and fixed receive slots to the NVLinkOneSided all-to-all`

```markdown
@coderabbitai summary

## Description

**Summary.** Part 2 of 10. Follows #TBD (1/10) and is followed by #TBD (3/10). It needs the wave-1 PR for MoE fake-op arguments and shared EP health.
- **Rank mask:** the dispatch and combine kernels read the active-rank mask through a device pointer at execution time. This allows CUDA graphs in rank-mask mode, so the capture refusal is removed.
- **Abort record:** the fence waits poll a pinned-host `MoeA2AAbortState` (requested/status). A released source is marked in the round flag.
- **Fixed receive slots:** each source writes into its own receive slot. One receive CTA per source writes the valid count, compact routing lists and per-source EPLB statistics. A device `expert_to_slot` lookup is added on the fence dispatch.
- **`MoeExecutionState`:** owns the mask and abort records. It handles retain/pin, rearm, request_abort, aborted, install_mask and close, and binds layers through `NVLinkOneSided.bind_execution_state`.
- The watchdog `clear()`s an observed abort and does not watch under CUDA graph capture.
- Clean-up: the unreachable `expert_to_slot` path of the CFT counted-write dispatch kernel is deleted. CFT now rejects `expert_to_slot` (`TLLM_CHECK` and `TORCH_CHECK`).

**Design.** Graphs are captured once and never recaptured. Membership changes only rewrite the device mask record. A source's slot rows are fixed, so a lost peer's late writes stay inside its own slot.

**FT-off delta:** kernel instantiations and workspace layout change (no new work at run time with the mask off); Python adds `None` checks per dispatch/combine. `wide_ep_ft.py` drops the CUDA-graph refusal, which never fires with FT off. Details:
- **Kernels:** the mask-free instantiations compile the new branches away.
  - No receive CTAs run, and rows per source equal the runtime maximum.
  - `wait_round_flag` gets a null abort pointer and still traps on timeout.
  - The fence dispatch is templated on `USE_EXPERT_TO_SLOT` (false when FT is off), which doubles the fence-dispatch instantiations (binary size).
- **Workspace:** `FLAG_VAL` reserves 8 bytes instead of 4, so later workspace offsets shift.
- **Op:** the expert-id and sanitize checks are restructured but equivalent for existing callers. `moe_a2a_initialize` gains `barrier=True`, which keeps today's behaviour.
- **Python, per dispatch/combine:**
  - `state is None` checks and one `kwargs.get`;
  - the CUDA-graph capture rejection call is removed;
  - the watchdog makes no CUDA query unless a shared watchdog exists.

**Performance.** FT-off rows first.
Measured on the full stack only (see the stack performance table); this PR alone is not benchmarked.

**Limitations / known gaps.**
- `MoeExecutionState` has no in-tree caller until 3/10, where the decoder runner binds it. It is tested directly here.
- There are no C++ gtests for the a2a kernels; the GPU unit tests below cover them.
- Known gaps (fix planned in a follow-up patch set before opening):
  - the fake `moe_a2a_dispatch` returns `[ep, runtime_max_tokens_per_rank, ...]`, but with fixed receive slots (`source_valid_counts`) the real op returns `MAX_NUM_TOKENS` rows per source, so the fake output shape does not match;
  - `abort_state` is declared read-only in the op schema (`Tensor?`), but the kernels write its status.

## Test Coverage

- l0_b200, pre-merge (all new files):
  - `tests/unittest/_torch/moe/test_moe_a2a_abort.py`: cancel/exclusion on replay, per-source valid counts, late arrivals, released sources, compact routing, slot-table replay, option checks, fence-timeout and corrupt-count traps
  - `tests/unittest/_torch/moe/test_moe_execution_state.py`: abort/rearm/exclusion after a peer is lost mid-collective; bind and pin; membership checks; detection-only mask publish
- `tests/unittest/_torch/moe/multi_gpu/test_moe_a2a_late_writes.py::test_killed_peer_writes_stay_in_its_storage` (l0_gb200_multi_gpus, pre-merge)
- `tests/unittest/_torch/modules/test_alltoall_watchdog.py` (cpu_only): clear(), no watch under capture, no CUDA query without a watchdog
- `tests/unittest/_torch/moe/test_moe_comm.py`: existing tests adapted to the device mask
- Build/test status: the final stack tree (v2 cef4670789; the stack tip matches it in product code, its test files are trimmed) was built (build 8114247, BUILD_SUCCESS). Unit groups on that tree: gpu_moe 128 passed, cpu 963 passed, executor 291 passed, eplb_gpu 15 passed. The remaining FT unit files, the load-balancer gtest and the integration test are running (results pending: job 8115508). This PR's own intermediate tree is not built yet (a build-only job is planned).

## PR Checklist

Please review the following before submitting your PR:
- PR description clearly explains what and why. If using CodeRabbit's summary, please make sure it makes sense.
- PR Follows [TRT-LLM CODING GUIDELINES](https://github.com/NVIDIA/TensorRT-LLM/blob/main/CODING_GUIDELINES.md) to the best of your knowledge.
- Test cases are provided for new code paths (see [test instructions](https://github.com/NVIDIA/TensorRT-LLM/tree/main/tests#1-how-does-the-ci-work))
- If PR introduces API changes, an appropriate PR label is added - either `api-compatible` or `api-breaking`. For `api-breaking`, include `BREAKING` in the PR title.
- Any new dependencies have been scanned for license and vulnerabilities
- [CODEOWNERS](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.github/CODEOWNERS) updated if ownership changes
- Documentation updated as needed
- Update [tava architecture diagram](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.github/tava_architecture_diagram.md) if there is a significant design change in PR.
- The reviewers assigned automatically/manually are appropriate for the PR.

- [x] Please check this after reviewing the above items as appropriate for this PR.

## GitHub Bot Help

To see a list of available CI bot commands, please comment `/bot help`.

Owners: @NVIDIA/trt-llm-devs @NVIDIA/trt-llm-infra-devs @NVIDIA/trt-llm-moe-devs @NVIDIA/trt-llm-multi-gpu-ci-review @NVIDIA/trt-llm-runtime-devs. C++: yes. Size: product +1084/−148, tests +993/−22.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

---

## PR 3/10

**Title:** `[None][feat] WideEP FT 3/10: Read all-to-all receive slots in place in the Cutlass and CuteDSL MoE`

```markdown
@coderabbitai summary

## Description

**Summary.** Part 3 of 10. Follows #TBD (2/10) and is followed by #TBD (4/10).
- **Cutlass:** `MoeInputSourceSlots` with an `expandInputRowsFromSourceSlotsKernel`, a `runMoe` overload, and `fused_moe(..., input_source_tokens, input_slot_tokens)`.
- **CuteDSL NVFP4:** the FC1 gather reads slot rows. The tuning key includes the slot geometry.
- **Scheduler:** backends that can read receive slots get them in the comm plan. Other backends get a compact copy (`DispatchSourceSlots.compact`). `do_finalize=False` is refused together with slots.
- The decoder runner binds `MoeExecutionState.for_model(model)` when peer-loss recovery is on. The engine exposes it as `moe_execution_state` and closes it on teardown.

**Design.** Reading the receive slots in place avoids a per-layer copy. The compact copy keeps the other backends correct.

**FT-off delta:** a second (unused) Cutlass expand-kernel instantiation, two unused CuteDSL launch arguments and key entries, and one cached env read at scheduler and runner init; details:
- **Cutlass:**
  - `expandInputRowsKernel` wraps the same body with `READ_SOURCE_SLOTS=false`, which adds a second instantiation per type (binary size).
  - The old `runMoe` signature forwards inline with empty slots.
  - `num_rows` is taken from `token_selected_experts`, which equals the input rows when FT is off.
- **CuteDSL:**
  - The tuning and compile-cache keys gain a constant.
  - The kernel takes two unused Int32 launch arguments.
- **Scheduler:** one cached `peer_loss_recovery_enabled()` read at init. Every new branch is skipped.
- **Runner:** one cached read at init, and a `None` check at teardown.

**Performance.**
Measured on the full stack only (see the stack performance table); this PR alone is not benchmarked.

**Limitations / known gaps.**
- Only Cutlass, for unquantized layers without LoRA, and CuteDSL NVFP4 read the receive slots in place. Other backends pay for the copy.
- Known gaps (fix planned in a follow-up patch set before opening):
  - the `do_finalize=False` refusal with receive slots fires after dispatch, so the peers can hang in combine. It is an unsupported configuration, and the check will move to construction;
  - the CuteDSL tuning key includes the per-iteration `runtime_max_tokens_per_rank`, so with FT on the tuning cache grows with each distinct value.

## Test Coverage

- `tests/unittest/_torch/moe/test_moe_slot_gather.py` (new; l0_b200, pre-merge):
  - FC1 slot reads match a compact copy
  - Cutlass BF16 slot reads
  - inconsistent slots are rejected
  - the scheduler chooses between slots and a compact copy
- `tests/unittest/_torch/thop/parallel/test_cute_dsl_moe.py::test_gather_helper_source_slot_rows` (l0_b200)
- Build/test status: the final stack tree (v2 cef4670789; the stack tip matches it in product code, its test files are trimmed) was built (build 8114247, BUILD_SUCCESS). Unit groups on that tree: gpu_moe 128 passed, cpu 963 passed, executor 291 passed, eplb_gpu 15 passed. The remaining FT unit files, the load-balancer gtest and the integration test are running (results pending: job 8115508). This PR's own intermediate tree is not built yet (a build-only job is planned).

## PR Checklist

Please review the following before submitting your PR:
- PR description clearly explains what and why. If using CodeRabbit's summary, please make sure it makes sense.
- PR Follows [TRT-LLM CODING GUIDELINES](https://github.com/NVIDIA/TensorRT-LLM/blob/main/CODING_GUIDELINES.md) to the best of your knowledge.
- Test cases are provided for new code paths (see [test instructions](https://github.com/NVIDIA/TensorRT-LLM/tree/main/tests#1-how-does-the-ci-work))
- If PR introduces API changes, an appropriate PR label is added - either `api-compatible` or `api-breaking`. For `api-breaking`, include `BREAKING` in the PR title.
- Any new dependencies have been scanned for license and vulnerabilities
- [CODEOWNERS](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.github/CODEOWNERS) updated if ownership changes
- Documentation updated as needed
- Update [tava architecture diagram](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.github/tava_architecture_diagram.md) if there is a significant design change in PR.
- The reviewers assigned automatically/manually are appropriate for the PR.

- [x] Please check this after reviewing the above items as appropriate for this PR.

## GitHub Bot Help

To see a list of available CI bot commands, please comment `/bot help`.

Owners: @NVIDIA/trt-llm-devs @NVIDIA/trt-llm-disagg-devs @NVIDIA/trt-llm-infra-devs @NVIDIA/trt-llm-moe-devs @NVIDIA/trt-llm-runtime-devs. C++: yes. Size: product +484/−73, tests +237.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

---

## PR 4/10

**Title:** `[None][feat] WideEP FT 4/10: Add ULFM survivor communicators and a bounded MPI Finalize`

```markdown
@coderabbitai summary

## Description

**Summary.** Part 4 of 10. Follows #TBD (3/10) and is followed by #TBD (5/10). It needs the wave-1 PR that moves to mpi4py 4.1.2.
- **Interrupting host collectives:** under recovery, host collectives raise `HostCollectiveInterrupted` when a peer fails.
- **`PeerFailureWatch`:** a native nanobind thread that detects a failure, arms the MoE abort and revokes.
- **`MPIDist` survivor operations:** `enable_peer_loss_recovery` checks ULFM, `MPI_THREAD_MULTIPLE` and the pure attention-DP layout. Also adds revoke, shrink, survivor watch, and broadcast/barrier among the survivors.
- **Degraded MPI session:** after a rank departs, the session accepts only shutdown, then leaves with a bounded `MPI_Finalize` or an Abort. The mgmn leader and worker nodes exit through `exit_fault_tolerant_process()` in `tensorrt_llm/llmapi/mpi_session.py`.

**Design.** Survivors agree on the failed set through ULFM shrink, so every survivor sees the same membership.

**FT-off delta:** one `None` check per MPIDist host collective plus cached env reads in the MPI session; details:
- Every `MPIDist` host collective does one `_host_world is None` check.
- broadcast, allreduce and broadcast_int64 take their communicator from `_world_comm()`, which returns `mpi_comm()`.
- **mpi_session:**
  - a cached `num_failed_ranks()` read per task and at shutdown;
  - the task-end barrier is wrapped in a try/except;
  - the async error callback is a partial.
- mgmn nodes call `exit_fault_tolerant_process()`, which returns at once.
- The nanobind module registers `PeerFailureWatch`.

**Performance.**
Measured on the full stack only (see the stack performance table); this PR alone is not benchmarked.

ULFM needs ob1 over TCP for host collectives. Its cost is part of the FT-on column.

**Limitations / known gaps.**
- The survivor APIs (`enable_peer_loss_recovery`, `install_survivors`, `live_ranks`, survivor watch and broadcast/barrier, departed gathers) have no in-tree caller until 7a/10, the executor recovery, which is their first caller. They are tested directly here.
- CI has no ULFM-enabled MPI image, so the ULFM test skips (`TLLM_REQUIRE_ULFM=1` turns the skip into a failure). An infra follow-up is needed.

## Test Coverage

- `tests/unittest/_torch/distributed/test_mpi_survivors.py` (CPU fakes, cpu_only; l0_cpu via the `unittest/_torch/distributed` entry)
- `tests/unittest/_torch/distributed/test_mpi_survivors_ulfm.py::test_survivors_recover_from_departed_ranks[exit|replan]` (needs a ULFM MPI)
- `tests/unittest/llmapi/test_mpi_session.py` (existing list): degraded world, bounded Finalize
- Build/test status: the final stack tree (v2 cef4670789; the stack tip matches it in product code, its test files are trimmed) was built (build 8114247, BUILD_SUCCESS). Unit groups on that tree: gpu_moe 128 passed, cpu 963 passed, executor 291 passed, eplb_gpu 15 passed. The remaining FT unit files, the load-balancer gtest and the integration test are running (results pending: job 8115508). This PR's own intermediate tree is not built yet (a build-only job is planned).
- `test_mpi_survivors_ulfm.py` cases skip in CI today: the CI image has no ULFM Open MPI and nothing sets `TLLM_REQUIRE_ULFM`. Evidence for them is from manual GB200 runs.

## PR Checklist

Please review the following before submitting your PR:
- PR description clearly explains what and why. If using CodeRabbit's summary, please make sure it makes sense.
- PR Follows [TRT-LLM CODING GUIDELINES](https://github.com/NVIDIA/TensorRT-LLM/blob/main/CODING_GUIDELINES.md) to the best of your knowledge.
- Test cases are provided for new code paths (see [test instructions](https://github.com/NVIDIA/TensorRT-LLM/tree/main/tests#1-how-does-the-ci-work))
- If PR introduces API changes, an appropriate PR label is added - either `api-compatible` or `api-breaking`. For `api-breaking`, include `BREAKING` in the PR title.
- Any new dependencies have been scanned for license and vulnerabilities
- [CODEOWNERS](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.github/CODEOWNERS) updated if ownership changes
- Documentation updated as needed
- Update [tava architecture diagram](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.github/tava_architecture_diagram.md) if there is a significant design change in PR.
- The reviewers assigned automatically/manually are appropriate for the PR.

- [x] Please check this after reviewing the above items as appropriate for this PR.

## GitHub Bot Help

To see a list of available CI bot commands, please comment `/bot help`.

Owners: @NVIDIA/trt-llm-devs @NVIDIA/trt-llm-runtime-devs. C++: yes. Size: product +921/−22, tests +892.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

---

## PR 5/10

**Title:** `[None][feat] WideEP FT 5/10: Add the external recovery authority`

```markdown
@coderabbitai summary

## Description

**Summary.** Part 5 of 10. Follows #TBD (4/10) and is followed by #TBD (6/10).
- `python -m tensorrt_llm.executor.recovery.authority`: a ZeroMQ ROUTER authority with HMAC-signed messages and per-session sequence replay checks.
- **Registration:** the authority admits only failure domains that exclude rank 0. A worker whose settings digest differs is refused.
- **Protocol:** `Stopped` → `Propose` → `Ready` (with a placement digest) → `Install` → `Installed` → `Commit`.
  - The authority re-plans on a further loss before Install.
  - It aborts when a deadline passes.
  - A `Status`/`StatusQuery` observer query is used by `/health` in 10/10.
- **Worker side:** `RecoveryClient`, plus the config fields `authority_timeout_s` and `digest()`.

**Design.** The authority holds no GPU and is not on the serving path. If it stops, the deployment keeps serving, but unprotected.

**FT-off delta:** `ZeroMqQueue.__init__` iterates an empty socket-options dict; nothing else on the default path. Details:
- `ZeroMqQueue.__init__` iterates an empty socket-options dict. Nothing else changes.

**Performance.** Not on the serving path.

**Limitations / known gaps.**
- Messages are authenticated but not encrypted. Bind the authority to a private interface.
- Nothing restarts the authority.
- `RecoveryClient` has no in-tree caller until 7a/10 (the executor recovery), its first caller; `Status`/`StatusQuery` is first used by `/health` in 10/10.
- Known gap (fix planned in a follow-up patch set before opening): the authority protocol has no version field, so a worker and an authority of different versions are not refused at registration.

## Test Coverage

- `tests/unittest/executor/test_recovery_authority.py` (new; l0_cpu, pre-merge): protocol, re-plan, deadlines, replay and foreign-message refusal, wire format
- `tests/unittest/executor/test_recovery_config.py`: timeout order and digest
- Build/test status: the final stack tree (v2 cef4670789; the stack tip matches it in product code, its test files are trimmed) was built (build 8114247, BUILD_SUCCESS). Unit groups on that tree: gpu_moe 128 passed, cpu 963 passed, executor 291 passed, eplb_gpu 15 passed. The remaining FT unit files, the load-balancer gtest and the integration test are running (results pending: job 8115508). This PR's own intermediate tree is not built yet (a build-only job is planned).

## PR Checklist

Please review the following before submitting your PR:
- PR description clearly explains what and why. If using CodeRabbit's summary, please make sure it makes sense.
- PR Follows [TRT-LLM CODING GUIDELINES](https://github.com/NVIDIA/TensorRT-LLM/blob/main/CODING_GUIDELINES.md) to the best of your knowledge.
- Test cases are provided for new code paths (see [test instructions](https://github.com/NVIDIA/TensorRT-LLM/tree/main/tests#1-how-does-the-ci-work))
- If PR introduces API changes, an appropriate PR label is added - either `api-compatible` or `api-breaking`. For `api-breaking`, include `BREAKING` in the PR title.
- Any new dependencies have been scanned for license and vulnerabilities
- [CODEOWNERS](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.github/CODEOWNERS) updated if ownership changes
- Documentation updated as needed
- Update [tava architecture diagram](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.github/tava_architecture_diagram.md) if there is a significant design change in PR.
- The reviewers assigned automatically/manually are appropriate for the PR.

- [x] Please check this after reviewing the above items as appropriate for this PR.

## GitHub Bot Help

To see a list of available CI bot commands, please comment `/bot help`.

Owners: @NVIDIA/trt-llm-devs @NVIDIA/trt-llm-infra-devs @NVIDIA/trt-llm-runtime-devs. Size: product +1036/−5, tests +854.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

---

## PR 6/10

**Title:** `[None][feat] WideEP FT 6/10: Restore a lost rank's experts into empty extra EPLB slots`

```markdown
@coderabbitai summary

## Description

**Summary.** Part 6 of 10. Follows #TBD (5/10) and is followed by #TBD (7a/10). It needs the wave-1 EPLB PRs: planner unusable slots, load publish, and host sharer expert lists.
- **Extra slots:** under peer-loss recovery, each rank gets `ceil(E / survivors) − slots_per_rank` appended *empty* slots, sized for the largest failure domain. They are excluded from routing and from EPLB updates until a restore fills them.
- **Host copies:** every node keeps a host copy of every expert.
  - The `/dev/shm` segment names are per job.
  - Startup checks space and reserves it.
  - Each segment is unlinked once every local rank has mapped it.
  - Pages are prefaulted in the background.
- **Recovery API:** `covers_loss_of`, `recovery_placements` (fills empty slots first, else evicts the most-replicated replica), `placement_digest`, `install_recovery_placement` (C++ `restorePlacement`), and `quiesce`/`reset_update_schedule` for online EPLB.
- **One-slot routing table** (`moe_load_balance_one_slot_table`) for a static placement with one replica per expert.
- **MoE layers:** slot rows, CuteDSL keys for the extra slots, and the dispatch slot table.
- `failure_domains(hosts, failure_domain)` in `pyexecutor/peer_loss_recovery.py`, the rank sets whose loss is recovered (one rank, or every rank of a node other than rank 0's), which sizes the appended slots. The module grows into the executor recovery in 7a/10.

**Design.**
- `restorePlacement` validates every layer before it changes any of them.
- An interrupted EPLB iteration releases the layers it never reached, so the worker cannot hang.

**FT-off delta:** EPLB passes `interrupted=False`, gains `oneSlotPerExpert` routing (true only where main already picks the same kernel) and changed error texts; MoE layers compute the same slot counts through new helpers; details:
- **Python:**
  - `end_iter` passes `interrupted=False`;
  - `finalize_model` joins prefault threads (all `None`);
  - `extra_slots_mode()` returns `None`;
  - the MPI comm setup moves before the shm name read.
- **C++:**
  - the route kernel gets `oneSlotPerExpert`, which is true when FT is off only if slots == experts; main already picks that kernel in that case;
  - `setInitialWeightAssignments` clears an empty unusable-slot set;
  - the update-weights error text changed;
  - `reconfigureMaskOnly` was refactored with the same checks.
- **MoE layers:** `interface.py` computes `expert_size_per_partition` from `slot_count_per_rank`, which gives the same values; the CuteDSL tuning keys gain a 0.

**Performance.**
Measured on the full stack only (see the stack performance table); this PR alone is not benchmarked.

Restore time is not measured on its own; it is inside the interrupt-to-resume times (rank kill 261-451 ms, node kill 832-1429 ms). Device memory held by the empty slots is not measured; the estimate is about 1.4 GiB/GPU for DeepSeek-R1 NVFP4.

**Limitations / known gaps.**
- Without replicas, every node still keeps full host copies of every MoE layer in `/dev/shm`. 9/10 (ring copies) reduces this.
- The recovery API's only caller arrives in 7a/10.

## Test Coverage

- `tests/unittest/_torch/moe/test_resident_placement.py` (new, cpu_only; l0_cpu via the `unittest/_torch/moe` entry): placement rules and restores
- `tests/unittest/_torch/moe/test_moe_load_balancer.py` (l0_b200, pre-merge): shm handling, quiesce, host copies, FT-off guards
- `tests/unittest/_torch/thop/parallel/test_cute_dsl_moe.py`: extra-slot layout keys (l0_b200)
- `tests/unittest/_torch/executor/test_peer_loss_recovery.py::test_failure_domains_are_every_rank_then_every_host_but_rank_0_s` (cpu_only, l0_cpu)
- C++: `MoeLoadBalancerTest.QuiesceReleasesTheLayersAnInterruptedForwardDidNotReach` (guards a real worker hang)
- Build/test status: the final stack tree (v2 cef4670789; the stack tip matches it in product code, its test files are trimmed) was built (build 8114247, BUILD_SUCCESS). Unit groups on that tree: gpu_moe 128 passed, cpu 963 passed, executor 291 passed, eplb_gpu 15 passed. The remaining FT unit files, the load-balancer gtest and the integration test are running (results pending: job 8115508). This PR's own intermediate tree is not built yet (a build-only job is planned).

## PR Checklist

Please review the following before submitting your PR:
- PR description clearly explains what and why. If using CodeRabbit's summary, please make sure it makes sense.
- PR Follows [TRT-LLM CODING GUIDELINES](https://github.com/NVIDIA/TensorRT-LLM/blob/main/CODING_GUIDELINES.md) to the best of your knowledge.
- Test cases are provided for new code paths (see [test instructions](https://github.com/NVIDIA/TensorRT-LLM/tree/main/tests#1-how-does-the-ci-work))
- If PR introduces API changes, an appropriate PR label is added - either `api-compatible` or `api-breaking`. For `api-breaking`, include `BREAKING` in the PR title.
- Any new dependencies have been scanned for license and vulnerabilities
- [CODEOWNERS](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.github/CODEOWNERS) updated if ownership changes
- Documentation updated as needed
- Update [tava architecture diagram](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.github/tava_architecture_diagram.md) if there is a significant design change in PR.
- The reviewers assigned automatically/manually are appropriate for the PR.

- [x] Please check this after reviewing the above items as appropriate for this PR.

## GitHub Bot Help

To see a list of available CI bot commands, please comment `/bot help`.

Owners: @NVIDIA/trt-llm-moe-devs @NVIDIA/trt-llm-runtime-devs. C++: yes. Size: product +1071/−117, tests +1086/−7.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

---

## PR 7a/10

**Title:** `[None][feat] WideEP FT 7a/10: Recover the executor from a lost WideEP rank`

```markdown
@coderabbitai summary

## Description

**Summary.** Part 7a of 10. Follows #TBD (6/10) and is followed by #TBD (7b/10). It needs the wave-1 PR for KV cache manager V2 step rollback.

`PeerLossRecovery` is the executor's recovery hook. When recovery is off, a do-nothing `RecoveryHook` takes its place.
- **Startup checks:** recovery fails closed for any configuration a discarded step cannot re-run. It checks that EPLB slots cover each failure domain and registers with the authority.
- **Step commit/discard:**
  - A step commits only if the MoE abort record is clear.
  - A discard restores each request's state and draft tokens, rolls back logits and additional outputs (`PyResult.checkpoint/rollback`) and reverts the KV cache, target and draft.
- **Recovery loop:**
  - shrink, then the authority protocol, re-planning on a further loss before Install;
  - online EPLB adopts rank 0's placement;
  - the restored placement and rank mask are installed;
  - no CUDA graph is recaptured.
- **Requests:**
  - interrupted response gathers are redelivered;
  - unbroadcast and undelivered requests are re-admitted;
  - a departed rank's request that delivered nothing runs again on a survivor;
  - a streamed one fails retryably (`peer_loss_retryable_message`);
  - a cancelled one ends cancelled.
- **Attention DP with absent ranks:** `departed=` on every per-step tp gather, balancing over the live ranks, and strict pins to a lost rank fail with HTTP 400.
- **Process ends:**
  - A rank whose own error corrupts its CUDA context SIGKILLs itself while the survivors can cover its loss.
  - A loss before serving starts ends the job.
- **Communicator:** departed-aware gathers, `live_ranks` and `install_survivors`.

**Reviewing order.** Start with step commit/discard and the KV cache revert (`commit_step`, `discard_interrupted_steps`, `PreparedStep`, `PyResult.checkpoint/rollback`, `revert_prepare_resources`); then the recovery protocol loop and placement install (`_recover`, `_shrink_again`, `install_recovery_placement`, attention DP with absent ranks); then replay and requests (re-run of a lost rank's undelivered requests, response-gather redelivery, re-admission of unbroadcast or undelivered requests).

**Why one PR.** It stays one PR of about 3.6k lines with tests: commit/discard, the protocol loop and request replay are only correct together. A step discarded without replaying its requests loses them, and a protocol without discard resumes on corrupt state. Split, no part would be testable or safe to enable alone. The quarantine and HTTP 503 path already moved to 7b/10.

**Design.** Survivors may decide differently on one step: a step commits only if every collective it ran read complete data. The executor's own hang detector bounds a recovery, so the hang timeout must exceed three times `recovery_timeout_s`. With recovery on, a pin to a rank that is not serving is failed or rebalanced; with FT off, a pin outside the attention-DP ranks (past the last rank, or negative) indexes the per-rank counts by position, exactly as main's lists do.

**FT-off delta:** the event loop and response publishing go through do-nothing hook methods, tp gathers take a no-op `departed=` keyword, and routing uses rank-keyed dicts with identical results; a pin outside the attention-DP ranks is looked up by position (main's list semantics: IndexError past the last rank, a negative pin counts from the last); details:
- **Event loop and hook:**
  - the loop runs through `RecoveryHook.run`, which just calls it;
  - hook calls are no-ops or guarded by `.enabled`;
  - `publish_responses` keeps the enqueue-then-terminate order.
- **Error paths:** forward and sampling errors check `isinstance(HostCollectiveInterrupted)`.
- **Gathers and routing:**
  - tp gathers take a `departed=` keyword, a no-op without a departed rank;
  - admission capacity uses `len(live_ranks)`, which equals `tp_size`;
  - per-rank active counts are a dict keyed by rank (same routing);
  - the response gather checks for a 3-tuple part.
- **Request queue:** a wake-item check runs on fetched items, and stats skip wake items.
- **Executor creation:** calls `watch_startup_peer_failures(dist)`, which returns at once.
- **Executor teardown on a failed creation:** `stop_startup_watch()` (no-op).
- **Communicator:** `Distributed.__init__` stores the tp ranks; each tp gather checks `_live_ranks is None` and returns as before; `tp_comm` does one more `None` check; `_comm_root(root)` returns `root`.
- **Other:** `_update_v2_context_resources` gains `and not enabled`; decoder inputs keep `tp_cp_allgather_int64` when recovery is off; a pin outside the serving ranks reads a cached env flag and then raises as on main; `exits_alone()` is checked only after a crash.

**Performance.**
Measured on the full stack only (see the stack performance table); this PR alone is not benchmarked.

Recovery time, rank-0 interrupt to resume (128 req/GPU, 15/15 kills): rank kill 261-451 ms (one 2785 ms Lustre log-write outlier), node kill 832-1429 ms.

**Limitations / known gaps.**
- Refused at startup:
  - every layout except pure attention DP (`tp == moe_ep == world`);
  - PP, CP and expert TP;
  - beam search, guided decoding and LM-head TP;
  - KV transceivers and connectors;
  - SWA scratch reuse and KV pool rebalancing;
  - PEFT, encoder-decoder models and sleep/wake;
  - non-MPI executors.
- Rejected per request: occurrence penalties, logits post-processors, and multimodal requests under overlap.
- A request interrupted twice is not quarantined until 7b/10.

## Test Coverage

- `tests/unittest/_torch/executor/test_peer_loss_recovery.py` (new, cpu_only; covered by l0_cpu `unittest/_torch/executor`): protocol order, re-planning, deadlines, online EPLB, commit/discard, KV revert, logits rollback, re-run and redelivery, startup watch, create checks
- `tests/unittest/_torch/executor/test_adp_absent_ranks.py` (new): departed gathers, balancing over survivors, unservable pins, FT-off parity
- `test_executor_request_queue.py::test_wake_releases_a_blocked_get_without_a_request`
- `test_py_executor.py::TestAdpBalanceExcludesPadDummies::test_balancing_continues_over_the_serving_ranks_after_a_loss`
- `test_adp_router.py`, `test_request_utils.py`, `test_adp_host_exchange.py`: adapted to the dict counts
- Communicator: `tests/unittest/_torch/distributed/test_mpi_survivors{,_ulfm}.py` and `test_mpi_session.py` additions
- `test_adp_absent_ranks.py::test_without_peer_loss_recovery_a_pin_past_the_last_rank_raises_as_on_main` and `::test_without_peer_loss_recovery_a_negative_pin_behaves_as_on_main` (FT-off parity of out-of-range pins)
- Build/test status: the final stack tree (v2 cef4670789; the stack tip matches it in product code, its test files are trimmed) was built (build 8114247, BUILD_SUCCESS). Unit groups on that tree: gpu_moe 128 passed, cpu 963 passed, executor 291 passed, eplb_gpu 15 passed. The remaining FT unit files, the load-balancer gtest and the integration test are running (results pending: job 8115508). This PR's own intermediate tree is not built yet (a build-only job is planned).
- End-to-end tests are in 7b/10.

## PR Checklist

Please review the following before submitting your PR:
- PR description clearly explains what and why. If using CodeRabbit's summary, please make sure it makes sense.
- PR Follows [TRT-LLM CODING GUIDELINES](https://github.com/NVIDIA/TensorRT-LLM/blob/main/CODING_GUIDELINES.md) to the best of your knowledge.
- Test cases are provided for new code paths (see [test instructions](https://github.com/NVIDIA/TensorRT-LLM/tree/main/tests#1-how-does-the-ci-work))
- If PR introduces API changes, an appropriate PR label is added - either `api-compatible` or `api-breaking`. For `api-breaking`, include `BREAKING` in the PR title.
- Any new dependencies have been scanned for license and vulnerabilities
- [CODEOWNERS](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.github/CODEOWNERS) updated if ownership changes
- Documentation updated as needed
- Update [tava architecture diagram](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.github/tava_architecture_diagram.md) if there is a significant design change in PR.
- The reviewers assigned automatically/manually are appropriate for the PR.

- [x] Please check this after reviewing the above items as appropriate for this PR.

## GitHub Bot Help

To see a list of available CI bot commands, please comment `/bot help`.

Owners: @NVIDIA/trt-llm-disagg-devs @NVIDIA/trt-llm-kv-cache-manager-devs @NVIDIA/trt-llm-runtime-devs. Size: product +1863/−105, tests +1761/−34.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

---

## PR 7b/10

**Title:** `[None][feat] WideEP FT 7b/10: Quarantine twice-interrupted requests, answer 503, add the peer-loss e2e tests`

```markdown
@coderabbitai summary

## Description

**Summary.** Part 7b of 10. Follows #TBD (7a/10) and is followed by #TBD (10/10). It needs the wave-1 SSE stream error events PR.
- **Quarantine:** a request that a loss interrupts a second time, without committing a step since, fails retryably instead of running again, since it may be what crashes the ranks.
  - Survivors track interrupted requests until one of their steps commits (`requests_cancelled` hook).
  - New `PyExecutor.fail_requests`.
- **Retryable errors over HTTP:** with fault-tolerance mode on (`TLLM_FAULT_TOLERANCE_MODE=1`, so also in detection-only mode), `is_peer_loss_retryable_message` maps to HTTP 503 with `Retry-After: 1` and code `peer_loss_retryable`. Streaming responses end with an SSE `error` event carrying that code, then `[DONE]`.
- **End-to-end tests:** DeepSeek-V3-Lite NVFP4, EP4 on GB200 under `mpirun --with-ft ulfm`, one test per core failure type.
- An interrupted request's committed steps are reported to rank 0 with the next response gather (a fourth entry of each rank's gather part), so a lost rank's progress counts too: a re-run request that progressed on a rank lost later runs again instead of being quarantined.
- The HTTP 503 mapping and the SSE error code key on `fault_tolerance_mode()`, a plain env read that cannot raise inside the error handler (an authority address without the mode is a worker misconfiguration).

**Design.** The error code travels inside the message text, the same way the context-length error crosses the IPC boundary.

**FT-off delta:** one no-op hook call per cancellation pass and one `fault_tolerance_mode()` env read on the error-response and SSE-error paths (both answer as on main); details:
- `_handle_canceled_requests` calls the no-op `requests_cancelled` hook.
- The SSE error event reads `fault_tolerance_mode()` (a plain, uncached env read; False) and answers as before. The OpenAI error response reads it only for a message that is not a context-length error, and answers as before.

**Performance.** No change from 7a/10.

**Limitations / known gaps.**
- The end-to-end tests need a ULFM MPI image and 4× GB200, so they are post-merge only. They skip without ULFM; `TLLM_REQUIRE_ULFM=1` turns the skip into a failure.
- Test lines exceed product lines here because of the required end-to-end tests.

## Test Coverage

- Unit, quarantine: `test_peer_loss_recovery.py`
  - `::test_a_request_two_losses_interrupt_fails_unless_it_progressed_between_them`
  - `::test_a_request_run_again_fails_when_it_loses_its_rank_again`
  - `::test_survivors_fail_the_sequences_of_the_requests_rank0_quarantines`
  - plus three progress-tracking tests
- Unit, HTTP (l0_cpu): `tests/unittest/llmapi/apps/test_stream_error_events.py`
  - `::test_a_peer_loss_failure_is_a_503_with_retry_after`
  - `::test_a_peer_loss_failure_streams_a_retryable_code`
- End to end: `tests/integration/defs/wide_ep_ft/test_peer_loss_recovery.py` (l0_gb200_multi_gpus, post-merge, TIMEOUT 45)
  - `::test_peer_loss_recovery[sync|overlap]`: a rank is killed mid-decode; the lost rank's requests run again; the survivors' greedy tokens match; no graph is recaptured
  - `::test_two_ranks_lost_together`
  - `::test_rank_lost_during_recovery`: re-plan
  - `::test_restore_into_appended_empty_slots`: online EPLB restore from host copies
- `test_peer_loss_recovery.py::test_a_request_run_again_that_reported_progress_runs_again_when_it_loses_its_rank_again`; `::test_a_survivor_reports_the_interrupted_requests_whose_step_committed_once` also checks the progress report
- `test_stream_error_events.py::test_without_fault_tolerance_a_tagged_message_is_an_ordinary_error[None|tcp...]`
- Build/test status: the final stack tree (v2 cef4670789; the stack tip matches it in product code, its test files are trimmed) was built (build 8114247, BUILD_SUCCESS). Unit groups on that tree: gpu_moe 128 passed, cpu 963 passed, executor 291 passed, eplb_gpu 15 passed. The remaining FT unit files, the load-balancer gtest and the integration test are running (results pending: job 8115508). This PR's own intermediate tree is not built yet (a build-only job is planned).
- The `wide_ep_ft/*` end-to-end cases skip in CI today: the CI image has no ULFM Open MPI and nothing sets `TLLM_REQUIRE_ULFM`. Evidence for them is from manual GB200 runs.

## PR Checklist

Please review the following before submitting your PR:
- PR description clearly explains what and why. If using CodeRabbit's summary, please make sure it makes sense.
- PR Follows [TRT-LLM CODING GUIDELINES](https://github.com/NVIDIA/TensorRT-LLM/blob/main/CODING_GUIDELINES.md) to the best of your knowledge.
- Test cases are provided for new code paths (see [test instructions](https://github.com/NVIDIA/TensorRT-LLM/tree/main/tests#1-how-does-the-ci-work))
- If PR introduces API changes, an appropriate PR label is added - either `api-compatible` or `api-breaking`. For `api-breaking`, include `BREAKING` in the PR title.
- Any new dependencies have been scanned for license and vulnerabilities
- [CODEOWNERS](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.github/CODEOWNERS) updated if ownership changes
- Documentation updated as needed
- Update [tava architecture diagram](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.github/tava_architecture_diagram.md) if there is a significant design change in PR.
- The reviewers assigned automatically/manually are appropriate for the PR.

- [x] Please check this after reviewing the above items as appropriate for this PR.

## GitHub Bot Help

To see a list of available CI bot commands, please comment `/bot help`.

Owners: @NVIDIA/trt-llm-devs @NVIDIA/trt-llm-disagg-devs @NVIDIA/trt-llm-infra-devs @NVIDIA/trt-llm-multi-gpu-ci-review @NVIDIA/trt-llm-qa-function @NVIDIA/trt-llm-runtime-devs. Size: product +161/−31, tests +852/−6.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

---

## PR 10/10

**Title:** `[None][feat] WideEP FT 10/10: Report FT states on /health and document WideEP fault tolerance`

```markdown
@coderabbitai summary

## Description

**Summary.** Part 10 of 10 of the core. Follows #TBD (7b/10). Wave 2 (8/10 and 9/10) stacks on it later.
- **`/health` states:**
  - `trtllm-serve` asks the authority for its state (`RecoveryStatusProbe`, cached for 1 s).
  - It returns 200 with `healthy`, `degraded`, `unprotected` or `unknown`, and 503 with `recovering` or `terminal`.
  - The JSON body carries the generation, live and failed ranks, the number of recoveries, the last recovery's stage times and the failed request count. Rank 0 reports that count in `Installed`.
- **Detection-only mode** (mode set, no authority) answers `unprotected`.
- **New guide** `docs/source/features/wide-ep-fault-tolerance.md` (core only), covering requirements, running, settings, coverage, limits, recovery behaviour, monitoring and probes, test maturity and cost. Also linked from `index.rst`, the feature-combination matrix and the wide-EP README.
- An authority phase this server does not know (`unknown`, e.g. another version) keeps `/health` at 200: the server still serves; only `recovering` and `terminal` return 503.
- The guide documents the executor's fixed hang detector (300 s without a step; startup refuses FT timeouts it would cut short), not an unexposed setting, and that `failed_requests` counts the lost ranks' failed requests only.

**Design.** `/health` only reports. Orchestration guidance is in the guide.

**FT-off delta:** one `from_env()` at server init (returns `None`) and a `getattr` plus one env read per `/health` before main's plain 200; details:
- At server init, `RecoveryStatusProbe.from_env()` returns `None`.
- `/health` does a `getattr` and a `fault_tolerance_mode()` env read, then returns main's plain 200.

**Performance.** Not on the serving path. The probe query is bounded at 0.5 s and cached for 1 s.

**Limitations / known gaps.**
- `/health` is unauthenticated, so the reason for an abort appears only in the authority's log.
- Known limits listed in the guide:
  - a loss after the commit, or a commit message that never arrives, ends the job;
  - a further loss during install can leave the survivors on mixed memberships;
  - speculative decoding with overlap can skip one KV update for the rejected draft tokens.
- Known gaps (fix planned in a follow-up patch set before opening):
  - `RecoveryStatusProbe.close()` has no caller, so the probe's thread pool and socket are never closed at server shutdown;
  - guide numbers to be sourced to the final runs before merge, or removed: "16 GiB per rank in our runs", "an estimated 1.4 GiB per GPU", "in milliseconds" / "within milliseconds … well under a second", GSM8K "within two standard errors", 503 "for about a second", and "64 hex characters" (the code accepts 32 or more).

## Test Coverage

- l0_cpu, pre-merge:
  - `tests/unittest/llmapi/apps/test_fault_tolerance_health.py`
  - `tests/unittest/executor/test_recovery_authority.py`: probe and state mapping
- Unit: `test_peer_loss_recovery.py::test_rank0_reports_how_many_requests_the_departed_rank_was_serving`
- End to end: `wide_ep_ft/test_peer_loss_recovery.py::test_authority_lost_while_serving` (l0_gb200_multi_gpus, post-merge): still serving and `unprotected` after the authority dies
- `test_fault_tolerance_health.py::test_health_returns_503_only_while_recovering_or_terminal` (new: `unknown` stays 200)
- Build/test status: the final stack tree (v2 cef4670789; the stack tip matches it in product code, its test files are trimmed) was built (build 8114247, BUILD_SUCCESS). Unit groups on that tree: gpu_moe 128 passed, cpu 963 passed, executor 291 passed, eplb_gpu 15 passed. The remaining FT unit files, the load-balancer gtest and the integration test are running (results pending: job 8115508). This PR's own intermediate tree is not built yet (a build-only job is planned).
- The GB200 integration case (`test_authority_lost_while_serving`) and the ULFM tests skip in CI today: the CI image has no ULFM Open MPI and nothing sets `TLLM_REQUIRE_ULFM`. Evidence for them is from manual GB200 runs.

## PR Checklist

Please review the following before submitting your PR:
- PR description clearly explains what and why. If using CodeRabbit's summary, please make sure it makes sense.
- PR Follows [TRT-LLM CODING GUIDELINES](https://github.com/NVIDIA/TensorRT-LLM/blob/main/CODING_GUIDELINES.md) to the best of your knowledge.
- Test cases are provided for new code paths (see [test instructions](https://github.com/NVIDIA/TensorRT-LLM/tree/main/tests#1-how-does-the-ci-work))
- If PR introduces API changes, an appropriate PR label is added - either `api-compatible` or `api-breaking`. For `api-breaking`, include `BREAKING` in the PR title.
- Any new dependencies have been scanned for license and vulnerabilities
- [CODEOWNERS](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.github/CODEOWNERS) updated if ownership changes
- Documentation updated as needed
- Update [tava architecture diagram](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.github/tava_architecture_diagram.md) if there is a significant design change in PR.
- The reviewers assigned automatically/manually are appropriate for the PR.

- [x] Please check this after reviewing the above items as appropriate for this PR.

## GitHub Bot Help

To see a list of available CI bot commands, please comment `/bot help`.

Owners: @NVIDIA/trt-llm-devs @NVIDIA/trt-llm-doc-owners @NVIDIA/trt-llm-infra-devs @NVIDIA/trt-llm-multi-gpu-ci-review @NVIDIA/trt-llm-qa-function @NVIDIA/trt-llm-runtime-devs. Size: product +184/−1, tests +289/−2, docs +285.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

---

## PR 8/10 (wave 2)

**Title:** `[None][feat] WideEP FT 8/10: Detect and fence hung WideEP ranks`

```markdown
@coderabbitai summary

## Description

**Summary.** Part 8 of 10, the first wave-2 PR. Stacks on #TBD (10/10) and is followed by #TBD (9/10).
- **Hang evidence:** workers report Python heartbeats, with host-wait state and the peers whose all-to-all waits exceed `hang_suspect_s`. A native `HeartbeatSender` (C++, outside the GIL) sends UDP heartbeats with the last finished iteration.
- **Authority `HangMonitor`:**
  - A rank is confirmed hung when every reporting peer has waited on it for `hang_confirm_s` while it finished no step.
  - A stall of every rank, or a hung rank 0, is reported terminal, and workers exit with code 75.
- **Fencing:** a confirmed hung rank is ended, in order, by:
  1. itself;
  2. a peer asking the MPI runtime to kill it (PMIx job control);
  3. the optional `--fence-command`.
- **Kills before agreement:**
  - Before the shrink, each survivor has the runtime kill the ranks its MPI view reports failed.
  - Before Propose, rank 0 kills any failed rank the authority has not fenced.
- **Config:** `hang_detection`, `hang_suspect_s`, `hang_confirm_s`, `heartbeat_s`, `heartbeat_timeout_s`. The shrink timeout includes the hang-resolution time.
- **Guide:** the hang sections.
- The runtime kills before the shrink and rank 0's fence of failed ranks before a proposal run only with `hang_detection`; without it, recovery uses no MPI runtime job control and a stopped process holds the agreement until it ends.

**Design.** A rank is blamed only when all of its peers wait on it. A rank whose own device waits on another rank is never blamed.

**FT-off delta:** two attribute writes around the request queue's blocking get, `in_host_collective = False` at `Distributed` init, and the `HeartbeatSender` binding registered; details:
- The request queue does two attribute writes around its blocking get.
- `Distributed.__init__` sets `in_host_collective = False`.
- The nanobind module registers `HeartbeatSender`.

**Performance.**
Measured on the full stack only (see the stack performance table); this PR alone is not benchmarked.

Time from SIGSTOP to serving again: not measured yet (a next step; the kill rows cover crashes, not hangs).

**Limitations / known gaps.**
- Two ranks hung at once are not confirmed.
- A network partition between the authority and a node looks like a stopped process.
- `hang_confirm_s` must exceed the slowest legitimate step.
- Known gaps (fix planned in a follow-up patch set before opening):
  - the CUDA-graph capture guard is an `assert` on the private `graphs._capture_allowed`, which `python -O` strips;
  - the `--fence-command` `subprocess.Popen` is never reaped.

## Test Coverage

- `tests/unittest/_torch/executor/test_hang_evidence.py` (new; l0_cpu `unittest/_torch/executor`, one GPU-only test)
- `tests/unittest/executor/test_recovery_hang_monitor.py` (new; l0_cpu)
- Unit `test_peer_loss_recovery.py`: hang-evidence start, fence before the plan, kills before the shrink, hang-timeout requirement
- Authority, config and launch-exit (status 75) additions; `test_mpi_survivors_ulfm.py[fenced]`
- End to end: `wide_ep_ft/test_peer_loss_recovery.py::test_hung_rank_fenced` (l0_gb200_multi_gpus, post-merge)
- `test_peer_loss_recovery.py::test_the_ranks_mpi_reports_failed_are_ended_before_the_shrink` gains a `hang_detection=False` case (nothing killed)
- Build/test status: the final stack tree (v2 cef4670789; the stack tip matches it in product code, its test files are trimmed) was built (build 8114247, BUILD_SUCCESS). Unit groups on that tree: gpu_moe 128 passed, cpu 963 passed, executor 291 passed, eplb_gpu 15 passed. The remaining FT unit files, the load-balancer gtest and the integration test are running (results pending: job 8115508). This PR's own intermediate tree is not built yet (a build-only job is planned).
- The GB200 integration case (`test_hung_rank_fenced`) and the ULFM tests (`test_mpi_survivors_ulfm.py[fenced]`) skip in CI today: the CI image has no ULFM Open MPI and nothing sets `TLLM_REQUIRE_ULFM`. Evidence for them is from manual GB200 runs.

## PR Checklist

Please review the following before submitting your PR:
- PR description clearly explains what and why. If using CodeRabbit's summary, please make sure it makes sense.
- PR Follows [TRT-LLM CODING GUIDELINES](https://github.com/NVIDIA/TensorRT-LLM/blob/main/CODING_GUIDELINES.md) to the best of your knowledge.
- Test cases are provided for new code paths (see [test instructions](https://github.com/NVIDIA/TensorRT-LLM/tree/main/tests#1-how-does-the-ci-work))
- If PR introduces API changes, an appropriate PR label is added - either `api-compatible` or `api-breaking`. For `api-breaking`, include `BREAKING` in the PR title.
- Any new dependencies have been scanned for license and vulnerabilities
- [CODEOWNERS](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.github/CODEOWNERS) updated if ownership changes
- Documentation updated as needed
- Update [tava architecture diagram](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.github/tava_architecture_diagram.md) if there is a significant design change in PR.
- The reviewers assigned automatically/manually are appropriate for the PR.

- [x] Please check this after reviewing the above items as appropriate for this PR.

## GitHub Bot Help

To see a list of available CI bot commands, please comment `/bot help`.

Owners: @NVIDIA/trt-llm-devs @NVIDIA/trt-llm-doc-owners @NVIDIA/trt-llm-infra-devs @NVIDIA/trt-llm-moe-devs @NVIDIA/trt-llm-multi-gpu-ci-review @NVIDIA/trt-llm-qa-function @NVIDIA/trt-llm-runtime-devs. C++: yes. Size: product +1581/−34, tests +1190/−23, docs +83/−17.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

---

## PR 9/10 (wave 2)

**Title:** `[None][feat] WideEP FT 9/10: Back extra EPLB slots with ghost memory and keep ring host copies`

```markdown
@coderabbitai summary

## Description

**Summary.** Part 9 of 10. Follows #TBD (8/10).
- **Ghost slots:**
  - `GhostSlotMemory` (C++, VMM) maps the extra slots to shared scratch until a restore backs them.
  - The backing memory is taken from the KV cache when the device lacks it (`PyExecutor.resize_kv_cache_quota`). An abandoned plan's backing is released.
  - With GDRCopy, host-accessible ghost memory lets online EPLB copy on the CPU.
- **Ring host copies:** for static placements without replicas, each node keeps only the next node's home experts, about 1/N of the copies.
  - A restore stages each expert through the NVLinkOneSided workspace in chunks, between staging barriers.
  - The barriers are counted in `Stopped`, and the copy plan goes into the placement digest.
- **Default change:** `extra_slots` changes from `Literal["empty"] = "empty"` to `Optional[Literal["empty", "ghost"]] = None`, where `None` resolves to `ghost`. It resolves to `empty` with online EPLB where ghost memory is not host accessible.
- **Guide:** ghost and ring sections.
- Rewrites 56 lines added by 6/10 (49 product, 7 test): the C++ `validatePlacement`/`installPlacement` gain a `slotsLoaded` argument and `restorePlacement` a `slotsLoaded = false` default, and the Python load balancer, `interface.py` and `test_resident_placement.py` move from the appended empty slots to ghost slots.

**Design.** No memory is held while healthy. A recovery restores into ghost slots after Ready, and ring staging runs after Propose, while no all-to-all runs.

**FT-off delta:** a per-slot `GhostSlotMemory::contains` check at EPLB weight registration, a lambda per CPU weight copy, a never-destroyed allocator singleton, and `None` checks in host-accessible/finalize/shutdown paths; details:
- **C++:**
  - `addSingleWeightSlot` (CPU-copy path) asks `GhostSlotMemory::contains` per slot at registration;
  - `copyWeightsCpu` routes rows through a lambda;
  - the `HostAccessibleDeviceAllocator` singleton is never destroyed (its map's `shared_mutex` already exists on main).
- **Python:** `make_tensor_host_accessible` checks `ghost_slots is None`, and finalize/shutdown branch on ring fields, which are empty when FT is off.

**Performance.**
Measured on the full stack only (see the stack performance table); this PR alone is not benchmarked.

KV cache capacity (empty vs ghost) and restore time (mapping and staging) are not measured on their own (next steps); the ghost-slot deployments' recovery is inside the interrupt-to-resume times in the stack header (CLAIMS 142).

**Limitations / known gaps.**
- With ring copies, losing a node, or only its copy node, leaves the next node's experts without host copies.
- Ghost host access on NUMA-node GPU memory is deferred.
- Two default/behaviour changes have no switch (both apply only with peer-loss recovery on):
  - ring host copies replace the full per-node host copies for static placements without replicas; there is no setting to keep full copies;
  - the `extra_slots` default moves from `empty` to ghost through `Optional[...] = None`; an explicit `empty` keeps the 6/10 behaviour.

## Test Coverage

- C++: `GhostSlotMemoryTest.*` (7 tests, new component)
- `tests/unittest/_torch/moe/test_ring_host_copies.py` (new, cpu_only; l0_cpu via the `unittest/_torch/moe` entry)
- `tests/unittest/_torch/moe/multi_gpu/test_ring_restore_staging.py` (l0_gb200_multi_gpus, pre-merge, TIMEOUT 10)
- `test_resident_placement.py`, `test_moe_load_balancer.py`: ghost-mode tests
- Unit `test_peer_loss_recovery.py`: ghost KV cut, quota resize, ring staging, re-plan staging
- End to end: `wide_ep_ft/test_peer_loss_recovery.py::test_restore_into_ghost_slots[static|online]` (l0_gb200_multi_gpus, post-merge)
- Build/test status: the final stack tree (v2 cef4670789; the stack tip matches it in product code, its test files are trimmed) was built (build 8114247, BUILD_SUCCESS). Unit groups on that tree: gpu_moe 128 passed, cpu 963 passed, executor 291 passed, eplb_gpu 15 passed. The remaining FT unit files, the load-balancer gtest and the integration test are running (results pending: job 8115508). This PR's own intermediate tree is not built yet (a build-only job is planned).
- The GB200 integration cases (`test_restore_into_ghost_slots[static|online]`) and the ULFM tests skip in CI today: the CI image has no ULFM Open MPI and nothing sets `TLLM_REQUIRE_ULFM`. Evidence for them is from manual GB200 runs.

## PR Checklist

Please review the following before submitting your PR:
- PR description clearly explains what and why. If using CodeRabbit's summary, please make sure it makes sense.
- PR Follows [TRT-LLM CODING GUIDELINES](https://github.com/NVIDIA/TensorRT-LLM/blob/main/CODING_GUIDELINES.md) to the best of your knowledge.
- Test cases are provided for new code paths (see [test instructions](https://github.com/NVIDIA/TensorRT-LLM/tree/main/tests#1-how-does-the-ci-work))
- If PR introduces API changes, an appropriate PR label is added - either `api-compatible` or `api-breaking`. For `api-breaking`, include `BREAKING` in the PR title.
- Any new dependencies have been scanned for license and vulnerabilities
- [CODEOWNERS](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.github/CODEOWNERS) updated if ownership changes
- Documentation updated as needed
- Update [tava architecture diagram](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.github/tava_architecture_diagram.md) if there is a significant design change in PR.
- The reviewers assigned automatically/manually are appropriate for the PR.

- [x] Please check this after reviewing the above items as appropriate for this PR.

## GitHub Bot Help

To see a list of available CI bot commands, please comment `/bot help`.

Owners: @NVIDIA/trt-llm-devs @NVIDIA/trt-llm-disagg-devs @NVIDIA/trt-llm-doc-owners @NVIDIA/trt-llm-infra-devs @NVIDIA/trt-llm-moe-devs @NVIDIA/trt-llm-multi-gpu-ci-review @NVIDIA/trt-llm-qa-function @NVIDIA/trt-llm-runtime-devs. C++: yes. Size: product +1878/−108, tests +1460/−23, docs +21/−13.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

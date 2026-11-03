# prmain/* wave 1: ready-to-open status

Base: upstream/main `0829689c52`. All branches are local only (nothing pushed, no PRs opened). One commit per branch, DCO `Signed-off-by: Andrew Rusli`, no co-author/AI trailers (user instruction overrides the attribution reminder for commits; PR bodies end with the Claude Code line per the reminder; drop it if the user prefers).

## Environment limits (apply to every branch)

- This host has no torch and no built TensorRT-LLM: `python3 -c "import tensorrt_llm"` -> `ModuleNotFoundError: No module named 'torch'`. No branch's test file could run against the real package.
- Substitutes used: `python3 -m py_compile` on every changed `.py`; stub harnesses that load only the changed functions/classes (AST-extracted from the branch's file) and run the branch's own test file under a scratch venv with `torch 2.14.1+cpu` + `pytest 9.1.1`; for C++ 05, the planner functions compiled with g++ against stub types, and the branch's two new gtest bodies run under minimal EXPECT macros.
- C++ branches 05 and 06: not separately built; "built as part of the full tree". No CMake/CUDA build was attempted. The real gtests (05) and the CPU-copy EPLB path (06) have not run.
- Nothing here has had a `/bot run`.

## Summary table

| Branch | Final SHA | Old SHA(s) | +/- | Tests here | pre-commit |
|---|---|---|---|---|---|
| 01-sse-stream-errors | 6ffa651379 | unchanged | +125/-64 (3 files) | stub: 3 passed | pass |
| 02-moe-fake-ops-shared-health | c29ff974f0 | bc3f700e7a + 020889b7ba (2 commits) | +36/-0 (3 files) | py_compile ok; schema/deepcopy checks pass; real test not run | pass |
| 03-mpi4py-4.1.2 | 192c29b269 | 94d23962d2 | +47/-54 (1 file) | bash -n ok; patch applies cleanly to mpi4py 4.1.2 source | pass |
| 04-kv-v2-step-rollback | 969c1ec487 | 017ceb51fd + 7159fc6b53 (2 commits) | +361/-0 (3 files) | py_compile only (test needs CUDA) | pass |
| 05-eplb-planner-unusable-slots | 98c6d90bab | 341d1ac4eb | +171/-15 (3 files) | g++ harness: 126 checks, 0 failures; 17596 random cases match main's planner | pass (clang-format pass) |
| 06-eplb-load-publish | 5e8929a8ac | ab1578460f | +22/-31 (2 files) | none (built as part of the full tree) | pass (clang-format pass) |
| 07-eplb-host-sharer-expert-lists | 4bac439bae | 22c11e2a39 | +139/-45 (2 files) | stub: 5 passed; main reproduces the bug | pass |

Copyright headers: every modified/new source file has 2026 (`2022-2026` for the C++ files); `l0_cpu.yml` has no header on main and none was added.

---

## 01 prmain/01-sse-stream-errors

- Final SHA: `6ffa651379` (unchanged). +125/-64, 3 files (`tensorrt_llm/serve/openai_server.py`, new `tests/unittest/llmapi/apps/test_stream_error_events.py`, `l0_cpu.yml` entry).
- Tests run:
  - `python3 -m py_compile` on both .py files: ok.
  - Stub harness: `_stream_error_events` and `_end_stream_on_error` AST-extracted from the branch's `openai_server.py`, then `python -m pytest test_stream_error_events.py -q` -> `3 passed, 1 warning in 0.21s` (the warning is the unregistered `cpu_only` mark outside the repo's conftest).
- pre-commit (`PRE_COMMIT_HOME=$TMPDIR/pc-cache pre-commit run --files <changed files>` on the final tree): exit 0, tree unchanged; 19 passed, 9 skipped. The `type-check` (mypy) hook prints `- hook id: type-check` with "(no files to check) Skipped" on every branch; it is not a failure.
- Fixes made: none.
- Remaining gaps:
  - The test covers the two helpers only. No test drives a real chat, harmony or Responses endpoint that fails mid-stream, and none covers the Completions change (per-prompt generator now emits only the error event).
  - Completions with several prompts: when one prompt fails, the other prompts' chunks can still follow its error event, and then one `[DONE]` ends the stream. On main each failing prompt emitted its own `[DONE]`.
  - If `_extract_metrics` raises after `[DONE]` in chat/Responses, the wrapper emits an error event and a second `[DONE]`. This is an edge case and was not tested.
  - The client-visible error message is `str(error)`. Completions already did this on main, but a reviewer may question exposing internal exception text, since `_record_generator_termination` deliberately avoids it.

Title: `[None][fix] End a failed OpenAI stream with an error event and one [DONE]`

```markdown
@coderabbitai summary

## Description

**Summary.**
- A streamed chat, harmony chat or Responses request that failed after its first chunk re-raised (Responses had no handler at all). Starlette then cut the connection mid-body, so OpenAI clients saw a connection error instead of the request's error.
- These three streams now go through one wrapper, `_end_stream_on_error`. It logs the traceback and ends the stream with an SSE `error` event (`{"error": {"message", "type": "server_error", "code", "param"}}`, which OpenAI clients raise on) followed by `data: [DONE]`.
- Completions streaming already sent an error event, but a failing prompt sent `[DONE]` itself and the outer wrapper then sent a second one. The per-prompt generator now sends only the error event, and the wrapper's single `[DONE]` ends the stream.

**Design.** The HTTP status is committed with the first chunk, so the error has to travel in-band. `_stream_error_events` builds the error event and `[DONE]`, and both the wrapper and the Completions path use it. The wrapper catches `Exception` only, so client disconnects and `asyncio.CancelledError` still propagate.

**Default behaviour (a behaviour change for every deployment, fault tolerance on or off).** This applies to every streamed chat, harmony chat, Responses and Completions request, including the Anthropic-compatible stream path that reuses them. A stream that completes is passed through unchanged. A stream that raises after its first chunk now ends with an SSE `error` event and `[DONE]` instead of a dropped connection. The event's `message` is the exception text (`str(error)`), as the non-streaming error responses already return; the full traceback goes only to the server log.

**Performance.** No performance impact: the wrapper adds one async-generator layer per streamed response.

**Limitations.** With several Completions prompts, the other prompts' chunks may follow one prompt's error event before the final `[DONE]`.

## Test Coverage

- `tests/unittest/llmapi/apps/test_stream_error_events.py` (l0_cpu, pre-merge, `cpu_only`):
  - `test_failed_stream_ends_with_an_error_event_then_done`: the error event's shape and the final `[DONE]`.
  - `test_a_stream_failing_mid_way_ends_with_one_error_event_and_one_done`: events already sent, then exactly one error event and one `[DONE]`.
  - `test_a_stream_that_completes_passes_through_unchanged`.

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

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

## 02 prmain/02-moe-fake-ops-shared-health

- Final SHA: `c29ff974f0`. Old: `bc3f700e7a` (fix) + `020889b7ba` (test), squashed into one commit and retitled (the old 103-character title was over the 90-character cap). +36/-0, 3 files.
- Tests run:
  - `python3 -m py_compile` on 3 files: ok.
  - Schema check (script compares the C++ `module.def` arguments in `moeAlltoAllOp.cpp` with the fake functions' parameters): on main `moe_a2a_dispatch MISMATCH`, `moe_a2a_combine MISMATCH`; on the branch `moe_a2a_dispatch match`, `moe_a2a_combine match`. The NVLinkOneSided caller on main passes these arguments positionally (`nvlink_one_sided.py:995`, `:1140`).
  - `EPGroupHealth` loaded alone from `ep_group_health.py`: on main `copy.deepcopy(h)` -> `TypeError: cannot pickle '_thread.lock' object`; on the branch `deepcopy is self: True`.
  - The real test `test_configurable_moe.py::test_layer_quant_override_shares_ep_group_health` was **not run** (needs the full package). CI collects it through l0_cpu's `unittest/_torch/moe` entry.
- pre-commit (`PRE_COMMIT_HOME=$TMPDIR/pc-cache pre-commit run --files <changed files>` on the final tree): exit 0, tree unchanged; 18 passed, 10 skipped. The `type-check` (mypy) hook prints `- hook id: type-check` with "(no files to check) Skipped" on every branch; it is not a failure.
- Fixes made: squashed to one commit; new title `[None][fix] Fix the fake moe_a2a op signatures and deep copies of EPGroupHealth`; commit body states both bugs and their consequences.
- Remaining gaps:
  - Two unrelated fixes share one PR. Reviewers may ask for a split; each is about 5 lines.
  - `__deepcopy__` returning `self` is a semantic choice a reviewer will question ("is deepcopy the right hook, or should ConfigurableMoE carry extra_attrs by reference?").
  - The fake-op fix has no test of its own (for example, a `torch.library.opcheck` or a FakeTensorMode call).

Title: `[None][fix] Fix the fake moe_a2a op signatures and deep copies of EPGroupHealth`

```markdown
@coderabbitai summary

## Description

**Summary.**
- The fake (meta) registrations of `trtllm::moe_a2a_dispatch` and `trtllm::moe_a2a_combine` were missing arguments the C++ schemas declare: `use_cft_counted_writes`, `expert_id_payload_index` and `invalid_token_expert_id` on dispatch, and `use_cft_counted_writes` on combine. NVLinkOneSided passes these arguments positionally, so a fake-tensor trace of dispatch or combine failed. The fakes now match the schemas.
- `ConfigurableMoE` deep-copies the `ModelConfig` when a layer overrides the quantization config. With an `EPGroupHealth` in `extra_attrs`, that copy raised `TypeError: cannot pickle '_thread.lock'`. A copy would also have split the EP group's membership state between layers. `EPGroupHealth.__deepcopy__` now returns the instance.

**Design.** `EPGroupHealth` describes the EP group, which every MoE layer shares, so a deep copy keeps the shared reference. No other code path deep-copies it.

**Default behaviour (fault tolerance off).** No `EPGroupHealth` is created, so the `__deepcopy__` change is never reached. The fake-op change only affects fake-tensor tracing (torch.compile), not eager execution.

**Performance.** No performance impact: fake kernels and a deepcopy hook only.

## Test Coverage

- `tests/unittest/_torch/moe/fused_moe/test_configurable_moe.py::test_layer_quant_override_shares_ep_group_health` (l0_cpu via `unittest/_torch/moe`): `deepcopy` returns the same health object, and a quantization-override backend gets a copied config whose `extra_attrs` still hold the same `EPGroupHealth`.

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

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

## 03 prmain/03-mpi4py-4.1.2

- Final SHA: `192c29b269`. Old `94d23962d2`. +47/-54, 1 file (`docker/common/install_mpi4py.sh`).
- Tests run:
  - `bash -n` on the script: ok.
  - Downloaded the mpi4py `4.1.2` tag tarball from GitHub into a scratch directory (nothing in it was executed). `git apply --check -v` of the script's embedded patch, before the fix: `Hunk #1 succeeded at 538 (offset 4 lines)`; after the fix it applies with no offset.
  - The patched `src/mpi4py/futures/_core.py` passes `py_compile`. `os` is already imported in `_core.py`.
  - No image was built and no MPI job was run.
- pre-commit (`PRE_COMMIT_HOME=$TMPDIR/pc-cache pre-commit run --files <changed files>` on the final tree): exit 0, tree unchanged; 11 passed, 17 skipped. The `type-check` (mypy) hook prints `- hook id: type-check` with "(no files to check) Skipped" on every branch; it is not a failure.
- Fixes made: the embedded patch's hunk header changed from `@@ -534,5 +534,42 @@` to `@@ -538,5 +538,42 @@`, so it applies with no offset (no content change); added a commit body.
- Remaining gaps (brutal):
  - The image rebuild and the `jenkins/current_image_tags.properties` bump are a separate infra step, not in this PR. Until both land, CI keeps testing mpi4py 3.1.5.
  - No runtime validation of mpi4py 4.x in TRT-LLM: MPIPoolExecutor / `mpi4py.futures` launch, `TRTLLM_USE_MPI_KVCACHE=1` device selection in the moved patch, and disaggregated MPI paths. mpi4py 4 changed some defaults, which needs a full multi-GPU CI run.
  - The PR does not say why mpi4py 4.1.2 is needed. State the concrete reason in the body before opening (placeholder below).
  - infra/oss-compliance will need to review the dependency bump.

Title: `[None][infra] Upgrade mpi4py to 4.1.2`

```markdown
@coderabbitai summary

## Description

**Summary.**
- Build mpi4py 4.1.2 instead of 3.1.5 in `docker/common/install_mpi4py.sh`.
- The `TRTLLM_USE_MPI_KVCACHE=1` device-selection patch is ported unchanged to mpi4py 4.x's layout (`futures/_core.py`, `_comm_executor_helper._manager`).
- Drops the workarounds 3.1.5 needed: the `setuptools < 69` pin and the aarch64 `Cython==0.29.37` pre-install.
- Why: <<state the concrete reason for 4.x, e.g. the MPI-4 API / Python version support it brings>>.

**Rollout.** This PR changes the install script only. Rebuilding the CI images and bumping `jenkins/current_image_tags.properties` is a separate infra step that follows it. Until then CI images still carry 3.1.5.

**Default behaviour.** Unchanged until the images are rebuilt. After that, every MPI launch uses mpi4py 4.1.2.

**Performance.** No performance impact expected: mpi4py is a binding layer. A full multi-GPU pre-merge run on the rebuilt image is the check.

**Limitations.** Not yet validated on a rebuilt image. The patch applies cleanly to the 4.1.2 source tarball.

## Test Coverage

- No new tests. Validation is the full pre-merge and post-merge multi-GPU pipeline on the rebuilt image (MPI session, `mpi4py.futures` worker pool, disaggregated MPI KV-cache transfer with `TRTLLM_USE_MPI_KVCACHE=1`).

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

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

## 04 prmain/04-kv-v2-step-rollback

- Final SHA: `969c1ec487`. Old: `017ceb51fd` (feat) + `7159fc6b53` (test), squashed into one commit. +361/-0, 3 files (`kv_cache_manager_v2.py` +66, new `tests/unittest/_torch/executor/test_kv_cache_revert.py` +294, `l0_b200.yml` entry).
- Tests run:
  - `python3 -m py_compile` on both files: ok.
  - `test_kv_cache_revert.py` (6 test functions, 7 cases) was **not run**: it needs CUDA and the built package, and none of it has run on GPU in this branch's form. On l0_cpu it is collected by the `unittest/_torch/executor` directory entry and skipped there (`skipif not cuda`).
  - All helpers the new code uses exist on main: `rewind_context_after_cache_drop`, `_inflight_draft_lens`, `_allocated_draft_lens`, `_after_shrink`, `py_ctx_pre_resize_cap`.
- pre-commit (`PRE_COMMIT_HOME=$TMPDIR/pc-cache pre-commit run --files <changed files>` on the final tree): exit 0, tree unchanged; 15 passed, 13 skipped. The `type-check` (mypy) hook prints `- hook id: type-check` with "(no files to check) Skipped" on every branch; it is not a failure.
- Fixes made:
  - Squashed into one commit, with a commit body.
  - The docstring said "peer-loss recovery, the only caller, refuses them", naming a caller that does not exist on main. It now reads "a caller must not use this with them".
  - The test module docstring no longer says "(WideEP peer-loss recovery)".
- Remaining gaps (brutal):
  - Public methods with no caller on main. kv-cache-manager-devs may refuse "dead code until PR X"; open it only after the FT PR exists as a draft.
  - The test has never run on GPU. Run it on a B200/H100 before `/bot run`.
  - `context_capacities` must be captured right after scheduling. This is a caller contract that only the docstring enforces.

Title: `[None][feat] Roll back a discarded step's allocations in KV Cache Manager V2`

```markdown
@coderabbitai summary

## Description

**Summary.**
- Adds `KVCacheManagerV2.context_capacities(batch)` and `KVCacheManagerV2.revert_prepare_resources(batch, context_capacities)`. Together they undo the scheduler's KV cache allocations for a step whose forward is discarded, so the step can run again on an unchanged cache.
- Steps are reverted newest first. This also handles the overlap scheduler, where step N+1 is prepared before step N commits.
- First caller: WideEP fault-tolerance PR (to be opened). It discards in-flight steps after a peer loss and replays them.

**Design.**
- A first context chunk is freed, and its context cursor is rewound with `rewind_context_after_cache_drop`, because preparing it may have matched a reused prefix.
- Later chunks shrink back to the capacity `context_capacities` recorded right after scheduling, and stay resident. This differs from `revert_allocate_context`, which suspends them.
- Generation requests give back the step's newest allocation, including draft slots recorded in `_inflight_draft_lens`.
- A discarded step committed no blocks for reuse. Its KV writes are overwritten when the step runs again.
- Caches freed or suspended by a later step (cancellation, recompute) are left alone.
- A paired draft pool is reverted by its own call.
- Not reverted: KV connectors and SWA scratch reuse. Callers must not combine the rollback with them.

**Default behaviour.** No change: nothing on main calls the new methods.

**Performance.** No performance impact: new methods with no caller on the default path.

**Limitations.** No in-tree caller in this PR (see above). Connector and SWA scratch-reuse state is out of scope.

## Test Coverage

- `tests/unittest/_torch/executor/test_kv_cache_revert.py` (l0_b200 pre-merge; real `KVCacheManagerV2`, CUDA required):
  - `test_revert_prepare_resources_round_trip`: free blocks and per-request capacity/history match after revert, and the re-prepared step matches the first preparation.
  - `test_reverted_context_is_not_offered_for_reuse`.
  - `test_reverted_first_chunk_that_reused_a_prefix_runs_again_unchanged`.
  - `test_two_overlapped_steps_revert_newest_first`.
  - `test_discarded_later_chunk_keeps_the_committed_chunks_reusable[sequential|overlap]`.
  - `test_a_draft_kv_pool_reverts_with_the_target_across_two_overlapped_steps` (DFlash draft pool with `KVCacheV2Scheduler`).

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

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

## 05 prmain/05-eplb-planner-unusable-slots (C++)

- Final SHA: `98c6d90bab`. Old `341d1ac4eb`. +171/-15, 3 files (`moeLoadBalancer.cpp/.h`, `cpp/tests/unit_tests/runtime/moeLoadBalancerTest.cpp` +117).
- Build: not separately built; "built as part of the full tree". No CMake/nvcc build was attempted, and the real gtests have not run.
- Tests run here (substitute, g++ C++17, host CPU):
  - Harness 1: the branch's `ReplicaInfo` and anonymous-namespace helpers, plus `doReplication` and `doPlacement`, copied verbatim (moeLoadBalancer.cpp lines 35-269) and compiled against stub `MoeLoadBalanceMetaInfo`, `MoePlacementCpuInfo` and `TLLM_CHECK_WITH_INFO` (throwing). Both new gtest bodies were copied verbatim and run under minimal EXPECT/ASSERT macros: `126 checks, 0 failures`.
  - Harness 2: main's and the branch's `doReplication`/`doPlacement` compiled side by side. The branch was called with no unusable mask on random shapes (1-16 ranks, 1-6 slots per rank, random loads including zeros, random dead-rank masks): `17596 cases, 0 differ` (replica counts, placements and throw/no-throw all equal).
- pre-commit: exit 0, 12 passed (including clang-format), 16 skipped.
- Fixes made:
  - Test comments: "(a ghost slot a restore backed)" removed, and "main's planner" changed to "the planner".
  - Constant `kGhostSlot` renamed to `kLastSlot` (test only).
  - Commit amended (same title and body).
- Remaining gaps (brutal):
  - Not built with the real toolchain, and the gtest binary has not run. Build `runtime-tests` and run `ctest -L ... MoeLoadBalancerUnusableSlotTest` before opening.
  - Nothing on main passes a mask ("no caller"), so it lands as groundwork.
  - In `doPlacement`, the slot-skip loop relies on `usableSlotCounts` to stay inside the rank's slots. That is correct, but the `TLLM_CHECK` comes after the loop, not inside it.

Title: `[None][feat] Let the EPLB planner leave slots marked unusable empty`

```markdown
@coderabbitai summary

## Description

**Summary.**
- `doReplication` and `doPlacement` take an optional `unusableSlotMask`: one byte per global slot (`rank * slotCountPerRank + localSlotId`), and the planner leaves every marked slot `-1`.
- Replication counts only usable slots, so the total replica count equals the number of usable slots.
- First caller: WideEP fault-tolerance PR (to be opened). It reserves slots that have no backing weights after a rank loss.

**Design.**
- `getUsableSlotCounts` gives each rank its number of fillable slots: 0 for a rank the dead-rank mask marks, `slotCountPerRank` when there is no mask.
- Placement fills each rank in slot order and skips masked slots (`nextLocalSlotId`). It offers a rank to the priority queue only while the rank has usable slots left.
- Fewer usable slots than experts, or a mask of the wrong size, raise `TllmException` as the dead-rank path already does.

**Default behaviour.** Unchanged. Every existing caller passes no mask, and the plan is then identical. The new gtest `AllUsableMaskPlacesExactlyAsNoMask` checks this, and a side-by-side comparison of main's planner with this branch's planner on 17,596 random shapes and loads found no difference.

**Performance.** No performance impact: host-side planning, O(epSize * slotCountPerRank) extra work only when a mask is passed.

**Limitations.** No in-tree caller in this PR.

## Test Coverage

- C++ `MoeLoadBalancerUnusableSlotTest.PlacementNeverFillsAnUnusableSlot`: masked slots stay -1 with and without a dead rank and under skewed and zero load, replica totals equal the usable slot count, and too-few-slots and wrong-size masks throw.
- C++ `MoeLoadBalancerUnusableSlotTest.AllUsableMaskPlacesExactlyAsNoMask`: an all-usable mask gives exactly the no-mask plan, with a hand-checked small case and seeded loads with and without dead ranks.
- Both are in `cpp/tests/unit_tests/runtime/moeLoadBalancerTest.cpp` (existing runtime gtest target).

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

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

## 06 prmain/06-eplb-load-publish (C++)

- Final SHA: `5e8929a8ac`. Old `ab1578460f`, amended to add a commit body; the tree is unchanged. +22/-31, 2 files.
- Build: not separately built; "built as part of the full tree". No tests were run here, and the PR has no new test.
- Reviewed by reading: the GPU path order is unchanged (prepare, updateWeights, copyPlacementInfoToGpu, event record and sync, signal) and so is the CPU path (prepare, copy task, wait, copyPlacementInfoToGpuByCpu, signal). `mUseGpuMemcpy` is read on the update thread at the same point as before.
- pre-commit: exit 0, 12 passed (including clang-format), 16 skipped.
- Fixes made: commit body added (no code change).
- Remaining gaps (brutal):
  - Not built.
  - On main no test covers the CPU-copy path (`mUseGpuMemcpy == false`); reviewers may ask for one.
  - It is a pure refactor whose only motive is a later caller (EPLB restore in the FT PR). Say so, or fold it into that PR if reviewers push back.

Title: `[None][chore] Share EPLB's slot load and routing publish in loadAndPublishPlacement`

```markdown
@coderabbitai summary

## Description

**Summary.**
- `SingleLayerMoeLoadBalancer::updateWeightsRoutine` and `updateWeightsRoutineByCpu` repeated the same planning and completion signalling. They differed only in how they load the changed slots and publish the placement: GPU memcpy, or the CPU copy threads.
- There is now one `updateWeightsRoutine`. It plans with `doReplication`/`doPlacement`, calls the new private `loadAndPublishPlacement` (which picks the copy path from `mUseGpuMemcpy`), and signals completion.
- First caller of `loadAndPublishPlacement` beyond `updateWeightsRoutine`: WideEP fault-tolerance PR (to be opened). It loads a restored placement without re-planning.

**Default behaviour.** No behaviour change. Both copy paths run the same calls in the same order.

**Performance.** No performance impact: the same work, one fewer lambda branch.

## Test Coverage

- Covered by the existing EPLB tests: `cpp/tests/unit_tests/runtime/moeLoadBalancerTest.cpp` and `tests/unittest/_torch/moe/test_moe_load_balancer.py` (l0_b200).
- No new test. The CPU-copy path (`mUseGpuMemcpy == false`) has no dedicated test on main.

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

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

## 07 prmain/07-eplb-host-sharer-expert-lists

- Final SHA: `4bac439bae`. Old `22c11e2a39`. +139/-45, 2 files (`moe_load_balancer.py`, new test).
- Tests run:
  - `python3 -m py_compile` on both files: ok.
  - Stub harness: `HostMoeTensorSharer` AST-extracted from the branch's `moe_load_balancer.py` (stubs: `tensorrt_llm.logger`, `MPI.Comm`, `advise_tensor_pageout`), then the branch's test file under torch 2.14.1+cpu: `python -m pytest test_moe_host_sharer_expert_ids.py -q` -> `5 passed, 1 warning in 4.70s`. This includes the real POSIX shared-memory test; `/dev/shm` was left clean (0 leftover segments).
  - Negative control on main's class: 3 experts over 4 local ranks, rank 0 `finalize_layer_weights()` -> `ValueError: 'size' must be a positive number different from zero` (bug confirmed).
  - The existing `test_moe_host_sharer.py` (needs mpi4py and MPI) and the `SingleLayerMoeLoadBalancer` paths were **not run**.
- pre-commit (`PRE_COMMIT_HOME=$TMPDIR/pc-cache pre-commit run --files <changed files>` on the final tree): exit 0, tree unchanged; 18 passed, 10 skipped. The `type-check` (mypy) hook prints `- hook id: type-check` with "(no files to check) Skipped" on every branch; it is not a failure.
- Fixes made:
  - Retitled to lead with the bug fix (per review): `[None][fix] Let EPLB host copies work on a node with more ranks than experts`. The commit body now opens with the failure.
  - Test file renamed from `test_ring_host_copies.py` (an FT concept not on main) to `tests/unittest/_torch/moe/test_moe_host_sharer_expert_ids.py`, with a neutral docstring and shm name prefix. CI collects it through l0_cpu's `unittest/_torch/moe` entry.
- Remaining gaps:
  - `keep_experts` and `unlink_shared_memory` have no caller on main besides `__init__` and `post_shutdown_cleanup`.
  - `expert_id in self.own_expert_ids` is a list scan per registered tensor. That is negligible at expert counts of 256-512, but a reviewer may ask for a set.

Title: `[None][fix] Let EPLB host copies work on a node with more ranks than experts`

```markdown
@coderabbitai summary

## Description

**Summary.**
- **Bug:** `HostMoeTensorSharer` gives each local rank a contiguous share of the experts. On a node with more ranks than experts, some ranks get none, and their `finalize_layer_weights` asked for a zero-size `SharedMemory` segment, which raises `ValueError`, so EPLB failed at startup. Such a rank now creates no segment, and the other ranks skip it when they map the node's segments.
- The sharer now splits a sorted list of expert ids over the node's ranks instead of the fixed range `[0, expert_count)`.
  - `keep_experts(expert_ids)` sets the list. The default is all experts, so the split is unchanged.
  - `rank_expert_ids(local_rank)` returns any local rank's part.
  - `SingleLayerMoeLoadBalancer.get_load_expert_ids` returns the sharer's part instead of computing the same split a second time.
- `unlink_shared_memory()` removes this rank's segment name. `post_shutdown_cleanup` uses it.
- First caller of `keep_experts`: WideEP fault-tolerance PR (to be opened). It keeps host copies of a subset of experts per node.

**Default behaviour.** Unchanged except for the bug fix. The default list is every expert, so each rank loads and shares the same experts as before, and shutdown unlinks the same segment.

**Performance.** No performance impact: setup-time bookkeeping on the host.

## Test Coverage

- `tests/unittest/_torch/moe/test_moe_host_sharer_expert_ids.py` (l0_cpu via `unittest/_torch/moe`, `cpu_only`):
  - `test_sharer_splits_every_expert_over_the_node_s_ranks_by_default[8-4|10-4|3-4]`: same split as before by default.
  - `test_sharer_splits_the_experts_it_holds_over_the_node_s_ranks`: explicit list split and registration asserts.
  - `test_a_rank_that_shares_no_expert_creates_no_shared_memory`: 3 experts over 4 ranks, end to end through real POSIX shared memory. It covers creating, mapping and unlinking the segments, and checks every rank reads every expert's weights.

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

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

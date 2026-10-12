# WideEP FT, round 3: adversarial review

Scope: `git diff HEAD` (uncommitted round-2 fixes) plus WIP commit ed20a243ad, in
`.wideep-ft-runs/claude/wt`. Line numbers refer to the working tree as of this review.
Abbreviations: PE = `tensorrt_llm/_torch/pyexecutor/py_executor.py`, PLR =
`tensorrt_llm/_torch/pyexecutor/peer_loss_recovery.py`, RM =
`tensorrt_llm/_torch/pyexecutor/resource_manager.py`, KV2 =
`tensorrt_llm/_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py`, K =
`cpp/tensorrt_llm/kernels/moe/communication/moeAlltoAllKernels.cu`.

## CONFIRMED bugs (ranked)

### C1. A torn request broadcast that succeeds on rank 0 loses requests (client hangs forever). High impact, low-to-moderate probability
Code: PE:6327-6332 (`_unbroadcast_requests` is captured only when the broadcast *raises*), PE:8347-8358
(`_requeue_unscheduled_requests`), PE:6540-6571 (pop + route are local).

Scenario (EP32, binomial bcast tree, D = rank 5, a child of rank 4 and not of root 0):
1. Rank 0 fetches request X and calls `RequestBroadcaster.broadcast`. D dies. Root's eager sends to
   its children 16/8/4/2/1 complete, so the broadcast **returns normally on rank 0**. Ranks 4 and
   5's subtree (and any rank whose path crosses D) raise `HostCollectiveInterrupted` with an empty
   `_unbroadcast_requests` (non-roots pass `[]`). ULFM explicitly allows this non-uniform outcome.
2. Rank 0 runs `_handle_special_queue_items`, adds X to its waiting queue, then pops and routes it
   locally (steps 3-6 have no collective). `requests_routed` records owner = rank k (for example
   rank 4, which is alive). Rank 0 then raises at the next collective (`_balance_adp_requests` or
   `_can_queue`).
3. During recovery, rank 0's `unscheduled` list is `waiting_queue` (X was already popped) plus
   `_unbroadcast_requests` (empty, because rank 0's broadcast did not raise). Rank 4 drops its
   waiting queue, which never held X.

Result: X is never served, and rank 0 never fails it because owner 4 is alive. The client waits
forever. The PE:8350 comment, "Rank 0 holds every such request", is false once rank 0 has popped
the items. Cancels and the shutdown or control items that rank 0 applied are also lost on the
failed subtree (a mirror of r2 H6).

Fix direction: rank 0 keeps the last broadcast batch as "unconfirmed" until a later collective
succeeds. After recovery, it re-broadcasts that batch with ids, and every rank skips the ids it
already admitted or routed. Alternatively, run an agreement after Shrink on the last broadcast
sequence number each rank received.

### C2. V1 revert does not rewind the context cursor, so a re-run with a smaller prefix match computes from a stale position (silent wrong output). High impact, low probability
Code: RM:1255-1259. The C++ side is `kvCacheManager.cpp:3836-3839`, where `addSequenceBatch` calls
`setPrepopulatedPromptLen(minPrepopulatedLen)`, together with `llmRequest.h:1028-1041`, which moves
`contextCurrentPosition` **only when prepopulatedPromptLen > 0**.

Scenario (V1, block reuse on, which create() now allows):
1. The first chunk of R matches P = 64 reusable tokens. `add_sequence_batch` sets
   prepopulated = cursor = 64.
2. A loss discards the step. The revert calls `impl.remove_sequence(R, None, False)`. The cursor
   stays at 64 and prepopulated stays at 64, so `is_first_context_chunk` stays true.
3. Before R is scheduled again, KV pressure evicts the free reusable blocks. They were released
   with refcount 0, and generation `add_token` calls in later iterations reclaim them.
4. The re-run calls `add_sequence_batch`, which matches 0 and calls `setPrepopulatedPromptLen(0)`.
   The cursor **stays at 64**. The forward computes [64, 64 + chunk) and attends over freshly
   allocated, unfilled blocks for [0, 64). The tokens are silently wrong.

The match only shrinks to exactly 0 on this path: a partial shrink to P' > 0 resets the cursor
correctly. V2 is safe because `_settle_context_cursor` → `_rewind_context_cursor` lowers the cursor
(KV2:1180-1200). Fix: in the V1 revert, rewind first chunks the way V2's
`rewind_context_after_cache_drop(req, tokens_per_block)` does (`llm_request.py:1694`). The new
`test_reverted_context_is_not_offered_for_reuse` does not cover a reverted first chunk that had a
prefix hit.

### C3. V2 allocations made before `step_prepared` are never reverted, and generation capacity drifts by +1 for the request's lifetime. Low impact, likely in recommended configs
Code: PE:4709-4714 (V2 `step_prepared` comes after `_schedule()`), PE:7036-7060 (with
`attention_dp_config.enable_balance`, `_balance_adp_requests` does `tp_allgather_int64` after
`scheduler.schedule_request` has already allocated), KV2 `try_allocate_generation`
(`new_capacity = pre_capacity + 1 + draft`, which is relative, not target-based).

Scenario: a loss interrupts `_balance_adp_requests`. `_uncommitted_step` is None, so nothing is
reverted. Each generation request keeps its +1 slot and its `_allocated_draft_lens` entry. Every
later step grows from the inflated capacity again, and `update_resources` only subtracts the draft
rewind, so capacity stays one ahead forever. The PLR:239-241 docstring ("freed with the request")
is accurate, but the drift is per request. Near `max_seq_len`, the extra slot can make a resize
fail and stall the request. Enable-balance is common in WideEP DeepSeek deployments. Fix: record
the V2 step inside the scheduler path (or before `_balance_adp_requests`), or have the revert
recompute the target capacity from `history_length` rather than undoing deltas.

### C4. The GPU round-trip tests for KV revert never run in CI. Test gap
`tests/unittest/_torch/executor/test_peer_loss_recovery.py:61` sets a module-level
`pytestmark = pytest.mark.cpu_only`. `test_revert_prepare_resources_round_trip` and
`test_reverted_context_is_not_offered_for_reuse` are `@requires_cuda`. The CPU stage
(`l0_cpu.yml:46`, `unittest/_torch/executor`) has no CUDA, so they skip there. GPU stages
deselect `cpu_only` (see the `l0_b200.yml` comment and `l0_cpu.yml:39-40`). The only real-manager
evidence for the V1/V2 rollback is therefore never exercised. Move these tests to a module that
is not `cpu_only` and add that module to a GPU list.

### C5. Stale comments. Nits
- PE:1495-1500, the `_run_event_loop` docstring: "the hang detector is stopped for the bounded
  recovery". Recovery now runs *under* the hang detector (PLR:212).
- `mpi_session.py:1046-1048`: "the exit status stays successful". Nothing verifies this. PRRTE
  can report the SIGKILLed rank as a failure at job end, and the integration test accepts any
  status (`_wait_for_teardown`). Either verify it or drop the claim.
- During this review the `test_create_rejects_...` parametrize still had the
  `("block reuse")` case, which would fail now that create() no longer rejects reuse. It was
  removed on disk while I was reviewing. Make sure the committed version has it removed.

## PLAUSIBLE risks (ranked)

P1. **The watcher callback can die before revoking.** PLR:227-232 runs `request_abort()` (which does
`_abort_stream.synchronize()`) before `dist.revoke()`. If the context already has a sticky error
(a trap, or an Xid on a survivor), `synchronize` raises in the watcher thread. `revoke()` is then
skipped and the thread exits silently. Wrap it so `revoke()` always runs (try/finally) and log the
abort failure.

P2. **Python-thread watcher plus GIL.** Abort delivery needs the watcher thread to get the GIL. A
main-thread blocking CUDA call that holds the GIL while the GPU sits in a stuck MoE kernel
(`tensor.tolist()` on a CUDA tensor, for example, which does not release the GIL in PyTorch) would
starve the watcher. The device timeout then traps the survivor, which is a second loss and
terminal. I found no such call on the NVLinkOneSided or static-EPLB hot path. Keep it as an
invariant, or move the abort request into the native watcher.

P3. **Withheld combine readiness is bounded only by ULFM detection.** K:1929-1931. A survivor B whose
peer A withheld readiness spins until B's *own* abort arrives. That requires B's watcher to observe
the failure through ULFM's detector, because revocations do not complete the private-comm
`Irecv`. B is in GPU wait, not in a host collective. This is bounded, but only if
`mpi_ft_detector_timeout` plus the 10 ms poll plus the abort copy stays below
`TLLM_MOE_A2A_TIMEOUT_SECONDS` (300 s default). Nothing checks this (r2 H8 is still open).
Without withholding, B would have completed in the case where D had already signalled readiness,
so withholding turns a "completes and commits" case into a "waits for detection" case.

P4. **Healthy-path cost of zeroing invalid rows.** K:1404-1411 zeroes every invalid row with
1-byte stores, one block per row, on every layer and every step. Invalid rows are the rows past
each source's count, which is most of `[ep, maxTokens]` (about 75% at top-8/EP32 in prefill).
At prefill `maxTokens` this is on the order of the full receive buffer per layer, written
byte-wise. Use vectorized 16 B stores, or zero only payloads that a consumer actually reads
unmasked. Measure it as part of K-S6.

P5. **No validation of hang-detector timeout against recovery timeout.** PLR:212 puts the whole
recovery under the executor's `HangDetector` with a single `checkpoint()`. If
`hang_detection_timeout` is below `TLLM_FAULT_TOLERANCE_RECOVERY_TIMEOUT_S` (120 s default), or the
recovery timeout is raised above the hang timeout (300 s default), a slow but healthy recovery is
hard-killed instead of failing with its own deadline error. Validate the ordering in create().

P6. **Forward error and loss in the same step.** PE:8288-8301 calls `_handle_errors` inside
`_forward_step`, which does a response collective. If a loss interrupts that collective,
`_uncommitted_step` still holds the batch. The terminations are deferred while
`_discard_uncommitted_step` reverts the batch, so V1 `remove_sequence` runs and the deferred
`_terminate_request` → `free_resources` later removes the same sequence again. That is a C++
error during or after recovery, which is terminal. It is rare.

P7. **V2 revert runs `conversation_manager.finish_request`.** KV2:5694 `free_resources` calls it for
a discarded first chunk. If a conversation manager is configured, the discarded request is
"finished" in its bookkeeping. create() does not reject it.

P8. **`OMPI_MCA_mpi_ft_enable` check is environment-only.** communicator.py:888 accepts only "1" or
"true". It rejects valid MCA spellings ("yes", "on", "enabled") and settings from a param file or
`--mca` that don't surface as that env var. `--with-ft ulfm` does set it to "1" (confirmed in
probe-ulfm/out.txt), so this fails closed. It is a UX issue only.

P9. **`logging.shutdown()` before `Abort(0)`.** `mpi_session.py:1052`. It blocks if another thread
holds a handler lock, for example in a write to a stalled pipe. That is the same exposure as the
previous `flush()`. Later log calls from daemon threads still work, because StreamHandler.close
does not close the stream. Low.

P10. **Theoretical `int` overflow.** K:1414, `stat = blockIdx.x * blockDim.x + threadIdx.x` in the
gather. It overflows only when `epSize * maxTokens >= 2^23`. Low.

## Checked and OK
- **Gather barriers:** `__shared__ aborted` is written by thread 0 and followed by `__syncthreads()`
  before any divergence. There are no early returns, and blocks beyond the token range still reach
  the barrier (K:1366-1374).
- **Combine barriers:** early returns happen before any barrier. Lanes that break on abort reach
  `__syncthreads_or`, and the result is block-uniform (K:2000-2006).
- **PDL:** `cudaGridDependencySynchronize` in the gather waits for the full dispatch grid, so the
  early trigger in dispatch does not let the gather read in-flight counters. This matches main.
- **Abort stickiness:** `requested` stays set until rearm, so if dispatch aborted, the gather's
  single read sees it.
- **Concurrent revokes:** the watcher's `revoke` and the main thread's `revoke` and `shrink` never
  run concurrently within a process. `_recover` stops and joins the watch first, and a revoke from
  another rank during Shrink is legal. `stop()` is idempotent, and the new watch starts only after
  Commit.
- **V2 dummy double free:** in V2, an `_pad_empty_attention_dp_batch` dummy that is freed by
  `_finalize_adp_dummy_allocation(False)` and then reverted is a no-op, because
  `_allocated_draft_lens` was already popped and the cache is gone. A V2 context dummy freed by the
  revert and then retired is also a no-op. In V1, preprepared dummies are excluded from `added`.
- **V2 first-chunk revert with a prefix hit:** re-preparing re-settles the cursor, including when
  the new match is 0.
- **Recording the V2 step twice** (at scheduling and after `prepare_resources`) stores the same
  batch, so the revert runs once. `step_released` on `can_queue=False` prevents a double revert.
- **HangDetector re-entry:** the loop's `with` has exited before `run()` re-enters, so there is no
  "already active" warning.
- **Static EPLB** launches no `wait_gpu_stage` spin kernels, and `MoeLoadBalancerIterContext` is a
  no-op, so an interrupted forward leaves no `in_iter` state.
- **Removing the combine fixed-slot check:** the generic `combinePayloadOffset + payloadSize <=
  sizePerRank` check still bounds the combine.
- **Late-writes test:** its rearm emulation matches `NVLinkOneSided.reset_local_workspace`.

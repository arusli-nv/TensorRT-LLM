# Review r2: NVLinkOneSided fence A2A execution descriptor / cooperative abort / fixed slots

Scope: `git diff 55279c109f` over moeAlltoAllKernels.{cu,h}, moeAlltoAllOp.cpp, cpp_custom_ops.py,
fused_moe/communication/{execution_state,nvlink_one_sided}.py, wide_ep_ft.py, test_moe_a2a_abort.py,
test_moe_execution_state.py. Worktree `.wideep-ft-runs/claude/wt`, not compiled. Line numbers are worktree lines.

Verdict: no hard blocker found in the kernel logic. The fixed-slot plus sticky-abort design holds up.
Dead-rank late writes only land in masked state. Timeout still traps. With FT off, codegen is unchanged
apart from kernel-param size. PDL is ordered correctly. Fake/schema agree. Three should-fix items are
correctness hardening and one is a perf trap on the abort path.

## Verified OK (with reasoning)

- **PDL ordering.** The dispatch kernel triggers before its flag wait (.cu:573). The gather calls
  `cudaGridDependencySynchronize` before any read (.cu:1357). griddepcontrol.wait waits for full
  completion and memory flush of the primary, not just its trigger, so the gather never sees counters or
  rows before the dispatch's wait has finished. Dispatch reads the descriptor mask only after its
  griddepsync (route at .cu:543, last-block logic after 524/539). Combine does the same (.cu:1894 before
  1897).
- **Sticky abort.** `requested` only goes 0 to 1 while kernels run and is cleared only in `rearm()`
  after `torch.cuda.synchronize`. So an aborted dispatch always makes its gather invalidate everything:
  `abort_requested` at .cu:1361 is read after the dispatch has finished. A late abort only invalidates
  more, which is safe.
- **Peer data completeness.** A peer always writes rows, counters and stats before its release fence and
  flag store, and abort only shortens waits. So any flag we observe implies complete data, even from a
  peer that later aborts.
- **Fixed slots.** Source r's rows sit at `recv_buffers[t][p] + r*stride + idx*bytes`
  (.cu:413-420). `stride = sum_p pad128(staticMax*bytes_p)` (op.cpp:614-620), and
  `validateFixedSlotWorkspace` enforces runtime <= static (op.cpp:170-180). Zombie writes from an
  excluded rank therefore only reach its own slot, its own recv_counters/flags column and its own stats
  row. All of these are masked in the gather (.cu:1368, 1397). Prepare-combine bounds its reads by grid
  size, so a corrupt count from a zombie cannot go out of bounds.
- **Gather count bound.** `count <= maxTokens && source_token < count` rejects -1 (reset), negative and
  oversize counts, and the test covers these cases. The Python `_check_dispatch_region` mirrors the native
  layout, and the `combine_payload_offset != dispatch_payload_end` check cross-checks it.
- **Rearm and flag epochs.** `rearm()` re-runs the init layout (zero row, both recv-counter banks = -1)
  on every survivor while all are quiescent: `_join_device_work`, then the Install/Installed/Commit
  handshake. That resets `flag_val` and the completion flags, so ranks that stopped in different rounds
  cannot carry a parity or equality mismatch forward.
- **Timeout.** Both dispatch and combine still `trap` (.cu:704, 1966). The abort branch is taken only
  when `aborted`, and the subprocess test checks the trap.
- **`__syncthreads_or` in combine (.cu:1977-1984).** Lanes leave the peer loop by `break`, not `return`
  (except on trap). All lanes execute the fence. Warps other than the first reach the barrier directly.
  Blocks that exit early (`local_token_idx >= local_num_tokens`) exit before the barrier, as on main.
  I found no deadlock or divergence hazard.
- **FT-off codegen.** For `ENABLE_RANK_MASK=false`, `descriptor` is the constant `nullptr`, so `aborted`
  folds to false and both the `if (aborted)` and the source-stride branches are removed. `__syncthreads()`
  is kept. The only differences are param size (+16 B DispatchKernelPointers, +8 B
  CombineKernelPointers, still far below the 32 KB limit) and twice as many `vectorized_dispatch_impl`
  instantiations (binary size only). The host op's non-descriptor path computes the same offsets and uses
  `from_blob`. Recommend a SASS diff of `moeA2ADispatchKernel<*,*,false>` and
  `moeA2ACombineKernel<*,*,*,false>` against main once the build finishes, to confirm.
- **Schema and fake.** `Tensor(b!)? execution_descriptor=None` is appended last, so existing positional
  callers still work. The fake signatures now match the schema positionally; main's dispatch fake was
  missing `use_cft_counted_writes` / `expert_id_payload_index` / `invalid_token_expert_id`, which this
  diff fixes.

## Blockers

None.

## Should-fix

1. **The op accepts a descriptor without sanitization args, which leaves expert ids uninitialized**
   (op.cpp:548, 626, 816, 848; .cu:1384).
   - In descriptor mode the recv tensors are `torch::empty`. Invalid rows get `invalidExpertId` only
     when `expertIdPayloadIndex >= 0`.
   - Python always passes the index when bound (nvlink_one_sided.py:936), but nothing in the op enforces
     it. A caller that passes a descriptor without `expert_id_payload_index`/`invalid_token_expert_id`
     gets garbage expert ids, which means out-of-range expert indices in MoE kernels.
   - The standalone `moe_a2a_sanitize_expert_ids` is not a substitute because it ignores the mask and
     the abort state.
   - Fix: `TORCH_CHECK(!executionDescriptor.has_value() || sanitizeExpertIds, ...)`.

2. **Invalid rows of the non-expert-id payloads are uninitialized memory** (.cu:1373-1392; header
   comment .h:95 says "left untouched").
   - On main, rows past `count` alias the workspace and hold stale but finite data. Now they come from
     `torch::empty` and can hold NaN/Inf bit patterns: hidden states, NVFP4/FP8 scale factors and
     `token_final_scales`.
   - This is harmless only if every MoE backend ignores rows whose expert id is invalid. Any whole-tensor
     op breaks that: a per-tensor amax/dynamic quant, a swizzle that reads scale factors, a debug
     reduction, or a backend that multiplies scales before the expert mask.
   - Fix options:
     - (a) In the gather, zero the invalid rows of all payloads. This costs the same as the copy that
       valid rows already pay.
     - (b) Audit the backends and state the contract explicitly. The header comment should at least say
       "uninitialized", not "untouched".
   - Suggestion: (a), for robustness.

3. **An aborted rank still publishes combine readiness, so live peers can consume its invalid
   contributions without observing the abort themselves** (.cu:1909-1928).
   - Scenario: rank B aborts dispatch L, and its gather invalidates everything. B's MoE output for
     peer A's tokens is therefore garbage, uninitialized per item 2. B's combine block 0 still writes
     its combine flag to A, and A pulls B's combine region.
   - If A's flags are all set, A never polls `requested`. A gets a corrupted output with status 0, and
     `commit_step` accepts it if A's `_peer_failed` watcher has not fired yet.
   - Today this is prevented only by a timing argument. Every collective still waits on the dead rank D
     (D stays in the mask until recovery), so A cannot finish a step that needs D's post-death signals.
     For A to finish, D must have passed the same flags A is waiting on, and detection latency (ms) is
     much larger than the gap between flag stores (us).
   - That argument is plausible but not a protocol guarantee. It assumes, for example, no detection
     triggers other than D's death, and that `request_abort` is the only release path.
   - Cheap hardening: in combine block 0, skip the readiness stores when `requested != 0`. Optionally do
     the same in dispatch, where flags precede the wait and so only later rounds matter. Peers then stay
     blocked until their own abort arrives, or trap at the timeout, which is fail-stop. This turns
     silent contamination into propagated abort.
   - At minimum, document the invariant in `MoeA2AAbortState`.

4. **Abort path of the gather issues one `atomicExch` per thread** (.cu:1361 with .cu:302-311).
   - `abort_requested` runs in every thread of every block. The grid is `ep*maxTokens` blocks x 256
     threads: 64x8192 prefill gives ~134M same-address atomics after an abort, which is tens to hundreds
     of ms added to recovery.
   - It is also one `ld.relaxed.sys` per warp on the healthy path.
   - Different threads of one block can also see different `aborted` values, which produces mixed rows.
     This is harmless because the request is sticky, but it is untidy.
   - Fix: thread 0 reads the request (and records status), then broadcasts via
     `__syncthreads_or`/smem. Or drop the status write in the gather; see nit 3.

5. **`request_abort` depends on a side-stream H2D copy not being serialized behind the spinning kernel**
   (execution_state.py:125-134).
   - This works with default `CUDA_DEVICE_MAX_CONNECTIONS` and no MPS throttling, and the threaded test
     covers that.
   - With a single hardware connection (e.g. `CUDA_DEVICE_MAX_CONNECTIONS=1`, which is common in
     Megatron-style envs), the copy can queue behind work that depends on the stuck kernel. The watcher
     then blocks in `synchronize()` until the device timeout traps.
   - It also issues `cudaStreamSynchronize` from a non-main thread, which can fail if a global-mode graph
     capture is in progress (low risk, since losses during warmup are terminal).
   - Fix options:
     - (a) Put `requested` in pinned, mapped host memory and have kernels poll it with `ld.relaxed.sys`
       (no stream involvement). Only the slow path pays PCIe/C2C latency.
     - (b) At least document or assert the connection requirement.

6. **Healthy-path perf on FT: the gather is a full extra read+write of every received row per MoE
   layer.**
   - Example: EP64, decode at 64 tokens, ~7.5 KB per row, gives ~30 MB per layer, roughly 20 us per
     layer and ~1 ms per step on 60 layers.
   - The spin loops also double their sys-scope loads, because `requested` is polled on every iteration
     (.cu:693, 1956).
   - Alternative to consider before this sets in:
     - Lay out the fixed slots payload-major at static capacity, `[ep, staticMax, bytes]` per payload.
       Fixed per-source rows still contain zombie writes.
     - Return the hidden-state and scale tensors as workspace views, padded to static max or sliced.
     - Copy and sanitize only the small expert-id (and maybe final-scale) payload into a private
       buffer. That buffer is the only one a zombie could poison in a way that matters.
   - Poll `requested` every N iterations, or only after the first ~us of spinning.

## Nits

1. **Comments.**
   - "Acquire load" above `ld.relaxed.sys` (.cu:684, 1946) is pre-existing but now sits next to new
     code; the acquire is the `fence.acquire.sys` in combine. Dispatch has no acquire fence after its
     wait and relies on kernel-boundary ordering, the same as main.
   - nvlink_one_sided.py:973 says "Graph replays bypass the host watchdog", but the watchdog is skipped
     for eager calls too whenever the communicator is bound. Reword to say that bound execution relies
     on the device timeout.
2. **Naming per CODING_GUIDELINES.**
   - The new gather kernel's locals (`source_rank`, `source_token`, `expert_ids`) and new struct members
     (`execution_descriptor`, `source_stride_bytes`) are snake_case.
   - `MoeA2AGatherParams` / `MoeA2AFenceDescriptor` use camelCase. Pick one: guidelines say camelCase,
     but the file is snake_case.
3. **`MoeA2AAbortState::status` has no production reader.**
   - `install_mask` checks `abort.any()`, and `requested` alone suffices for that.
   - The gather sets status even when nothing was cancelled, so it means "observed", not "cancelled".
   - Either drop it (simpler kernels, fewer atomics) or document it as diagnostic only.
4. **Duplicated init logic.** `reset_local_workspace` (nvlink_one_sided.py:626-635) reimplements
   `moeA2AInitializeOp`'s zero plus 0xFF memset (op.cpp:307-311), and the test repeats it by hand
   (test_moe_a2a_abort.py, `test_cancel_and_exclusion_on_replay`). Expose a native reset, or call
   `moe_a2a_initialize`, to keep one source of truth.
5. **Inaccurate alias annotation.** The dispatch schema still annotates outputs as `Tensor(a!)[]` and
   `Tensor(a!)` aliases of the workspace, but in descriptor mode they are fresh allocations. Eager mode
   is fine; `opcheck`/schema-check mode would flag it. Worth a comment.
6. **Unnecessary check in combine.** `validateFixedSlotWorkspace` in combine (op.cpp:946-949) does not
   affect the combine layout. It only re-checks capacity; drop it or say why it is there.
7. **Combine abort zeroing** uses byte stores (.cu:1994-1998). This is fine on the abort path, but a
   `vec_t` zero store would be more idiomatic.
8. **Upward dependency.** `wide_ep_ft.recovery_enabled` imports `tensorrt_llm.executor.recovery.client`
   from MoE code. `get_wide_ep_ft_options` also drops the `extra_attrs["ep_group_health"]` fallback,
   which is a silent behavior change.
9. **Docs.** MOE_DEVELOPER_GUIDE.md does not describe descriptor execution, fixed slots, or abort
   semantics, but the guide is the stated entry point for MoE communication changes.

## Tests: coverage and gaps

Covered:
- Graph replay sees abort and mask edits.
- Fixed slots shared across two captured batch sizes, with tombstones proving excluded state is
  untouched.
- Count bounds.
- Over-capacity rejection.
- Descriptor validation.
- Fake and schema agreement.
- Abort releases a pending dispatch or combine at 0 and N tokens.
- Timeout still traps.
- `request_abort` from a thread while a kernel spins, then rearm, install and resume.

Gaps:
- **No real multi-process transport.** Every test is single process with simulated peers, so
  cross-rank behaviour is untested: a peer racing ahead after abort, rearm across ranks, and item 3.
- **`ep_size > 32`** (several peers per lane, so the `return`/`break` paths run with lanes still
  spinning) is not tested.
- **Abort arriving mid-gather or mid-combine** (racing completion) is not tested, nor is a large grid
  that fills all SMs. `_pending_wait` releases via a `fill_` kernel on another stream, which needs a free
  SM; only the execution-state test uses the real copy-engine path.
- **Invalid-row content of non-id payloads** (item 2) and the missing-sanitize-args case (item 1) are
  not asserted.
- **No FT-off regression or perf check.** Existing tests exercise the path, but nothing benchmarks or
  diffs SASS.
- `test_descriptor_schema_and_fake_signatures` asserts `"abort_state" not in names`, which tests for
  absence of an old name (history). It also hard-codes `meta` length 15.

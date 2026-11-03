I've read all five commits and checked the main claims against the stack tree. Here is the review.

# Review: WideEP fault tolerance, PRs 12, 13, 14, 15, 24

## PR 12: rank mask through a pointer, plus abort record

**[major] Op contract break not in the commit message.**
- `moeAlltoAllOp.cpp:resolveActiveRankMask` now requires a CUDA uint64 tensor on the workspace device. It used to require a CPU tensor.
- The ABI changed for every rank-mask caller of `moe_a2a_dispatch` and `moe_a2a_combine`. The message only lists the FT-off defaults.
- The local-rank-bit check is gone from the op and from `moe_a2a_{dispatch,combine}_launch`. It now lives only in `_publish_rank_mask` and `MoeExecutionState.install_mask`.
- `release_workspaces` clears `departed` bits with no check that the local rank is kept.
- Failure scenario: a mask without the local bit reaches the kernel. This rank then skips its own flag and its own receive count (`moeAlltoAllKernels.cu:665-671`).
- Fix: keep a device-side or host-side assertion in `release_workspaces` (`for rank in departed: assert rank != own`). Name the CPU-to-CUDA mask change in the message.
- Commit-message reason: none stated.

**[minor] `resolveAbortState` cache comment is wrong.**
- The comment says "once per record", but `thread_local cachedHost/cachedDevice` is a single-entry cache (`moeAlltoAllOp.cpp:~724`).
- Two alternating abort records, such as two states or two threads, re-query every call. The result is still correct, so this is wasted complexity.
- Fix: call `cudaHostGetDevicePointer` directly, or key a small map by host pointer. I did not measure how expensive the call is.

**[minor] `moe_a2a_dispatch_launch` has no check that `expert_to_slot` goes with the rank mask.**
- The launch checks only the CFT case (`moeAlltoAllKernels.cu:1357`).
- PR 24 later adds a rank-mask-only instantiation, so a C++ caller with `expert_to_slot` and no mask silently launches the non-table kernel and misroutes.
- Today only the op reaches this, and PR 24 adds its check in the op (`moeAlltoAllOp.cpp:808`). The check belongs in `moe_a2a_dispatch_launch` where the instantiation is chosen.

**[minor] The workspace layout changes for every caller, including FT-off users.**
- `FLAG_VAL` grows from 4 to 8 bytes, which shifts all later offsets and the workspace size.
- The message states this, so it is justified: "The workspace layout change (8-byte FLAG_VAL) applies to every caller."
- I found no other C++ consumer of the old offset. The Python side reads offsets from `metainfo`.

**Checked and fine**
- The `released` write is lane-local and nothing reads it later in the same kernel.
- The mask read before `cudaGridDependencySynchronize` is justified by the in-code comment, and `install_mask` and `release_workspaces` both synchronize before writing the mask.
- Mask-free instantiations fold `abort` to `nullptr`.
- The `routed_slot_ids` shared array is published by the `__syncthreads()` at `moeAlltoAllKernels.cu:613` before it is read.
- `wait_round_flag` still traps on timeout.

**Tests:** `test_moe_a2a_abort.py` is good. It covers replay, abort, the released marker, and the timeout trap. It has no negative test for the dropped local-rank check.

**Verdict: approve with changes.** Add the local-bit assertion in the publisher path and move the `expert_to_slot` check into the launch function.

## PR 13: lane-parallel mask test and 8-byte round-state load

**[minor] The EPLB stats-copy loop is duplicated.**
- `moeAlltoAllKernels.cu:676-714` now has two copies of the same copy body. One is ballot-based and one is serial.
- The no-mask copy could use the same loop, with the ballot replaced by an all-ones mask under `if constexpr`.
- The only cost is maintainability, but it is a divergence risk for the next edit.

**[minor] The perf claims have no reproducer.**
- The message quotes "2.1 us of the 3.0 us" and "1.6 us less" at EP32/64 tokens.
- The PR adds no benchmark or test and gives no link.
- `load_round_state` has no new test of the 8-byte alignment beyond a `static_assert` on the index. The alignment depends on `reserve(..., 8, 8)` in `moeAlltoAllOp.cpp:238`.

**Checked and fine**
- `__ballot_sync(~0U)` is reached only through the warp-uniform `is_last_token` (broadcast with `__shfl_sync`), so the full mask is valid.
- The FT-off path is unchanged: `else if constexpr (ENABLE_EPLB)` keeps the old loop.

**Verdict: approve.** The duplication is optional to fix. Ask for the benchmark or a link.

## PR 14: MNNVL remap among survivors

**[major] Failed remap leaves a window where a live peer maps a rank's fresh pages that this rank then unmaps.**
- `mnnvl_memory.py:remap` runs `_create_and_map_handles` and then a second `gather(error)` as an outcome exchange. The failure path unmaps, but peers that mapped successfully may already have written into the pages.
- This is only safe if recovery treats any remap failure as fatal.
- The allocation is reset after a successful retry, so this probably holds. The docstring says "a later attempt may remap it", while the cleanup failure case marks the allocation BROKEN with only a log line.
- Fix: state in the docstring that all live ranks must retry together. Today the handle exchange has no generation or round tag, so one rank retrying while others have given up would pair with the wrong exchange.
- Commit-message reason: none stated.

**[minor] `MnnvlMemory.pin()` and `unpin()` change lifetime semantics for any caller.**
- `close_mnnvl_memory` now returns early on `pin_count != 0` with `logger.warning("Retaining pinned MNNVL allocation")` (`mnnvl_memory.py:866`).
- That path is reached only if something pinned the allocation, which only recovery does. FT-off is unaffected, as the message states.
- Check: `MoeExecutionState` keeps a reference to `memory` in `_workspaces` until `close()`. So `__del__` cannot run while pinned, and `close()` unpins before clearing, which frees the allocation. No leak.

**[minor] `_LiveRowsComm` is an adapter implementing only `Get_rank`, `Get_size` and `allgather`.**
- It must stay in sync with whatever `_create_and_map_handles` calls on its communicator.
- No test pins that interface. The fake driver tests use it, which gives partial coverage.

**Tests:** strong CPU tests with a fake driver. They cover failure ordering, a cleanup failure leaving the allocation BROKEN, and a retry. The multi-process late-write test is in PR 15.

**Verdict: approve with changes.** Document the exchange-round invariant for retries.

## PR 15: `MoeExecutionState` and the bound path

**[major] Auto-selected CFT turns bound execution into a hard failure.**
- `bind_execution_state` raises `ValueError("Bound execution requires the fence path ... Set TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT=0")` (`nvlink_one_sided.py:~741`).
- `select_cft_counted_writes` enables CFT automatically on supported driver and GPU (`nvlink_one_sided.py:140-166`).
- `for_model` is called from the decoder runner whenever `peer_loss_recovery_enabled()` (`runner.py:225`).
- On CFT-capable hardware, a user who enables recovery gets an exception during model load unless they know to set the env variable.
- Fix: when peer-loss recovery is enabled, make `select_cft_counted_writes` return False and log once, instead of raising later.
- Commit-message reason: none stated.

**[major] Bound mode drops the host watchdog for eager calls too, but the docs say otherwise.**
- The new `NVLinkOneSided` docstring says "The watchdog observes eager launches only."
- The code skips `watch_collective` whenever `state is not None`, eager or captured (`nvlink_one_sided.py:1109-1112` and the combine path).
- Bound eager runs therefore get no host watchdog, and the comment's claim that bound execution "relies on the abort record... and on the device timeout" is the only coverage.
- The commit message does not say that eager bound execution loses the watchdog.
- Fix: either keep `watch_collective` in the bound eager case when not capturing, or correct the docstring and the message.

**[major] Behaviour change at config time for FT-only users, not stated.**
- `get_wide_ep_ft_options` no longer raises on `use_cuda_graph=True` with `TLLM_FAULT_TOLERANCE_MODE=1`.
- Detection-only FT (no recovery, so no bound state) now fails later, as a `RuntimeError` raised at capture time inside `dispatch`, instead of at config time.
- Fix: keep the config-time check for the unbound case (`health is not None and no recovery`), or state the change in the message.

**[minor] `capture_active_rank_mask` and `active_rank_mask_snapshot` are still computed when bound.**
- The snapshot is stored in `_dispatch_state` (`nvlink_one_sided.py:1119`) but the kernels ignore it in bound mode.
- Combine's "fail closed on generation change" now has no effect when bound.
- Either skip the capture for bound layers or document why it is kept.

**[minor] `device_visible_host_zeros` is a thin wrapper over `torch.zeros(...).pin_memory()`.**
- The name promises a device-mapped address, but the mapping is done separately by `cudaHostGetDevicePointer` in the op.
- It relies on torch's pinned allocator producing mapped memory. If that ever changes, the first symptom is an opaque CUDA error.
- Fix: rename it or inline it, and cover the mapping in a test.

**[minor] The abort polling cadence and the host store carry the correctness argument.**
- `request_abort` is `self.abort[_ABORT_REQUESTED] = 1`, a torch `__setitem__` on a pinned tensor, from any thread.
- That is fine, but nothing tests thread-safety against `rearm`'s `abort.zero_()`. The docstring claims "an abort requested while the workspaces reset is kept", and the order inside `rearm` (clear the abort, then reset the workspaces) does support it.

**[minor] CPU tests got a blanket `pytestmark = pytest.mark.cpu_only`.**
- It is added to the whole `test_alltoall_watchdog.py` file. Confirm no test in that file needs CUDA.

**Tests:** `test_moe_a2a_late_writes.py` is a genuinely good negative-control design (it runs `killed`, `writing` and `no-remap`). It sits in `l0_gb200_multi_gpus`. It fully builds the comm with `NVLinkOneSided.__new__` and hand-set fields, which will break silently when the constructor changes.

**Verdict: request changes.** Fix the CFT hard failure, correct or restore eager watchdog coverage, and decide the config-time check for detection-only FT.

## PR 24: expert-to-slot table in the dispatch

**[major] A silent fallback if `ENABLE_RANK_MASK` is false.**
- `moeAlltoAllKernels.cu:~1435`: with the mask off, `launch(std::false_type{})` runs even if `params.expert_to_slot != nullptr`.
- The op-level `TORCH_CHECK` (`moeAlltoAllOp.cpp:808`) is the only guard. Add `TLLM_CHECK_WITH_INFO` in `moe_a2a_dispatch_launch` (see PR 12) so the instantiation choice is checked where it is made.
- The generic lambda inside the `SWITCH_BOOL` macro is also harder to read than adding `USE_EXPERT_TO_SLOT` to the existing switch.

**[minor] Python-side caching of `one_slot_table` can go stale.**
- `SingleLayerMoeLoadBalancer.one_slot_table` caches the first eager result, including `None` (`moe_load_balancer.py`, `_one_slot_table_queried`).
- A later placement change from replicas to one slot per expert is never noticed. The docstring accepts this: "the routing kernel serves every placement".
- The unit test `test_one_slot_table_keeps_a_none_result` encodes it.
- It is only safe if `routesEachExpertToOneSlot` cannot flip after the first call. C++ `commitOneSlotRouting` and `checkRoutingKernelServes` enforce that for the committed case, but not for the uncommitted None case.

**[minor] The table is a `from_blob` view with no owner.**
- `moeLoadBalanceOneSlotTable` returns a view of `globalSlotIds`, "valid while the layer's balancer exists" (per the code comment).
- Nothing in Python ties the tensor's lifetime to the balancer. If a captured CUDA graph outlives a balancer re-creation (restore or shutdown), it reads freed memory.
- Fix: keep the balancer alive from `_one_slot_table`'s owner, and assert on shutdown that no graph is live.

**[minor] `ExternalCommMoEScheduler.__init__` reads `peer_loss_recovery_enabled()`, which is `lru_cache`d.**
- The value is fixed per process, so the scheduler field is redundant with the call. Test monkeypatching is the only reason for it.

**FT-off behaviour:** `routesEachExpertToOneSlot` returns false without `mPeerLossRecovery`, `oneSlotPerExpert` is false, and the scheduler checks `_peer_loss_recovery` first. The FT-off path is unchanged, as the message says.

**Tests:** good. There is a bit-exact comparison against pre-lookup slot ids with a table rewritten in place under a replayed graph, plus the scheduler gating test.

**Verdict: approve with changes.** Move the check into the launch function and clarify the table's ownership.

## Summary

| PR | Verdict |
|---|---|
| 12 | approve with changes |
| 13 | approve |
| 14 | approve with changes |
| 15 | request changes |
| 24 | approve with changes |

I read everything with Read, Grep and Glob only, and modified nothing. I did not build or run any of it. The kernel-ordering and ownership points above come from reading the code, not from execution.

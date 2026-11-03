# Code-owner review: PRs 16–19 (WideEP fault-tolerance stack)

I read all four diffs and checked the claims below against `/tmp/claude-161238/wt-stack`. I did not read the last ~700 lines of 16.diff (C++ tests), the tail of 17.diff or 18.diff, or any of 19.diff beyond what is quoted here. I did not run any code. Line numbers are tree lines unless they say "diff".

With `TLLM_FAULT_TOLERANCE_MODE` unset, I found no behaviour change. The gating is consistent everywhere I checked:
- The C++ `isPeerLossRecoveryEnabled()` requires both the mode and the authority variable.
- The Python `peer_loss_recovery_enabled()` is `lru_cache`d.
- `SingleLayerMoeLoadBalancer.host_copies` defaults to `updates_enabled`.
- `loaded_slot_layout` returns its inputs unchanged when `num_extra_local_experts` is 0.
- The `isSupported()` refactor is logically equivalent to the original.
- The `updateWeightsRoutine` / `...ByCpu` merge into `loadAndPublishPlacement` keeps the same call order.
- With no unusable mask, `doPlacement`'s `nextLocalSlotId` equals the old `currentRankSlots`.

---

## PR 16: ghost slot memory and C++ restore

**1. major — the commit message does not match the gating the code implements.**
- Location: diff 16 commit message; `moeLoadBalancer.cpp:1208`, `moeLoadBalancer.cpp:507`, `HostMemoryMoeWeightUpdater::finalizeWeights`.
- The message says "mFaultToleranceMode is set at finalizeModel from TLLM_FAULT_TOLERANCE_MODE". No such member exists. The member is `mPeerLossRecovery`, and it needs the authority variable too.
- The gate is read from two places: the cached `mPeerLossRecovery`, set at `finalizeModel`, and a live `isPeerLossRecoveryEnabled()` call. The live call is in `setInitialWeightAssignments`, `addSingleWeightSlot`, `finalizeWeights` and `HostAccessibleDeviceAllocator::getInstance`.
- Failure scenario: `setInitialWeightAssignments` runs before `finalizeModel`, so it relies on the live env. If the env changes between the two, the layer accepts `-1` slots and then runs `finalizeModel` with `mPeerLossRecovery == false`.
- Fix: read the gate once at `MoeLoadBalancer` construction, store it, and pass it down. Fix the message.
- Stated reason: none. The message is inaccurate, not justified.

**2. major — one PR bundles too much, and its tests are costly to review.**
- It adds 710 lines of VMM/GDRCopy code (`ghostSlotMemory.cpp`), a new `HostAccessibleDeviceAllocator` API, the restore/quiesce state machine, and about 900 lines of gtests.
- A code owner cannot validate the ghost-piece and scratch-swap logic and the worker-thread locking in one pass.
- Fix: split into (a) `GhostSlotMemory` plus the allocator `register`/`unregister` API, (b) `restorePlacement`/`quiesce`, and (c) the `endIter(interrupted)` and abandoned-stage path.
- Stated reason: none.

**3. minor — a busy-wait polls a `std::function` and, on abandon, writes the GPU-owned signal word from the host.**
- Location: `moeLoadBalanceKernels.cu:659-668`.
- This is safe only under the precondition "device work of that iteration is done". `kGPU=0` and `kCPU=1`, so `fetch_or` is idempotent, but a stray late device `+= kCPU` (`.cu:143`) would flip the owner back.
- The only protection is the header comment and the Python docstring on `quiesce`. Nothing in C++ enforces that the device work is done.
- Fix: have `quiesce` assert, or take a device-sync handle, instead of relying on caller discipline.
- Stated reason: the header documents the precondition ("only once the device work of that iteration is done"), so this is partly justified.

**4. minor — `moeGhostSlotStorage` copies the whole old tensor, empty slots included, into the new storage.**
- Location: `moeLoadBalanceOp.cpp` in diff 16, around line 2266.
- Ghost chunks are mapped to scratch at that point. The copy writes junk into scratch memory shared by every range, and it wastes bandwidth.
- Fix: copy only the non-empty slot runs, or document that scratch is write-tolerant. The `ghostSlotMemory.h` comment does state that strays hit scratch, so this is mostly justified.

**5. minor — `routesEachExpertToOneSlot` and `mOneReplicaPerExpert` are stale after EPLB updates.**
- `mOneReplicaPerExpert` is refreshed only in `setInitialWeightAssignments`, `reconfigureMaskOnly` and `installPlacement`, not in `updateWeightsRoutine`.
- It is read only when `mLayerUpdatesPerIter == 0`, so it is currently correct.
- It is a trap for future callers. Fix: a one-line comment on the member, or compute it lazily.

**6. minor — style.**
- Location: `moeLoadBalancer.cpp:1841` and `moeLoadBalancer.cpp:1945` in diff 16.
- Two `TLLM_CHECK_WITH_INFO(\n mPeerLossRecovery, "…")` calls are split across lines where clang-format would join them. Pre-commit may catch these.

Verified OK:
- `mWorkerDoneIterId` and `mIterId` are both -1 for static routing, because Python never calls `start_iter` there (`moe_load_balancer.py:1781`). `restorePlacement` therefore works without a prior `quiesce`.
- The `mPending` `Piece*` pointers stay valid when the `allocation.pieces` vector is moved into the map.

**Verdict: request changes.** Please fix #1 and consider #2.

---

## PR 17: extra slots and Python restore

**1. major — `extra_slots_mode()` is a getter that can run a collective on first call.**
- Location: `moe_load_balancer.py`, `extra_slots_mode` → `_ghost_slot_memory_host_accessible()` (an MPI `allgather`) in diff 17.
- It is currently safe only because `register_weight_slots_after_to_cuda` pre-calls it on every rank, and only when `_largest_loss is not None and is_dynamic_routing()`.
- Failure scenario: any later first-time call from a path that is not rank-symmetric, or after a peer died, hangs. A hang is the worst failure mode for a fault-tolerance feature.
- Fix: compute `ghost_host_accessible` once in `register_weight_slots_after_to_cuda` and store it on the instance. `extra_slots_mode` then becomes pure.
- Related: the module-global `_ghost_host_accessible` is never reset. A second `MoeLoadBalancer` in the same process (draft model, tests) inherits the cache. Store it on the instance.
- Stated reason: the docstring says "The first call is collective", so this is acknowledged but not justified as a design.

**2. major — node identity comes from `socket.gethostname()`, but host copies are shared through `shared_mpi_comm`.**
- Location: `_gather_hosts` in diff 17; used by `failure_domains` and `ring_copy_nodes` (PR 18).
- `shared_mpi_comm` comes from `Split_type(COMM_TYPE_SHARED)`, which is what actually decides who shares `/dev/shm`.
- Failure scenario: ranks in separate containers or pods that share a hostname, or one physical node reporting different hostnames. `failure_domain=node` then groups the wrong ranks, and ring copy nodes and `host_copy_ranks` differ between ranks of one shared comm. `rank_expert_ids(local_rank)` then disagrees and the shm layout is corrupt.
- Fix: derive the node key from the shared-comm split (e.g. allgather the shared comm's lowest global rank), not the hostname.
- Stated reason: none.

**3. minor — `end_iter` now always passes `interrupted=` to the binding.**
- With recovery off the value is always `False`, so this is harmless. The existing test was edited to `assert_called_once_with(0, interrupted=False)`.
- It also resets `func_called_count` and `statistic_flag_tensor` on every layer when interrupted. Only recovery reaches this path.

**4. minor — `restore_lost_experts` and `recovery_placements` are untested for some edge cases.**
- I saw tests for rank-out-of-range and layer-count mismatch. I did not confirm tests for `lost` experts larger than any placed id (the `bincount(minlength=…)` path), for ties, or for determinism across ranks. Determinism is the property recovery depends on.
- Fix: add a seeded-random property test that all ranks compute identical placements.

**5. minor — the `ValueError` for `_gather_hosts` blocks PP or attention-DP deployments with recovery on.**
- The message is clear and the check is correctly gated. It is a design limit, not a bug.

**6. minor — `placement_digest` hashes `placement.tobytes()` without fixing the dtype.**
- Location: `placement_digest`.
- `recovery_placements` builds placements as int32, and `base` is coerced to int32. A caller that passes int64 placements would produce different digests on different ranks.
- Fix: cast to `np.int32` inside the digest.

**Verdict: approve with changes.** Please fix #1 and #2.

---

## PR 18: ring host copies and staging

**1. major — the weight-reload path leaks the old shared-memory mapping.**
- Location: `moe_load_balancer.py:296-321`.
- Under recovery, `finalize_layer_weights` calls `self.unlink_shared_memory()` ("A weight reload shares the layer again under the same name"). That only removes the name.
- It then does `self.own_shm = shm` without closing the previous segment. The old segment stays mapped in this process and in every peer's `imported_shms`, and `/dev/shm` usage grows on each reload.
- Peers also append duplicate entries to `imported_shms`.
- Fix: `if self.own_shm: self.own_shm.close()` before reassigning, and clear `imported_shms` before re-import.
- Stated reason: the message does not mention reload semantics, so this is not justified.

**2. major — no cap or visibility on the host memory cost.**
- Layers with replicas keep every expert's host copy on every node. Ring layers keep about 1/nodes of the experts.
- Before this PR a static placement kept no host copies. `_check_shm_space` raises a clear error, but only after model load has started, and it checks per rank before the other ranks of the node have allocated. `_reserve_shm` catches the overfill, but only at `posix_fallocate` time.
- Fix: log the total per-node `/dev/shm` requirement at `add_layer` and fail early (fail-fast sizing before weight loading).
- Stated reason: the message and the error text acknowledge the requirement ("enlarge it, e.g. with docker run --shm-size"). The fail-late timing is not justified.

**3. minor — `_unmap_pages` uses `ctypes.CDLL(None).madvise(MADV_DONTNEED=4)` with a hard-coded constant.**
- Location: `moe_load_balancer.py:86-95`.
- It is Linux-only and relies on `MAP_SHARED` tmpfs semantics, which hold for Python's `SharedMemory`. Failure is logged at debug level only. `_SHM_MOUNT` is also hard-coded.
- Fix: use `mmap.MADV_DONTNEED` (available in Python 3.8+ on Linux), which removes the ctypes code.
- Stated reason: the perf numbers in the message (21–46 s vs 2.7–5.7 s) justify the behaviour, not the ctypes implementation.

**4. minor — `assert` is used for a runtime input check.**
- `restore_copy_plan`: `assert all(end > align(begin) …), "no aligned staging room"`. This is stripped under `python -O`. Use `ValueError`.
- `live[turns[node] % len(live)]` raises `ZeroDivisionError` if `placements` did not come from `recovery_placements`, which filters uncopied experts.

**5. minor — ring-layer restrictions are enforced at `add_layer` time with a model-wide `ValueError`.**
- Mixing ring and replica layers fails the model build. This is gated on recovery being on, so there is no effect otherwise.
- The check is correct (`ring != (self.copy_nodes is not None)`). It limits which models can enable recovery, and the limit should be stated in the PR description.

**6. minor — `stage_restored_slots` calls `stream.synchronize()` and a barrier once per chunk.**
- The chunk count is roughly (bytes per target) / (half of the workspace row). Python overhead is bounded, but I did not verify the 2.7–5.7 s claim.
- The plan arithmetic (halves, cursor, no overlapping offsets, no infinite loop in `_next_piece` loop) checks out on reading.
- The GB200-only multi-GPU test is the sole check of bit-exactness. `tests/unittest/_torch/moe/test_ring_host_copies.py` exists, and I did not read it. I did not confirm it covers weights larger than one half or a plan with several targets, which is where the offset logic can break.

Verified OK:
- With recovery off, `keep_experts(range(E))` gives the same split as the old `expert_start`/`expert_end`.
- `get_load_expert_ids` returns the same list as the removed `load_expert_ids`.
- The pre-barrier unlink in `finalize_model` is gated on recovery.

**Verdict: request changes.** Please fix #1 and #2.

---

## PR 19: tune and key MoE tactics on loaded slots

**1. major — a tactic tuned on `num_loaded` slots is run over all local slots, and no GPU test checks this.**
- The design assumption, from the docstring: "a slot without tokens adds no tile and the tactics fix only tile shapes". It is plausible.
- The tests are metadata-only: meta-device tensors, a mock tuner, and `loaded_tactics <= tactics`. There is no run of `fused_moe` or the CuTe DSL ops with extra slots, empty versus restored, compared against the no-extra output.
- Failure scenario: after a restore fills the extra slots, a kernel tile config valid for the loaded layout reads or writes wrongly for the larger layout. Nothing would catch it before production.
- Fix: add one GPU numerics test (extra slots empty, then filled, versus the baseline).
- Stated reason: the message says "tune and key … on the loaded slots only, as they run". "As they run" is not verified by a test.

**2. minor — coverage is uneven across backends.**
- Direct-metadata and count-native CuTe DSL paths pass 0. The Rubin gather path does not pass the kwarg at all (`nvfp4.py:473-476`). Other paths (`TrtllmCutedslGroupedGemmBf16Impl`, TRTLLM-gen, DeepGEMM) do not pass it either.
- Those keep keying on all slots. That is safe, but the PR title overstates the scope. It only claims "Cutlass and CuTe DSL", which is accurate.

**3. minor — public op schema grows.**
- `trtllm::fused_moe` and three CuTe DSL ops gain a trailing `num_extra_local_experts: int = 0`, plus the fakes. The default keeps callers compatible. The fused-MoE op schema is not versioned.

**4. minor — duplicated plumbing.**
- The same `(num_experts, num_local, offset)` re-derivation exists in four runners (`tuning_layout`), and `loaded_slot_layout` assumes `num_experts % num_local == 0` and `offset == rank * num_local` without asserting either.
- Fix: assert both in `loaded_slot_layout`.

**5. minor — behaviour preserved with recovery off.**
- The cache key `(unique_id(), 0)` in `tuning_config_cache` is internal.
- `unique_id()` is unchanged, so persisted autotuner caches stay valid.
- Both GEMM calls in `fused_moe` now share one `tuner_inputs` list instead of two separate ones. I did not check whether `choose_one` or any pre-hook mutates its `inputs` argument.

**Verdict: approve with changes.** Please add the GPU numerics test (#1) before merge.

---

## Summary

| PR | Verdict | Required before merge |
|----|---------|----------------------|
| 16 | request changes | Fix gate source and message (#1); consider splitting (#2) |
| 17 | approve with changes | Make `extra_slots_mode` collective-free (#1); node identity from the shared comm (#2) |
| 18 | request changes | Fix the shm leak on reload (#1); fail-fast `/dev/shm` sizing (#2) |
| 19 | approve with changes | Add a GPU numerics test with extra slots (#1) |

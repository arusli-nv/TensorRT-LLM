C++/CUDA review, red team 3, lane "code-review-cpp" (read-only)
Target: v2 = cef4670789 vs upstream/main 0829689c52. Diff: `git diff upstream/main v2 -- cpp/ ':!cpp/tests'` (26 files, about +4000/-272), plus cpp/tests separately.
Method: static, line by line. Nothing was built or run. Line numbers refer to the v2 files.
Format: severity | CONFIRMED (read in the code) or PLAUSIBLE (depends on the caller or the runtime) | file:line | failure scenario | minimal fix.

== SUMMARY ==
No blockers. One major item, which is a correctness hazard that only Python mitigates. Seven minor items and twelve nits.
Several round-2 findings are still open in v2, including the fix-minors fix #9: moe_a2a_initialize still ignores CUDA errors.
FT-off parity holds. The mask-free instantiations compile every new branch away, and expandInputRowsKernel is unchanged.
moeComputeRouteDevice selects the same kernel. isSupported is equivalent. The FT-off costs are build size plus one mutex per slot add (N3, N4).

== MAJOR ==
M1 | PLAUSIBLE (Python mitigates it today) | runtime/moeLoadBalancer/moeLoadBalancer.cpp:1452-1456 (workerThread); :601 (maybeStartUpdateWeights); kernels/moe/loadBalance/moeLoadBalanceKernels.cu:657 (abandoned wait)
  Scenario: after endIter(N, interrupted=true), quiesce sets mAbandonedIterId. The worker then sets the CPU stage itself for the layers the forward never reached.
  The worker then falls through to maybeStartUpdateWeights() for every such layer whose update is scheduled in iteration N.
  updateWeightsRoutine runs doReplication and doPlacement on load factors from a partial iteration. Some ranks gathered layer k's statistics before the raise and some did not, so the ranks compute different placements.
  Each rank then installs its own placement. Routing becomes inconsistent across ranks: tokens go to slots that hold other experts, and the outputs are silently wrong.
  resetUpdateSchedule() clears the load factors, but only after the update has already been applied.
  Today this is safe only because peer_loss_recovery installs rank 0's placement on every rank afterwards: restore_placement, or reconfigure_mask_only on ranks that already match.
  Any path that resumes after an interrupted iteration without that install (a non-peer-loss raise, or a future caller) diverges.
  Round-2 maintainer review PR6 #76 reported this; it is still open.
  Fix (about 3 lines): in maybeStartUpdateWeights, return early when `mMoeLoadBalancer->mAbandonedIterId.load() == mIterId`, and clear that layer's expertLoadFactor. Add a gtest assertion to QuiesceReleasesTheLayersAnInterruptedForwardDidNotReach that the placement did not change.

== MINOR ==
m1 | CONFIRMED (code), PLAUSIBLE (impact) | kernels/moe/communication/moeAlltoAllKernels.cu:611-612 vs :690-713 (receive_source)
  Each receive CTA calls cudaTriggerProgrammaticLaunchCompletion right after cudaGridDependencySynchronize. Only after that does it write source_valid_counts, compact_expert_ids/weights, eplb_stats_out, recv_counters[...] = 0 and flag_val[kReleasedRoundIndex].
  Under PDL, a dependent kernel's GDS only guarantees visibility of writes made before the trigger. A PDL-launched consumer (TRTLLM_ENABLE_PDL=1 MoE kernels) can therefore read stale counts or compact lists.
  This is round-2 cpp.md A1, still open. Fix: move the trigger to the end of receive_source, after a __syncthreads() that follows thread 0's stores. The receive CTAs run off the critical path, so this costs nothing.
m2 | CONFIRMED | thop/moe/communication/moeAlltoAllOp.cpp:1203 and :1218 (schemas)
  `Tensor? abort_state` is written by the kernels (record_cancelled sets status = kCancelled), but the schema declares it read-only.
  Functionalization, torch.compile or graph-partition passes may reorder or elide reads of status around the op.
  Fix: declare it `Tensor(f!)? abort_state` in dispatch and `Tensor(b!)? abort_state` in combine. This is round-2 A3, still open.
m3 | CONFIRMED | moeAlltoAllOp.cpp:317 and :323 (moeA2AInitializeOp)
  cudaMemsetAsync and cudaDeviceSynchronize are unchecked. The new barrier=false path is the recovery reset of one rank's workspace after a peer loss.
  At that point a timed-out kernel may have trapped and left a sticky error. Initialization "succeeds", and the failure surfaces later as an unrelated op error, outside the recovery's error handling.
  The fix-minors commit f499b5ef8d fixed this, but it is not in v2. Fix: wrap both calls in TLLM_CUDA_CHECK, or cherry-pick f499b5ef8d.
m4 | PLAUSIBLE | runtime/moeLoadBalancer/ghostSlotMemory.cpp:674-705 (releasePending)
  (a) It unmaps backed pieces without the cudaDeviceSynchronize that free() does at :558, and cuMemUnmap does not wait for queued work.
      peer_loss_recovery calls release_uninstalled_backing() when a second rank fails mid-recovery. If stage_restored_slots raised between its copy_ and stream.synchronize(), queued copies still target the slots.
      Unmapping under them is an illegal-address fault, which is sticky and kills the survivor.
  (b) A failed releaseWithTag still sets piece->backed = false. The manager has dropped the entry (bad handle), so the next back() fails closed, but the state is reported wrongly.
  Fix: call TLLM_CUDA_CHECK(cudaDeviceSynchronize()) before the loop. On failure, call dropPiece(*piece) so the piece ends on scratch, as back() does.
m5 | CONFIRMED (code), PLAUSIBLE (trigger) | moeLoadBalancer.cpp:979-984 (finalizeWeights) and :1090 (updateWeights)
  With FT on, a layer without host weights is accepted even with online EPLB. An update that fills a slot then evaluates `mHostWeights[name][newExpertId]`.
  operator[] inserts an empty vector and then indexes past its end, which is out of bounds and UB. With CPU copies, every copy thread calls it at once, so the std::map insertion is also a data race.
  It is latent because Python drops host weights only for static layers. Round-2 A2, still open.
  Fix: in updateWeights, add TLLM_CHECK_WITH_INFO(hasHostWeights(), ...) before the slot loop, and use mHostWeights.at(name).
m6 | CONFIRMED | nanobind/runtime/heartbeatSender.cpp:98, :106, :121
  - The interval is converted before TLLM_CHECK(intervalS > 0) runs at :100. duration_cast of inf/NaN or a huge value is UB.
  - nb::init runs getaddrinfo with the GIL held, so a slow DNS lookup stalls every Python thread at startup.
  - If the std::thread constructor throws, the socket leaks.
  Fix: check before converting, and bound the interval (for example to ≤ 3600 s). Add nb::call_guard<nb::gil_scoped_release>() to the init. Wrap the socket in a guard until the thread starts. This is round-2 #39, still open.
m7 | PLAUSIBLE | thop/moe/loadBalance/moeLoadBalanceOp.cpp:175-186 (moe_load_balance_one_slot_table)
  - The op returns from_blob over the balancer's globalSlotIds with no lifetime tie, so a cached tensor dangles after shutdown.
  - It uses c10::cuda::current_device() instead of the balancer's device.
  - It has a side effect: routesEachExpertToOneSlot() sets mOneSlotRoutingUsed, which later restricts restorePlacement. That side effect is invisible to the schema `(int) -> Tensor` and to the CompositeExplicitAutograd registration, so a compiled graph can drop or reorder the call.
  Fix: derive the device from the balancer. Document that the table borrows from the balancer, or hold a shared_ptr in the DataPtr context. Set mOneSlotRoutingUsed through a separate explicit binding instead of a getter-shaped op.

== NITS ==
N1 | CONFIRMED | moeAlltoAllKernels.cu:587-592 (checked_source_count)
  observe_abort() already records kCancelled, so record_cancelled is duplicated. The ptrs.abort null checks are dead, because the launcher accepts source_valid_counts only on the rank-mask fence path and the op requires abort_state.
  Fix: add a launcher TLLM_CHECK(params.source_valid_counts == nullptr || params.abort != nullptr), then drop both null checks and the second record. (Round-2 A5.)
N2 | CONFIRMED (benign race) | moeAlltoAllKernels.cu:711 vs :896 (released_round in the token CTA)
  In the same grid, a receive CTA does a plain store to flag_val[1] while the last token CTA does a plain load of it. Under the PTX model this is a data race.
  Outcomes: this rank skips publishing for this round, or publishes. Both are safe only because the host requests the abort on every rank anyway.
  Fix: use ld.relaxed/st.relaxed (or volatile) for the marker, and add a comment stating that host contract.
N3 | CONFIRMED | moeAlltoAllKernels.cu:1601
  SWITCH_BOOL(expert_to_slot) doubles every fence-dispatch instantiation, FT-off included (build time and binary size; no runtime cost).
  Fix: nest it under ENABLE_RANK_MASK, or TLLM_CHECK(!expert_to_slot || enable_rank_mask) and instantiate only <MASK=true>.
N4 | CONFIRMED | moeLoadBalancer.cpp:861 (addSingleWeightSlot)
  The FT-off CPU-copy path now takes the GhostSlotMemory singleton mutex for every slot. It is harmless, but it is FT machinery on the FT-off path.
  Fix: gate it on mFaultToleranceMode or the env, and fold it into the single FT gate (round-2 #6, still open: env reads in setInitialWeightAssignments :511 and finalizeWeights :981 against the cached member elsewhere).
N5 | CONFIRMED | ghostSlotMemory.cpp:261 (ScratchSwap::setup)
  `catch (std::exception const&) {}` is an empty catch. Per CODING_GUIDELINES it should log. Fix: TLLM_LOG_WARNING before the rethrow. (Round-2 maintainer nit.)
N6 | CONFIRMED | ghostSlotMemory.h:34-50 and the public API comments; the new declarations in moeLoadBalancer.h
  They use `//` where Doxygen `//!` is expected on public interfaces, and the 16-line design block belongs in the dev guide. moeLoadBalancer.h now pulls cuda.h and virtualMemory.h into a widely included header.
  Fix: use `//!`; forward-declare GhostSlotMemory::Range as a std::pair alias in a small header. (Round-2 #41.)
N7 | PLAUSIBLE | moeAlltoAllKernels.cu:727-735 (receive CTAs at blockIdx ≥ token_ctas)
  Freedom from deadlock assumes that blocks are dispatched in blockIdx order, so that the spinning receive CTAs never hold SMs that this rank's token CTAs need.
  CUDA does not guarantee that order. Under MPS, green contexts or concurrent kernels holding SMs, two ranks could each fill their SMs with receive CTAs. They would then wait until the abort or the 300 s trap.
  Fix: document the assumption next to grid_size (:1546), or let receive CTAs spin only after an atomic counter shows that every token CTA has published.
N8 | CONFIRMED | moeLoadBalancer.cpp:1330-1340 (restorePlacement)
  If installPlacement throws at layer k, layers < k are installed, the dead mask is not updated, and the pending backing is neither kept nor released. The comment says the process ends; nothing enforces it.
  Fix: in a catch, call releasePending only if no layer was installed; otherwise rethrow with a "fatal: partial install" message that Python treats as fatal.
N9 | CONFIRMED | moeLoadBalancer.cpp:879 (appendSlotRanges)
  `(mHeight - 1) * mPitch + mWidth` underflows for mHeight == 0 (size_t), which backs the rest of the allocation. Fix: skip empty slots.
N10 | CONFIRMED | hostAccessibleDeviceAllocator.cpp:409 (registerHostAccessible)
  If recordAllocation throws (for example on a duplicate key), the new GdrMemDesc and the GDR pin and map leak. Fix: use a unique_ptr and unmap/unpin on failure.
N11 | PLAUSIBLE | nanobind/runtime/peerFailureWatch.cpp:66-68, :179
  If the thread constructor throws, the posted MPI_Irecv leaks, and the comm is later freed with a pending request.
  report() acquires the GIL. If the watch is still running at interpreter finalization, gil_scoped_acquire on a finalizing interpreter can hang or abort the thread.
  Fix: cancel the request on a constructor failure, and check Py_IsFinalizing() (or a stop flag) before acquiring.
  Round-2 C (destroyed on its own thread, std::terminate) is FIXED by stopForDestruction plus detach. The member teardown after detach was checked: the GIL is re-held, and the unique_lock no longer owns the destroyed mutex.
N12 | CONFIRMED | moeAlltoAllOp.cpp:185-191 (resolveAbortState)
  The thread_local cache is keyed by host pointer only. This is fine for one device per process, but a reused pinned block on another device or context would return a stale device pointer. Fix: key it by (ptr, device), or drop the cache (cudaHostGetDevicePointer is cheap and runs only at capture time).

== TESTS (cpp/tests/unit_tests/runtime/moeLoadBalancerTest.cpp, +1034) ==
- Good: the planner, restore, quiesce (including the interrupted forward), the FT-off rejections, and the GhostSlotMemory scratch/back/fail/rollback and graph-capture behavior are all exercised with the real classes. ScopedFaultToleranceMode works because isFaultToleranceModeEnabled() reads the env on each call.
- Gap: there is no C++ test for the A2A FT kernels (abort, fixed slots, receive_source, compact copy, expert_to_slot) or for expandInputRowsFromSourceSlotsKernel. Coverage is Python only, and the multi-GPU late-write test needs a multi-GPU stage. Confirm these files are listed in a test-db stage.
- Gap (M1): no test asserts that the placement is unchanged after an abandoned iteration.
- Weak: AllUsableMaskPlacesExactlyAsNoMask has a golden only for 4 experts; the other cases compare the new planner with itself (round-2 #10). Fix: add goldens from upstream doPlacement.
- Flaky risk: FreedAndFailedAllocationsLeaveNoMemoryBehind sizes slots at 3/4 of free memory and asserts on cudaMemGetInfo deltas. A parallel ctest process on the same GPU breaks it (round-2 #22). Fix: mark it RUN_SERIAL, or use bookedBytes() only.
- QuiesceWaitsForTheWorkerAndTheGpuStage expects quiesce(100 ms) to be false, which assumes the worker is blocked. That holds by construction, so this is fine.

== CHECKED, NO FINDING ==
- Memory ordering: the receive CTA does fence.acquire.sys and then __syncthreads before every thread reads the count and rows. Abort polling uses ld.relaxed.sys through the mapped pointer. The host abort store is __atomic SEQ_CST. Status uses st.relaxed.sys.
- Bounds: copy_compact_routing takes the vector path only when all 4 pointers are 16-byte aligned, and has a scalar tail. Staged EPLB stats plus the tail loop never dereference stats_row unless count_valid holds. Offsets are int64.
  The op guarantees slot_tokens (workspace max) ≥ runtime max_tokens_per_rank (:565, :630); the launcher check is still missing (round-2 A6 nit).
- CUDA graphs: the mask, abort record, one-slot table and ghost ranges keep fixed addresses and are read at execution time. The combine mask is read before GDS; the PDL chain makes that safe because prepare's GDS precedes it.
- Kernel params: DispatchKernelPointers is about 11 KB (recv_buffers alone are 256x4x8 B). It already needs the large-param path upstream; the net change is +48 B.
- moe_kernels: the source-slot remap excludes min-latency, LoRA, W4AFP8, fused FP8 block quantization and input_sf (moe_kernels.cu:4531; op :643-655). Every expandInputRowsKernelLauncher caller passes the new argument.
- Lock order: GhostSlotMemory, then the VMM manager, then the allocator; copy threads take only the allocator lock. No inversion was found.
- Static EPLB: startIter is never called, so mIterId == mWorkerDoneIterId == -1 and restore and quiesce pass. Round-2 PR6 #74 (restore race) is FIXED by the mWorkerDoneIterId check (:1318).
- Prior items still open in v2: A1 (m1), A3 (m2), #9 (m3), A2 (m5), #39 (m6), A5 (N1), #6/#8 (N4), #41 (N6), #22/#10 (tests), #26 (FT-off unreachable store at k.cu:931; harmless), A6.

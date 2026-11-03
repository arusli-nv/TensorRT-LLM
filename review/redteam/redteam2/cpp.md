C++/CUDA red team, round 2 (read-only), 2026-10-20 20:20 PT
Target: fix-final8-rc1 (78836d1723) plus the four C++ commits of fix-minors (f499b5ef8d, fc7419efdc, 1f048122cc, dda2d51984).
Base: merge-base(upstream/main, fix-final8-rc1) = 0829689c52. upstream/main has not changed any of the 26 cpp/ files since the base, so "FT-off identical to main" was checked against the base.
Method: static review of `git diff base fix-final8-rc1 -- cpp/` (26 files, +3960/-287) and `git diff fix-final7 fix-minors -- cpp/`. Nothing was built or run.
Format: severity | file:line (rc1) | issue | concrete fix

== 0. PROCESS ==
- The four fix-minors C++ commits are NOT in rc1 (rc1 = fix-final7 + a1d93a6f86 + 78836d1723). `git merge-tree rc1 fix-minors` is clean, so cherry-pick them before the release candidate is final.

== A. NEW FINDINGS ==

MINOR
1. minor (suspected; depends on how the consumer is launched) | cpp/tensorrt_llm/kernels/moe/communication/moeAlltoAllKernels.cu:609-612 (receive_source) | Each fixed-slot receive CTA calls cudaTriggerProgrammaticLaunchCompletion right after its grid-dependency wait. Only after that does it write source_valid_counts, compact_expert_ids/compact_weights, eplb_stats_out and flag_val[kReleasedRoundIndex]. Under PDL, a dependent kernel's cudaGridDependencySynchronize only guarantees visibility of writes made before the trigger. So a PDL-launched consumer can read stale counts or compact lists. Main's token CTAs also trigger before their peer waits (:818), so main already relies on the next kernel not overlapping. The receive CTAs' own outputs are new, though. | Move the trigger to the end of receive_source, after thread 0's stores, with a __syncthreads() between them. Or state in DispatchKernelPointers that these outputs need a consumer that does not use PDL.
2. minor | cpp/tensorrt_llm/runtime/moeLoadBalancer/moeLoadBalancer.cpp:976-985 (HostMemoryMoeWeightUpdater::finalizeWeights) | With FT on, a layer with no host weights is accepted for online EPLB too. An update that changes a slot then reads mHostWeights[name][newExpertId] (:1090) from an empty vector: out-of-bounds and UB. This is latent: Python drops host weights only for static layers (_has_restore_slots). | Accept empty host weights only with FT on and `mLayerLoadBalancer->mMoeLoadBalancer->mLayerUpdatesPerIter == 0`. Alternatively, TLLM_CHECK(hasHostWeights()) in updateWeights before the first copy.
3. minor | cpp/tensorrt_llm/thop/moe/communication/moeAlltoAllOp.cpp:1202 (dispatch schema), :1217 (combine schema) | The kernels write abort_state (status = kCancelled), but both schemas declare it as a read-only `Tensor?`. Every other output buffer added here is annotated, for example `Tensor(b!)? source_valid_counts`. Functionalization and torch.compile may then treat the record as unmodified. | Declare it `Tensor(f!)? abort_state` in dispatch and `Tensor(b!)? abort_state` in combine.
4. minor (lean; removable) | moeAlltoAllKernels.cu:1167-1172, 1205-1221 (CFT routed_id and staging rewrite), :1590-1606 (CFT SWITCH_BOOL) | USE_EXPERT_TO_SLOT is unreachable in the CFT kernel. expert_to_slot is passed only under peer-loss recovery (moe_scheduler._dispatch_slot_table), and recovery refuses CFT (bind_execution_state). The branch doubles every CFT instantiation (TOP_K x EPLB x MASK x COMPACT) for dead code. For the same reason, the fence kernel's USE_EXPERT_TO_SLOT=true instantiations are only needed with ENABLE_RANK_MASK. | Add TLLM_CHECK(!params.use_cft_counted_writes || params.expert_to_slot == nullptr) in moe_a2a_dispatch_launch and the matching TORCH_CHECK in the op. Delete the CFT routed_id and staging-rewrite code and its SWITCH_BOOL (about 25 lines). Optionally nest the fence SWITCH_BOOL(expert_to_slot) under ENABLE_RANK_MASK, which halves the new FT-off instantiations.

NIT
5. nit (removable) | moeAlltoAllKernels.cu:580-597 (checked_source_count) | When observe_abort() returns true it has already recorded kCancelled, so the following record_cancelled is a duplicate. ptrs.abort is never null on this path (the op requires abort_state with fixed slots, :~646), but moe_a2a_dispatch_launch does not check it. This is the unfinished half of prior #4. | Add TLLM_CHECK(params.source_valid_counts == nullptr || params.abort != nullptr) in the launcher. Then drop both null checks and the duplicate record: `if (count == -1) record_cancelled(ptrs.abort); else if (!observe_abort(ptrs.abort)) { printf; trap; } return -1;`
6. nit | moeAlltoAllKernels.cu:1529-1532 | The launcher does not check slot_tokens >= max_tokens_per_rank. A smaller slot lets one source's rows overflow into the next source's slot, which defeats the late-write isolation. Latent: the op always passes the workspace max. | TLLM_CHECK(params.slot_tokens == 0 || params.slot_tokens >= params.max_tokens_per_rank).
7. nit (removable) | cpp/tensorrt_llm/kernels/moe/cutlass/include/moe_kernels.h:~567-587 | The default body of the new runMoe(..., MoeInputSourceSlots) overload is dead code. CutlassMoeFCRunner is the only implementer of cutlass_kernels::CutlassMoeFCRunnerInterface; the internal runner has its own interface. | Make the overload pure virtual and drop its 19-line body.
8. nit | moeLoadBalancer.cpp:861 (addSingleWeightSlot) | With FT off, main's CPU-copy path now goes through the GhostSlotMemory singleton and its mutex on every slot add, via contains(). This is harmless, but it is FT machinery on the FT-off path. | `if (!gpuAccess && tensorrt_llm::mpi::isFaultToleranceModeEnabled() && GhostSlotMemory::getInstance().contains(...))`. Better: the single FT gate of prior #6.
9. nit | moeLoadBalancer.cpp:1766-1785 (restoreBackingBytes, backRestoredSlots, releaseUninstalledBacking) | Unlike restorePlacement and quiesce, these three do not check mModelFinalized. | Add TLLM_CHECK_WITH_INFO(mModelFinalized, ...).
10. nit (test) | cpp/tests/unit_tests/runtime/moeLoadBalancerTest.cpp:~491-525 (AllUsableMaskPlacesExactlyAsNoMask) | Only the 4-expert case is a golden. The 16-expert and dead-rank cases compare the new planner with itself (nullptr vs all-usable mask), so a regression in FT-off doPlacement would pass. | Add goldens produced by upstream/main's doPlacement for those loads.
11. nit (test coverage) | A2A kernels and expandInputRowsFromSourceSlotsKernel | There are no C++ tests. Coverage is only in Python: test_moe_a2a_abort.py, multi_gpu/test_moe_a2a_late_writes.py, test_moe_slot_gather.py and test_nvlink_one_sided.py. | Confirm that these files are listed in a GPU (and multi-GPU) test-db stage. Otherwise the FT kernel paths run in no CI stage.
12. nit (duplication) | cpp/tensorrt_llm/runtime/moeLoadBalancer/ghostSlotMemory.cpp:43-72 (MapConfigurator) | This copies main's UnicastConfigurator, plus an unmap when cuMemSetAccess fails. | Add that rollback to UnicastConfigurator and reuse it, if touching virtualMemory is acceptable. Otherwise keep it and say so in the PR.

== B. PRIOR FINDINGS (CODE_REDTEAM_FINAL / MINORS_FIX), status on rc1 + fix-minors ==
- #3 major, FinalizeRejectsWeightSlotsWithoutHostWeights cannot fail: FIXED in rc1 (setUseGpuMemcpy(true) at test:~690, positive control FinalizeAcceptsWeightSlotsWithoutHostWeights at :710).
- #4 fixed slots without an abort record: FIXED at the op (moeAlltoAllOp.cpp TORCH_CHECK(!fixedSlots || abortState)); the launcher check is still missing (folded into A5).
- #6 two FT gates (env in setInitialWeightAssignments :511 and finalizeWeights :981, cached member elsewhere): OPEN (skipped in MINORS_FIX).
- #7 resetUpdateSchedule without an FT check: FIXED in fc7419efdc (fix-minors only).
- #8 static quiesce after an interrupted iteration: FIXED in 1f048122cc (fix-minors only). The new gtest is meaningful: on rc1 code it waits for mWorkerDoneIterId == mIterId with no worker thread and returns false.
- #9 moe_a2a_initialize ignores CUDA errors: FIXED in f499b5ef8d (fix-minors only); rc1 :318-325 still unchecked.
- #22 FreedAndFailedAllocationsLeaveNoMemoryBehind sizes at 3/4 of free memory (test:~1150): OPEN.
- #23 one-slot table device and lifetime: FIXED in dda2d51984 (fix-minors only).
- #26 marker store in a branch also present in the FT-off instantiation: OPEN, at moeAlltoAllKernels.cu:931. It is unreachable there (wait_round_flag traps first). Wrap it in `if constexpr (ENABLE_RANK_MASK)`.
- #27 header does not state the latency assumption: OPEN (moeAlltoAllKernels.h MoeA2AAbortState).
- #28 FLAG_VAL reserves 8 bytes with FT off: OPEN (disclose in the PR).
- #39 HeartbeatSender: OPEN. The interval is cast in the init list (heartbeatSender.cpp:98) before the check at :100; getaddrinfo runs with the GIL (init has no gil_scoped_release); the socket leaks if the std::thread constructor throws (:121).
- #40 `while (!isSlotUsable(...)) ++localSlotId;` has no bound: OPEN, moeLoadBalancer.cpp:250.
- #41 `//` instead of Doxygen `//!` on the public interface: OPEN (ghostSlotMemory.h, new moeLoadBalancer.h declarations).
- C (parts/2-ulfm) PeerFailureWatch destroyed on its own thread: OPEN. stop() returns false at peerFailureWatch.cpp:105, then ~std::thread runs on a joinable thread and calls std::terminate. Also, the posted MPI_Irecv leaks if the std::thread constructor throws (:83).

== C. CHECKED, NO FINDING ==
- FT-off parity against the base (= upstream/main for these files):
  - The dispatch and combine mask-free instantiations compile the new branches away: `released` is false, rows_per_source is max_tokens_per_rank, receive CTAs exist only under ENABLE_RANK_MASK, and wait_round_flag gets abort=nullptr and still traps.
  - expandInputRowsKernel keeps its name and arguments as a wrapper over the same body; the source-slot path is its own kernel.
  - HostAccessibleDeviceAllocator::isSupported/init are equivalent after the refactor. The leaked singleton has a trivial destructor.
  - doPlacement fills ranks in the same order when no slot is masked. updateWeightsRoutine and the GPU/CPU publish paths are the same sequence. workerThread, waitCpuStage and the updateWeights range checks are unchanged with FT off.
  - moeOp num_rows checks are equivalent. moeComputeRouteDevice's oneSlotPerExpert can be true with FT off only when slots == experts, which is the old branch anyway.
  - The rank-mask CPU-to-device change affects only enable_rank_mask=True. The local-rank check moved to nvlink_one_sided._publish_rank_mask (:938).
- Memory ordering:
  - The receive CTA uses fence.acquire.sys before reading counts and rows, extended to the CTA by __syncthreads.
  - The abort record is read with ld.relaxed.sys through the mapped device pointer of pinned memory.
  - The released marker is written and read in stream-ordered kernels.
  - The combine abort uses __syncthreads_or, and an aborted block zeroes its output.
- Out-of-bounds checks: compact copy (vectorized only when 16-byte aligned, scalar tail), staged EPLB rows (kStagedStats then tail loop), and GhostSlotMemory slotOf/unbackedPieces chunk math (hi clamps at allocation.size, then -1).
- CUDA-graph safety:
  - The mask, the one-slot table and the ghost ranges keep fixed addresses and are rewritten in place.
  - Ghost pieces stay mapped (scratch or backed); the CapturedGraphReadsWhateverIsMappedBehindASlot test covers this.
  - resolveAbortState caches the device pointer per thread.
- moeWaitSignalForCpuStageHost(abandoned) sets kCPU with the host's fetch_or. That is safe only once the interrupted iteration's device work is done, a precondition the quiesce doc states; the Python caller is not re-verified here.
- Error handling: GhostSlotMemory back/allocate/free roll back on failure and keep pieces on scratch; tests BackingThatCannotBeAllocatedChangesNothing and FailedPieceIsDroppedAndEndsOnScratch cover it. registerHostAccessible unpins on a failed map.
- Headers: every new file has the 2026 SPDX header; the modified allocator files have 2022-2026.

== COUNTS ==
New: blocker 0, major 0, minor 4, nit 8 (12 total).
Prior C++ items: 6 fixed (#3 in rc1; #7, #8, #9, #23 only on fix-minors; #4 at the op), 9 open (#6, #22, #26, #27, #28, #39, #40, #41, PeerFailureWatch destructor).
Removable lines: about 25 (A4 CFT expert_to_slot), 19 (A7 base runMoe body), about 5 (A5 duplicate record and null checks).

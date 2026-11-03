# Bound dispatch: receive by warp (H1/H3) and stage rank-local tables before the PDL wait (H2). Design + red team

Base: /tmp/claude-161238/wt-v5 (v5 @ f96a8c1a53). `a2a.cu` = cpp/tensorrt_llm/kernels/moe/communication/moeAlltoAllKernels.cu. Inputs:
kernels_a.md, moe_path.md, graph_b.md (same dir). Design only: nothing was built or run. Numbers are estimates or come from those docs' ns32
trace (device 0, which does NOT reproduce the step delta). None is a CLAIMS row.
## 1. Change A: receive teams (one warp per source)
- **Today** (a2a.cu:601-715, 727-735, 1538-1543): grid = token CTAs + ep_size. Each receive CTA: thread 0 spins on one flag, then `fence.acquire.sys`,
  `__syncthreads`, count, the 2x2 KB copy, and `source_valid_counts[s]`.
- **New:** the host sets `receive_sources_per_cta` S. S = 8 (one warp per source) when `max_tokens_per_rank*top_k <= 1024` words, which covers decode
  up to 128 tokens/rank at top-8 in 1-2 copy round trips per warp. S = 1 otherwise: today's CTA per source, so prefill-size copies (up to 64K words)
  never land on one warp. Grid = token CTAs + ceil(ep/S). EP32 at 64 tokens: 96 -> 68. S and the grid depend only on the capture-time arguments
  (local_num_tokens, max_tokens_per_rank, top_k, ep_size). Each graph replays one shape (req 3).
- **Per team:** lane 0 waits (`wait_round_flag`, with its own 100k-cycle abort poll), then does the fence, the checked count and the count store. All
  lanes load the count and copy. The team sync is `__syncwarp()`. It is `__syncthreads()` only when S=1 (one team per CTA). No barrier ever spans two
  sources (req 1, 2). Spare warps of the last CTA exit after the PDL calls, and no barrier follows them.
- **Fences:** Lane 0's `ld.relaxed.sys` + `fence.acquire.sys` pairs with the remote `fence.release.sys` + `st.relaxed.sys` flag (a2a.cu:886-903): the
  PTX fence-fence synchronizes-with pattern at sys scope. `__syncwarp` (`bar.warp.sync`) is a morally strong barrier that orders memory among its
  participants (CUDA documents this). So causality order reaches the other 31 lanes' later loads. This is today's fence + `__syncthreads` chain at
  warp scope. Each warp's fence covers only its own source's count and rows, which are all that warp reads. Arrival and the count address go through
  smem (`flag_arrived[team]`, `count_slot[team]`) before the `__syncwarp`. `__shfl_sync` is not a memory-ordering op, so it is not relied on.
- **PDL:** all threads still `griddepcontrol.wait` and then `launch_dependents` at entry. The trigger is per CTA, unchanged. The `active` mask read
  moves before the wait. The mask is rewritten only between collectives, and combine already reads it pre-wait (a2a.cu:1973-1986). `*flag_val` and the
  recv-counter zeroing stay after the wait, because PrepareDispatch writes them.
- **Guarantees, unchanged:** The sender addressing of fixed slots and per-signature regions (a2a.cu:786-794) is untouched. A masked source gets count
  0, counter 0, ids -1 and weights 0. Abort release is per warp. A released source gets the masked rows and `kReleasedRoundIndex`.
  `checked_source_count` runs per lane 0. EPLB stats rows are strided by team lane.
- **Occupancy:** one kernel, so the register count is the max over its paths (60 today). ENABLE_EPLB adds about 6 regs (`kStagedStats` 2 -> 8, so 256
  experts take one round trip per warp). Static smem +72 B. Dynamic smem +32 B + 4*num_routed_experts (change B; 1.3 KB at 320 slots).
- **SM budget** (dispatch + routing cluster + aux g56; kernels_a.md section 1): FT-new: 116 / **132** / 148 at 24 / 32 / 40 req/GPU, all <= 152.
  FT-old: 160 / 176 at 32 / 40.
- **Expected:** If H1 holds, ft-new matches off at 32/40, recovering up to the ~6 us/layer H1 share. The aux g56 ">6 us" fraction should drop from 64%
  to ~5%. The local floor and the H3 tail stay about the same (one round trip after the last flag). Caveat: graph_b.md finds the aux GEMM off the
  critical path in that trace, so H1's step share is unproven. The A/B decides it.
## 2. Change B: stage the mask and expert_to_slot before griddepcontrol.wait (H2)
- **Today** (a2a.cu:762-784, 366): wait, then `token_selected_experts` -> `expert_to_slot[e]` (L2) -> mask word (L2) -> `atomicAdd`. The last-token
  loops (852, 868, 897, 922) reload the mask.
- **New:** before the wait, warp 0 copies the 4 mask words and `num_routed_experts` table ints into dynamic smem, then `__syncwarp`. Routing and the
  last-token loops (all warp 0) read the copies. `STAGED` defaults to false, so the CFT call (a2a.cu:1172) is unchanged.
- **Legality:** Dispatch starts only after PrepareDispatch triggers. PrepareDispatch triggers after its own wait (a2a.cu:496-497), so everything
  stream-ordered before it is complete and flushed. The only kernel still running, PrepareDispatch, writes `flag_val[0]` and the counters, never the
  mask or the table. The table (`one_slot_table`, lb:957) exists only with `updates_enabled=False`. FT recovery rewrites it between steps
  (moeLoadBalancer.cpp:768-797), with host/event ordering as today. Not hoisted: `released_round` (a2a.cu:891): a receive team of the same launch can
  set it under an abort. `*flag_val` and `token_selected_experts`: written by PrepareDispatch and the router.
- **Expected:** 2 dependent L2 hits leave each token CTA's send path. The worst case (prepare already done) adds 1 round trip before the wait, a net
  of at least -1 round trip. Estimate: -0.3 to -0.8 us on the send-done proxy (FT p50 7.78 us). Not the full 1.6 us; the rest may be the 96-CTA grid
  (change A).
## 3. Change T (graph_b.md item 2): receive-side trigger after the copy
- **What:** delete the receive `cudaTriggerProgrammaticLaunchCompletion()`, so the CTA exit triggers. That is equivalent to an explicit trigger after
  the copy, since the copy is the CTA's last work.
- **Cost and benefit:** routing becomes resident at dispatch end, not ~6 us early. This frees 8 SMs during the wait. But every rank then pays the
  cluster launch serially on each layer, as off does after sanitize. Estimate +0.5 to +1.5 us/layer.
- **Alone on FT-old:** 96+56 = 152 just fits, so it is a clean H1 test, but likely a net loss if H1 is false.
- **With A:** the budget already fits (132), so T only adds latency.
- **Recommendation: build A+B first and keep the entry trigger. Run T only as a diagnostic arm (ft-old+T at 32/GPU).** If T alone recovers the delta,
  SM residency is confirmed. If A+B recovers it and T does not, the receive CTAs are the cause.
## 4. Tests
- **Existing, must pass (GPU):** multi_gpu/test_nvlink_one_sided.py: all CASES and `test_dispatch_through_slot_table`.
  multi_gpu/test_moe_a2a_late_writes.py: 3 modes x 3 kill delays. test_moe_a2a_abort.py: notably compact routing (520,8) for the S=1 path, the
  aborted-source rows, and `test_replay_reads_the_slot_table_and_mask_written_before_it` (guards B under replay), plus the corrupt-count and timeout
  traps.
- **New, in test_moe_a2a_abort.py** (single GPU, simulated peers; EPLB stats on and off):
  1. `test_receive_teams_cover_every_source[ep 5,8,9,37,72 x (cap,top_k) (8,1),(128,8),(129,8)]`: every count and compact row; at ep 72 a
     source in mask word 1 is masked. Covers a partial last warp group, >32 sources and the S=8/S=1 boundary.
  2. `test_a_dead_source_holds_only_its_own_warp` (req 4), ep 37: source 12 (CTA 1, warp 4) is active and silent; source 13 (same CTA)
     publishes late on a side stream. Assert not done; set `abort[0]=1`; assert done within 10 s, every published source (incl. 13) has its
     count and rows, source 12 has count 0 / ids -1 / weights 0, `abort[1]` is cancelled and `kReleasedRoundIndex` is set.
  3. `test_receive_grid_is_fixed_per_shape`: profiler grid = tokens + ceil(ep/S) at ep 5 and 37, over two graph replays.
- **GPU A/B** (one job, fixed nodes; import smoke, then build): Arms: off | ft-old | ft-new (A+B) at 24/32/40 req/GPU, plus ft-old+T at 32. Standard
  1k/1k, decode-only trim-mean step, 3 repeats, warm restart. If H1 holds: ft-old - off is ~0 at 24 and > 0 at 32/40; ft-new - off is ~0 at all three.
  nsys in separate cases on >= 2 ranks: grid 68, aux g56 fraction, send-done proxy, and dispatch end -> routing end p90/p99. Key on
  `moeA2ADispatchKernel`. Safety on ft-new: a rank kill and a node kill (recovery, outputs, no trap), plus the late-writes test in the same job.
## 5. Red team
- **Intra-CTA deadlock:** none. There is no CTA barrier after the team split, so a spinning lane 0 holds only its own warp. Lanes 1-31 park at
  `__syncwarp`, which is safe under Volta+ independent scheduling. Spinning warps cost a few issue slots, as main's waiter does.
- **Cross-CTA deadlock:** none. Receive CTAs still follow every token CTA in blockIdx (68 CTAs on 152 SMs). Routing launches only after all token CTAs
  trigger, so it cannot starve them.
- **Abort while waiting:** each lane 0 polls independently. `record_cancelled` and the `kReleasedRoundIndex` stores are idempotent. A source that
  arrives as the abort lands keeps its count, as today.
- **ep_size not a multiple of 8 or 32:** spare warps exit with no barrier after them. ep < 8 uses 1 CTA. ep 256 uses 32 CTAs and mask word 3. New test
  1 covers this.
- **Masked-source fast path:** the pre-wait mask read is legal; the counter zeroing stays after the wait. **S=1** is today's path.
- **CFT:** untouched. It is a separate kernel and `STAGED=false`. CFT rejects `source_valid_counts` (a2a.cu:1513-1516), and B adds smem only on the
  fence path.
- **EPLB stats rows:** `kStagedStats` 8 gives one round trip at 256 experts. The remainder loop is kept, and `stats_row`/`out` do not alias.
- **Stale L1:** each source's rows are read only by its own warp, after its own acquire.
- **Not addressed:** the H3 serial tail and R2 (the routing template 512). These come next if a delta remains at 32.

## 6. Proposed diff (hand-written against v5; offsets approximate; `...` marks mechanical edits; review before build)
```diff
--- a/cpp/tensorrt_llm/kernels/moe/communication/moeAlltoAllKernels.h
@@ -165 +165,2 @@ struct DispatchKernelPointers
     int* source_valid_counts{nullptr};
+    int receive_sources_per_cta{1}; // kReceiveWarpsPerCta (a warp per source) or 1 (the CTA); fixed per launch shape
--- a/cpp/tensorrt_llm/kernels/moe/communication/moeAlltoAllKernels.cu
@@ -332,336 @@ route_dispatch_token
... template gains `bool STAGED = false` (last) ...
-    int* topk_target_indices, uint32_t lane_mask)
+    int* topk_target_indices, uint32_t lane_mask, int32_t const* staged_expert_to_slot = nullptr,
+    uint64_t const* staged_mask = nullptr) // STAGED: smem copies made before the grid dependency wait
@@ -349,366 @@
-            ? ptrs.expert_to_slot[expert_id] : num_experts;
+            ? (STAGED ? staged_expert_to_slot : ptrs.expert_to_slot)[expert_id] : num_experts;
-        should_send = should_send && is_rank_active(ptrs.active_rank_mask, target_rank);
+        should_send = should_send && is_rank_active(STAGED ? staged_mask : ptrs.active_rank_mask, target_rank);
@@ -531 @@ copy_compact_routing(..., int thread, int threads): threadIdx.x -> thread, blockDim.x -> threads (539, 566)
@@ -601,715 @@ receive_source
+constexpr int kReceiveWarpsPerCta = 8;
+__device__ __forceinline__ void team_sync(int team_size) { team_size == warpSize ? __syncwarp() : __syncthreads(); }
-__device__ void receive_source(DispatchKernelPointers const& ptrs, int source, int source_tokens, int rank_id,
+__device__ void receive_source(DispatchKernelPointers const& ptrs, int receive_cta, int source_tokens, int rank_id,
 {
+    int const team_size = blockDim.x / ptrs.receive_sources_per_cta;
+    int const team = threadIdx.x / team_size, lane = threadIdx.x % team_size;
+    int const source = receive_cta * ptrs.receive_sources_per_cta + team;
+    // The mask changes only while no collective runs, so it is read before the grid dependency wait.
+    bool const active = source < ep_size && is_rank_active(ptrs.active_rank_mask, source);
     cudaGridDependencySynchronize();
     cudaTriggerProgrammaticLaunchCompletion();
-    bool const active = is_rank_active(ptrs.active_rank_mask, source);
-    __shared__ bool flag_arrived;
-    __shared__ int const* count_slot;
-    if (threadIdx.x == 0)
+    if (source >= ep_size)
+        return; // spare warps of the last CTA; teams share no barrier
+    __shared__ bool flag_arrived[kReceiveWarpsPerCta];
+    __shared__ int const* count_slot[kReceiveWarpsPerCta];
+    if (lane == 0) // body unchanged, with flag_arrived[team] and count_slot[team]
-    __syncthreads();
-    bool const arrived = flag_arrived;
-    int const raw_count = arrived ? *count_slot : 0;
+    team_sync(team_size); // extends lane 0's acquire to its team, which reads only this source
+    bool const arrived = flag_arrived[team];
+    int const raw_count = arrived ? *count_slot[team] : 0;
-        constexpr int kStagedStats = 2;
+        constexpr int kStagedStats = 8;
     ... threadIdx.x + k * blockDim.x -> lane + k * team_size (2x); remainder loop uses lane, team_size ...
-            count * TOP_K, source_tokens * TOP_K);
+            count * TOP_K, source_tokens * TOP_K, lane, team_size);
-    if (threadIdx.x == 0)
+    if (lane == 0)
@@ -736 @@ moeA2ADispatchKernel, after the receive branch (the inner `extern __shared__` at 754 moves here)
+    // The mask and expert_to_slot change only while no collective runs. Warp 0 stages them before the grid
+    // dependency wait, which orders this kernel only after moeA2APrepareDispatchKernel.
+    extern __shared__ int smem[];
+    auto* const staged_mask = reinterpret_cast<uint64_t*>(smem + 2 * TOP_K);
+    int32_t* const staged_expert_to_slot = smem + 2 * TOP_K + 2 * kRankMaskWords;
+    if (thread_idx < warpSize)
+    {
+        for (int w = thread_idx; ENABLE_RANK_MASK && w < kRankMaskWords; w += warpSize)
+            staged_mask[w] = ptrs.active_rank_mask[w];
+        for (int e = thread_idx; USE_EXPERT_TO_SLOT && e < ptrs.num_routed_experts; e += warpSize)
+            staged_expert_to_slot[e] = ptrs.expert_to_slot[e];
+        __syncwarp();
+    }
@@ -772 @@ route call: <..., USE_EXPERT_TO_SLOT, true>(..., kRoutingLaneMask, staged_expert_to_slot, staged_mask)
@@ -852,868,897,922 @@ last-token loops (warp 0): is_rank_active(ptrs.active_rank_mask, X) -> is_rank_active(staged_mask, X)
@@ -1540 @@ moe_a2a_dispatch_launch
-        // One receiving CTA per source (receive_source).
-        grid_size += params.ep_size;
+        static_assert(kBlockSize == kReceiveWarpsPerCta * 32);
+        int const per_cta = params.max_tokens_per_rank * params.top_k <= 1024 ? kReceiveWarpsPerCta : 1;
+        kernel_ptrs.receive_sources_per_cta = per_cta;
+        grid_size += (params.ep_size + per_cta - 1) / per_cta;
@@ -1546 @@
     int shared_bytes = routing_bytes;
+    if (!params.use_cft_counted_writes) // staged mask and expert_to_slot
+        shared_bytes += kRankMaskWords * sizeof(uint64_t)
+            + (params.expert_to_slot != nullptr ? params.num_routed_experts * sizeof(int32_t) : 0);
```

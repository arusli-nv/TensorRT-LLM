# Review: 046965f51b (pre-wait route loads) and b0ad558bcb (Tier<384,8> for Renormalize)

Read-only review of /tmp/claude-161238/wt-recvfix (v5-recvfix). Nothing was built or run on a GPU. clang-format 16 and ruff
(pre-commit cache) report no changes on the touched files. a2a.cu = cpp/tensorrt_llm/kernels/moe/communication/moeAlltoAllKernels.cu,
pol = .../moe/trtllmGen/routing/RoutingCustomPolicy.cuh, rck = .../routing/RoutingCustomKernels.cuh. Line numbers are at b0ad558bcb.

**Verdicts: 046965f51b: build. b0ad558bcb: build. Before it lands, add one post-top-k gtest for the shape it targets (F3).**

## 046965f51b

### (a) PDL legality: legal in every case checked
- Order: moeAlltoAllOp.cpp:889/892 launches prepare and then dispatch back to back on one stream, both through
  launchWithPdlWhenEnabled. Prepare runs `cudaGridDependencySynchronize()` before `cudaTriggerProgrammaticLaunchCompletion()`
  (a2a.cu:520-522, and the new comment pins that order). So no dispatch CTA starts until every prepare CTA has passed its wait,
  which means prepare's predecessor has completed and its memory is flushed. Graph capture turns this into a programmatic
  edge with the same semantics. With PDL off, dispatch is fully serialized.
- What reads before the wait (a2a.cu:811-823, 343-363): `token_selected_experts[t*K+k]`, `expert_to_slot[e]` and
  `active_rank_mask[rank>>6]`. Prepare writes only send/recv counters, local_token_counter and flag_val (a2a.cu:516+). All
  three of those are still read after the wait. Old vs new: the post-wait reads saw "prepare complete"; the pre-wait reads
  see "prepare's predecessor complete". These differ only by prepare's own writes, which these loads do not touch, so there
  is no new window.
- Writers of the three inputs:
  - token_selected_experts: the router / top-k kernel, or the slot ids, stream-ordered before prepare.
  - Bound mask: `install_mask` brackets its copy_ with `torch.cuda.synchronize` and runs only while the model is idle
    (execution_state.py:188-203).
  - Non-bound mask: `_publish_rank_mask` copies on the current stream before the op (nvlink_one_sided.py:1048-1072, 1327).
    That is a memcpy, so a full dependency, and it raises if the mask changes during capture.
  - expert_to_slot (`one_slot_table`, moe_load_balancer.py:957-973, only with updates disabled): recovery rewrites it with
    cudaMemcpyAsync on the LB stream followed by cudaEventSynchronize (moeLoadBalancer.cpp:703-706, 811-815), or by host
    writes (789-805, 822-824). Neither is stream-ordered with the compute stream. That was already true for the post-wait
    read: its safety rests on "the model is idle", not on PDL. Unchanged.
- Between replays: no device kernel writes active_rank_mask or the slot table. The only writers are the two Python
  `copy_` calls and the LB copies (searched cpp/ and tensorrt_llm/). A previous layer's combine writes neither. The
  routing input of layer L is a fresh router output, not an a2a buffer. A graph launch is a full dependency on the work
  queued before it.
- Precedent: receive_source already reads the mask before its wait (a2a.cu:648), and so does combine (a2a.cu:2052).
- Pre-existing (out of scope): if a kernel earlier in the chain is PDL-launched and never waits, prepare's wait covers
  only its direct predecessor. That hazard is the same before and after this commit.

### (b) Compiler sinking: affects performance only, not correctness
- CUDA's sm_90 runtime header defines `cudaGridDependencySynchronize()` as `asm volatile("griddepcontrol.wait;" ::: "memory")`.
  This is from memory: there is no CUDA toolkit on this host to check. The memory clobber stops NVVM from moving the plain
  `ld.global` loads (the pointers are not `__restrict__`) across the wait.
- Sinking a load below the wait would still be correct. It would only lose the speedup. Confirm in SASS
  (`cuobjdump -sass`, moeA2ADispatchKernel<8,*,true,*,true>): the 2-3 dependent LDGs should come before the
  griddepcontrol.wait instruction.

### (c) FT-off and CFT: same results
- ENABLE_RANK_MASK=false: `load_dispatch_route` runs after the wait, inside the `thread_idx < TOP_K` branch
  (a2a.cu:834-838), and does the same loads and arithmetic as before. The only reordering is the compact-routing smem `-1`
  init plus `__syncwarp`, which now runs after the global loads. They do not alias. `target_active` is unused.
- Rank mask on: `should_send = valid && first-of-target && (valid && mask bit)` is the same boolean as before. Every valid
  lane now loads its mask word, where before only the first lane per target did. The mask cannot change during the kernel.
- `valid_expert = target_rank >= 0` matches the old `expert_id in [0, num_experts)`, because compute_target_rank_id >= 0
  for valid ids.
- CFT (a2a.cu:1238-1241): the loads stay after the wait at a2a.cu:1207, and USE_EXPERT_TO_SLOT=false. Same results.
- local_num_tokens == 0 and receive CTAs: no new pre-wait loads.

### (d) Register pressure: negligible (estimated)
- Across the wait, lanes < TOP_K hold the route: 3 values (id, rank, flag) on top of local_token_idx and thread_idx. Those
  values are dead before vectorized_dispatch, which is the token path's register peak. The kernel's register count is the
  max over its paths, and the receive path (~60 regs, kernels_a.md) is larger.
- Check with `--resource-usage` on the RANK_MASK=1 instantiations against b643ad8a29: expect no change and no spill.

### (e) test_replay_routes_by_wide_slot_table_entries_to_mask_word_1_peers (test_moe_a2a_abort.py:553-641)
- What it tests: each replay uses the table and mask written just before it. The tokens' slots change between the two
  rounds (130/131 <-> 140/141). The active peer flips between 65 and 70, both in mask word 1. It asserts the sent slot ids,
  the token order, the local count, the active peer's counter and flag, and that the masked peer's workspace is untouched.
  A stale (baked or cached) table or mask would fail it. This is a real test of the staged values.
- What it cannot test: the PDL ordering itself. Its writers (index_put, `_set_mask`) run before `graph.replay()`, behind
  a full graph-launch dependency, so the test passes on the parent too. On one coherent GPU no single-GPU test can catch a
  pre-wait race. Only the multi-GPU tests and the rank and node kill smoke can.
- Gaps (T1): top_k=1, so it never has several lanes on one target. Neither the "every lane loads the mask" change nor
  `__match_any` is exercised, and there are no invalid ids (-1, or >= num_routed, which map to num_experts). Suggested: a
  top_k=4 variant with two experts on the masked rank, one on the active rank, and one id of -1. "More than a warp of
  entries" only mattered to the dropped staging loop. It is harmless now.

## b0ad558bcb

### (f) Tier<384,8> in PolicyTraits<NoOp, Softmax> (pol:586)
- Order: (256,16) < (384,8) < (512,8), sorted by E and then K. 257-384 experts with topK <= 8 now pick 384. topK 9-16
  still picks (512,16). Shapes with <= 256 experts are unchanged. RenormalizeNaive is turned into this policy at run time.
- numThreadsHist: `getMaxNumExperts(320)` already returned 384 (pol:427-460), so RoutingFromTopKIds.cu:70-71 and
  rck run() passed 384 while the tier was 512. DefaultRoutingLaunchConfig (pol:676) raised that to 512. Now both are 384,
  which is more consistent than before. queryDispatchedMaxExperts (pol:848) returns 384.
- Which kernels change: only the PolicyTraits launches.
  - Post-top-k (RoutingFromTopKIds.cu:42-150): block kernel at <= 4 tokens; dyn-block at 5-12 tokens
    (smem 12*384*3 B = 13.8 KB); cluster-1024 at 1025-8192 tokens (rck:1723-1735, LAUNCH_ROUTING_CUSTOM).
  - Scores path: the small-token kernel at <= 8 tokens. With TLLM_ROUTING_SMALL_TOKEN_FASTPATH=0, a 1-token input moves
    from coop-block (tier 512) to the classic block kernel (tier 384 is below CoopBlockKernelSingleTokenMinNumExperts=512).
  - Unchanged, because they have their own tier lists without 384: cluster 256/512 (rck:1543-1555, so post-top-k at 13-1024
    tokens stays at 512) and HistogramScores (rck:1830).
  - Init, histogram, offsets and coop already used the 384 tier through the NO_POLICY ladders (pol:776+, rck:1942-1980).
  - The 1024-token boundary matters for us: the bound routing input is [32*64, 8] = 2048 rows (moe_path.md R3), so it takes
    the cluster-1024 path and gets the 384 tier. A shape with <= 1024 rows would get no change.
- Power-of-2 assumptions: none found.
  - Every changed kernel derives `NumThreadsBlock = min(E,1024)` and `ExpertsPerThread = E/NumThreadsBlock`, and
    static_asserts `E % 32 == 0` and `E % NumThreadsBlock == 0` (rck:150-155, 856-862, 1230; RoutingKernel.cuh:233-236). 384
    passes all of them.
  - cub::BlockScan<.,384>, the warp scan over 12 warps and top-k N=12 (RoutingKernelTopK.cuh:435, N <= 64) take any warp
    count.
  - The small-token pack (`<<12`) depends on the token count, not E. Its 12 warps are <= 32.
  - Non-power-of-2 tiers already ship for this policy (160, 576). The same kernels run at 384 for SigmoidBias (Kimi K2,
    pol:610), including cluster-1024 and coop-block.
- Shared memory and occupancy: smaller than at 512 everywhere (smemExpertCount/Offset are 1.5 KB each, against 2 KB). The
  single-CTA kernels launch 384 threads instead of 512. Cluster-1024 keeps 8x1024 threads, so its occupancy is the same.
- Other routing methods and models that change:
  - Any method that goes through runPostTopKPipeline (custom, DeepSeek, Llama4) with 257-384 experts or slots and topK <= 8:
    DeepSeek-V3 with EPLB at 288/320 slots, and Kimi K2 (384, top-8) with precomputed top-k.
  - Any Renormalize model with 257-384 experts on the scores path at <= 8 tokens.
  - Results should be identical (same algorithm), apart from order nondeterminism the kernels already have. Binary and
    compile time grow by about 5 kernels x 3 dtype pairs.
- Test coverage:
  - Existing gtests now really run the new tier (routingCustomTest.cpp:1078-1150; the comment there already claimed
    Tier<384,8>, which until now resolved to 512): BlockLevelE384 plus the TopKAsInput variants (1 token).
  - Not covered:
    - F3: the target shape. Add `Renormalize, TopKAsInput, numExperts 320, topK 8, tokens 2048` (cluster-1024 at 384, with
      320 < tier padding), plus 8 tokens (dyn-block). Run them with the E384 gtests in the A/B job.
  - Nit: routingCustomTest.cpp:397 lists the Renormalize tiers as "128, 160, 256, 512, 576, 2048". That list was already
    missing 1024 and is now also missing 384. Add a 384 row to CoopBlockKernelSelectionTable (expected: classic at 1-4
    tokens).

# FT bound vs main unbound NVLinkOneSided dispatch/combine at 64 tokens/rank: where ~6 us/layer of wait comes from

Code: worktree /tmp/claude-161238/wt-v5 (branch v5 @ f96a8c1a53). `a2a.cu` = cpp/tensorrt_llm/kernels/moe/communication/
moeAlltoAllKernels.cu, `op` = cpp/tensorrt_llm/thop/moe/communication/moeAlltoAllOp.cpp. Builds on moe_path.md (same dir).
Trace: /tmp/claude-161238/ns32/{off,ft}.sqlite, device 0, n=2900 decode layers per arm (dispatch grid 64/96). This trace
does NOT reproduce the step delta (see moe_path.md). Use it for structure and floors only. Scripts were throwaway, run in $TMPDIR.
TARGET_INFO_GPU.smCount = **152** (not 148).

## 1. Grid and CTA roles (a2a.cu:718-735, 1533-1541, 1604)
- Main: grid = local_num_tokens = 64 token CTAs, 256 threads, **32 regs**. The last token CTA (atomic counter) writes
  recv counters, runs `fence.release.sys`, publishes 32 flags (`st.relaxed.sys`), then 1 warp (lane = peer) spins on 32 local
  flags. Then a separate `moeA2ASanitizeExpertIdsKernel` (grid 8) runs.
- FT: grid = 64 token CTAs + 32 `receive_source` CTAs (blockIdx >= 64, one per source), **60 regs** (trace). Token CTAs
  publish the same way. The last one then returns without waiting (a2a.cu:910-913). Each receive CTA: thread 0 spins on
  its source's flag, `fence.acquire.sys`, `__syncthreads`, count load, then a compact copy of 64x8 ids and 64x8 weights from
  the fixed slot, then a `source_valid_counts[s]` store (a2a.cu:607-715).
- Residency: 256 threads with tiny smem, so dispatch CTAs alone never form a second wave. The binding resource is the
  **register file of the co-running kernels**. Each of these needs an empty SM:
  - aux-stream shared-expert `nvjet_..._15` (g56): 256 thr x 255 regs = 65280 regs.
  - routing cluster `routingIndicesClusterKernel` (g8): 1024 thr x 60 regs = 61440 regs.
  A dispatch CTA on an SM blocks both from that SM.
  - Off at 32/GPU: 64 dispatch + 8 sanitize + 56 aux = 128 <= 152. Routing launches later, after sanitize.
  - FT: 96 + 8 routing (launched early, see PDL) + 56 aux = **160 > 152**.
  - Trace: aux nvjet g56 runs >6 us in **64% of FT layers vs 5% off**. p50 is 5.15 -> 6.62 us, p10 is 4.99 in both, so
    the distribution is bimodal: a partial second wave.
  - The same budget by load (FT vs off): 8/GPU 16+32+8+56 = 112 (both fit). 64/GPU off 128+64 = 192 (off already
    oversubscribed). 128/GPU both far over. So FT crosses the SM budget alone **only near 32/GPU**, which matches the
    peak shape.

## 2. PDL placement
- Token CTAs (both arms): `griddepcontrol.wait` before routing (a2a.cu:762), trigger after the payload send
  (a2a.cu:818). The flag publish and main's wait come after the trigger.
- FT receive CTAs: wait and trigger at entry (a2a.cu:611-612). The dependent launches once every CTA has triggered, i.e.
  once the last token CTA has sent. So both arms launch the dependent at "local send done". FT's dependent is the routing
  cluster, which then sits resident, spinning in its own wait for ~10 us (start p50 7.78 us, end at dispatch end), on 8
  exclusive SMs. Off's dependent is sanitize (8 small CTAs). Off routing starts at 29.6 us p50, after dispatch.
- Receive CTAs do **not** trigger later. The trigger is not what delays routing. What delays routing in FT is the
  dispatch grid's *completion* (routing's wait), which now includes the per-source tail (section 4).
- **Local send-done proxy** (dependent start minus dispatch start) is p5/p50/p90 **5.63/6.18/7.04 off vs 7.23/7.78/8.58
  FT: +1.6 us on every layer**. Every rank pays this, so every peer's flag lands ~1.6 us later. On the receiver that shows
  as cross-rank wait, not as a longer local kernel. (Confound: the cluster launch of FT's dependent may itself wait for
  8 free SMs in a GPC.)

## 3. Sender prologue (token CTA, a2a.cu:334-391, 776-799)
FT adds dependent global loads before the first payload byte leaves:
`token_selected_experts[k]` -> `expert_to_slot[expert]` (USE_EXPERT_TO_SLOT) -> `__match_any` ->
`active_rank_mask[target>>6]` (ENABLE_RANK_MASK; the mask word depends on target_rank) -> `atomicAdd(send_counters)`.
That is two extra L2 round trips (~0.5-0.8 us each) per CTA on the send critical path. The last token CTA also loads
`released_round(flag_val)` and the per-target mask before publishing (a2a.cu:891-897).
`dispatch_payload_offset/size` only moves base pointers (op:628-631), so it costs nothing.
Fixed slots: each sender writes one 64-row region per peer in both layouts, so sender-side page or TLB footprint per peer
is the same. The 29 MB stride matters only to the receiver's FC1 gather (+0.26 us, G1).

## 4. Fences, per-source barrier, slowest source
- Release side is identical: one `fence.release.sys` and then relaxed sys flag stores.
- Acquire side:
  - Main: no fence in the dispatch waiter. Ordering comes from kernel completion before sanitize and routing.
  - FT: each of 32 CTAs runs `fence.acquire.sys` after its flag, then dependent loads (count, then the 2x2 KB slot
    rows), compact stores, and the count store.
- Barrier: yes. Routing's `griddepcontrol.wait` needs the whole dispatch grid done. So FT's critical path is
  last-arriving source -> its CTA's fence + ~3 dependent memory round trips -> CTA exit -> grid flush. Main has
  last flag -> warp exit -> sanitize launch and run.
- Trace:
  - "dispatch end minus dependent start" p5 is 5.31 off vs 6.40 FT.
  - Dispatch-start-to-routing-end floor (p1/p5/p10) is **17.34/18.91/20.22 off vs 19.52/20.74/21.73 FT: +1.5 to +2.2
    us**. About 1 us of that is the R2 routing template (moe_path.md).
  - Dispatch kernel min is 9.76 vs 13.02 us. This is not comparable on its own, because off's tail sits in sanitize.
- Abort poll: `observe_abort` reads pinned host memory only after 100k cycles (~50 us), so it is absent in steady state.

## 5. Combine (a2a.cu:1948-2080; op:1015-1120)
- Same geometry in both arms: 64 CTAs, packed rows at the runtime maximum of 64 (no fixed slots on combine), the same
  readiness-flag protocol, and the same single `fence.acquire.sys` per block.
- FT's additions are hidden: mask words load before the grid wait, and there is a `__syncthreads_or`.
- Floor is equal: min 15.23 vs 14.98, p5 19.20 vs 19.10 us.
- Combine is where any rank's *local* delay before FC2 end (routing, FC1, FC2, SM contention) becomes everyone's wait,
  as the max over 32 ranks. FT adds no combine cost of its own.

## 6. Top 3 hypotheses for ~6 us/layer that peaks at 32/GPU

**H1. SM-budget overflow from 32 long-lived receive CTAs plus the early routing cluster (fits the peak).**
- Mechanism:
  - The receive CTAs pin 32 SMs for the whole cross-rank wait. With the aux g56 GEMM and the resident routing cluster
    (both need a full RF), FT needs 160 SMs vs 152.
  - The partial second wave delays the aux GEMM (seen) and, at the floor, the routing cluster placement and FC1 start.
  - Per-rank jitter becomes a max-of-32 combine wait.
  - Off fits at 32/GPU, while at 64/GPU and above both arms overflow. That gives the peak.
- Check:
  - Prediction sweep with no code change: FT vs off at **24/GPU** (48+32+8+56 = 144, fits) and **40/GPU** (80+32+64 =
    176 vs off 144). The delta should collapse at 24 and stay at 40.
  - Trace check already done: aux g56 >6 us in 64% vs 5% of layers.
  - Follow-up query on new paired traces: per layer, regress (FC1 start minus dispatch end) on aux g56 overlap.

**H2. Slower sender prologue: +1.6 us on every rank's flag publish.**
- Mechanism: two extra dependent loads (`expert_to_slot`, then the mask word indexed by target), plus the
  `released_round` and mask loads before the publish. Every receiver's wait rises by the same shift.
- Peak shape: this alone is flat with load, so it explains the floor but not the peak.
- Check: kernel toggle that loads the mask words into registers (`load_rank_mask_words`, as combine does) and stages
  `expert_to_slot` (<=512 ints) into smem *before* `griddepcontrol.wait`. Metric: the send-done proxy (section 2) should
  fall from 7.78 back to ~6.2 us p50 on the existing query.

**H3. Per-source barrier tail.**
- Mechanism: the slowest source's CTA must do fence + count + 2x2 KB copy + store after the last arrival, then grid
  completion, before routing can leave its wait.
- Effect: about +1 us of serial latency after the last flag (floor +1.5 to +2.2 us including R2). It is larger when
  the slowest source arrives late in the receive CTA's issue order.
- Check: kernel toggle where receive CTAs do only wait + `source_valid_counts`. The routing kernel would read the fixed
  slots directly (stride `slot_tokens`, rows >= count masked). Metric: "dispatch end minus dependent start" p5, 6.40 ->
  ~5.3 us. Alternative check with no code change: query the existing ft.sqlite for (routing end minus dispatch end)
  vs dispatch duration. If the tail is a constant after the last arrival, it does not scale with the wait.

## 7. General fix that keeps FT's guarantees
1. **Receive by warp, not by CTA.**
   - One warp per source: `ceil(ep_size/8)` CTAs of 8 warps (grid 64+4 = 68 at EP32), placed after the token CTAs.
   - Lane 0 spins on the source's flag, then `fence.acquire.sys` and `__syncwarp`. The warp copies the 64x8 ids and
     weights (16 words per lane) and stores `source_valid_counts[s]`.
   - Fixed per-source slots, per-signature regions, rank-mask skip (count 0), abort release (`kReleasedRoundIndex`)
     and the checked count are unchanged and timing-independent.
   - This removes 28 SM-pinning CTAs (H1) and shortens the tail (H3).
2. **Hoist rank-local tables before `griddepcontrol.wait`.**
   - Token CTAs: load the mask words (already safe, since the mask only changes between collectives) and stage
     `expert_to_slot` in smem, which removes the H2 round trips. The table is static while the graph is bound; it is
     rewritten only at EPLB updates, which are stream-ordered before dispatch.
3. Optional: keep the receive warps' PDL trigger at entry, but this only helps once (1) frees the SMs. Revisit the R2
   template so that 320 slots do not select MaxNumExperts=512.

Validate with a same-node A/B at 24, 32 and 40 req/GPU (fixed nodes, standard 1k/1k). Read the send-done proxy, the
aux g56 >6 us fraction and the combine p50 from nsys on >=2 ranks.

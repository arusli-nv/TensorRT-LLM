# v5-exp: token-major fixed slots, hoisted bound-dispatch loads, late receive PDL trigger (2026-10-22)

Branch v5-exp in /tmp/claude-161238/wt-exp, on v5 5ebf7f72a2. Three commits: e680506a0f (T: token-major), 0e0af0f845 (H: hoists),
eb3950a3f8 (P: PDL trigger). Not built or run. `a2a` = cpp/tensorrt_llm/kernels/moe/communication/moeAlltoAllKernels.cu,
`op` = cpp/tensorrt_llm/thop/moe/communication/moeAlltoAllOp.cpp, `nvl` = .../fused_moe/communication/nvlink_one_sided.py. Lines are at eb3950a3f8.

## T. Token-major fixed slots (e680506a0f)
- Dispatch write (a2a:821-825): with slot_tokens > 0, row = t * ep_size + source (was source * slot_tokens + t). Off/CFT/unbound keep
  rank_id * max_tokens_per_rank + t (same branch condition as before).
- Compact routing copy (a2a:542-600, call a2a:698-707): lists are [T * ep, top_k] in the same token-major order, so the receive CTA of
  source s copies rows s, s+ep, ... (TOP_K words each, stride ep*TOP_K words) from the receive slots to the same rows of the lists.
  int4 vectors when TOP_K % 4 == 0 and the bases are 16-byte aligned, else words; loads still issue before the count is known.
  source_valid_counts unchanged (a2a:735-740).
- FC1 gather / compact(): **no mapping is left.** Rows [0, T*ep) of the slots are exactly the routing tokens, so nvl:1395-1398 hands
  every backend `recv.flatten(0,1)[:T*ep]` (packed, same base address, graph-stable). Removed: scheduler `_compact_receive_slots`,
  `MoECommPlan.source_slots`, `DispatchSourceSlots`, `supports_dispatch_receive_slots` (impl_blocks, CuteDSL, Cutlass), the CuteDSL
  source_slots tuner/forward plumbing, the Cutlass `input_source_tokens` kwargs, and execution_state's unfinalized-combine refusal
  (packed rows make it moot). FT now tunes and runs FC1/FC2 on the same packed path and tactic keys as off (plateau #9 goes away).
  The op-level features (`_source_slot_row`/`read_source_slots`, cute op `source_tokens/slot_tokens`, Cutlass `MoeInputSourceSlots`)
  stay, unused by WideEP, because their tests (test_moe_slot_gather.py, test_cute_dsl_moe.py) are files I may not delete.
- Combine (op:940-959, 1128; a2a:1950-1953, 2516-2526): new op arg `token_major` (nvl:1462, 1511 pass `state is not None`). Fence combine
  reads peer row t*ep+rank as base `ep_rank*stride` + `target_index * (ep*stride)` through the existing stride_per_token launch arg (no
  device change in moeA2ACombineKernel); prepare-combine maps row g to (g % ep, g / ep) for the recv-count skip. CFT + token_major is
  rejected (bound never uses CFT). Op returns fixed-slot recv tensors as [capacity, ep, W] (op:904-907, fake in cpp_custom_ops.py).
- Workspace sizing: unchanged (ep * slot_tokens rows per payload per signature region); source_write_ranges already covers the whole region.
- Guarantees: rows of source s are exactly {t*ep+s : t < slot_tokens}, disjoint across sources and inside the payload's region, so a dead or
  late source writes only its own rows (any T, any layer). Per-signature regions still needed and kept (widths differ => a row index of one
  signature overlaps other sources' bytes of another). Masked/aborted sources: count 0 and ids -1/weights 0 on all their compact rows (copy
  fills rows >= count), so no expert reads their rows, as before. No hang: no new wait. Graphs: addresses fixed (slice of fixed buffers);
  combine stride is a capture-time launch arg. MTP/draft T=1: rows [0, ep) packed. Stale rows of unsent tokens are visible to backends
  that reduce over all rows (e.g. dynamic per-tensor quant), exactly as the old compact copy exposed them.
- Found by: `grep -rnE "slot_tokens|source_tokens|source_slot|_source_slot_row|read_source_slots|supports_dispatch_receive_slots|
  dispatch_source_slots|DispatchSourceSlots|compact\(|source_valid_counts|rows_per_source|rowsPerSource|input_(source|slot)_tokens"`
  over cpp/, tensorrt_llm/, tests/, docs/; then every `max_tokens_per_rank` product in a2a (`grep -n "max_tokens_per_rank"`: dispatch
  write, prepare-combine, combine origin_offset, CFT push/sanitize (off-only)), and every caller of moe_a2a_dispatch/moe_a2a_combine
  (nvl only in product code); `git log -S read_source_slots / supports_dispatch_receive_slots / input_slot_tokens` for the commits that
  added readers (10ce178a1a, 1c347eea2a, 9d662e2b93, d1e64775c3).

## H. Hoists (0e0af0f845)
- Last token CTA mask (a2a:856-865, 887, 903, 935, 960): warp 0 of each token CTA loads the 4 mask words into registers before the token
  counter atomicAdd; all four loops test `is_rank_active_in_words`. Why correct: the mask guards against touching a dead peer's memory. It is
  rewritten only while no collective runs (install_mask brackets its copy with synchronize; combine already reads it before its grid wait,
  a2a:2019), so every read inside one launch returns the same words; reading earlier cannot admit a masked peer.
- released_round (a2a:921-925): read before `fence.release.sys` instead of after. It guards against publishing flags after this rank released
  a round. Writers: earlier kernels (stream-ordered, visible after this kernel's griddepcontrol.wait, which precedes this code) and receive CTAs
  of the same launch. The release fence orders only this thread's prior writes; it creates no order with a concurrent receive-CTA write, so
  the post-fence read could equally miss it. Same guarantee; the round trip overlaps the fence drain.
- Receive acquire (a2a:291-312, 656-663): receive CTAs poll with `ld.acquire.sys` (wait_round_flag<true>) and drop `fence.acquire.sys`.
  Same ordering: the writer's `fence.release.sys; st.relaxed.sys flag` is a release pattern; an `ld.acquire.sys` that observes it is an
  acquire pattern and synchronizes-with it at sys scope (PTX memory model), exactly like `ld.relaxed.sys` + `fence.acquire.sys`; bar.sync
  extends it to the CTA as before. Abort/timeout paths unchanged. Other wait_round_flag callers keep relaxed polls.
- Not hoisted: `*flag_val`, counters (written by PrepareDispatch), payload/count reads (need the acquire).

## P. Receive PDL trigger (eb3950a3f8)
- a2a:640-643 (wait only at entry), a2a:741-745 (trigger after source_valid_counts / compact copy). Token CTAs' trigger (a2a:850) unchanged.
  Dependent (routing) launches once every receive CTA has finished its source, i.e. at ~grid end; it no longer pins 8 SMs early. No
  correctness change: routing's griddepcontrol.wait already waited for full completion. Cost: routing launch latency is no longer hidden.

## Tests changed (none run)
- test_moe_a2a_abort.py: `_combine` passes token_major with controls; quarantine test poisons rows t*2+1; recv indexing [t, s]; compact
  asserts view(capacity, ep, k); fake-shape test expects [capacity, ep, W]; new `_first_tokens` helper.
- multi_gpu/test_moe_a2a_late_writes.py: victim-owned ranges are rows t*2+1 per payload; combine token_major; survivor rows [:, 0].
- test_moe_execution_state.py: combine token_major; unfinalized-combine test now expects a bind; removed compact-copy and plan.source_slots tests.
- test_moe_slot_gather.py docstring; peer_loss_driver.py comment; docs wide-ep-fault-tolerance.md requirement line.
- test_nvlink_one_sided.py bind test goes through comm.dispatch/combine: unchanged, still covers the bound round trip.
- py_compile + ruff/yapf/isort/clang-format 16 clean on touched files (pre-existing ruff findings only).

## Risks to check on GPU
- ld.acquire.sys per poll may cost more per iteration (L1 invalidate) than relaxed + one fence; A/B it (H alone vs T alone).
- P removes the early routing launch; kernels_a/graph_b predict +0.5-1.5 us/layer if SM residency was not the cost.
- Fence asm without a "memory" clobber (pre-existing, main's code) means source order around it is not a compiler guarantee; check SASS that
  the released/mask loads issue before MEMBAR and that no store sinks past it.
- First build: check the copy_compact_routing int4/int32 instantiations for all SWITCH_TOP_K values and the op schema change.

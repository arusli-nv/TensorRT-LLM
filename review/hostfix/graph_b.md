# FT on vs off: Python-level construction of one decode MoE layer (graph topology, streams, events, PDL)

Code: worktree /tmp/claude-161238/wt-v5 (branch v5 @ f96a8c1a53). Paths are relative to it.
`sched` = tensorrt_llm/_torch/moe/fused_moe/moe_scheduler.py, `nvl` = .../communication/nvlink_one_sided.py,
`cute` = .../fused_moe_cute_dsl.py, `lb` = .../moe_load_balancer.py, `ifc` = .../interface.py,
`dsv3` = tensorrt_llm/_torch/models/modeling_deepseekv3.py.
Trace: /tmp/claude-161238/ns32/{off,ft}.sqlite. These are device 0 only, 50 graph replays, 59 MoE layers per replay
(58 main + 1 MTP). This trace does NOT reproduce the claim (step period off 15.92 vs ft 15.22 ms; see moe_path.md).
Use it only for structure and per-segment deltas. Scripts are in $TMPDIR/g (lay.py, ms.py, se.py, emax.py, step.py).

## 1. Verdict
At the Python level, FT builds the same per-layer graph as off: same streams, same event forks and joins, and the
same PDL edges, with two exceptions:
(a) off has one extra node, `moeA2ASanitizeExpertIdsKernel`, between dispatch and routing; FT drops it;
(b) the kernel variants differ: dispatch grid 96 and routing template 512 (both covered in moe_path.md).
FT adds no stream, no event wait, no memset and no host branch inside the captured region. No PDL chain is broken.
The trace confirms this: graph node counts per kernel match (59 dispatch/combine/prepare-combine and 58
routing/memset nodes in both arms), and the gap distributions at each edge match (section 3).

## 2. Per-layer launch order (decode, NVLinkOneSided + CuteDSL NVFP4, single chunk)
| # | Launch (stream) | Off | FT | Code |
|---|-----------------|-----|----|------|
| 0 | fork for shared expert: Main event record, aux `MoeShared` waits; shared gate/up GEMM, silu, quant, down GEMM on aux | same | same | dsv3:1247-1254 (`maybe_execute_in_parallel`) |
| 1 | router GEMM, `deepseek_v3_topk` (main) | same | same | sched:465-505 |
| 2 | LB stage hooks | absent (no LB) | static LB: `_load_balancer_*` all gated on `_using_dynamic_load_balancer()`, which is False, so no event, no `wait_stream`, no kernel | sched:431,553,702,744; ifc:644-707 |
| 3 | `route()` (moe_load_balance_routing) | absent | skipped: `expert_to_slot = one_slot_table()` is a host-cached device table, so no kernel | sched:567-574; lb:957-975 |
| 4 | `quantize_input` (main) | same | same | sched:635-637 |
| 5 | rank-mask publish (H2D copy) | n/a (no rank mask) | skipped when bound (`state is not None`); the kernels read `state.active_rank_mask` | nvl:1326-1327 |
| 6 | `moe_a2a_dispatch` = PrepareDispatch + Dispatch, both PDL | grid 64 | grid 96, compact routing outputs, abort/mask records | nvl:1342-1366 |
| 7 | host watchdog `watch_collective` | eager only | skipped when bound (graph replay never runs it) | nvl:1372-1375 |
| 8 | `moe_a2a_sanitize_expert_ids` (PDL) | **runs** | **not launched** (`state is None` gate) | nvl:1401-1408 |
| 9 | `_compact_receive_slots` | n/a | no-op: CuteDSL reads the slots in place (`supports_dispatch_receive_slots`) | sched:652-653, 1093-1107; cute:2190 |
| 10 | `moe_sort` -> `routingIndicesClusterKernel` (main, PDL via mUsePdl) | num_experts 256, from received ids | num_experts=num_slots 320, from compact lists; same op | cute:1371-1379 |
| 11 | memset fork: Main record, `MoeOutputMemset` (or reserved remainder) stream waits, memset, records | same | same | cute:1381-1410 |
| 12 | FC1 gather GEMM (main) | packed rows | `source_tokens/slot_tokens` kwargs (index remap only) | cute:1440-1442 |
| 13 | join memset event, FC2 finalize GEMM (main) | same | same; `num_extra_local_experts`=2 | cute:1474-1476 |
| 14 | `moe_a2a_combine` (PDL) | eager watchdog | bound: none | nvl:1510-1531 |
| 15 | join shared-expert aux event, add | same | same | dsv3:1247-1254 |
Async x-quant (`quantize_input_async`) is not taken in either arm: it requires `moe.comm is None` (sched:442-450).
`try_fused_route_quant` is only overridden by TRTLLM-Gen W4A8-MXFP4, so CuteDSL gets None in both arms. The
`not _using_load_balancer()` gate (sched:470) therefore changes nothing here.
Static LB per iteration: `MoeLoadBalancerIterContext.__enter__/__exit__` call start_iter/end_iter only when
not static (lb:1854, 1873). No `stream.synchronize()` in the static path; lb:1613/1620 belong to recovery.

## 3. Trace evidence (device 0, per-layer, n=2850 off / 2900 ft)
Edges, p50 (p90):
- routing end to FC1 start: off 0.22 (0.42), ft 0.32 (0.48). FC1 follows routing by about 0.2 us in both,
  so the PDL/graph edge is the same.
- memset start after FC1 start: 98% of layers in both (memset fork contends with FC1). Memset start minus FC1 start
  is 16.19 vs 16.61, memset end minus FC1 end -3.14 vs -3.17, FC2 end minus FC1 end 15.30 vs 15.42. Same in both.
- routing end to PrepareCombine start (all local GPU work): off 42.91 (50.18), ft 43.04 (50.72), mean +0.26.
- dispatch end to routing end: off 6.46 (6.78; sanitize included), ft 5.92 (**8.64**). FT has the better median
  and the worse tail: p90 +1.9, p99 9.25 vs 7.58.
- Shared expert (aux): the down GEMM starts about 8 us after dispatch start in both arms and ends before routing ends
  in both (-22.5 off vs -10.2 ft), so it stays off the critical path. Its duration grows 5.15 to 6.62 us in FT:
  the 96-CTA dispatch and the early-resident 8x1024 routing cluster take SMs from it.
- Bootstrap of E[max over 32 draws] from device 0's own distribution: recv to PrepareCombine 67.8 off vs 66.8 ft.
  Recv to routing end 7.44 vs 9.11 (+1.7).
- Step: graph span 15188 vs 14572 us; inter-replay gap 660 vs 658 us (identical host/eager part).
Trace caveat: every FT replay is missing the CUPTI record of one `moeA2APrepareDispatchKernel` node (graph node
...122 before layer 32's dispatch; FT has 58 prepare nodes, off 59). A parser that windows on PrepareDispatch merges
two layers once per step: the "228 us dispatch" outliers at p99 are this artifact. The moe_path.md FT window
medians may include 50 such windows; re-key on `moeA2ADispatchKernel` or graphNodeId.

## 4. Top 3 hypotheses for ~6 us/layer of cross-rank wait (in this scope)
None of these is a structural FT addition. Each has to act through the slowest of 32 ranks, so judge them by tails
and max over ranks, not by device-0 medians.

1. **The routing step after recv has a fatter tail (compact lists + MaxNumExperts=512 template; early-resident
   cluster).** Device 0 p90 is +1.9 us and E[max32] is +1.7 us. Every rank's FC1 waits on it, and every combine
   waits on the slowest rank's FC2. Of all FT-only per-layer terms, this one is on the critical path of every rank.
   - Check, one variable: lb arm with FT off and num_slots 320 (ghost-free, 10 slots/rank) vs 256, then compare
     "dispatch end to routing end" p90/p99 and the step. Trace-only: on a reproducing multi-rank pair, the per-layer
     max over ranks of (routing end - dispatch end).
   - Fix: give the post-top-k cluster kernel a tier that fits 320 slots (e.g. a 384 instantiation in
     RoutingCustomPolicy) instead of 512. The ids stay in [0, 320), so masks, ghosts and abort handling are
     unchanged.
2. **Early PDL residency during the dispatch window (moe_path D5).** Receive CTAs call
   `cudaTriggerProgrammaticLaunchCompletion` at entry, so the 8-CTA x 1024-thread routing cluster becomes resident
   about 6 us before dispatch ends (start 7.97 vs dispatch end 14.14). In the off arm, routing launches after
   sanitize. This contention is visible in the shared-expert GEMM (+1.5 us) and can stretch the receive CTAs'
   copy/fence and the routing tail on loaded ranks.
   - Check, one variable: move the receive-CTA trigger to after the compact copy (a2a kernel one-liner, see
     moe_path a2a.cu:610-613), FT only. Compare the shared down-GEMM duration, the routing-end tail and the step.
   - Fix: trigger after the copy. Routing still does `griddepcontrol.wait` on full dispatch completion, so no FT
     guarantee changes. The routing launch overlaps the last copy instead of the whole wait.
3. **Fixed receive slots chosen at bind time (`_bound_layout`: slot_tokens = max_num_tokens_per_rank, nvl:1180-1214;
   bind_layers nvl:828-874).** The bound workspace is about 32 x capacity x 4 KB per rank, versus a packed 64-row
   stride off. Sender writes stay one contiguous chunk per peer in both layouts. The difference is on the receiver:
   the FC1 gather and the compact copy touch 32 regions about 29 MB apart (TLB/L2). This shows as +0.26 us FC1 on
   device 0, and possibly more on ranks with more remote traffic. It ranks third because nothing here inflates the
   sender side.
   - Check, one variable: FT arm `max_num_tokens` 2048 vs 8192 (slot stride drops 4x; graphs and decode batch
     unchanged), plus an off control at 2048.
   - Fix: size the bound capacity from the decode worker's actual max_num_tokens (config, not code). Keep one stride
     for all graphs: per-batch strides would alias a lost source's late write across layouts.

Ruled out in this scope: streams, events, PDL breaks, extra memsets, LB hooks, `route()`, mask publish, watchdog,
the async-quant path and `try_fused_route_quant`, routing path selection (single cluster up to 8192 tokens in both
arms, RoutingFromTopKIds.cu:96), and GEMM tactic. The remaining wait terms (dispatch-to-recv flag wait, combine with
rank mask/abort poll) are kernel internals covered by the parallel agent. A per-step (not per-layer) cause outside
the MoE layers, such as host-side FT work in a host-bound regime, also fits "peaks at 32/GPU". It can only be judged
on a reproducing trace: compare the inter-replay gap (660 vs 658 us here).

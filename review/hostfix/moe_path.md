# FT on vs off: per-MoE-layer GPU work at decode (DEP32, R1 NVFP4, CUTEDSL, NVLinkOneSided, 32 req/GPU)

Code: worktree /tmp/claude-161238/wt-v3 (branch v3 @ a02f262be7). Paths below are relative to it;
`sched` = tensorrt_llm/_torch/moe/fused_moe/moe_scheduler.py, `nvl` = .../communication/nvlink_one_sided.py,
`a2a.cu` = cpp/tensorrt_llm/kernels/moe/communication/moeAlltoAllKernels.cu, `cute` = .../fused_moe_cute_dsl.py,
`lb` = .../moe_load_balancer.py.
Trace numbers: /tmp/claude-161238/ns32/{off,ft}.sqlite. Device 0 only, 50 decode steps, n=2850 decode MoE layers
per arm. Medians come from a per-layer window (moeA2APrepareDispatchKernel to moeA2ACombineKernel, dispatch
grid 64/96). This trace does NOT reproduce the claim. Its step period was off 15.92 ms vs ft 15.22 ms, and its
58-layer span 13.29 vs 12.65 ms, so FT was faster here. Use it for per-kernel deltas only, not for the step delta.

Local tokens per rank at decode = 32 req x 2 (MTP1) = 64. That gives dispatch grid 64. Backend rows =
32 x 64 = 2048, and 16384 (token, expert) pairs.

## 1. Kernel set (whole trace, decode)
The kernel lists are identical except for two things:
- FT has no `moeA2ASanitizeExpertIdsKernel`. Off runs it on every layer (grid 8, 21.9 us avg, mostly flag wait).
- `moeA2ADispatchKernel` grid is 64 off vs 96 FT (+32 receive CTAs).
FT adds no extra kernels, memcpys (1200 vs 1200) or memsets (50 vs 50). There are no load-balancer kernels: static
routing leaves every LB stage a GPU no-op. The `routingIndicesClusterKernel` template differs (item R2).
Both arms use the fence path, not CFT. Off's separate sanitize kernel shows CFT was off in that run, and the FT
bound path requires fence (a2a.cu:1512-1516).

## 2. Per-layer differences

| # | Item | Off | FT | file:line | est. us/layer (trace median) |
|---|------|-----|----|-----------|------------------------------|
| R1 | top-k -> slot | expert ids sent as-is (no LB) | `expert_to_slot` table passed to dispatch; `route()` kernel skipped (the lb arm runs it) | sched:567-571, lb:957, a2a.cu:776-784 | ~0 (8 L2-hit loads per token CTA) |
| R2 | routing sort `moe_sort` -> routingIndicesClusterKernel | num_experts=256, local 8: template MaxNumExperts=256, smem 8 KB | num_slots=320, local 10: template **MaxNumExperts=512, smem 16 KB** | cute:1371-1379 | **+1.06** (recv-done to routing-end: 4.86 vs 5.92) |
| R3 | routing input | received slot ids, sanitized in place | compact lists from receive CTAs (expert -1, weight 0 on empty rows), [32*64, 8] | nvl:1202-1215, 1292-1294 | in R2 / D2 |
| D1 | dispatch variant | fence, ENABLE_RANK_MASK=0, then separate sanitize kernel | fence, ENABLE_RANK_MASK=1, USE_EXPERT_TO_SLOT=1, abort record, fixed slots | nvl:1179, 1216-1240, 1272-1278; a2a.cu:1598-1607 | dispatch start to recv done: **-12.4** (30.62 vs 18.24); a cross-rank wait term, varies by run |
| D2 | receive path | last token CTA: 1 warp polls 32 flags; sanitize waits again | +32 CTAs (`receive_source`), one per source: flag wait, `fence.acquire.sys`, compact copy (64x8 ids+weights), `source_valid_counts[s]`; last token CTA returns without waiting | a2a.cu:606-715, 727-735, 905-908, 1531-1540 | in D1 |
| D3 | rank mask / abort | none | per-target `is_rank_active` (device mask); abort record in **pinned host memory**, polled only after a wait passes kAbortPollCycles=100k cycles (~50 us) | a2a.cu:289-320, kernels.h:62, execution_state.py:45 | ~0 at median waits of 18-35 us; ~1 us C2C read on waits over 50 us |
| D4 | receive layout | packed: row = s*64+i (runtime max); hidden ~7.3 MB contiguous | **fixed slots: row = s*8192+i** (capacity = max_num_tokens). 32 regions of 64 rows, ~29 MB apart (3.5 KB/row), for each of 4 payloads | nvl:1059-1097 (slot_tokens=max_num_tokens_per_rank), a2a.cu:786-794 | not separable; shows in D1 and G1 |
| D5 | PDL | token CTAs only | receive CTAs call `cudaTriggerProgrammaticLaunchCompletion` at entry, so the routing kernel (8x1024 cluster) becomes resident and spins early | a2a.cu:610-613 | ~0 (routing launches about 11 us before dispatch ends) |
| D6 | host per dispatch | watchdog `watch_collective`, mask publish | bound: none (graph replay) | nvl:1200, 1244-1247 | 0 GPU |
| G1 | FC1 gather GEMM | reads packed rows | reads receive slots in place (`source_tokens=64, slot_tokens=8192` index remap); no compact copy (`_compact_receive_slots` is a no-op because CuteDSL supports slots) | cute:1440-1442, 2190; sched:1093-1107 | **+0.26** (27.36 vs 27.62) |
| G2 | local slots | esp=8, extra=0 | esp=10, `num_extra_local_experts=2` (ghost: unmapped weights, no tokens, no tiles); tactic key uses the 8-slot layout | cute:1136-1140; cute_dsl_custom_ops.py:213-245, 344 | FC2 **+0.14** (16.93 vs 17.07); same tactic (persistent grid 152, same kernel) |
| G3 | tile scheduling vs capacity | tiles from `num_non_exiting_tiles` (device) | same; permuted-row bound adds 2x(tile-1)=254 rows for 2 extra slots. Nothing scans the 8192-row capacity | cute_dsl_custom_ops.py:330-345 | ~0 |
| G4 | moeOutputMemset / finalize | grid 912 | grid 912 | cute:1381-1410 | 0 |
| C1 | combine | fence, no mask | fence, ENABLE_RANK_MASK=1 (mask check per peer, abort poll, `__syncthreads_or`) | nvl:1381-1397; a2a.cu:2031-2065 | ~-0.8 (32.67 vs 31.90; wait-dominated) |
| L1 | load balancer per layer/iter | absent | static: `done_wait_gpu_stage`, `update_statistic`, `get_local_statistic_tensor`, `start_set_cpu_stage` all gated on `updates_enabled` (False), so no events, stream waits or kernels; iter context is a no-op | lb:728-790, 823-837, 1854-1880; sched:553-562, 702 | 0 |
| A1 | aux-stream overlap | shared-expert GEMM nvjet 56-CTA 5.25 us | 6.21 us (+0.96), quantize 4.09 to 4.40, running beside a 96-CTA dispatch | trace only | off critical path in the sample (aux ends 10-30 us before FC1 starts) |

Sum of local (non-wait) deltas: R2 +1.06, G1 +0.26, G2 +0.14, so **~+1.5 us/layer** (~0.09 ms per step over 59
layers). Nothing in the code adds 6-9 us of GPU work per layer, and no FT-only node is inserted into the CUDA graph.
The rest of the claimed delta has to be in the cross-rank wait terms: dispatch start to recv done, and combine.
Those wait terms moved by -12.4 us/layer between these two runs, i.e. their run-to-run spread is larger than the
claimed effect.

## 3. Top 3 suspects for 6-9 us/layer

1. **Fixed-slot receive layout (D4) charged as cross-rank wait (D1/C1).** Each source's rows sit 29 MB apart
   (capacity 8192), not packed in ~7 MB. Every peer's remote writes, and this rank's G1 reads, touch 32 x 4
   scattered regions. Translation or locality costs on remote NVLink writes would show up on the *receiver* as
   later flag arrival (wait time), not as longer kernels. That fits (d): per-kernel time is not higher.
   - Check, one variable: FT arm only, `max_num_tokens: 2048` vs `8192`. This shrinks the slot stride 4x; decode
     at 64 tokens/rank and the graphs are unchanged. If FT at 32/GPU improves by ~0.3 ms, D4 is the cause.
     Control: one off run at 2048.
   - Log/trace check first: in a reproducing pair, capture nsys on >=2 ranks and compare per-layer medians of
     "dispatch start to last receive-CTA end" and "combine dur" (the window method in section 4).
2. **Barrier structure of the bound dispatch (D2/D5).** 32 per-source CTAs each wait, fence and copy, and the
   routing kernel cannot start until the slowest receive CTA finishes. Off waits with one warp and then the
   sanitize kernel. With skew across 32 ranks, the extra per-source acquire fence and copy are serialized behind the
   last arrival on every layer. That adds 1-3 us more than the packed path, and it shows only when skew is
   present, so it can be absent from this trace.
   - Check: in the existing ft.sqlite, take (routing start - dispatch end) and (dispatch end - last token CTA)
     per layer. In a reproducing multi-rank trace, regress per-layer (recv done to FC2 end) on dispatch wait. A
     kernel-only check: nsys with `TLLM_FAULT_TOLERANCE_MODE=1` on a single node (EP4) at fixed skew.
3. **Routing-sort template jump to MaxNumExperts=512 (R2).** Confirmed in the trace by the demangled name: 256 vs
   512, smem 8 vs 16 KB. It costs +1.06 us/layer here and could be larger when not hidden by PDL. Fact (b) (320
   slots ~ 256 slots under lb) argues it is small, which is why it ranks third.
   - Check: already readable in sqlite (`routingIndicesClusterKernel` demangledName, gridX=8, sharedMemoryExecuted
     8192 vs 16384). One-variable A/B: lb arm (FT off) at num_slots 320 vs 256, with the same kernel instance
     switch. If the delta is ~0, rule R2 out at the step level.

## 4. Queries used
- Kernel set: `CUPTI_ACTIVITY_KIND_KERNEL JOIN StringIds ON shortName`, grouped by (name, gridX, gridY, blockX),
  diffed between off and ft.
- Per-layer window medians (decode, dispatch grid 64/96): off vs ft
  - span: 116.06 vs 105.57
  - dispatch to recv done: 30.62 vs 18.24
  - recv done to routing end: 4.86 vs 5.92
  - FC1: 27.36 vs 27.62
  - FC2: 16.93 vs 17.07
  - combine: 32.67 vs 31.90
- Step: 58 main-model MoE layers plus 1 MTP layer per step. The last main-model combine to MTP dispatch gap is
  659 vs 661 us.

## Floor after the receive fix (A) and per-lane prologue staging (B'), estimated

Per MoE layer, from the ns32 per-kernel deltas above and plateau.md (n=1 trace, 32/GPU, device 0):
- Cross-rank chain, the two halves targeted: sender prologue ~+1.6 us (B': two dependent loads move before the
  dependency wait, so ~1.0-1.3 us of it should hide behind the prepare kernel) and the per-source receive tail
  +1.5-2.2 us (A: a warp per source instead of a CTA, and 68 CTAs instead of 96, so ~0.5-1 us left for the acquire
  fence and the compact copy behind the slowest source).
- Local work that stays with A + B' (~1.5 us/layer, ~0.09 ms/step):
  - routing sort in the 512-expert template (320 slots > 256): +1.06 us. Removed only by a 384-expert tier (an
    upstream routing-kernel change) or by keeping FT's ghost slots out of the routing table;
  - FC1 gather reading the fixed slots in place: +0.26 us (inherent to timing-independent slots);
  - FC2 with 10 local slots instead of 8: +0.14 us (the ghost restore slots);
  - dropped sanitize: -0.09 us (FT gains it back).
- So the expected floor after A + B' is ~1.5-2.5 us/layer, ~0.1-0.15 ms/step at 32/GPU (~0.7-1% of a 14 ms step), against
  today's ~0.35 ms. These are estimates to check against RECVAB and the A + B' A/B, not CLAIMS numbers.

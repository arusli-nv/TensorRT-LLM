# What makes the no-EPLB FT gap a 24-48 req/GPU plateau that drops at 64 (CLAIMS 162)

Code: worktree /tmp/claude-161238/wt-v5 (branch v5 @ 5ebf7f72a2). The measured run used the v3 wheel (bcea8da496).
`git diff bcea8da496 HEAD` is empty for routing/, communication/ kernels, cute_dsl_custom_ops.py and
fused_moe_cute_dsl.py. It touches only nvlink_one_sided.py (the per-signature workspace layout), so the kernel paths
below are the measured ones. Paths: `rt` = cpp/tensorrt_llm/kernels/moe/trtllmGen/routing/, `a2a.cu` =
cpp/.../moe/communication/moeAlltoAllKernels.cu, `nvl` = tensorrt_llm/_torch/moe/fused_moe/communication/nvlink_one_sided.py,
`cute` = .../fused_moe/fused_moe_cute_dsl.py, `ops` = tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py.
Builds on moe_path.md, kernels_a.md and graph_b.md (same directory).

Token counts used below: T = tokens/rank = 2 x req/GPU. R = rows the routing and GEMM see = 32T. E = expanded rows = 8R.

| req/GPU | 16 | 24 | 32 | 40 | 48 | 64 |
|---|---|---|---|---|---|---|
| T | 32 | 48 | 64 | 80 | 96 | 128 |
| R | 1024 | 1536 | 2048 | 2560 | 3072 | 4096 |
| E | 8192 | 12288 | 16384 | 20480 | 24576 | 32768 |
| gap ms, trim (median) | .21 (.21) | .33 (.35) | .32 (.27) | .39 (.27) | .27 (.19) | .14 (.03) |

Read the medians as well as the trimmed means. The medians peak at 24 and then fall steadily; they do not show a flat
plateau followed by a cliff. With n=1 per point, the 48->64 step is within the spread CLAIMS 160 shows inside a single
run (trim 0.37 vs median 0.50 at 32/GPU).

## 1. Every branch on token count in the decode MoE path

| # | Branch | file:line | Cutoff | Which arms | Switch inside 16..64? | Predicted gap shape |
|---|---|---|---|---|---|---|
| 1 | post-top-k path: dyn-block / static block / single cluster / coop / histogram | rt/RoutingFromTopKIds.cu:75, 85-86, 96 | block R<=4; dyn-block R<=12; single cluster R<=8192 | both, same R | no: single cluster at every load | none |
| 2 | cluster block size: 512-thread when post-top-k and R<=1024, else 1024-thread | rt/RoutingCustomKernels.cuh:1749-1754 (score limits :1500-1501 do not apply post-top-k) | **R<=1024, i.e. only 16/GPU** | both | **yes, between 16 and 24** | at 16 the 8 cluster CTAs need 512x~60 regs (half an SM), not a full SM. FT's early-resident cluster (PDL trigger at receive-CTA entry, a2a.cu:610-613) contends less. Predicts a smaller gap at 16 only |
| 3 | expert tier | rt/RoutingCustomKernels.cuh:1536-1547 (no 384 tier in the Softmax cluster lists) | num_experts 256->256, 320->512 | FT only | no: fixed by slots, not by T | constant +~1 us/layer (moe_path R2) |
| 4 | cost of the 512 tier vs T | rt/RoutingKernel.cuh:234 (ExpertsPerThread = 1 for both 256 and 512 at 512 or 1024 threads), :242, :319-353 | per-thread expanded loop = E/8192 = 1, 1.5, 2, 2.5, 3, 4 iterations; fast path (no bound checks) only when E is a multiple of 4x8192 | both, same E | the 64/GPU point (4 iterations) is the only fully fast-path one | the tier adds a fixed cost (smem init over 512, DSMEM gather for 320 vs 256 threads); the T-dependent part is identical in both arms. Predicts no T shape in the gap |
| 5 | dispatch geometry | a2a.cu:1533-1541 (grid T, +ep_size receive CTAs if bound), kBlockSize 256 at :1438; one CTA per token, block-wide fanout per payload :803-812 | none (1 token/CTA at every T) | FT +32 CTAs | no | fixed +32 CTAs. The SM-count model (kernels_a H1) puts the threshold at T>56 and is refuted at 24 |
| 6 | off-only sanitize | a2a.cu:2555-2558, kernel :2512-2547 | grid = 32T/256 = 4,6,8,10,12,16 | off only | linear, no cutoff | off's post-dispatch tail grows ~linearly with T. Would erode the gap slowly, not stepwise |
| 7 | CFT vs fence | nvl:64-67, 186-198, 1281 | CFT only if runtime T <= 128 (and driver/build support) | off could use it; FT bound requires fence | no: 128 <= 128 keeps CFT at 64/GPU. Both runs were fence anyway (off ran sanitize, moe_path) | none in range |
| 8 | outer tactic (tile 128/256) bucket | cute:409-413 (tiles), 447-451 (bucket = last pow2 of R); tuner input x[:num_tokens] for FT, cute:1263-1270 | R buckets 1024 \| 2048 \| 4096 -> {16,24} \| {32,40,48} \| {64} | both, tuned separately per process | **yes, at 64** (and between 24 and 32) | changes in both arms. It differs between arms only if per-process profiling picks different tiles. Its buckets split 16/24 together and 24/32 apart, which does not match a 24-48 plateau |
| 9 | FC1/FC2 inner tactic buckets | ops:277-282 (tiles = (E+(tile-1)*nle)/tile), 346-357; FT key adds pow2(source_tokens), slot_tokens, extras, ops:4302-4305 | permuted-row bucket = pow2 of inferred tokens (same as #8); FT source_tokens pow2 = 32 \| 64 \| 128 -> {16,24} \| {32,40,48} \| {64} | FT profiles on slot-strided input | at 64 (both arms re-bucket) | same caveat as #8. Ghost slots add 2x(tile-1) rows to the bound, but no tiles. At 32/GPU the FC1/FC2 tactic matched (moe_path G2) |
| 10 | GEMM tile count | routing writes num_non_exiting_tiles, RoutingKernel.cuh:400-417 | rows/expert ~T: tile 128 gives 1 tile/expert to T~96, then ~1.5 at T=128 | both, same counts (ghosts get 0) | soft at 64 | none differential |
| 11 | CUDA graph padding | no FT-specific graph sizing (no FT hook in pyexecutor graph batch sizes; graph_b: same nodes) | config buckets | both | depends on the YAML, not on FT | none differential |
| 12 | combine | a2a.cu:2396-2505 | grid = T, packed rows at runtime max, in both arms; FT adds only mask/abort checks | both | no | none (floor equal, kernels_a s5) |

So between T=96 and T=128 nothing switches for FT alone. The switches at 64/GPU (#4 fast path, #8/#9 tactic buckets,
#10 tile count) hit both arms. The only cutoff that lines up with a feature of the curve is #2: the 512-thread routing
cluster used only at 16/GPU (R=1024). Nothing is constant from 48 to 96 that is not also constant from 96 to 128, apart
from the tactic bucket, and that bucket also contains 32 and 40.

## 2. Best explanation for the whole curve

The FT cost has two layers:
- (a) a fixed per-layer local cost of ~1.5-2 us. This is the routing tier, the FC1 slot gather and FC2 extras
  (moe_path s2), about 0.09-0.12 ms/step.
- (b) a latency term that sits on the cross-rank critical path and is exposed only in a middle load window:
  - the sender prologue: +1.6 us on every rank's flag publish (kernels_a s3);
  - the per-source receive barrier tail: fence, count, then the compact copy before routing's
    griddepcontrol.wait releases (kernels_a s4: +1.5 to +2.2 us floor).
  Both are dependent-latency chains of fixed length.
- High load (64/GPU): what remains is (a). CLAIMS 154 has the same pair at 64 and 128/GPU, +0.11 and +0.10 ms
  = 1.9 and 1.7 us/layer over 59 layers, which matches (a). As T grows, the dispatch and combine waits are set by
  per-rank payload volume and the max over 32 ranks of FC1/FC2 time. A fixed few-us chain then moves the slowest
  rank less, so (b) is absorbed.
- Middle loads (24-48): (b) is fully exposed. 0.27-0.39 ms is 4.6-6.6 us/layer, which is about (a)+(b).
- Low load (16/GPU, and ~0 at 8/GPU, CLAIMS 154): the gap falls. Two code-grounded contributors:
  - the 512-thread routing cluster (#2) halves the footprint of FT's early-resident cluster;
  - more likely, the step is closer to host-bound at low load (8/GPU has the same 11.9 ms step in both arms,
    CLAIMS 154/158), so part of a GPU-side shift is hidden behind host scheduling under the overlap scheduler.

This is a reading of exposure, not a kernel-variant switch. It predicts that the per-layer local delta is the same at 48
and 64/GPU and that only the cross-rank wait delta shrinks. Confidence is moderate: no code cutoff supports a
FT-specific switch at 96->128, and with n=1 a gradual slope (as the medians show) fits as well as a cliff.

## 3. One-variable checks

1. Variant-switch test (decisive for "a switch exists"). Take a same-node off/ft pair at 48 and 64 req/GPU with nsys on
   device 0 plus one remote rank. Diff the kernel set by (demangledName, gridX, blockX, sharedMemoryExecuted) per arm
   and load, using the moe_path s4 query.
   - Prediction: the FT-minus-off diff is identical at both loads: dispatch grid T+32 vs T, sanitize absent,
     routing template 512 vs 256.
   - A different FC1/FC2 kernel name between arms at 64 (#8/#9) or a routing block size other than 1024 would be the
     switch. In the same traces, take the per-layer p5 of (dispatch start -> FC1 start) FT-off. Prediction: ~+2 us at
     both loads, while the combine-wait median delta falls at 64.
2. Zero-GPU check of #8/#9: grep the LOADCURVE rank-0 logs (STD-8150634) for the autotuner's chosen tactics, if they were
   logged at debug level. If both arms have the same tile and tactic per bucket, #8/#9 are out.
3. Low-end check (host-bound hiding). In the off arm at 16/GPU, compare the GPU graph span with the step period (an
   nsys step query as in graph_b s3). If the step period exceeds the span by more than ~0.2 ms, the GPU-side FT shift is
   partly hidden there. Code toggle to isolate #2 instead: `PostTopKClusterBlockDim512MaxNumTokens` 1024 -> 0
   (rt/RoutingCustomKernels.cuh:1749) in both arms at 16/GPU. If the gap rises to ~0.33 ms, #2 explains the low end.

The existing ns32 sqlite (32/GPU only) can confirm the 32/GPU kernel diff (done in moe_path: same GEMM tactic, cluster
8x1024 in both arms, template 256 vs 512). It cannot test load dependence.

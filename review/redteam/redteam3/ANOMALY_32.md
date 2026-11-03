# ANOMALY_32: w/o EPLB FT cost of +0.33 ms/step at 32/GPU (read-only analysis, rc1 build-rc1-78836d1723)

Runs: STD-8110144 (G0) and STD-G1-8110144 (G1), n=1 case per arm per group, R1 NVFP4 DEP32, std-*.yaml (MTP1 forced
acceptance, CUTEDSL). R = .../.wideep-ft-runs/claude/mvp/runs. Scratch scripts: $TMPDIR/a32/{hd,dist,corr}.py.

## 1. What the delta is: device-side, a uniform shift (HIGH confidence)
Source: rank-0.log decode steps (num_ctx 0, num_gen = 2 x load). prev_device_step_time is attributed to the step it measures.
Values are medians in ms.

| load | G0 off -> FT device | G1 off -> FT device | G0 host body (off -> FT) |
|---|---|---|---|
| 8 | 12.088 -> 12.067 (**-21 us**) | 12.091 -> 12.068 (**-23**) | 11.894 -> 11.882 |
| 32 | 14.501 -> 14.816 (**+315**) | 14.476 -> 14.829 (**+353**) | 13.735 -> 14.168 |
| 128 | 27.490 -> 27.681 (+191) | 27.645 -> 27.648 (+3) | 24.380 -> 24.531 |
| 512 | 80.283 -> 80.785 (+502) | 80.136 -> 80.822 (+686) | 70.52 -> 70.96 |

- At 32/GPU the whole device distribution moves: q5..q95 go from 14.26..14.73 to 14.58..15.05 (G0). It is not a tail.
  The device IQR is about 0.15 ms, and off-to-off across node groups agrees within 25 us. The effect reproduces;
  it is not node-set noise.
- **Not host:** rank-0 host body vs device step correlates at -0.13/-0.07 (G0) and +0.05/+0.02 (G1). Steps with a short host
  body (<12.6 ms) still show the full shift: device medians 14.44 -> 14.79. The host body is wide (11.3..15.1 ms) and
  has about 1 ms of slack against the device step in both arms. The host-median "steady_step" (+3.1%) overstates the
  cost. The device cost is +2.2/+2.4%.
- **Not attention/KV:** the shift is flat across KV fill (harness kv_bins.py, +2.28/+2.43%). ADP dummies are 0 in both arms
  at 32/GPU (rank-0 log), and the graph batch is the same (32).
- **Same-run controls at 32/GPU:** online FT cost is -3 us (G1: 15.594 -> 15.591; the off arm already has an LB with 288
  slots, 9/rank). Offline FT cost is +122 us (G0: 14.858 -> 14.980). The anomaly is specific to "no LB at all" (off) vs
  "static LB 256 + 64 ghosts" (FT).

## 2. Decomposition at 32/GPU (us/step)
| # | piece | est. us/step | evidence | load shape | confidence |
|---|---|---|---|---|---|
| A | Routing tier 512 (320 slots) vs 256 | +40..50 | ledger §1a item 2: +0.68..0.85 us/layer (+39..49), measured at static 32/GPU. rc1 `runPostTopKPipeline` (RoutingFromTopKIds.cu) picks the post-top-k cluster kernel BlockDim512 at <=1024 rows (8/GPU = 512 rows) and BlockDim1024 above (32/GPU = 2048, 128/GPU = 8192). The tier cost may differ by variant, which would leave ~0 at 8/GPU | fixed per layer at >1024 rows | MEDIUM |
| B | 10 vs 8 local slots in the MoE GEMM (padding bound, FC1 per-row source-slot divide) | +15 | ledger §1a item 4 (+0.06/+0.20 us per launch, same at 32 and 256/GPU) | fixed | MEDIUM |
| C | FT dispatch receive path vs main's sanitize tail; combine rank-mask variant (R2 in) | ~0 | ledger: net 3a+3b = -1. R2 removed the +41 mask load. rc1 includes R2 (08ba507229) and the R3 restore (911d8f2a1d). Online FT (same FT kernels) costs ~0 at 32/GPU here | token-scaled (ledger: +1.6/layer at 32, +4.8/layer at 256) | MEDIUM |
| D | MTP layer (bf16, CUTLASS MoERunner) tactics. Off ran self-tuned 8-expert keys (PIN_CHECK FAIL, 38 keys); FT ran pinned 10-expert keys | 0..+50 (bounded) | bucket = sum(all_rank_num_tokens) = 2048 at 32/GPU. gemm1 off '45' (86.3 us tuned) vs FT '146' (98.3 us tuned, measured with 10 loaded experts). gemm2 '298' vs '296'. The tactics also differ at 512 and 8192 (8 and 128/GPU). The whole MTP MoE is ~140 us/step, so it cannot carry 330 | one call per step | LOW-MEDIUM |
| E | CuteDSL main-layer tactics (58 layers) | **0 at 32/GPU** | pinned.json across all 32 ranks: buckets 512 and 2048 use tile '128' for the 256-, 288- and 320-slot keys, with similar tuned times (2048: 50.5 / 56.4 / 54.9 us) | — | HIGH (ruled out at 8 and 32) |
| F | Static-LB path cost vs no LB: expert->slot routing, LB stats/signal plumbing, LB-mode dispatch. Off runs main's non-LB path | **unmeasured; the prime suspect for the remainder** | the only difference left that online FT vs online-off does not share (both have an LB there, cost ~0). No profile exists of std-noeplb vs any LB arm with MTP | ? | LOW (untested) |
| | **sum A..E** | **+55..115** | | | |
| | **remainder** | **+200..300** | not explained by any measured component. Candidates: F; cross-rank combine wait (ledger §1a remainder of +36 at static 32/GPU, sign flips across builds); the MTP layer's FT dispatch/expand (expandInputRowsFromSourceSlotsKernel, a1d93a6f86) | | |

The static nsys pair FT-COST-PROF-B-8092284 (aae00e6c65, 32/GPU, **no MTP**, st1_58.txt in /tmp/claude-161238/q120)
gives GPU step +120.8 us. Its parts: dispatch +2.20/layer, routing +666.9/step against sanitize -684.7/step, combine
+17, rest +0.33/layer, expert_gemm +2. Host step +58.8 (12.72 vs 12.67 ms GPU: near host-bound at 32/GPU without MTP).
Since then, rc1 contains R2/R3. The pair has no MTP layer and no rc1 kernels, so it bounds A+B+C at about +60..120
(final 547 static at 32/GPU: +63, CLAIMS #134). It cannot see anything MTP- or rc1-specific. No STD (MTP) nsys exists.

## 3. Why ~0 at 8 and 128
- **8/GPU (-21/-23):** 512 routing rows take the post-top-k BlockDim512 cluster variant, where the tier penalty is
  likely small. FT's removal of the sanitize tail is most exposed at low load (the ledger notes the send phase is
  short). The two roughly cancel. The CuteDSL tactics are identical. MEDIUM.
- **128/GPU (+191/+3): masked by a tactic confound, not cost-free (MEDIUM-HIGH).** At bucket 8192 (=32 x 256 rows), all 32
  ranks' **256-slot** CuteDSL keys (FT off) pinned tile **'128'** (tuned mean 89.2 us, max 98.4). The 288- and 320-slot keys
  pinned **'256'** (80.0 / 79.4 us, max ~81). By tuner times, FT gets a faster MoE tactic on 58 layers: up to
  ~9-18 us/layer, i.e. **~-0.5..-1.0 ms/step** (tuner synthetic data; the real gain is smaller but signed). The 8192 row
  of the 256 table is non-monotone ('256' at 4096 and 16384, '128' at 8192), so it looks like a tuning artifact. A
  fixed-or-growing FT cost of the 32/GPU size is therefore plausibly hidden at 128/GPU. At 512/GPU (bucket 32768) all
  tables pick '256', and FT shows +502/+686 us, consistent with an unmasked cost that grows with load.
- Net shape: the FT cost grows with load from ~0 (8) to +0.33 (32) to ~+0.6 ms (512). 128 is the exception, offset by the
  off arm's worse tactic. So the 32/GPU number is not a 32-only anomaly. 8/GPU is the special case.

## 4. Ruled out
Host/Python FT hooks (device-side shift, corr ~0); KV/attention (flat across fill); ADP padding (0 dummies);
CuteDSL tactics at 8/32 (identical tile 128); node set (G0/G1 agree within 40 us); the compact-copy revert (R3 is in rc1:
911d8f2a1d is an ancestor of 78836d1723 by message; online FT at 32/128 is ~0/+78); the old 2x DispatchSourceSlots.compact
copies (CUTLASS now reads slots in place, a1d93a6f86).

## 5. Decisive cheap experiment (one group of 8 nodes, ~40-45 min)
Three arms on the rc1 build, mirrored. Use the std-seed2 table so every arm is PIN_CHECK OK: it holds the 8-expert MTP
keys. Run 32/GPU plus a short 128/GPU.
```
R=.../.wideep-ft-runs/claude; B=build-rc1-78836d1723
E="MVP_BUILD=$B FINAL_BUILD=$B TLLM_TEST_POPULATE=timed FINAL_BASE_BUILD=build-devs-f4afe429ba WIDEEP_POLLUTION_CHECKS=1 \
   FT_COST_E2E_ROUNDS=1 FT_COST_E2E_ROUND_MIN=50 FT_COST_E2E_SEED=$R/launch/std/std-seed2.json \
   COST_LOADS=32,128 COST_LOAD_ITERATIONS=32=15,128=5"
bash $R/bundle.sh "0:8:3000:$E FT_COST_E2E_ARMS='std-noeplb-lb std-noeplb std-noeplb-ft' bash $R/mvp/harness/run_job.sh STD"
```
(std-noeplb-lb = std-noeplb-ft.yaml with FT off: a static LB with 256 slots, 8/rank, tier 256, the same CuteDSL and MTP keys as
off. job.sh:2938.) Read the rank-0 prev_device_step_time medians (`$TMPDIR/a32/hd.py`) and ab_stats-*-lb*.txt.
Decision at 32/GPU:
- lb ≈ off and FT − lb ≈ +0.33 ms: FT-specific (tier 512, 10 slots, FT kernels). Next: a 32/GPU nsys pair lb vs FT.
- lb ≈ FT: the static-LB path in main's code costs it, not FT. Next: nsys pair off vs lb.
- At 128/GPU, lb and off share tile '128': FT − lb gives the true FT cost at 128, and lb − off ≈ 0 confirms the confound.
STD mode has no `--profile-after-s` (job.sh STD `load=(...)`). Adding nsys to this job needs a harness knob. Do not
edit job.sh while a job runs (jobs run /tmp copies). Cheapest follow-up profile: the same three configs through a
C-COMBINE2-style block with `TLLM_TEST_NSYS_RANKS=all TLLM_TEST_NSYS_CAPTURE_END=repeat:2:async --loads 32
--profile-after-s 20`, analysed with `nsys_ab.py --layers 59`.
Also fix the tactic confound in the finals: the 256-slot CuteDSL 8192 bucket ('128') should be re-tuned or checked.
Otherwise the w/o-EPLB FT cost at 128/GPU is biased low.

## Routing/sort kernel variant by token count
Code read on branch v2 (cef4670789); file:line below are v2 paths. Note: the STD runs used rc1 (78836d1723). I did not diff rc1
against v2 for these files. The routing code paths below match the rc1 description in §2 row A (BlockDim512 at <=1024 rows).

**Which rows reach the sort.** CuteDSL calls `moe_sort(num_experts=self.num_slots, local_num_experts=esp, tile 128)`
(fused_moe_cute_dsl.py:1371-1379). That becomes `moe_topk_sort_impl` with DeepSeekV3 and n_group 1
(cuteDslMoeUtilsOp.cpp:123-135), then `routingCustom::run` (runner.cu:72-230), then `runPostTopKPipeline`
(RoutingCustomKernels.cuh:2102-2110). The sort input is the post-dispatch routing, ep_size x runtime_max rows, in both arms. FT
sorts its compact lists, which have the same row count (nvlink_one_sided.py:1059-1092). Rows = 32 x tokens/rank: **512 / 2048 /
8192 / 32768**, with 8x as many expanded ids.

**Tier mapping.** The post-top-k pipeline forces policy None+Softmax (RoutingFromTopKIds.cu:60-65). Its tier list has no 384
(RoutingCustomPolicy.cuh:578-593, first covering tier wins, :691), so 256 slots map to `Tier<256,8>` and 320 map to
`Tier<512,8>`. The coop kernel is the exception: it dispatches on mNumExperts with a 384 case (RoutingCustomKernels.cuh:1954-1963),
so 320 slots map to 384 there.

**Variant selection** (RoutingFromTopKIds.cu:75-149; constants RoutingCustomPolicy.cuh:385-391, RoutingKernel.cuh:45; cluster
variant RoutingCustomKernels.cuh:1749-1768; GB200 has 148 SMs, so coop uses 140 blocks):

| tok/rank | rows | variant (both arms) | 256 slots | 320 slots | per-thread loop trips |
|---|---|---|---|---|---|
| 16 | 512 | single-cluster `routingIndicesClusterKernel`, **BlockDim512** (rows <= 1024, :1749-1754), 8 CTAs x 512 thr | Tier<256,8>; smem counts+offsets 2 KB | Tier<512,8>; 4 KB | 4096 ids / 4096 thr = 1 (compiled max 8) |
| 64 | 2048 | single-cluster, **BlockDim1024** (rows > 1024 and > the 64/128 score-path limits :1500-1501, :1758-1768), 8 x 1024 thr | Tier<256,8>, 2 KB | Tier<512,8>, 4 KB | 16384 / 8192 = 2 |
| 256 | 8192 | single-cluster, **BlockDim1024** (rows <= 8192 = 8 x 1024) | same | same | 65536 / 8192 = 8 |
| 1024 | 32768 | **coop** `routingIndicesCoopKernel` + InitExpertCounts (:114-132). 32768 <= 140 x thr x 64/8 | 140 x **256** thr, 2 KB | 140 x **384** thr, 3 KB | 262144 / 35840 -> **8** vs / 53760 -> **5** |

**Why the tier barely matters in the cluster kernel** (routingPermutation, RoutingKernel.cuh:225-517):
- ExpertsPerThread = 1 for both tiers at BlockDim 512 and 1024 (:234). The per-expert phases (zero-fill, the 8-rank DSMEM gather,
  the numCta writes) run on 256 vs 320 active threads out of 512 or 1024 threads. That is 8 vs 10 warps in parallel, so the
  latency is the same.
- The BlockScan runs over NumThreads, not experts (:247, :421).
- The expanded-id loop and its atomics depend only on tokens and local hits (:318-354). Both arms have 8 real experts per rank.
  The 2 ghost slots get no rows.
- The only tier-dependent items are 2 KB of extra static smem and the unused score-path register arrays (VecSize 8 -> 16,
  :1563). The register arrays matter only under the 1024-thread launch bound (64 regs/thread), i.e. at 2048 and 8192 rows. That
  fits the ledger's measured +0.68..0.85 us/layer at 32/GPU (§2 row A) as a codegen effect, not extra work.

**Estimated FT-minus-off routing delta:**

| tok/rank | per layer | per step |
|---|---|---|
| 16 | ~0..+0.2 us | ~0..+12 us |
| 64 | +0.7..0.85 us (ledger) | **+40..50 us** |
| 256 | same variant as 64, so same | **+40..50 us** |
| 1024 | ~0 to -0.5 us (FT's 384-thread coop needs 5 trips instead of 8) | **~0 to -30 us** |

Per step means 58 MoE layers plus the MTP layer.

The MTP layer is bf16 CUTLASS. Its fused prologue only runs at <=256 rows (moe_kernels.cu:486), so it never runs here. The
three-step sort's `computeNumTokensPerBlock` (:581-592) picks 64/128/256/512 threads for 8 experts and 128/256/512/1024 for 10
experts at the four loads. That is a uniform 2x block shift, not a threshold, and costs <2 us/step.

**FT's own path has no switch near 64 tokens:**
- The only token threshold is CFT counted writes at <=128 tokens/rank (nvlink_one_sided.py:64-66, 172-184). It is **inert**:
  rank-0.log of both STD-8110144 arms says "CFT counted writes disabled: NVIDIA driver 580.126.20 is below required 615.00".
  FT's fixed slots, abort record, source_valid_counts and expert_to_slot also require the fence path (moeAlltoAllKernels.cu:1505-1516).
- Fence dispatch launches local_tokens + ep_size receive CTAs of 256 threads (:1533-1542). That is 48 / 96 / 288 / 1056 CTAs,
  all co-resident at every load.
- Fence combine is one block per token (:2459-2465).
- The rank mask is fixed width. The compact copy is source_tokens x 8 ints per source (:671-681). The expert_to_slot lookup is
  one load per lane.
- All of these scale linearly or stay fixed. None changes variant.

**Conclusion: no.** The routing variant switch (BlockDim512 at <=1024 rows, BlockDim1024 above) falls between 16 and 64
tokens/rank. So it can explain a tier penalty that is ~0 at 16/rank and present at 64/rank, but only about +40..50 us of the
+315..353. The same BlockDim1024 variant also runs at 256/rank, so routing predicts the same +40..50 there, not ~0. The ~0 at
128/GPU is still best explained by the CuteDSL 8192-bucket tactic confound (§3). At 1024/rank routing is ~0 or slightly
favourable to FT. That leaves +250..300 us at 32/GPU (and the +0.5..0.7 ms at 512/GPU) unexplained by any token-count kernel
variant in the routing, sort, dispatch or combine code. §2's remainder candidates stand: F (static-LB/FT plumbing), the
cross-rank combine wait, and MTP-layer FT dispatch/expand.

**Prediction for std-noeplb-lb at 32/GPU** (static LB with 256 slots, 8 per rank, FT off):
- Routing tier is 256, local slots are 8, and the CuteDSL 256-slot and 8-expert MTP keys are the same as off. So rows A, B, D and E
  of §2 are 0 versus off.
- Its only device-side additions are:
  - one `moeComputeRouteNoRedundant` kernel per MoE layer, 1 CTA x 256 threads for 512 ids with PDL (moe_scheduler.py:567-572 ->
    moeLoadBalanceKernels.cu:626-636);
  - LB stats calls, which are no-ops for static routing because they are gated on dynamic LB (interface.py:649-676).
- FT does not pay that kernel: it fuses expert_to_slot into dispatch.
- Expect **lb ≈ off + 0.06..0.18 ms**: device step ~14.56..14.68 ms against G0 off 14.50. Expect **FT − lb ≈ +0.15..0.27 ms**.

How to read the result:
- lb within ~40 us of off: the cost is FT-specific beyond routing (tier <= 50 us, slots ~15 us), so the next step is a 32/GPU
  nsys pair of lb vs FT.
- lb ≈ FT (~+0.3 ms): main's static-LB mode costs more than its one route kernel explains, likely graph or stream structure
  rather than kernels, so the next step is nsys of off vs lb.
- At 128/GPU, lb shares off's 256-slot tile '128' keys. So lb − off ≈ +0.06..0.18 ms there, and FT − lb should show FT's true
  cost net of the tactic confound.

## seed2 tactic diff, 256- vs 320-slot keys (dev agent, 04:26 PT)
Source: launch/std/std-seed2.json, rank_0 and rank_17 (identical choices on both ranks).
Bucket = tokens = rows of compact routing [ep * source_tokens]: the dispatch output, the same with FT on or off.
At 32/GPU with MTP-1 that is 32 ranks * 64 = 2048, so the bucket is n=2048.
- n=2048, both 256 and 320 slots: run_moe tile 128; act (128,128)(1,1); finalize (128,128)(1,1). IDENTICAL.
- Tactics differ only at other buckets:
  - n=512 (8/GPU): finalize (1,1) off vs (1,2) on.
  - n=4096: run_moe tile 256 off vs 128 on.
  - n=8192 (128/GPU): tile 128 off vs 256 on.
  - Measured cost at 8 and 128/GPU is small, so a tactic flip doesn't track the cost.
=> The seed2 tactic/tile hypothesis is REFUTED for 32/GPU.
Remaining FT-specific device candidates at 32/GPU, for the nsys pair to diff:
(a) FC1 gather reads NVLinkOneSided receive slots in place (source_tokens/slot_tokens stride) instead of compact rows.
(b) 10 local slots, 2 of them empty ghost slots: more padded permuted rows (17536 vs 17280) and empty-expert tiles.
(c) the larger fixed-slot dispatch/combine buffers.
Caveat: the bucket is last_pow2(32*source_tokens), so any source_tokens in 64..127 maps to n=2048; only steps with fewer than 64 tokens per rank (n=1024: same tactics) or 128+ (n=4096: tile differs) map elsewhere.

## lb ~ off confirmed in one group (STD-G1-8118134, finals_readout section 7; host loop-step median; 04:47 PT)
- 32/GPU: noeplb 13.796 ms; lb 13.824 ms (+0.20% vs noeplb); ft 14.344 ms (+3.76% vs lb, +3.97% vs noeplb).
- Other loads, lb vs noeplb: +0.89 / +0.56 / +0.86% at 8 / 128 / 512.
- So the static 256-slot LB path costs <=~0.9%, and at 32/GPU it costs ~0.2%: the 32/GPU cost is FT-specific.
- n=1 group. This group's FT-off-vs-FT-on cost at 32/GPU (+3.97%) is larger than F1's +2.14%.
- Same group, stall share on vs off: 8 / 32 / 128 = +1.72 / +1.49 / +0.59 pt. This fails the stall rule (HL <= +0.5 pt) at n=1.

## nsys attribution (NSYS-BOTH 8119878, STD-G1-8119878, ranks 0+5, 48 steps; review/finals/nsys32_*.md; 06:00 PT)
- lb vs off: GPU step +45 us (+0.31%). lb ~ off holds on the device too.
- ft vs lb: GPU step +318 us (+2.14%), +5.49 us/layer.
  - combine +3.38 us/layer: same 64-CTA grid; the cross-rank floor is +2.99, so it is intrinsic kernel time, not waiting.
  - dispatch +2.78 us/layer: a 96-CTA grid instead of 64. Bound (FT) dispatch copies compact routing (expert ids + weights) per source and skips the sanitize kernel (nvlink_one_sided.py:1177-1235).
  - expert section +0.90 us/layer: same tactics; FC1/FC2 +18-21 us/step each.
  - rest -1.58 us/layer.
- Next (after 07:00): a dispatch/combine microbench, FT vs off, sweeping tokens/rank around 64, to explain why only 32/GPU is hit.

## a2a microbench, EP4 on one node (A2A-BENCH 8121588; build-v2-cef4670789/a2a-microbench/*.json; 07:10 PT)
- Max over ranks of per-rank medians, 200 iterations, eager with launches queued behind a GPU sleep, fence path, low-precision combine, payload in workspace.
- FT on vs off differs by <= ~1 us (the event-timer step) at every tokens/rank 16-256, at both 32 experts (8 vs 10 slots per rank) and 256.
- Example at 64 tokens/rank, 32 experts: dispatch 19.1 -> 20.1 us, combine 17.4 -> 17.4 us.
- So the DEP32 cost (+2.8 us/layer dispatch, +3.4 us/layer combine) does not reproduce at EP4 within one NVLink domain node.
- Reading: the FT a2a cost scales with the EP size or the multi-node fabric. The bound dispatch adds per-source work (32 sources at DEP32, 4 here). The combine is floor-bound at DEP32.
- Next: the same bench at EP32 (8 nodes), timed by CUDA-graph replay of 20 round trips for sub-us resolution.

## PDL A/B (PDL-AB 8121996, STD-8121996, w/o EPLB 32/GPU, one group, order off, ft, ft-nopdl, off-nopdl; 08:22 PT)
Rank-0 decode-only median, num_active >= 0.9 x 32, healthy-32 window; n=1 group:
- host step: off 13.875 / ft 14.150 / ft-nopdl 14.399 / off-nopdl 14.212 ms
- device step: off 14.512 / ft 14.870 / ft-nopdl 15.178 / off-nopdl 14.970 ms
- FT cost with PDL on: host +1.98%, device +2.47%. With PDL off: host +1.31%, device +1.39%.
- Turning PDL off slows both arms (FT off +2.43%, FT on +1.76%).
- Reading: PDL accounts for roughly a third of the gap at most (n=1, within group noise). Most of the FT cost stays with PDL off, so PDL co-residency is not the main mechanism. The FT dispatch/combine slowdown in serving remains open.

## Other-stream overlap (nsys32 8119878 rank 0, 2655 calls; 12:12 PT)
Per call, the time other-stream kernels overlap:
- dispatch: lb 11.8 / FT 15.3 / off 14.0 us. The same set in every arm: shared-expert nvjet GEMM (56 CTAs), quantize (128), silu_and_mul (128), device_kernel (32).
- combine: 0 us in every arm.
So the combine slowdown (30.9 -> 33.3 us) is intrinsic to the FT combine path, not contention. Dispatch contention is similar across arms (off has no slowdown at 14.0 us overlap), so it explains little.

## Capacity sweep (A2A-CAP 8128811; EP32, 8 nodes, 256 experts, PDL on, CUDA-graph round trip FT minus off, us; n=1)
| tokens/rank | capacity 128 | capacity 8192 (serving) |
|---|---|---|
| 32 | +1.05 | +2.58 |
| 64 | -1.08 | +1.95 |
| 128 | +0.87 | +1.46 |
At serving's capacity, FT costs +1.5-2.6 us per round trip in every cell. At 128 the delta stays within ~+-1 us (noise).
- This partly reproduces serving's +6.2 us/layer: about a third of it at 64 tokens/rank.
- Supports the fixed-slot stride (bound receive slots span the full capacity per source) as one FT-specific contributor. The rest is not reproduced; candidates are skewed routing and the serving neighbours.
- Eager per-call numbers are noisy again (dispatch 82-125 us) and not used.
- Next: repeat (n>=3) to firm up the delta, and add skewed routing to the bench.

## Capacity n=3 and served skew (A2A-CAP2, job 8129075): stride lead retracted
- EP32, 8 nodes, PDL on, graph round trip of NVLinkOneSided dispatch+combine, FT minus off, n=3 runs
  (build-v2-cef4670789/a2a-microbench-cap2). Served routing replays layer 45's per-expert load from STD-8109649
  (max/mean 20.9; tools/a2a_expert_load.json).
- Mean FT-minus-off, cap 8192 vs cap 128: uniform t32 +0.16 vs +0.57 µs, t64 +1.50 vs −0.49; served t32 +1.17 vs +1.96,
  t64 +0.07 vs +0.82. The run-to-run spread is ±2 µs, and capacity has no consistent effect, so the n=1 capacity
  result (+1.5-2.6 µs at 8192) was noise. Skew adds no FT cost either.
- Status of the w/o EPLB 32/GPU FT cost (+2.68%, ~+6.2 µs/layer):
  - Ruled out: bench config, receive-slot capacity/stride, routing skew, PDL (accounts for at most a part:
    +1.31% off vs +1.98% on), and other-stream contention.
  - About +1 µs per round trip reproduces in isolation; the rest arises only in full serving.
  - Open item; not chased further unless a cheap idea appears.

# Healthy path ~0: gap map and fix plan (monitor, 2026-10-22)

User direction (2026-10-22 ~09:10 PT): FT costs ~0 on the healthy path; late-write containment is paid during recovery.
Expert-to-slot indirection = main's mechanism; a small explained no-EPLB residual is acceptable. CFT out of scope (fence only).
"Solve all the gaps" with focused agents. Basis: code reads of wt-land 9f2f1892f0 vs main 0829689c52 (monitor-verified spot
checks) + CLAIMS 151/159/164/167. Numbers: R1 NVFP4 DEP32 GB200, no EPLB, 32 req/GPU, decode trim mean.

| Gap | Should be | Today | Code cause (9f2f1892f0) | Fix (agent) |
|---|---|---|---|---|
| G1 late-write containment on the data path | 0 healthy, paid in recovery | bound path +0.04/+0.10 same-run, ~0.16-0.22 inferred (CLAIMS 167) | fixed t*ep+s rows, +ep receive CTAs with fence.acquire.sys, compact copy, released_round, eplb_stats_out snapshot, per-signature regions (a2a K:627-741, 816-820, 912-918; nvl:385-422) | G1a: recovery-time same-address remap of every peer-writable allocation (dead rank keeps only orphaned pages). G1b: healthy dispatch/combine = main's fence path + rank mask; delete [L] machinery; hoist mask loads before griddepsync |
| G2 ghost slots | 0 | +0.06 (CLAIMS 164, n=1-2) | 320 slots -> moe_sort tier (512,8) vs (256,8); 10 local experts grow permute capacity; bucket inference 10 vs tactic key 8 (UNVERIFIED) | G2: tier/capacity by routable experts, post-restore correct on same graphs |
| G3 autotuner | identical tactics, clean | sound but narrow (weight slicing; CuteDSL fused only) | cute_dsl_custom_ops.py:211-243 | G3: routable-experts-per-rank + ConstraintSpec; extend or reject other backends |
| G4 host per-step | 0 | ~0 decode (CLAIMS 159/164) | commit_step full-sync fallback (plr:703-706), watch MPI_Test 5 ms, rank-0 response bookkeeping | G4: trims + hazard fix |
| G5 context steps / transport (C21/C22) | ~0 | v2: online ctx +15-25%, whole-window +1.4-2.4%; v5 unmeasured | FT ob1/TCP+ULFM vs off UCX on ~6-8 pre-launch host collectives; payloads scale with prompts | G5: payload/transport design + isolating experiment; VAL-H whole-window on v5 |
| indirection | as main | dispatch lookup ~0.3-0.5 us/layer (est.) | K:354-358 | none beyond G1b's choice (user: as good as main is fine) |
| CFT | out of scope | both arms fence on driver 580 | - | Limitations line only |

Workflow: monitor agents draft design + patch.diff in /tmp/claude-161238/g*.XXXX (no builds); monitor verifies; dev integrates on a
branch off v5-land, builds once, gates, A/B vs off (+ SPLIT2's same-run decomposition as the before picture). v5-land stays the
fallback freeze candidate.

## Divergence audit (FT-on vs FT-off on v5-land + G1a/G1b; user principle: FT on runs FT off's path)
Order of work (monitor-reviewed; code-read only, UNVERIFIED items flagged in the agent report):
1. Transport (largest, KEEP in cheapest form): FT forces ob1 btl self,tcp (even intra-node) vs FT-off UCX on ~8 host
   collectives/step. Same-node A/B: FT-off on UCX | FT-off on ob1/tcp | ob1/self,sm,tcp (| RDMA BTL under ob1 if ULFM allows).
2. Pin CFT off in both arms (bound refuses CFT; on driver 580 both use fence anyway, keep the pin for parity).
3. Routing tier for 320 slots (no-EPLB only; online 288 and 320 both use Tier<512>): G2's 384 tier; sized by SPLIT2 noghost.
4. Extend loaded_slot_layout to Cutlass (MTP layer key still uses 10 local slots; fused_moe_cutlass.py:432).
5. Host trims: split_unservable_requests only after a loss; derive finished/streaming on rank 0; inputs.py tp_allgather
   branch -> tp_cp path with departed; cache GhostSlotMemory::contains at registration.
6. FT-off vs main (C30): ADP balance change (py_executor ~7107) is the gate fix, by design; confirm whether mpi4py 4.1.2 is
   installed in both arms (requirements.txt unchanged, so harness-installed) -> must be identical in off and ft arms.
Autotuner verdict: tuning_layout is a workaround for the ghost-slot shape divergence (tactics match; shapes don't), correct
and cheap; the root (ghost slots visible) stays because removing them needs per-loss weight realloc + recapture of ~25 graph
sizes (est. 2-10 s, UNVERIFIED) — over the ~1-2 s budget and v7's no-recapture invariant. Ghost slots vanish entirely with
failure_domain=rank on the 288 recipe (31x9 >= 256).
Must remain: abort poll + released-round load; ob1/ULFM; fixed extra slots (zero memory, FT-off tuning key, matching tier);
_interruptible/watch checks; commit/replay bookkeeping.

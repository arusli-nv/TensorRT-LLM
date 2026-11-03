# Healthy-zero integration plan (dev, 2026-10-22 09:15 PT)

Inputs: the monitor-verified patches G1a/G1b/G2/G3/G4/G5 (review/HEALTHY_ZERO.md), each a patch.diff on 9f2f1892f0.
Branch: v5-zero off v5-land 9f2f1892f0 (worktree wt-zero). User order (09:20 via monitor): finish the fixes and finalize
the code first; then rebase once onto current main, before the freeze build, followed by one rebuild, the gate and a short
confirm A/B; the freeze hash is the rebased build. C30 baseline until then: main 0829689c52 (BUILD-MAIN). After freeze:
one branch wide-ep-ft-mvp on the fork and one deletion list for the user. v5-land stays the fallback freeze candidate.

Order (one commit per gap, each builds and passes its own tests; conflicts resolved toward the patch's design):
1. G1b healthy dispatch/combine = main's fence path + rank mask (deletes fixed rows, receive CTAs, compact copy,
   released_round, per-signature regions; reverts token-major). Then G1a recovery-time same-address remap, which is
   what keeps late writes off live data once G1b removes the data-path confinement. G1a+G1b land together or not at
   all (G1b alone loses containment).
2. G2 routing tier / capacity bucket. 3. G3 autotuner. 4. G4 host trims. 5. G5 context transport (may be its own A/B).

Gate list (build job, before any GPU A/B):
- gtests: MoeLoadBalancer* + GhostSlotMemory* + routing (RoutingCustomKernelTest for the G2 tier).
- groups gpu_moe, cpu, executor; extras: the build list in launch/build_land_cmd.txt, plus whatever G1a/G3 add.
- Late-write containment tests must change meaning, not be deleted: test_moe_a2a_late_writes (a dead source writing
  after recovery must hit orphaned pages, never a live buffer), test_moe_a2a_abort, test_nvlink_one_sided bind/ep4
  dead-middle-rank, test_ring_restore_staging, test_moe_execution_state. Every patch states which tests change.
- Known failures only via tools/pytest_plugins/known_failures.txt; integration tests on.
GPU, in this order (each gated on the previous):
- KILL-SMOKE (8 nodes): rank5 + node1 + rank0 + stop (SIGSTOP with hang detection on: fenced while alive, the case
  the remap exists for) + the C32 spray / late-write trigger row (TLLM_TEST_LATE_WRITE_IN_DECODE) on the new build; pass = resume, no client failure, outputs after recovery match the C17 perturbed-control rule.
- ZERO-AB (16 nodes, same nodes, mirrored): off | lb | ft-zero | ft-v5land at 32 and 128; readout tools/trim_readout.py;
  pre-registered (monitor-amended 09:20):
  - headline ft-zero - off with its decomposition (lb - off = indirection, rest = FT); ft-zero may beat lb if G1b keeps
    the dispatch-side lookup instead of route();
  - trim: ft-zero - lb <= 0.05 ms at 32, and <= 0.4 % of off's trim at 128, same sign in both groups; one group pass +
    one fail -> a second round, never a call from n=1;
  - stall ms/step: ft-zero within lb + the off-vs-off spread (FT's extra stalls, CLAIMS 159/163, do not show in trim);
  - client and whole-window within 1 %.
- Then VAL-H/K/P re-pointed at the v5-zero wheel.

## G3 scope (monitor, 09:35; after G1/G4)
- Ghost count from ghost_slot_count, not placement.
- Extend loaded_slot_layout to Cutlass (the standard recipe's MTP layer runs Cutlass MoE with the FT extra slot; it tunes on
  10 local slots today, a key that differs from FT-off's). No guard or rejection of Cutlass (it would break the finals config
  at startup); DeepGemm checked before any guard (GLM-5.3 FT configs use it). No renames.
- Online EPLB 288 + FT: 1 ghost slot per rank, ghost (GDRCopy host mappings), not empty; the readout keeps the "appends 1
  ghost slots" / GDRCopy line check. A/B warms both arms past the first EPLB update cycle.

## Design principle (user via monitor, 2026-10-22 ~10:05)
While healthy, FT on runs exactly what FT off runs. Every FT change should mostly be "stay on the FT-off path", with the
fewest changes; failure-specific work is paid at recovery. When integrating, prefer main's code over FT variants wherever
the recovery need allows. The monitor's audit of every remaining FT-on vs FT-off divergence (remove/defer/keep) drives the
next integration commits.

## Stated gaps (monitor-accepted, 10:10)
- A double loss inside the remap window (a survivor dying after posting its remap outcome) fails recovery on every rank
  and ends the job (fail-closed).
- Peer-failure detection lag 5 -> 20 ms (G4); the monitor tells the user.
GPU remap test (monitor draft gt.Uvuf, test_moe_a2a_recovery.py, 4 cases, 1 node x 4 GPUs) is re-fit to v5-zero and runs
in KILL-SMOKE's in-job gate, ahead of the kill rows.

## Divergence audit work order (monitor, 10:15; HEALTHY_ZERO.md "Divergence audit")
1. Transport A/B (FT off: UCX | ob1 self,tcp | ob1 self,sm,tcp), same nodes; packed into ZERO-AB/KILL-SMOKE or its own job.
2. CFT pinned off in both arms. 4. Cutlass loaded_slot_layout (MTP key). 5. Host trims (split_unservable after loss only,
   finished/streaming on rank 0, inputs.py tp_cp path with departed, GhostSlotMemory::contains cached) -> Python follow-up
   commits on v5-zero, shipped as an overlay on BUILD-ZERO's wheel.
3. G2 384 routing tier: C++, gated on SPLIT2's noghost number.
6. mpi4py: verified identical in every arm: install_wheel.sh:48-62 installs the same cached mpi4py 4.x (ULFM check) for
   every case regardless of mode.
Must stay: abort poll + released_round load, ob1/ULFM, fixed extra slots (zero memory, FT-off key, matching tier),
_interruptible/watch, commit/replay bookkeeping.

## Next-rebuild C++ batch (FT-off byte identity with main; monitor 10:25)
1. `__shared__ routed_ids[TOP_K]` (.cu:575) under `if constexpr (USE_EXPERT_TO_SLOT)`.
2. Revert the `::: "memory"` clobbers on main's fences (.cu:689-691, :1828-1830) on our branch; record them in PR_PLAN as a
   separate upstream fix.
3. Revert the unrelated TLLM_CUDA_CHECK wraps in moeA2AInitializeOp.
Keep the FLAG_VAL 2-word reservation. Batched with G2's routing tier (if SPLIT-equivalent noghost data says so) into the
next rebuild; FT-off main parity is proven by VAL-P on the final build.
Design dependencies noted: a survivor that consumed a released rank's dispatch waits in combine until its own host aborts
(every survivor's watchdog must fire; KILL-SMOKE covers it); resolveAbortState caches by host address (Op.cpp:186-192), MINOR.
- Red team on dbeab13f9b/6fd091c588/82c6711fbe (monitor 10:30): no blockers, no FT-off change; fixes queued (commit_step
  None-path sync, Cutlass AutoTuner shape test, executor routing-guard test). No manifest/schema regeneration needed
  (FaultToleranceConfig is env-YAML, not llm_args). Doc: on CFT-capable drivers, perf claims state FT-off pinned to fence.
- 10:35 monitor decision: split_unservable stays (strict-pin error semantics) and 31251e0ca6 is not reversed; both dropped
  from the work order. W4AFP8 quint4x2 slicing: one CPU shape test if cheap, else a doc Limitations line.
- C++ batch committed on v5-zero (95d6a64a73 -> 7a03252e92, unbuilt): routed_ids helper (FT-off no longer instantiates it),
  main's fence clobbers reverted (upstream fix in PR_PLAN), moeA2AInitializeOp back to main, GhostSlotMemory device-addressed
  flag set at registration (copyWeightsCpu = main). Remaining FT-off vs main kernel divergence: kernel-arg structs (mask by
  pointer + new fields), FLAG_VAL 8 bytes (kept), constant-folded FT branches, USE_EXPERT_TO_SLOT instantiations, op schema
  optional args. Measure off (ZERO-AB) vs main (VAL-P) before any refactor.
- Post-failure perf (monitor code read, 12:10): one fix queued (online EPLB: set_warm_up_iter_count(~20) right after
  reset_update_schedule() at recovery, if honoured mid-run). Rest near ideal: no-EPLB restore leaves the busiest rank ~+9 %
  MoE work (10 vs 9.14 experts), reruns fair-share spread, ghost KV cut ~1.5 GB/rank after a node loss. KILL-SMOKE's per-rank
  load readout checks the ~5-8 % of ideal estimate. Also check in kill logs: host allgather time after Shrink() vs before.
- Red team on b4672a55d0/1d675a2281/f9e6bc0f8b (12:35): no blocker/major; every staging target is remapped first on every
  path (first recovery, re-plan, Install wait; online EPLB never stages; sleep/checkpoint refused); targets install only
  complete chunks. MINORs: PROTOCOL_VERSION bumped to 2 (1bdb4baa6c, Python, rides the next overlay/build); TODO a two-round
  late-writes GPU test (second victim that imported the first remap's pages keeps writing through a second remap) before the
  re-plan claim goes into finals; drop or assert the now practically dead staging-room check; hang evidence may read staged
  bytes during staging (pre-existing, evidence only).

## After code final (monitor 15:44)
1. Option-A doc line on v5-zero (after the v5-split squash agent).
2. Rebase v5-zero onto upstream main 9cc37929e8 (rerere has the 3 conflicts; notes /tmp/claude-161238/rebase-notes.UowY).
3. Freeze build (verify_wheel) + BUILD-MAIN at the rebase target (C30 baseline; 8168840 at 0829 becomes obsolete).
4. Push (user, via monitor; confirm with the monitor right before running): from the main checkout's `fork` remote
   (git@github.com:arusli-nv/TensorRT-LLM.git; never origin = NVIDIA): tag cef4670789 as wide-ep-ft-mvp-v2-backup and push the
   tag to fork, then `git push fork <rebased branch>:wide-ep-ft-mvp --force-with-lease=wide-ep-ft-mvp:cef4670789`.
   First fetch the rebased branch from the work clone into the Lustre checkout (`git -C <lustre-checkout> fetch
   /tmp/claude-161238/work/repo <rebased-branch>:wide-ep-ft-mvp-rebased`) and check `git rev-parse` equals the freeze hash.
5. VAL-H/K/P on the freeze hash.

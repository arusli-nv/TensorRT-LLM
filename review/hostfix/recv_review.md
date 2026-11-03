# Review: b72dee1a7b (receive teams + pre-wait staging), parent f96a8c1a53

Read-only correctness red team of /tmp/claude-161238/wt-recvfix. `a2a.cu` = cpp/tensorrt_llm/kernels/moe/communication/moeAlltoAllKernels.cu
(line numbers at b72dee1a7b). Nothing was built or run on a GPU. ruff 0.9.4 and clang-format 16 (the pre-commit versions) were run
read-only from the pre-commit cache.

**Verdict: build as is for the GPU A/B.** No correctness bug found. Before the commit is pushed, fix the formatting
(F1; pre-commit fails today). F2 is a performance risk the A/B has to measure. T1 and T2 are test gaps worth closing.

## 1. Memory ordering (a2a.cu:636-672): sufficient
- Chain: remote `fence.release.sys` + flag store (a2a.cu:927-944) -> lane 0 `ld.relaxed.sys` observes the flag (a2a.cu:299)
  -> lane 0 `fence.acquire.sys` (a2a.cu:653) -> program order -> `bar.warp.sync` (a2a.cu:668) -> lanes 1-31 load the count
  and rows (a2a.cu:673, 690, 708).
- The PTX ISA (bar.warp.sync) and the CUDA guide (__syncwarp) both say the barrier guarantees memory ordering among its
  participants. In the PTX model, causality order is transitive across scopes: the sys-scope fence-fence
  synchronizes-with plus the CTA-level barrier ordering puts the remote writes in causality order before every lane's
  reads. This is the same chain the S=1 path (and the parent's code) builds with `__syncthreads`, at warp scope.
- Compiler: `fence.acquire.sys` is `asm volatile` with no "memory" clobber (pre-existing). Every row load comes after
  `__syncwarp`/`__syncthreads`, and those are compiler memory barriers, so the loads cannot be hoisted. No change needed.
- L1: rows of neighbouring sources can share a 128 B line (slot_tokens*top_k*4 is not 128 B aligned), and warps of one
  CTA share L1. Each team's own acquire fence orders its own reads, which covers that. Same as the parent, where several
  receive CTAs could share an SM.
- Optional, not required: if you want an argument that does not rely on barrier ordering, every lane can re-load the
  flag with `ld.acquire.sys` after the `__syncwarp` (one local load). A per-lane `fence.acquire.sys` alone would not help,
  because those lanes never observed the flag. I do not recommend changing anything.

## 2. Barriers and deadlock (a2a.cu:619-672): OK
- `team_size = blockDim.x / S`. S is 8 or 1, and `kBlockSize == 8*32` is static_asserted (a2a.cu:1589). With S=8 the
  only barrier is `__syncwarp`, inside one team. With S=1 there are no spare threads (source = receive_cta < ep_size),
  so `__syncthreads` is CTA-uniform. `ptrs.receive_sources_per_cta` is a kernel-uniform parameter.
- Spare warps return after the PDL calls (a2a.cu:631-634). No later barrier counts them. `copy_compact_routing` and the
  stats loops contain no barrier.
- A silent source: its lane 0 spins and lanes 1-31 park at `bar.warp.sync`, which is fine with Volta+ independent
  scheduling. Other warps run on. `wait_round_flag` polls the abort per lane 0 (a2a.cu:308-315), so the abort releases
  every stuck team.

## 3. Abort accounting (a2a.cu:733-743, 584-603): no loss and no double count
- Every team writes only its own `source_valid_counts[source]`. The `flag_val[kReleasedRoundIndex]=1` store and
  `record_cancelled` (a status store, a2a.cu:265-268) are idempotent. Per-source multiplicity is the same as one CTA per
  source in the parent.
- `checked_source_count` runs, and can trap, on lane 0 only. The other lanes' copy uses `count_valid ? raw : 0`, so a
  corrupt count copies no rows before the trap. As in the parent.

## 4. Staging before griddepcontrol.wait (a2a.cu:771-788, 626): legal
- The direct predecessor is always `moeA2APrepareDispatchKernel`, launched just before on the same stream
  (moeAlltoAllOp.cpp:889-892). It runs `cudaGridDependencySynchronize()` before its trigger (a2a.cu:496-497), so when
  dispatch starts, everything stream-ordered before PrepareDispatch is complete and flushed: the previous combine, the
  router, and the EPLB statistic kernels. PrepareDispatch writes only flag_val, send/recv counters and the token
  counter. Those are still read after the wait (a2a.cu:641, 875).
- Mask: `install_mask` (execution_state.py:188-203) brackets its `copy_` with `torch.cuda.synchronize`, and the model
  is idle. That is safe against any graph replay.
- expert_to_slot: `one_slot_table` exists only with `updates_enabled=False` (moe_load_balancer.py:965). It aliases
  `globalSlotIds`, which `copyPlacementInfoToGpu` writes on the LB stream or `copyPlacementInfoToGpuByCpu` writes from
  the host (moeLoadBalancer.cpp:768-805). Neither is stream-ordered with dispatch, so the pre-wait read has exactly the
  ordering the post-wait read had: both come after PrepareDispatch's wait. No new window.
- This rests on one invariant that is not stated where it lives: PrepareDispatch must wait before it triggers. Put a
  one-line note on moeA2APrepareDispatchKernel so nobody moves its trigger above the wait.

## 5. Shared memory (a2a.cu:771-777, 1597-1603): matches
- Device: mask at byte 8*TOP_K (8-byte aligned for any TOP_K), table at 8*TOP_K + 32. Host: 8*top_k + 32 + 4*E, only when
  `fence_rank_mask`, which on the non-CFT path equals ENABLE_RANK_MASK. `SWITCH_TOP_K` maps values exactly (a2a.cu:87+),
  so TOP_K == top_k. E is `expertToSlot.size(0)` == `ptrs.num_routed_experts`, the bound the routing lookup uses
  (a2a.cu:358).
- The fence launch never opts in above 48 KB. E would have to exceed ~12K for the launch to fail. Nit: add a TORCH_CHECK
  on expert_to_slot length, or check `shared_bytes <= kDefaultDynamicSmemBytes`.
- Static smem: +72 B (`flag_arrived[8]`, `count_slot[8]`). Receive CTAs get the dynamic allocation but do not touch it.

## 6. Grid and S (a2a.cu:1585-1594): consistent
- S comes from `max_tokens_per_rank*top_k <= 1024` on the host and reaches the device only through
  `kernel_ptrs.receive_sources_per_cta`, so the two sides cannot disagree. The grid is `local_num_tokens (min 1) +
  ceil(ep/S)`, a function of the capture arguments only.

## 7. FT-off and CFT: unchanged
- ENABLE_RANK_MASK=false: staging and the staged-mask reads are `constexpr`-dead, `STAGED=false`, no receive CTAs, and
  `shared_bytes` is unchanged (fence_rank_mask is false). Moving the `extern __shared__` declaration does not change
  codegen.
- CFT: separate kernel. It calls `route_dispatch_token<...,false>` with the defaults (a2a.cu:1219). `shared_bytes` adds
  nothing because fence_rank_mask is false under CFT. Rank mask on without fixed slots: S stays at its default of 1, no
  receive CTAs, staging only.

## 8. Tests (tests/unittest/_torch/moe/test_moe_a2a_abort.py:597-833)
- `test_receive_teams_cover_every_source`: covers a partial CTA (ep 5, 9), >32 sources (37), the S=8 / S=1 split
  ((128,8) = 1024 words, (129,8)), masked source 70 in mask word 1 at ep 72 on the receive side, and EPLB stats with 300
  experts (the stats remainder loop at S=8). Good.
- `test_a_dead_source_holds_only_its_own_warp`: source 12 and source 13 are both in receive CTA 1 (S=8, 8 words), and
  13's count lands while 12 spins. That would catch a barrier spanning teams. The abort, released-round and aborted-row
  checks are right. Needs concurrent streams (stated in the test).
- `test_receive_grid_is_fixed_per_shape`: the kineto chrome trace exports `args.grid` for kernel events, including
  graph-replayed kernels. The name filter `moeA2ADispatchKernel<` excludes `_Cft<` and `moeA2APrepareDispatchKernel`.
  Robust enough. Not covered: local_num_tokens=0 (grid 1 + receive CTAs).
- T1 (gap): no single-GPU test stages expert_to_slot with more than 32 entries. The only table test uses 2 entries
  (test file:463), so the staging loop's lane stride runs only one iteration. Add E >= 64 (e.g. 72) to
  `test_replay_reads_the_slot_table_and_mask_written_before_it`, or add a variant of it.
- T2 (gap): at ep 72, the token side's staged mask in word 1 (publish and counter loops, a2a.cu:899/944) is never
  asserted. Assert that workspace row 70's dispatch completion flag and recv counter for rank 0 stay untouched.
- Single-GPU tests run in one coherent L2, so they cannot test item 1. Only multi-GPU tests do:
  multi_gpu/test_nvlink_one_sided.py and test_moe_a2a_late_writes.py. They must run in the A/B job.

## F2. Performance risk: the staging loop (a2a.cu:783-786)
- Every token CTA's warp 0 copies all E table entries before the wait, one load and one smem store per iteration. The
  runtime bound and a generic source pointer probably keep the loads serialized, so E=256 costs about 8 and E=320 about
  10 dependent L2 round trips (~2-3 us). PrepareDispatch is one tiny CTA and probably hides less than that, so this can
  cost more than the 1-2 dependent loads it removes.
- Cheaper and equally legal: `token_selected_experts` is written by the router before PrepareDispatch, so lanes < TOP_K
  can load their own `token_selected_experts[k]`, then `expert_to_slot[e]` and the mask before the wait. That is
  2 round trips total. Or at least unroll the staging loop (load into registers, then store).
- Check it on the send-done proxy in the A/B (ft-new with and without B) before trusting change B.

## 9. Style
- F1, ruff-format (v0.9.4, file not in legacy list): 7 new hunks over 100 columns: the `_rank_mask_controls` return,
  `_Peers.__init__` words/gathered/counters, the `test_receive_teams_cover_every_source` def, and the `_receive_round`,
  `on_time` and `_setup` calls. `ruff check` passes.
- F1, clang-format 16: 2 hunks in a2a.cu: the stats remainder `for` (a2a.cu:725-726 fits on one line) and the
  `route_dispatch_token<...>` call (a2a.cu:817-821).
- The commit fails pre-commit as is. Run pre-commit, re-stage, and commit, then confirm the binary is unchanged.
- Copyright years are 2025-2026 / 2026: OK. Comments describe the code, not its history: OK. Slight wording nit at
  a2a.cu:531-532 ("with threads threads of which this is thread").

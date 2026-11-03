# PR stack prep (started 2026-10-30, autonomous; user away)
User decisions (2026-10-30): rebase onto current main (up/main 78145f88ea); stacked per-PR fork branches
(wide-ep-ft/m1..m5 to main, wide-ep-ft/01..NN to feat/wide-ep-ft-mvp), wide-ep-ft-mvp = tip, backup tag old tip.
NEVER touch NVIDIA/TensorRT-LLM, never open PRs. Push optional (user may push at end). 2 Slurm slots free to use.
Include: a2a FT cost cuts (S1 ballot stats loop, F8 flag load; CLAIMS 210), all main fixes (M1-M5).
Cleanup: FT-off red-team items (review/diag/redteam_pr.md), fold v5-pr-fixes, split executor-recovery commit.
Exit loop: blind vs-main quality judge + per-PR code-owner review agents: no unexplained significant flaw.
Cleanup of stale branches/worktrees/builds: TRASH or deletion list (no rm).

Work: clone /tmp/claude-161238/work/repo, worktree /tmp/claude-161238/wt-stack, branch ft-stack.
Source: v5-pr-fixes 59eee8140a (28 commits on 9cc37929e8).

## Steps / status
1. rebase onto up/main  [ ]
2. main PRs M1-M5 split out  [ ]
3. fold v5-pr-fixes  [ ]
4. FT-off red-team fixes  [ ]
5. split executor recovery  [ ]
6. a2a S1/F8  [ ]
7. tests: static per-commit, build, unit per commit, GPU smoke, healthy pair  [ ]
8. review loop  [ ]
9. branches + PR_STACK.md + descriptions + push script  [ ]
10. workspace cleanup list  [ ]

## Log
- 2026-10-30 step 1 done: ft-rebase (worktree /tmp/claude-161238/wt-stack) = v5-pr-fixes replayed on up/main 78145f88ea,
  28 commits; conflicts resolved by hand in 8 commits. Main changes absorbed: MpiSession one-task-per-rank +
  new RemoteMpiCommSessionServer.serve (FT degraded-world logic ported into it), CuteDSL/Cutlass MoE split into
  cutedsl/ and cutlass/ packages (tactic keying ported), attention-DP helix CP vote + router.pop_requests +
  KV-balanced admission (departed=, dict counts, placement by position -> logical rank), executor loop
  batch_outputs-None restructure (commit_step inside), broadcast prefer_cpu, Responses streams already end via
  guard_responses_stream (M2 narrowed to chat/completions).
  TODO from rebase: M2 must update main's test_chat_stream_logs_a_failure_and_raises_it (stream now ends with
  error event, not raise); verify _make_route_client disagg needs prometheus fixture; rerere is ON in the clone
  (old resolutions replay; always resolve from index stages).
- 10-30 10:30 PT: STACK-BUILD 8479899 submitted (ft-rebase ef99fed236, snapshot overlays/build-src-stack, launch/stack_build_cmd.txt, token stack-build-ef99fed236-441121866f). Partial copy overlays/build-src-rebase-f1d38eaa42 is garbage (cp stopped) -> queue for deletion.
- 10-30 11:18 PT: 8480026 built C++ OK but failed verify_wheel (6 stale .py in build-root/wheel-staging/build from an older main): moved that dir to build.stale-1030. Review round 1 fixes landed (REVIEW_R1_DECISIONS.md). STACK-BUILD 8480761 submitted at 966d3788b3.
- Blind quality round 1 (BLIND_R1.md, key agents/blind_key.json): judge picked the new file in 6/6 pairs, by style
  clustering, at 55-85%; overall scores new vs main: 8/5, 7/5.5, 7/8, 6.5/6, 8/8, 7/7 (new not worse in 5 of 6). Fixed:
  contradictory comment, docstring break, explicit Sent check comment, empty-workspace error, is_active reuse, typing.
  Kept (WHY): rollback paths ignore cleanup failures (TLLM_CU_CHECK_FREE_RESOURCE throws; deleters must not).
- 10-30 12:10 PT: STACK-INTEG 8481305 (966d3788b3, 1 node EP4 DeepSeek-V3-Lite NVFP4, ULFM): 14 passed; MPI4PY_INSTALL_OK.
  STACK-BUILD 8480791 (966d3788b3): build OK, gtests 42 + mpiUtils pass, gpu_moe 103 pass, eplb_gpu 15 pass. Failures: tests
  already fixed after 966d (recovery_authority ids, py_executor fakes, adp_absent_ranks, peer_loss_recovery fakes) plus 3 real:
  test_nvlink_one_sided slot-table [fence] (expert_to_slot without rank mask; test now binds an execution state),
  test_ring_host_copies (assert->ValueError not reflected in test), test_mpi_session degraded async rank-0 failure (serve raised
  before ending the degraded world; fixed in mpi_session.serve: a stop/drain after an async failure with failed ranks goes
  through _stop_degraded_world).
- 10-30 12:20 PT: STACK-A2A 8482068 PASS (CLAIMS 244): no stats +0.54 us, stats +1.39 us (pre-registered <=1.0/<=2.0), 6/6 checks.
- pre-commit (PRE_COMMIT_HOME=$TMPDIR/pchome copy, no network) at tip and per commit (pc_each.sh): formatting folded; communicator
  import isort/yapf fight fixed with main's `# yapf: disable` idiom. All 26 commits clean.
- Harness: testhooks.py engine patch resolves ModelEngine (new main renamed PyTorchModelEngine). STRESS needs a seed with full
  autotuner metadata (empty metadata KeyErrors at load) -> use the smoke's tuned cache as STRESS_SEED.
- 10-30 12:50 PT: STACK-BUILD 8482658 submitted at b66685fb5c (build-src-stack index rebuilt via read-tree after a killed checkout).

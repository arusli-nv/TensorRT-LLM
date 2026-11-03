# Red-team lane: tests (QA lead) -- v2 cef4670789
Read-only review. Sources: `git diff upstream/main v2 -- tests cpp/tests` (41 files, +12.6k), test-db yml,
jenkins/L0_Test.groovy (mark expr), build-final-8d0ec8cf1c (job 8113399), logs/v2-cpu-6a673be32f.log, mvp/CLAIMS.md, mvp/runs/INTEGRATION-*.

## 1. Bottom line
- No result exists for v2 head. Newest evidence: 8d0ec8cf1c (ft-final2, job 8113399) and 6a673be32f (v2 CPU subset). Neither is a
  full green run. The 12-test GB200 integration suite last passed 12/12 at 547a6aa7ed (INTEGRATION-...-8101557,
  10-20 16:06 PT, n=1). 547a6aa7ed is **not an ancestor of v2** (rebased). Since then FT product code changed by about
  +511/-401 lines (peer_loss_recovery, py_executor, moe_load_balancer, nvlink_one_sided, authority, communicator).
- In upstream CI the end-to-end tests skip: no ULFM in the image, and nothing in jenkins/docker sets TLLM_REQUIRE_ULFM.
  So the CI story is "CPU mocks + single-process kernel tests + 2 MNNVL tests". All real recovery evidence is manual GB200 runs.

## 2. Latest container results
| Run | Result |
|---|---|
| gtest moeLoadBalancerTest (8d0ec8cf1c) | 39/39 pass |
| gpu_moe group (a2a_abort, exec_state, slot_gather, late_writes, kv_revert, watchdog) | 127 passed |
| cpu group | 953 passed, **5 failed**, 5 skipped. Failures in test_peer_loss_recovery: stale fakes (`_end_ranks` called twice; `PyExecutor has no attribute event_loop`). Test-only fixes 6a673be32f/aae00e6c65 are after this tree |
| executor / eplb_gpu | 291 / 15 pass |
| extra | survivors, **survivors_ulfm (real ULFM, 3 params, passed: custom OMPI)**, absent_ranks, hang_evidence, kv_revert, late_writes (3) rc=0; peer_loss_recovery rc=1 (same 5) |
| extra: test_nvlink_one_sided | **first case F, then the job step was Killed** (srun abort, about 00:00 PT). test_ring_restore_staging and later extras **never ran** on this tree. Investigate before any claim |
| v2 CPU subset @6a673be32f | 376 passed, **1 failed**: test_adp_absent_ranks::test_without_peer_loss_recovery_a_negative_pin_behaves_as_on_main. cef4670789 changes only that test (to "check against last rank's capacity"), with **no rerun** |
Note: cef4670789 fixes a failing "behaves as on main" test by editing the test, not the product. That needs a reviewer's eye: confirm the
old expectation was wrong rather than relaxed to match a behaviour change in the FT-off path (healthy-path parity is a hard invariant).

## 3. CI story (which stage runs what)
- Marker routing: `CPU-*` stages run `-m cpu_only`; GPU stages run `not cpu_only` (L0_Test.groovy:1694). Directory entries filter by marker.
- **l0_cpu (pre-merge)**: test_peer_loss_recovery (110 tests), test_adp_absent_ranks, test_mpi_survivors (fake ULFM),
  test_ring_host_copies, test_resident_placement, test_alltoall_watchdog, recovery_{authority,hang_monitor,config}, health,
  stream errors, launch_exit, test_mpi_session (ISOLATION). **test_mpi_survivors_ulfm is collected here and silently SKIPS** (no ULFM).
- **l0_b200 (pre-merge, 1 GPU)**: test_moe_a2a_abort, test_moe_execution_state, test_moe_slot_gather (SM100/103 + cutlass-dsl, else skip),
  test_kv_cache_revert, test_moe_load_balancer.
- **l0_h100 / dgx_b300 / gb300_multi (dir unittest/_torch/executor, not cpu_only)**: test_hang_evidence (GPU case skips without CUDA).
- **l0_gb200_multi_gpus pre_merge (4 GPU)**: test_moe_a2a_late_writes, test_ring_restore_staging. Both **skip if MNNVL is unavailable**
  (`_mnnvl_skip_reason`). They are the only real multi-rank FT tests that CI could run.
- **l0_gb200_multi_gpus post_merge**: 12 wide_ep_ft integration tests, TIMEOUT 45 each (about 25 min). Every one **skips** on: no
  `mpirun --with-ft ulfm` / ompi_info ftmpi / mpi4py<4, a missing model, or non-GB200. Net: 0 executed in CI today.
- C++: moeLoadBalancerTest uses legacy `add_gtest` without OWNER/LABELS (AGENTS.md wants `add_tllm_gtest` + LABELS). It builds only
  under google-tests, not runtime-tests/ctest -L.

## 4. Coverage vs risk
| Risk path | Real coverage | Mock-only / gap |
|---|---|---|
| Recovery protocol races (authority <-> workers) | Integration (manual): one loss, two together, loss after Stopped, authority lost while serving. Authority over real ZMQ with fake workers | No test of a loss **during Install/restore copy** or during Commit. Authority dying **mid-recovery** is untested (only while serving). Sequential loss after a recovered serve (generation 2) runs only in manual DEP32 twokill |
| KV revert of discarded steps | test_kv_cache_revert (real KVCacheManagerV2, draft pool, overlap 2 steps); integration exact token count | **DSACacheManagerV2 is admitted by `create` (peer_loss_recovery.py:395) but its indexer-K revert has no test**. In test_peer_loss_recovery, kv_cache_manager is a Mock and `discard_interrupted_steps` is mocked |
| Restore copy (ring host copies -> slots) | test_ring_restore_staging bit-exact over MNNVL (lost_idle/lost_writing); placement/ring logic CPU | Integration checks only "expected word + coherent text". A wrong expert's weights in a restored slot would likely still pass. No end-to-end checksum of restored slots. `restore_copy_plan` returns [] in the default recovery fixture |
| a2a abort / late writes | late_writes: real SIGKILL at 3 delays (MNNVL); a2a_abort fence/timeout | a2a_abort multi-rank cases are single-process (layout/masking only, per its docstring). nvlink_one_sided currently fails/hangs (section 2). No abort during a **captured-graph combine** across real ranks except late_writes' single replay |
| Hang detect / fence | test_hung_rank_fenced (SIGSTOP, hang_confirm_s=10); hang_monitor unit tests | **No negative control in any suite.** CLAIMS #109: a GIL hog falsely fenced at T=10; the shipped default is 30, while the integration test runs 10. All-rank stall and 2-rank stall controls exist only as manual runs |
| Request accounting | Unit: rerun/quarantine/cancel/gather (heavy mocks); integration: rerun count == lost*per_rank, failed==0, under_load 8/rank | Streaming clients, cancel-during-recovery and the /health state machine are covered only by unit tests. No end-to-end check that every client stream gets exactly one terminal event |
| Rank 0 / authority-host loss (fail-closed) | none e2e | No test that killing rank 0 ends the job promptly with errors to clients (no hang). Fail-closed paths are the least exercised |
| Multi-node, failure_domain=node | manual DEP32 only (CLAIMS #119/#121) | Integration is 1 node with failure_domain=rank |

## 5. Vacuous or over-mocked tests (would pass with the feature broken)
- test_peer_loss_recovery.py builds `PyExecutor.__new__` with Mock dist (shrink/allgather/broadcast are lambdas), a Mock load balancer,
  Mock KV manager, mocked `_join_device_work`, `_watch` and `discard_interrupted_steps` (381 mock sites in 110 tests). Many tests assert
  an **event order** (`events` list) or call counts. They catch protocol-ordering regressions but not wrong data. The 5 failures above were
  fake drift, not product bugs: the fakes track implementation details and churn with every refactor.
- test_mpi_survivors.py models ULFM with fake communicators. The only real-ULFM unit test (survivors_ulfm) skips in CI.
- test_moe_a2a_abort multi-rank cases: one process owns every peer, so peer transport is not exercised.
- Integration restore tests: weight correctness is inferred from greedy text containing a keyword.

## 6. Flakiness (timing)
- test_peer_loss_recovery deadline tests: `_timeout_s=0.2` against `sleep(0.3)` / `_shrink_timeout_s=0.3`. The 100 ms margins are prone to
  flaking on a loaded CPU runner (l0_cpu runs in parallel).
- test_recovery_authority: probe "answers within its timeout" (0.4-0.5 s), sleep(0.01) polling loops, join(timeout=5).
- late_writes kill_delay 0.003 s and ring staging deadlines are inherently racy, but they assert invariants, not timing. Lower risk.
- Integration: HEALTH_SETTLE_S=30, fixed sleep(5)/sleep(2) in teardown/driver, hang_confirm_s=10 on a shared node, n=1 per release.
  test_rpc::test_incremental_task_async timed out once (CLAIMS #118).

## 7. Top 10 recommendations (value order)
1. **Rerun the full matrix on v2 head** (gtest + unit groups + extras + 12 integration) in one GB200 job. Triage the nvlink_one_sided
   F/hang first. Effort S (one 4-GPU job, about 1.5 h). Runs: manual GB200. This blocks any "final" claim.
2. **Make CI execute the e2e suite**: build an image (or post-merge stage) with ULFM OMPI + mpi4py>=4 and set TLLM_REQUIRE_ULFM=1 on that
   stage, so a skip becomes a failure. Also set it for l0_cpu's survivors_ulfm, or move that test to the ULFM stage. Effort M-L (infra).
   Runs: l0_gb200_multi_gpus post_merge.
3. **Restored-slot checksum assertion**: after a restore, compare each restored slot's bytes (or a hash) to the host copy / checkpoint
   for that expert. Do it under a debug flag in the integration driver, or as an exact-logits check against a healthy run on the same
   post-loss placement. Effort M. Runs: integration GB200.
4. **DSACacheManagerV2 revert test** (indexer K pool + overlap 2 steps), mirroring test_kv_cache_revert. Alternatively, reject DSA in
   `create` until tested. Effort S-M. Runs: l0_b200.
5. **Hang negative controls in integration**: an all-rank 45 s host stall and a GIL hog on one rank at the shipped hang_confirm_s
   (30) must NOT fence. Also run test_hung_rank_fenced at the default value, not 10. Effort M. Runs: post-merge GB200 (long; one case).
6. **Loss during restore copy / Install**: a worker hook (like WIDEEP_FT_TEST_KILL_AFTER_STOPPED) that kills rank 3 after Install starts.
   Assert a replan without [2,3] and bit-exact outputs. Effort S (hook already exists). Runs: integration GB200.
7. **Sequential losses with serving in between** (generation 2) and **authority killed mid-recovery**. Assert clean recovery or a bounded
   fail-closed exit. Effort S-M. Runs: integration GB200.
8. **Rank-0 kill fail-closed test**: every client gets an error within N s, the process tree exits, and no orphans remain. Effort S.
   Runs: integration GB200.
9. **De-flake and de-mock the CPU suite**: replace the 0.2 s/0.3 s deadline races with an injected clock or events. Replace the
   hand-rolled PyExecutor fakes with one shared fixture built from a real (CPU) PyExecutor skeleton, so fake drift (the 5 failures)
   stops recurring. Add one real-data request-accounting test (each stream ends with exactly one terminal response). Effort M.
   Runs: l0_cpu.
10. **CI hygiene**: move moeLoadBalancerTest to `add_tllm_gtest` with OWNER/LABELS (runtime). Make the MNNVL tests (late_writes,
    ring_restore_staging) fail rather than skip on a GB200 stage that should have MNNVL (env guard like TLLM_REQUIRE_ULFM). Add
    a2a_abort/execution_state to a GB200 single-GPU list as well. Effort S. Runs: CI config.

## 8. Manual GB200 evidence (for reference; all n=1-3, never on v2 head)
- Integration 12/12 @547a6aa7ed and @066b5189a9 (V3-Lite NVFP4, EP4, 1 node). 10/10 @b18f0b64ed (#116).
- DEP32 R1 kill matrices (#119, #121: 9/9 and 12/12 PASS, n=3/row) and GSM8K after a kill (#124, PASS 3/3) were all on aae00e6c65,
  an earlier lineage.

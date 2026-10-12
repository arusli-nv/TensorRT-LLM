# WideEP peer-loss recovery: PR split

Each line becomes one commit on `wideep-ft`, rebased on the latest main. PRs are not opened until
the user says so. Order is merge order; every commit must build and pass its own tests, with no
code that only a later PR uses. Commit 44eded2a5c (frontend admission) is gone: ed20a243ad
already reverts it.

## Independent fixes, based on main
- U1 `[None][fix] propagate fatal RPC worker errors` (e80f9cbed6): rpc_proxy_mixin, rpc_worker_mixin,
  ray/executor, llm.py, their tests, l0_cpu entry.
- U2 `[None][fix] bound RPC socket teardown with unreachable peers` (952dcac056): rpc_client, test_rpc.
- U3 `[None][fix] retain DLPack metadata until the last storage alias` (e13dab7bcd).
- U4 `[None][fix] skip request updates without a sampling result` (4dc8d8803f).
- U5 `[None][fix] keep EPGroupHealth identity across ConfigurableMoE deepcopy`: configurable_moe.py.

## Feature series
1. `[None][infra] upgrade mpi4py to 4.1.2`: install_mpi4py.sh (patch re-ported to futures/_core.py),
   jenkins/current_image_tags.properties after the image rebuild, test_mpi_session KV-startup
   test. Until this image is in CI, the ULFM and integration tests skip.
2. `[None][feat] WideEP recovery authority`: executor/recovery/ (protocol, client, authority) plus
   the cached environment check that later PRs import; test_recovery_authority, l0_cpu entry.
   Torch-free at import time; no dependency.
3. `[None][feat] ULFM survivor communicators for host collectives`: communicator.py (MPIDist
   recovery, PeerFailureWatch, logical gathers with fill values, `live_ranks`), `_utils`
   `is_mpi_peer_failure`, fill_value in fake dists, test_mpi_survivors, test_mpi_survivors_ulfm.
4. `[None][feat] fault-tolerant execution descriptor for the NVLinkOneSided all-to-all`: .cu/.h/op,
   cpp_custom_ops.py, test_moe_a2a_abort (l0_b200), test_moe_a2a_late_writes (l0_gb200_multi_gpus).
   Description notes the fake-op signature repair on main. Coordinate with #19610 and #15895.
5. `[None][feat] MoE execution state for WideEP recovery`: execution_state.py, nvlink_one_sided.py,
   ep_group_health.py, wide_ep_ft.py, _mnnvl_utils.py, runner.py hook, model_engine
   `moe_execution_state` property and close, MOE_DEVELOPER_GUIDE. Needs 2 (env check) and 4.
6. `[None][feat] resident-replica EPLB placement after a rank loss`: moe_load_balancer.py,
   model_engine `install_resident_placement`, own test file split out of
   test_peer_loss_recovery. Needs 3 and 5.
7. `[None][refactor] let attention-DP routing skip absent ranks`: adp_router.py, request_utils.py,
   cuda_graph_runner.py, the py_executor unservable-request hunks, test_adp_host_exchange and
   the ADP tests split out of test_peer_loss_recovery. Needs 3. The one FT-off-visible change:
   a strict pin to an out-of-range rank gets an error response instead of IndexError.
8. `[None][feat] keep the MPI session serving after a peer loss`: mpi_session.py, test_mpi_session.
   Needs 2 and 3. Inert unless recovery is enabled.
9. `[None][feat] recover the executor from a WideEP peer loss`: peer_loss_recovery.py, py_executor
   hooks, resource_manager.py and kv_cache_manager_v2.py rollback, test_peer_loss_recovery,
   test_py_executor update, tests/integration/defs/wide_ep_ft/ with its l0_gb200 post-merge
   entries, docs/source/features/wide-ep-fault-tolerance.md and index entry. Needs everything.

## Mechanics
- Build each commit from the final tree with per-file checkouts onto a fresh branch from main;
  split shared files (py_executor.py: U4, 7, 9; model_engine.py: 5, 6; test_peer_loss_recovery.py:
  6, 7, 9) by hunk.
- Per commit: pre-commit, then CPU unit tests; on the tip: GPU unit tests, integration test, EP32.
- PR descriptions (drafted in pr-descriptions/, not posted): rationale and history, FT-off parity
  (SASS opcode diff, perf A/B), FT-on cost (1.4% all-active), dropped admission pause, multi-node
  `mpirun` teardown caveat, health stays "healthy" while degraded.

## Fold notes (from drafting the descriptions)
- PR 4 takes `NVLinkOneSided.calculate_required_workspace_size(fixed_source_slots=...)` so that
  test_moe_a2a_late_writes.py passes there.
- U5 carries the EP_GROUP_HEALTH_KEY rename and a deepcopy test in test_configurable_moe.py.
- PR 6 moves test_check_routing_excludes_* into a GPU-listed test_resident_placement.py.
- The MOE_DEVELOPER_GUIDE link to the user doc moves to PR 9, where the doc lands.
- FT-off throughput A/B against a main-only build: EP32 batch 2.

# WideEP peer-loss recovery — PR split plan

Each line becomes one commit on `wideep-ft` (rebased on latest main) and one PR. Order = merge order.
Every commit must build and pass its own tests. Commit 44eded2a5c (frontend admission) is dropped.

## Independent fixes (based on main, any order)
- U1 `[None][fix] propagate fatal RPC worker errors` — rpc_proxy_mixin, rpc_worker_mixin, ray/executor, llm.py,
  test_rpc_executor_mixin, test_rpc_worker_mixin, l0_cpu entry (from e80f9cbed6)
- U2 `[None][fix] bound RPC socket teardown with unreachable peers` — rpc_client, test_rpc (952dcac056)
- U3 `[None][fix] retain DLPack metadata until the last storage alias` — _dlpack_utils, test_mnnvl_memory_lifecycle (e13dab7bcd)
- U4 `[None][fix] skip request updates without a sampling result` — py_executor, test_py_executor (4dc8d8803f)
- U5 `[None][fix] ConfigurableMoE deepcopy memo` — configurable_moe.py (split out of 72f22284b0)

## Feature series
1. `[None][infra] upgrade mpi4py to 4.1.2` — install_mpi4py.sh (re-ported futures patch), test_mpi_session KV-startup test.
   Needs an image rebuild; everything below needs it at runtime only (FT off works on 3.1.5).
2. `[None][feat] fault-tolerant execution descriptor for NVLinkOneSided all-to-all` — .cu/.h/op.cpp,
   cpp_custom_ops.py, test_moe_a2a_abort.py, l0_gb200_multi_gpus entry. (squash 273d31cfbe + e4c6cefa41 + a0311cb56a + rework)
   Coordinate with draft PRs #19610 (NVLinkOneSided overhaul) and #15895 (abort ABI).
3. `[None][feat] MoE execution state for WideEP recovery` — execution_state.py, nvlink_one_sided.py,
   ep_group_health.py, wide_ep_ft.py, _mnnvl_utils.py, MOE_DEVELOPER_GUIDE, test_moe_execution_state,
   test_ep_group_health, l0_gb200_multi_gpus entry.
4. `[None][feat] ULFM survivor communicators for host collectives` — communicator.py, _utils.py
   (is_mpi_peer_failure), test_mpi_survivors.py, test_mpi_survivors_ulfm.py.
5. `[None][feat] WideEP recovery authority` — executor/recovery/*, test_recovery_authority, l0_cpu entry. Torch-free;
   can land first.
6. `[None][feat] keep the MPI session serving after a peer loss` — mpi_session.py, test_mpi_session. Depends on 4, 5.
7. `[None][refactor] let attention-DP routing skip absent ranks` — adp_router.py, request_utils.py,
   cuda_graph_runner.py, test_adp_host_exchange. The one FT-off-visible change (strict pin to an out-of-range
   rank becomes an error response instead of IndexError).
8. `[None][feat] resident-replica EPLB placement for a lost rank` — moe_load_balancer.py, model_engine.py,
   runner.py. Depends on 3.
9. `[None][feat] recover the executor from a WideEP peer loss` — peer_loss_recovery.py, py_executor.py,
   resource_manager.py, kv_cache_manager_v2.py, executor_request_queue.py, test_peer_loss_recovery,
   tests/integration/defs/wide_ep_ft/ (+ list entry), user doc. Depends on all.

## Mechanics
- Fold the WIP commit and later fixes into the commits above with a scripted per-file checkout onto a fresh
  branch from main: for each target commit, `git checkout <final> -- <files>`; files shared by several commits
  (py_executor.py: U4 + 9) need hunk-level splitting.
- After folding: per-commit build + unit tests (CPU), then one GPU run on the tip.
- PR descriptions: rationale/history there, not in code comments; state FT-on costs (gather copy per MoE layer,
  10 ms watcher polling, commit-gate sync) and FT-off parity (SASS diff, perf A/B).

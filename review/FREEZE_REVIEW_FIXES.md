# Final review of freeze 759787cb17: consolidated fix list (monitor, 2026-10-22 night)

Four read-only reviews (a2a/remap; recovery orchestration; comms/launcher/executor; MoE LB/ghost/autotuner). Monitor
spot-checked the HIGH series items and the impl_blocks gating. Goal (user): clean, confident code we are proud of.

## A. Product fixes (runtime change -> one refreeze; rerun only affected VAL rows)
| # | Sev | Where | Fix |
|---|---|---|---|
| A1 | MED | communicator.py:94 _is_transport_error | with recovery on, any exception from a host collective goes through _failure_evidence; re-raise unchanged only without evidence (covers mpi4py lowercase unpickle: allgather/allreduce/tp_allgather small) |
| A2 | MIN | mnnvl_memory.py remap ~985 | catch Exception (not only RuntimeError) before the outcome exchange |
| A3 | MED | moe_load_balancer.py add_layer ~1225 | ring mode decided once per balancer (assert all layers agree); mixed-layer test |
| A4 | MED | impl_blocks.py ~434 | FT-on static routing: no make_tensor_host_accessible migration; set_use_gpu_memcpy(True). Verify allocation change on GB200 |
| A5 | MED | peer_loss_recovery.py:1048 / protocol.py:96 | /health failed_requests counts quarantined requests (allgather before Installed); fix docstring |
| A6 | LOW | peer_loss_recovery.py:399/553 | hang_detector.checkpoint() in _shrink_again (or bound covers a re-shrink) |
| A7 | LOW | peer_loss_recovery.py:1494 | RecoveryTimeout, not TimeoutError |
| A8 | LOW | py_executor.py:8589 | StepChange restores exclude_last_generation_logits |
| A9 | LOW | kv_cache_manager_v2.py ~4447 | revert_prepare_resources calls _after_shrink (or refuse sharing under FT, stated) |
| A10 | LOW | openai_server stream | _extract_metrics error after [DONE] must not emit a second [DONE] |
| A11 | LOW | mgmn_worker_node.py:21 | set_inheritable(200) only with --flashinfer-lock-fd (FT-off delta) |
| A12 | LOW | nvlink watchdog _capturing (3496635495) | FT-off identical to main |
| A13 | LOW | hostAccessibleDeviceAllocator getInstance | no singleton leak with FT off (or justify) |
| A14 | LOW | ExtraSlots ghost + online EPLB w/o GDRCopy | warn/reject silent GPU-memcpy switch |
| A15 | LOW | finalize_layer_weights (rlhf) | close own/imported shm before re-sharing |
| A16 | LOW (C++) | moeLoadBalancer.cpp:660 | routesEachExpertToOneSlot pure query + explicit commit |
| A17 | LOW (C++) | moeAlltoAllOp.cpp resolveAbortState | resolve device pointer once in MoeExecutionState |
| A18 | LOW (C++) | moeAlltoAllOp.cpp FLAG_VAL 8 B | state in PR (FT-off layout delta) or move released marker to a rank-mask-only slot |
| A19 | LOW (C++) | restorePlacement | start CPU copy threads only when CPU copies are used |
C++ items A16-A19: include only if a C++ rebuild happens anyway; otherwise PR follow-up.

## B. Series hygiene (tree byte-identical; per-commit import/flake8 F821 smoke over the series)
B1 d05405ee21 uses FT names before they exist (code=None there). B2 42e2c587a3 / 9d5cf2ea2a import helpers added in
2c66efa1ab. B3 3496635495 needs 296e22f6dc APIs (reorder); retitle. B4 expert_to_slot spread over ab0e0d233f/3496635495/
006c0bbfdc -> all in 006c0bbfdc. B5 unrelated hunks (cuda_graph_runner vote, watchdog) to owners. B6 75a01579ef/db0d957ad0
Cutlass num_extra_local_experts before b3c6397ae3. B7 0faefbbc4f needs fad4ce76f4 executor methods; forward imports of
hang_evidence/pmix_fence. B8 squash 5330b899f0 into 0faefbbc4f; Status/StatusQuery to e5a37b7d32; PROTOCOL_VERSION starts at 1.
B9 b3c6397ae3 unrelated CuteDSL moe_output sizing split out.

## C. Lean / comments / tests (with B)
Dead: extra_slot_count, initial_weight_assignments-for-ghosts, BAR1/pynvml log + hostMappings binding, staging_halves lists,
redundant class attrs, unreachable _publish_rank_mask capture branch, watchdog clear/restart under bound (confirm), 4-tuple
unwrap in _enqueue_responses -> hook. Comments: _has_restore_slots rename, restore_copy_plan docstring, client.py:104,
NVLinkOneSided docstrings, op error wording; test names test_r1_*. Optional: split peer_loss_recovery.py into
peer_loss/{hook,startup,requirements,request_ledger,memory,recovery}.py. Tests: listed per review (transport garble via
allgather, remap non-RuntimeError, mixed layers, Install race, failed_requests, re-shrink vs hang detector, FT-off parity).

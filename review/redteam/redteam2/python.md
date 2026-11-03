# WideEP FT product-Python red team (round 2), 2026-10-20 20:17 PT

Scope: non-test Python under tensorrt_llm/ changed between the base (merge-base with upstream/main, 0829689c52) and
fix-final8-rc1 (78836d1723), plus fix-minors' Python commits (72447c6fc2..1b7b0b5318 on fix-final7 2c3af30ce8).
That is 48 files, about +7.4k/-0.4k lines. Fix-final8-rc1 differs from fix-final7 only in the log line of
peer_loss_recovery.py, so the fix-minors Python applies to it unchanged.
Method: read-only `git show` and `git diff`. No torch or GPU here, so every finding is static.

Line references: `R:` is 78836d1723. `M:` is fix-minors, for files fix-minors changes (py_executor, peer_loss_recovery,
moe_load_balancer, openai_server, fault_tolerance_status, authority, hang_monitor).

Format: severity | file:line | issue | fix

## Prior findings, briefly re-verified (not re-reported)
- CODE_REDTEAM_FINAL #1 (CuteDSL cache key bakes source_tokens): FIXED. The cache_key holds `source_tokens > 0`, and
  source_tokens/slot_tokens are runtime args (R:cute_dsl_custom_ops.py:4517,4559).
- #10 (lazy world allgather in add_layer): FIXED. Hosts are gathered once in MoeLoadBalancer.__init__ (R:moe_load_balancer.py:1162-1166).
- #11, #12, #15, #16, #18, #19, #20: fixed as MINORS_FIX.txt states. Checked: M:moe_load_balancer.py:546; M:openai_server.py:272,1207;
  M:fault_tolerance_status.py (one probe thread, and ZeroMqQueue binds its thread at first use, so the fix holds);
  M:py_executor.py:5077; M:peer_loss_recovery.py:759-768,1063.
- Still open as before: #29 (aborted() runs .any() every step), #32 (sys.modules reflection, R:mpi_session.py:140),
  #36, #33, B1 (peer_loss_recovery.py is 1657 lines). #37 is only half fixed: `world_group` is now freed, but `group`
  still leaks if `mpi_comm().Get_group()` raises (R:_utils.py:625).

## Findings

### Correctness / fail-closed
1. minor | M:tensorrt_llm/_torch/pyexecutor/peer_loss_recovery.py:761-768,1063 | `_rerouting` is only ever added to (`|=`)
   and is only drained when an id gets routed. Two cases leave an id in it for good: a re-run id that never reaches
   route_requests (cancelled while waiting, or removed by `split_unservable_requests`), and ids left over from an
   earlier recovery. From then on, every step on every rank scans every routed item. `own_rank` is `dist.rank`, but
   `routed` is keyed by TP rank (py_executor uses `all_ranks_new_requests[self.dist.tp_rank]`). The two are equal only
   because recovery requires pure ADP. | Fix: assign with `=` instead of `|=` at each recovery, or intersect with the
   ids rank 0 re-admitted. Drop ids on cancel/fail. Compare against `dist.tp_rank`.
2. minor (suspected) | R:tensorrt_llm/_torch/pyexecutor/hang_evidence.py:122-139 + R:engine/runners/decoder/graphs.py:543 |
   CUDA graphs can still be captured lazily inside the serving loop (`needs_capture` -> `cuda_graph_runner.capture`,
   `torch.cuda.graph` in the default "global" error mode). Meanwhile the hang-evidence thread calls `Event.query()` and
   issues async D2H copies plus `Event.record` on a side stream every heartbeat. In global capture mode, a
   capture-unsafe call from another thread can invalidate the capture, and the capture then raises on the serving
   thread. | Fix: check whether `cudaEventQuery` or a side-stream copy during a foreign global capture is legal on the
   target driver. If it is not, skip `read()` while `torch.cuda.is_current_stream_capturing()` is set on the serving
   thread (export a flag), or capture with `capture_error_mode="thread_local"` under recovery.
3. minor | R:tensorrt_llm/_torch/pyexecutor/peer_loss_recovery.py:695-698 | `commit_step` falls back to
   `torch.cuda.current_stream().synchronize()` when `sample_state` or its `sampler_event` is None. In the overlap loop,
   commit_step for the previous batch runs after the current forward was enqueued. The fallback therefore waits for
   the current step's forward and removes the overlap for any sampler without an event. | Fix: refuse such samplers in
   `create()`, or record an event at the step's end in step_prepared/forward and wait on that.

### FT-off parity with upstream/main
4. major (parity / one concern per PR) | M:tensorrt_llm/serve/openai_server.py:286,2419,2928,3211,3381 | Behavior
   changes with FT off:
   (a) Chat and Responses streams that fail mid-stream used to re-raise (main logs and raises, so the connection
   aborts). They now always end with an SSE `error` event plus `[DONE]`.
   (b) The completions generator no longer emits its own `[DONE]`. Main sent `[DONE]` twice on an error, and with
   several prompts it sent `[DONE]` mid-stream.
   Both are reasonable fixes to main, but they reach every deployment, have nothing to do with FT, and the FT tests
   do not cover them (section C already lists "error event can follow [DONE]"). | Fix: either gate
   `_end_stream_on_error` on `peer_loss_recovery_enabled()` and restore main's completions body, or split the
   streaming-error change into its own upstream PR with serving tests.
5. nit | R:tensorrt_llm/_torch/pyexecutor/request_utils.py:134-138, scheduler/adp_router.py:372 | With FT off,
   `all_ranks_num_active_requests` changes from a list to a dict. A pin of `-1` used to index the last rank and fail
   later in route_requests. It now raises KeyError at the waiting-queue pop: same outcome (the loop errors), different
   place and type. | Fix: validate `attention_dp_rank` in [0, tp_size) at admission (`_validate_request`) so both
   paths reject it the same way, then fix test_adp_absent_ranks (#46).

### Per-step hot-path cost
FT off adds only O(1) Python calls per step: about 12 no-op hook calls, the `_interruptible` attribute check on each
host collective, and three `live_ranks` tuple builds. It adds one `_note_step_change` call per scheduled request, and
about 4 calls per MoE layer per eager forward (`_dispatch_slot_table`, `_compact_receive_slots`, `_build_comm_plan`,
and 2x `_publish_rank_mask(None)`). CUDA-graph replays skip the per-layer ones. Estimated total: well under 50 us per
step. Nothing syncs.

FT on, healthy: one sampler-event sync per step that main already does; one `MPIX_Comm_get_failed` per host
collective (about 5-7 per step); a per-response pass in `gathering_responses`; one 2-word `.any()` per step (#29).
6. nit | R:tensorrt_llm/_torch/moe/fused_moe/moe_scheduler.py:261,1089-1101,1132-1134 + R:nvlink_one_sided.py:925 |
   Each MoE layer of each eager forward makes isinstance/backend/dict lookups to find the bound-slot layout, and calls
   `_publish_rank_mask` even when FT is off. | Fix: cache one bool per scheduler at construction
   (`moe.comm._execution_state is not None`) and skip all three helpers when it is False. Have the caller skip
   `_publish_rank_mask` when the mask is None.
7. nit | R:tensorrt_llm/_torch/moe/fused_moe/communication/nvlink_one_sided.py:933 | In detection-only mode
   (MODE=1, no authority), every dispatch and combine of every layer runs `mask.tolist()` and builds a tuple to
   compare against the last words. That is two host round-trips of Python per layer per eager step. | Fix: compare
   `EPGroupHealth.generation` (an int) and re-publish only when it changes.
8. nit | R:tensorrt_llm/_torch/distributed/communicator.py:290-292 (used at R:py_executor.py:6679,7089, R:cuda_graph_runner.py:302) |
   `Distributed.live_ranks` builds `tuple(range(tp_size))` on every access, about three times per step with FT off. |
   Fix: cache it in `__init__`, or add `num_live_ranks` and compare against `tp_size`.
9. nit | R:tensorrt_llm/_torch/pyexecutor/executor_request_queue.py:348-349 | Every `get_from_request_queue` call
   (FT off too) makes two extra passes over the fetched items to filter out wake items, which are only ever enqueued
   under recovery. | Fix: filter only when `items and items[-1].is_wake_request` or similar, or keep a "woken" flag
   that `wake()` sets.
10. nit | R:tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py:4302 | The tuning-config cache key includes
    `source_tokens` (the per-forward max tokens per rank), so FT on adds one TuningConfig entry per distinct token
    count. The growth is bounded by max_num_tokens. | Fix: key on `(source_tokens > 0, slot_tokens)` and pass the
    geometry to `infer_shape_source_slot_rows` at call time, or accept the bound and say so.

### Duplication of existing helpers
11. nit | R:py_executor.py:6944-6961, py_executor.py:9028,9046 (main), R:peer_loss_recovery.py:1542 and
    `_cancelled_responses` | "Error LlmResponse for a RequestQueueItem" (request_id, error_msg,
    `getattr(item.request, "client_id", None)`) is now built in five places. | Fix: add one
    `_queue_item_error_response(item, msg)` helper and use it in all five.
12. nit | R:scheduler/adp_router.py:226-262 (`_assign_explicit_dp_ranks`), R:adp_router.py `split_unservable_requests`,
    R:request_utils.py:125-138 | Three copies of "read `py_scheduling_params`, then `attention_dp_rank` and
    `attention_dp_relax`", each with its own FT branch on a missing rank. | Fix: add one
    `_attention_dp_pin(item) -> (rank, relax)` helper and decide servability in one place.
13. nit | R:tensorrt_llm/_torch/pyexecutor/py_executor.py:9338-9343 | `_enqueue_responses` detects a recovery
    `ResponseContribution` by duck-typing (`tuple`, `len == 3`, `list` first), and `responses_gathered` unwraps the same
    tuples again. | Fix: have the hook unwrap: `contributions = hook.responses_from(responses_list)`. Or use a small
    NamedTuple type and an isinstance check, so no type-shape guessing runs in main's gather loop.

### Module organization (> 800 lines, or grown past it)
14. minor | R:tensorrt_llm/_torch/moe/fused_moe/moe_load_balancer.py (1161 -> 1923 lines, +762) | The FT additions
    are about 40% of the module: the shm reserve/populate helpers, `restore_lost_experts`, `staging_halves`,
    `_next_piece`, `ring_copy_nodes`, `restore_copy_plan`, `stage_restored_slots`, `recovery_placements`,
    `placement_digest` and the ghost-slot accessors. | Fix: move the pure planning functions plus
    restore_copy_plan/stage_restored_slots into `moe/fused_moe/eplb_recovery.py`, about 600 lines. Keep only the
    native-binding wrappers on MoeLoadBalancer.
15. minor | M:tensorrt_llm/executor/recovery/authority.py (880 lines, new) | One file holds the state machine
    (`RecoveryAuthority` and `Phase`, :77-645) and the transport/CLI (`serve`, `_send`, `_receive_datagrams`,
    `datagram_socket`, `router_queue`, `command_fence`, `main`, :646-880). | Fix: split it into `authority.py` (state
    machine, unit-testable without zmq) and `authority_server.py` (sockets, fence, main).
16. minor (open B1, refined) | R:tensorrt_llm/_torch/pyexecutor/peer_loss_recovery.py (1657 lines) | There is a
    natural three-way split. | Fix: (a) the executor hooks and request/response bookkeeping (:195-880) stay in
    peer_loss_recovery.py; (b) the `_recover` protocol driver, shrink, fencing and placement (:876-1400) go to
    `peer_loss_recovery_protocol.py`; (c) startup-watch/registration (`_register`, `watch_startup_peer_failures`,
    `failure_domains`) go to `executor/recovery/` (see 18).
17. nit | R:tensorrt_llm/llmapi/mpi_session.py:75-205 (+207 lines in an already-1211-line module) | The FT exit path
    (`_num_failed_ranks`, `_is_departed_peer_failure`, `_mpi_finalize`, `_finalize_and_exit`,
    `exit_fault_tolerant_process`, `_leave_degraded_world`) is self-contained. | Fix: move it to
    `executor/recovery/mpi_exit.py`. Have peer_loss_recovery register its stop hook there, which also removes the
    sys.modules reflection (#32).
18. nit | R:moe_load_balancer.py:1203-1204 | The MoE module lazily imports `failure_domains` from
    `_torch.pyexecutor.peer_loss_recovery`, a layering inversion (moe -> pyexecutor). | Fix: move `failure_domains`
    next to `fault_tolerance_config` in `executor/recovery/config.py`.

### Docstrings / comments
19. nit | R:moe_load_balancer.py:75-80 | Measurement history in a docstring ("a restore's GPU copies of 1.4 GB took
    ~400 ms instead of ~35 ms"). | Fix: say "prefaults the pages so the first restore copy does not page-fault" and
    move the numbers to the PR.
20. nit | R:kv_cache_manager_v2.py:4326-4344 (revert_prepare_resources, 19-line docstring), R:peer_loss_recovery.py
    class docstring (25 lines) and `_recover` (19 lines) | These are much longer than the code they describe and
    restate invariants already given in `commit_step`/`MoeA2AAbortState`. | Fix: trim each to a contract (what is
    reverted, preconditions, raises) and link to the one place that explains the invariant.

### Concrete removable code
21. nit | R:tensorrt_llm/_torch/distributed/communicator.py:85-88,365-401 (+ `departed=` on every TorchDist method,
    :1520,1640,1660) | The `_DEPARTED_GROUP` path and `departed` in `tp_cp_allgather` has no use: recovery requires
    cp_size == 1 (`enable_peer_loss_recovery`), and `decoder/inputs.py` already bypasses tp_cp under FT. TorchDist
    accepts `departed` and ignores it. No caller passes `departed=` to tp_cp_allgather; the autotuner's calls pass
    none, so after a loss they raise either way. | Fix: delete the tp_cp_allgather departed branch (about 15 lines). Keep
    `departed` on the abstract signature only where MPIDist implements it.
22. nit | R:tensorrt_llm/_utils.py:1441-1443 | `device_visible_host_zeros` has one caller (execution_state.py:45) and
    no test uses it. | Fix: inline `torch.zeros(2, dtype=torch.int32).pin_memory()` and keep its comment at the call
    site.
23. nit | R:tensorrt_llm/_torch/pyexecutor/py_executor.py:1569-1576 | `_run_event_loop` is a one-line wrapper. |
    Fix: call `self._peer_loss_recovery.run(self.event_loop)` at :1483 and update the one test that mocks it
    (test_peer_loss_recovery.py:1024).
24. nit | R:tensorrt_llm/_torch/pyexecutor/peer_loss_recovery.py:1096-1098 | `_end_departed_rank(rank)` only wraps
    `_end_ranks([rank])[rank]`. | Fix: inline it at its single call site (:1091) and mock `_end_ranks` in the tests.

## Counts
blocker 0, major 1, minor 6, nit 17 (24 total).

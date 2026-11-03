# WideEP FT series cleanup plan (read-only review, 2026-10-23 ~13:45 PT)

Base: `/tmp/claude-161238/wt-split`, `v5-final` = `1117f7e0e7` (frozen, benchmarked), 32 commits on upstream main
`9cc37929e8`. The product diff is 10047+/597- over 70 files (Python 7591+/376-). Tests are not counted.
Inputs: CODING_GUIDELINES.md, AGENTS.md, `review/PR_DRAFTS.md`, `review/CODE_AUDIT_20261023.md`,
`review/FREEZE_REVIEW_FIXES.md` (B/C), plus 4 read-only sub-reviews (comments, dead code/test hooks,
env/duplication, C++). The monitor spot-checked the key claims: openai_server.py:1219, moeLoadBalancer.cpp:662,
moe_scheduler.py:266, the hostAccessibleDeviceAllocator NUMA branch, pair.h, encode_datagram,
is_mpi_peer_failure_class, live_helpers and exit_fault_tolerant_process.
Paths below are relative to the worktree, and line numbers are at `1117f7e0e7`.

Classes:
- **P**: pure. Comments, docstrings, provably dead code, renames, moving code with identical logic.
- **S**: structural, with identical behaviour; tests prove it.
- **B**: behaviour change. Not part of this cleanup; it goes on the follow-up PR list.

## 0. Bottom line

- The series is already lean. No product test-only hooks remain, every added def/binding has a product caller, and
  the earlier C-list dead items (`extra_slot_count`, BAR1/pynvml, `hostMappings`, `staging_halves`,
  `_has_restore_slots`, the 4-tuple unwrap) are already gone. A16 is already done.
- Most of the gain is in **comments**, about 170 lines (P). Dead code and test-only defaults add about 15 (P) plus
  about 45 (S). Reuse and deduplication add about 30 (P/S). C++ adds about 10 (P) plus about 30 (S), and only
  pays off if a rebuild happens anyway.
- **Estimated net reduction:**
  - P-only Python: **about 195 lines** (about 1.9% of the product diff).
  - Adding Python S: **about 290**.
  - Adding the C++ items: **about 330**.
  - The peer_loss_recovery split adds about 40-60 lines back (headers, imports, delegation).
  - **Realistic total: about 250-290 lines net.**
- An equivalence tool that makes P items provable instead of argued: an AST comparison that strips docstrings
  (`ast.dump` of each touched module, with docstring `Expr` nodes removed, must be equal before and after). Every
  comment/docstring item and every "move with identical logic" item should pass it (per module, or for the moved
  function bodies).

## 1. Recommended order (keeps every step provable)

1. **Restack first, tree-identical** (section 2). Proof: `git rev-parse NEW^{tree}` == `git rev-parse 1117f7e0e7^{tree}`.
   This changes no byte of the benchmarked build.
2. **Python P items** as `--fixup` commits into the commit that introduced each line, then autosquash with
   `GIT_SEQUENCE_EDITOR=: git rebase -i --autosquash 9cc37929e8`, or `git rebase --autosquash` on git >= 2.44,
   which needs no -i. Proof: the AST-equality script, plus `pytest` of the files listed, plus pre-commit.
3. **Python S items**, each with its own test run (listed per item).
4. **C++ items** only if a C++ rebuild is planned anyway (section 7).
5. Optional: the peer_loss_recovery split (section 4), as the last commit of PR 20/21 or as its own follow-up
   commit.

**User decision D1:** the code-freeze rule says a product-code change means refreeze plus rerun. P items that pass
AST equality do not change behaviour, so no rerun should be needed. S items change bytecode, so decide whether
they need (a) unit tests only, (b) a 1-node GPU smoke, or (c) a rerun of the affected VAL rows. The monitor
recommends (a) for pure-Python S and (b) for anything in `_recover`/communicator.

## 2. The 3 restack problems in PR_DRAFTS.md, plus the 3 later fix commits (P, series hygiene)

All of these are verified at HEAD. None of them changes the final tree.

| # | Problem (verified) | Fix | Proof |
|---|---|---|---|
| R1 | `e74cadb97c` calls `self._note_step_change(request)` unguarded in the overlap completion update, but `_step_changes` is `None` with FT off. That raises AttributeError with FT off + overlap until `ddc2658518` (which moves the guard into `_note_step_change`); `65741c820a` fixes the multimodal test fake. | Squash `ddc2658518` + `65741c820a` into `e74cadb97c`. Expect a trivial `py_executor.py` context conflict when `db6f10dc0b` is replayed. | tree hash equal; `pytest tests/unittest/_torch/executor/test_py_executor.py test_multimodal_scheduler.py test_peer_loss_recovery.py` at the squashed commit |
| R2 | `34611ef619` changes the op check from `t.is_cpu()` (main, moeAlltoAllOp.cpp:162) to `t.is_cuda() && t.device()==workspace.device()` (:161). Main's `capture_active_rank_mask` still builds the mask on CPU, so every rank-mask dispatch (EPGroupHealth or MODE=1) fails at `34611ef619` and `6f5f1d080b`, until `2b5005b9a1` publishes the device record. FT off is unaffected. | **Recommended: merge PR 10 + PR 12** (`34611ef619` + `2b5005b9a1`; `6f5f1d080b`/PR 11 then goes after or before both, since it only needs the abort record). Alternative: P5a keeps a host-to-device copy of the CPU mask, which `2b5005b9a1` then removes. That adds throwaway code to a PR and is not recommended. | tree hash equal; `test_nvlink_one_sided.py`, `test_moe_a2a_abort.py`, `test_moe_execution_state.py` at each commit (GPU) |
| R3 | `17a047ea76` (ADP burst rule) uses `len(live_ranks)` (py_executor.py:7041 at HEAD), a local introduced by `9cf51e9801`, so it cannot go to main alone. | Move `17a047ea76` before `9cf51e9801` and write `self.dist.tp_size` there. `9cf51e9801` then changes that one token to `len(live_ranks)` when it introduces `live_ranks`. The final tree is unchanged. Send PR 19 to main (it is not FT and changes scheduling for every user). | tree hash equal; `test_py_executor.py` ADP tests at both commits |
| R4 | New since PR_DRAFTS: `3d10673082` + `1117f7e0e7` (record started owners before the quarantine, requeue after the commit) touch only code from `db6f10dc0b` (blame: `_requeue_unscheduled_requests`/`_quarantine` are `db6f10dc0b`). | Squash both into `db6f10dc0b` (PR 21). The test hunk of `3d10673082` goes with it. | tree hash equal; `test_peer_loss_recovery.py` |
| R5 | New: `0e5a22ef51` (propagate a failed remap exchange) touches only `mnnvl_memory.py` lines from `6f5f1d080b`. | Squash into `6f5f1d080b` (PR 11). Clean: no other commit touches that file. | tree hash equal; `tests/unittest/_torch/test_mnnvl_memory_remap.py` |
| R6 | The remaining fix-ups from PR_DRAFTS A2: `7e8dfb37df` belongs to `ffc8bc30c3`, and `e25e638054` + `6e311f2f04` belong to `c86566c184`. | Squash as listed. | tree hash equal |

After R1-R6 the series is 23 commits, one per PR. PR 10+12 merge, so there are 24 PRs. Owed and not done by
anyone: a per-commit import/collection smoke. Run `python -c "import tensorrt_llm._torch.pyexecutor.py_executor"`
plus `pytest --collect-only` on the touched test files at each commit, inside the real container
(tools/import_smoke.sh). A per-commit C++ build applies to `34611ef619` / `ecfd7d3934` / `b0a8905b44`.

## 3. Prioritized item list

Columns: est. lines = net removed (negative means added). Risk: none, low, or med.

### Top priority (P, highest value per edit)

| # | file:line | change | class | lines | risk | equivalence |
|---|---|---|---|---|---|---|
| 1 | restack R1-R6 | section 2 | P | 0 | low (conflicts) | tree-hash equality |
| 2 | tensorrt_llm/serve/openai_server.py:1219 | delete the measured-number comment "Recovery from a peer loss takes about a second." | P | 1 | none | AST equality |
| 3 | cpp/.../moeLoadBalancer/moeLoadBalancer.cpp:662 | "routing stays main's replica-choosing kernel" becomes "routing keeps the replica-choosing kernel" (branch-relative wording) | P (C++) | 0 | none | build only; do it with a rebuild |
| 4 | tensorrt_llm/_torch/moe/fused_moe/moe_scheduler.py:266-270 `_dispatch_slot_table` | stale docstring: the gate is C++ `routesEachExpertToOneSlot` (FT mode + one replica per expert + no layer updates) plus the caller's `_peer_loss_recovery` check (:565). Rewrite in 2 lines and point to `one_slot_table`. | P | 2 | none | AST equality |
| 5 | tensorrt_llm/_torch/pyexecutor/peer_loss_recovery.py:387-391 `create` | the docstring lists 4 of the ~20 requirements. Say "the configuration must make a discarded step re-runnable (requirements below)" and keep the spec-dec / penalty sentences. | P | 2 | none | AST equality |
| 6 | tensorrt_llm/_torch/pyexecutor/py_executor.py:5603-5604 `resize_kv_cache_quota` | delete the garbled cross-reference sentence ("…refuses a KV cache transceiver and so remote-tail attention DP") | P | 2 | none | AST equality |
| 7 | py_executor.py:422-423 and :7032-7036 (ADP burst rule) | drop "A burst reaches every rank within 1-3 decisions" (an observed number) and cut the 5-line closed-loop rationale to 2 lines. The rationale goes in the PR 19 text. | P | 4 | none | AST equality |
| 8 | request_utils.py:121-125; adp_router.py:237-238; peer_loss_recovery.py:164 | history phrasings: "instead of leaving it queued forever", "as in a per-rank list", "no longer ends" | P | 2 | none | AST equality |
| 9 | peer_loss_recovery.py:701-724 `commit_step` | 24-line docstring down to about 10. The second paragraph re-derives the `MoeA2AAbortState` invariant stated in moeAlltoAllKernels.h:64-71, so replace it with "see MoeA2AAbortState". | P | 14 | none | AST equality |
| 10 | long-docstring trims, about 30 sites (list in 3.1) | trim to the contract and drop narration | P | ~140 | none | AST equality |

### 3.1 Comment/docstring trims (all P; equivalence by AST equality)

| file:line | current to proposed |
|---|---|
| peer_loss_recovery.py:1774-1791 `install_recovery_placement` | 18 to 11. Also drop "Raises ValueError: rank out of range", which this function does not raise itself (unverified for the callee). |
| peer_loss_recovery.py:1380-1395 `_remap_workspaces` | 16 to 9 |
| peer_loss_recovery.py:1615-1629 `_deliver_responses` | 15 to 9 |
| peer_loss_recovery.py:1200-1209 `_shrink` | 10 to 5 |
| peer_loss_recovery.py:1541-1554 `_requeue_unscheduled_requests` | 14 to 10 |
| peer_loss_recovery.py:81-85 `ResponseContribution` | 5 to 3 |
| tensorrt_llm/_torch/pyexecutor/hang_evidence.py:3-22 module docstring | 20 to 8 |
| tensorrt_llm/executor/recovery/authority.py:100-118 class, :128-133 `fence` arg | 18 to 8, 6 to 3 |
| tensorrt_llm/executor/recovery/hang_monitor.py:47-62 | 16 to 9 |
| tensorrt_llm/_torch/distributed/mnnvl_memory.py:966-981 `remap` | 16 to 9 |
| tensorrt_llm/_torch/distributed/communicator.py:108-119, 1010-1021, 1036-1048, 1059-1070 | 12 to 6, 12 to 7, 13 to 9, 12 to 7 |
| tensorrt_llm/llmapi/mpi_session.py:157-164 (also reflow the 123-char line 162), 218-228, 239-250 | 8 to 5, 11 to 6, 12 to 6 |
| tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py:214-223; torch_custom_ops.py:232-240 (duplicates the former; keep a "see") | 10 to 4, 9 to 3 |
| tensorrt_llm/_torch/moe/fused_moe/communication/execution_state.py:31-43 | 13 to 7 |
| moe_load_balancer.py:1395-1408 `recovery_placements` | 14 to 10 |
| reflow only: nvlink_one_sided.py:1025 (122 chars), executor/ipc.py:64 (165 chars) | 0 |
| C++ (only with a rebuild): ghostSlotMemory.h:34-49 (16 to 7), heartbeatSender.cpp:83-89 (7 to 4; also drop the `encode_datagram` mention if item 22 moves it), peerFailureWatch.cpp:92-96, moeLoadBalancer.h:351-355, moeAlltoAllKernels.cu:540-543 (names Python symbols; reuse the 1-line wording at :1765), moeAlltoAllKernels.h:258/:339 (main's "independent of future execution-abort handling" is now stale), moeLoadBalanceOp.cpp:229-231 (ambiguous "contents are not kept"), ghostSlotMemory.cpp:313-315 (the "untested" status note in a log message) | about 30 |

Total comment work: about 170 lines across about 47 sites in 21 files.

### 3.2 Dead code / test-only defaults

| # | file:line | finding (callers verified) | class | lines | risk | equivalence test |
|---|---|---|---|---|---|---|
| 11 | tensorrt_llm/_utils.py:613 `is_mpi_peer_failure_class` | one caller, `is_mpi_peer_failure` (:621); inline it | P | 3 | none | test_mpi_survivors.py; collection |
| 12 | _utils.py:1445 `device_visible_host_zeros` | one caller (execution_state.py:49); inline it. Do **not** replace it with `maybe_pin_memory`: that one skips pinning under Confidential Compute, which would be B. | P | 4 | none | test_moe_execution_state.py, test_moe_a2a_abort.py |
| 13 | executor/recovery/hang_monitor.py:185 `live_helpers(count=2)` | `count` is never passed | P | 1 | none | test_recovery_hang_monitor.py |
| 14 | llmapi/mpi_session.py:203 `exit_fault_tolerant_process(status=0)` | `status` is never passed (product :270, mgmn_leader_node.py:121; tests :1399/:1403) | P | 2 | none | test_mpi_session.py |
| 15 | hang_monitor.py:78-79 | the `<= 0` check cannot fail (PositiveFloat config, one caller) | P (defensive) | 2 | none | test_recovery_hang_monitor.py |
| 16 | communicator.py:988 `revoke` | `self._tp_comm is not None` is always true after the `_host_world` check (:985, setter :973-976) | S | 0 | low | test_mpi_survivors.py |
| 17 | communicator.py `shrink` | the 3-way comm choice reduces to `_shrunk[0] if _shrunk else self.tp_comm` (the property already returns `_survivor_comm`, :1233) | S | 2 | low | test_mpi_survivors.py, test_mpi_survivors_ulfm.py |
| 18 | moe_load_balancer.py:1428 | `and layer.host_copies` is always true on the recovery path (constructor :1229) | S | 1 | low | test_resident_placement.py, test_ring_host_copies.py |
| 19 | moe_load_balancer.py:450-484, :997-1023, `placement_digest(copy_plan=None)` | defaults only tests rely on (`host_copies=None`, `slot_count_per_rank=None`, `ghost_slot_count=0`, `host_copy_ranks=None`, `restored_per_rank=None` and its zeros branch). Make them required; update the tests. | S | 6 | low | test_moe_load_balancer.py, test_resident_placement.py, test_ring_restore_staging.py |
| 20 | authority.py:136-156, 492 | `fence_departed=False`, `recovery_timeout=20.0`, `config_digest=None` (and its fallback) are test-only; `main()` (:854) passes everything. About 13 test constructions change. | S | 7 | low | test_recovery_authority.py, test_recovery_hang_monitor.py |
| 21 | hang_evidence.py:208-211 `HangEvidenceReporter.__init__` | test-only defaults (`suspect_s`, `heartbeat_s`, `receive`, `pmix_ranks`); product (peer_loss_recovery.py:617-628) passes all of them | S | 4 | low | test_hang_evidence.py |
| 22 | protocol.py:42, 286-291 `encode_datagram` | only tests call it (the product encoder is C++ HeartbeatSender). Move it to a test helper and keep it as the reference encoder that test :590 compares against C++. | S | 8 | low | test_recovery_hang_monitor.py |
| 23 | client.py:132-136 `RecoveryClient.from_env` returning None | unreachable from product (both callers run after `create` checked enablement); test_recovery_authority.py:631 asserts None | S | 2 | low | test_recovery_authority.py |
| 24 | mnnvl_alltoall_workspace.py:302-322 | `_stop_watchdog()` / `synchronize()` + `_start_watchdog()` around remap guard a watch queue that bound layers never fill (bound layers skip `watch_collective`, nvlink_one_sided.py:1108/1255). **Unverified:** that no unbound client shares that workspace lifecycle. | S | 5 | med | test_mnnvl_alltoall_workspace.py, test_moe_a2a_recovery.py (GPU) |
| 25 | peer_loss_recovery.py:1792-1793 | the `own_rank in failed` raise is a defensive duplicate of `_recover`'s survivor check (:1022). Keep it unless the user wants it gone; it is cheap. | S | 2 | low | test_peer_loss_recovery.py:2761 |
| 26 | py_executor.py:5911 | `if self._is_kv_manager_v2:` before `step_released()` is redundant (`create` requires V2; the non-overlap loop :4990 calls it unguarded) | S | 1 | low | test_py_executor.py |
| — | py_executor.py ~12 `if self._peer_loss_recovery.enabled:` guards around do-nothing hooks | **Keep.** They save a Python call per step with FT off (healthy-cost priority). | — | 0 | — | — |
| — | `_free_device_bytes` (peer_loss_recovery.py:101), `_native_peer_failure_watch` (communicator.py:843) | small monkeypatch seams. **Keep:** removing them only moves the patch target in 9 tests. | — | 0 | — | — |
| — | executor_request_queue.py:70 `or self.is_wake_request` | possibly redundant, unverified (readers at request_utils.py:54, py_executor.py:2761, peer_loss_recovery.py:786). Leave it. | — | 0 | — | — |

### 3.3 Duplicated logic and reuse (main or series-internal)

| # | file:line | change | class | lines | equivalence test |
|---|---|---|---|---|---|
| 27 | client.py:132-136, fault_tolerance_status.py:107-111, hang_evidence.py:166-171 | the same "read AUTHORITY env + `load_key()`" sequence 3 times becomes one `authority_endpoint_and_key()` in client.py. Also drop the alias `AUTHORITY_ENV = FAULT_TOLERANCE_AUTHORITY_ENV` (client.py:35) and import the `_utils` name. | P | 7 | test_recovery_authority.py, test_fault_tolerance_health.py, test_hang_evidence.py |
| 28 | hang_evidence.py:58 `_FLAG_MASK` | equals main's `alltoall_watchdog._COMPLETION_FLAG_MASK`; import it | P | 1 | test_hang_evidence.py |
| 29 | communicator.py:96-104 `_failure_evidence` and mpi_session.py:95-111 | near-identical "poll `Get_failed` until timeout" loops become one `_utils.wait_failed_ranks(comm, timeout, poll_s)`. Pass the different poll intervals (1 ms / 10 ms) explicitly so behaviour stays identical. | S | 6 | test_mpi_survivors.py, test_mpi_session.py, test_mpi_survivors_ulfm.py |
| 30 | communicator.py:1109, :1155, :1375 | three copies of the departed-rank re-indexing become one helper | S | 4 | test_mpi_survivors.py:411-498 |
| 31 | mnnvl_memory.py:913-963 `checkpoint_prepare` / `release_backing` | extract the shared `_release_mapped(record)` core (synchronize, unmap+release, BROKEN on error, UNMAPPED) | S | 8 | test_mnnvl_memory_remap.py, checkpoint tests, test_mnnvl_alltoall_workspace.py |
| 32 | fault_tolerance_status.py:124-150 vs client.py:162-245 | share the "receive one authenticated message or None by a deadline" primitive. The skip/raise policy stays at each caller. | S | 6 | test_fault_tolerance_health.py, test_recovery_authority.py |
| — | the flush+SIGKILL at py_executor.py:1517-1519 and the flushes at mpi_session.py:170/271/306, vs main's `hang_detector._best_effort_flush_streams` | reuse would swallow flush errors, which is a (safer) behaviour delta | **B** | (6) | follow-up |
| — | `ProgressRowReader` / `lagging_peers` vs alltoall_watchdog readers; `remap` vs `checkpoint_restore`; `_leave_degraded_world` vs `propagate_hard_kill`; datagram HMAC vs `ZeroMqQueue` HMAC; `MapConfigurator` vs main's `UnicastConfigurator` | semantics differ by design | **B / keep** | 0 | — |

## 4. peer_loss_recovery.py (1815 lines): proposed split with exact boundaries

The file is not bloated; it is three concerns plus orchestration. Only three functions exceed 100 lines: `create`
(384-494, 111 lines), `_recover` (929-1117, 189) and `_deliver_responses` (1609-1715, 107).

**Constraint:** tests patch module attributes of `peer_loss_recovery`: `_free_device_bytes` (8x), `time.monotonic`,
`logger.*`, `fault_tolerance_config`, `is_mpi_peer_failure`, `_startup_watch`, `atexit` and
`peer_loss_recovery_enabled`. Also, `mpi_session.py:165-166` looks the module up **by name**
(`sys.modules.get("tensorrt_llm._torch.pyexecutor.peer_loss_recovery")`) to call `stop_startup_watch`. Every move
must update those patch targets and that string, or the tests silently patch a module that no longer uses the
name.

| New module | Takes (HEAD lines) | ~lines | class | notes |
|---|---|---|---|---|
| `pyexecutor/recovery_hook.py` | type aliases `Responses`, `ResponseContribution`, `StepChange` (80-91); `PreparedStep` (201-224); `RecoveryHook` (227-285) | ~120 | P (pure move) | Imported by py_executor.py:113 and test_kv_pool_rebalance.py. Gives the FT-off executor contract its own small file. `_T` is duplicated (1 line). |
| `pyexecutor/peer_loss_startup.py` | `_REGISTRATION_TIMEOUT_S` (78), `_tested_spec_dec` (105-116), `_register` (135-154), `_startup_watch` / `watch_startup_peer_failures` / `_end_job_before_serving` / `stop_startup_watch` (157-198), and the body of `create` 399-494 as `check_and_register(executor) -> Optional[(client, admitted)]` | ~230 | P for the module-level functions; S for carving `create` (same statements, new function boundary) | `PeerLossRecovery.create` shrinks to about 6 lines. Update the mpi_session.py:166 string, py_executor_creator.py:59, and the test patches of `_startup_watch`/`atexit`/`peer_loss_recovery_enabled`/`fault_tolerance_config`. |
| `executor/recovery/config.py` (existing) | `FailureDomain` (88) + `failure_domains` (119-132) | moves ~16 | P | **This fixes the layering import** (section 5). |
| `pyexecutor/peer_loss_requests.py` | the request ledger: state 344-381; hooks 756-926 (`fetching_requests` … `defer_terminations`); 1529-1745 (`_departed_requests`, `_requeue_unscheduled_requests`, `_started`, `_record_started_owners`, `_quarantine`, `_deliver_responses`, `_readmit_undelivered_requests`); helpers 1747-1764 | ~470 | S | Two options. (a) **Mixin** `_RequestLedger`, which `PeerLossRecovery` inherits: methods move verbatim and `self.*` names are unchanged, so the ~70 test accesses (`recovery._request_owners` x27, …) keep working. Close to P; AST-equal per moved method. (b) **Composition** `RequestLedger` owned by the hook: a cleaner boundary, but ~70 test edits and new delegation, so S. **Recommend (a) for this cleanup and (b) as a follow-up if reviewers ask.** |
| `pyexecutor/peer_loss_memory.py` (optional) | `_GHOST_MARGIN_BYTES`, `_REMAP_RETURN_INTERVAL_S`, `_free_device_bytes`; `_release_uninstalled_backing`, `_restore_placements`, `_back_restored_slots`, `_make_room`, `_remap_workspaces`, `serving_idle`, `_return_remap_room`, `_stage_ring_restore` (1290-1483); `install_recovery_placement` (1767-1802) | ~250 | S (mixin as above) | The 8 `_free_device_bytes` patches move here. Optional: it leaves the orchestration at ~750 lines either way. |
| `peer_loss_recovery.py` (remains) | `PeerLossRecovery`: `__init__` (non-ledger state), `create`, `validate_request`, `run`, peer watch, hang evidence, step hooks (587-753), `_recover`, `_log*`, `_wait_for_proposal`, `_end_ranks`, `_shrink*`, `_check_recoverable`, `_watch_survivors`, `_rank0_placements`, `_among_survivors`, `_join_device_work`, `_check_deadline`, `_expect_generation` | ~750 (with memory split) / ~1000 (without) | — | |

`_recover` (189 lines), S, optional: extract the plan loop 988-1050 into `_agree_on_plan(failed, deadline) ->
(failed, generation, placements)` and the install/commit section 1052-1098 into `_install_and_commit(...)`. This
is the same statement order and roughly +15 lines. Equivalence: `test_peer_loss_recovery.py` (it drives `_recover`
end to end with `_FakeClient`) plus `multi_gpu/test_moe_a2a_recovery.py`.

Net effect of the split: about +40 to +60 lines (imports, headers, the `_T` duplicate), with no behaviour change.
The value is reviewability, not size. **User decision D2:** split now (inside PR 20/21, which makes the largest PR
even larger to review as a move) or as a follow-up PR right after landing. The monitor leans to the follow-up:
(i) move-only PRs are easy to review on their own, and (ii) it keeps the restacked PRs closer to the benchmarked
bytes. The layering move (section 5) should happen now regardless.

## 5. Layering import (P)

`moe_load_balancer.py:1168-1169` lazily imports `failure_domains` from `tensorrt_llm._torch.pyexecutor.peer_loss_recovery`.
That is a MoE module reaching up into the executor package, and it keeps PR_SPLIT leftover #8 open.

- **Fix:** move `FailureDomain` + `failure_domains` (peer_loss_recovery.py:88, 119-132) to
  `tensorrt_llm/executor/recovery/config.py`, next to `FaultToleranceConfig.failure_domain`.
  `moe_load_balancer.py` already imports `fault_tolerance_config` from it (:27), so the lazy import becomes a
  top-level one.
- **Callers to update:** peer_loss_recovery.py:478, moe_load_balancer.py:1168, and the test
  test_peer_loss_recovery.py:2577-2586 (or move that test to test_recovery_config.py).
- **Where it lands:** in `d12f179f47` (PR 14), where the stub first appears, so no commit ever has the upward import.
- **Lines:** about 0 net. **Proof:** AST equality of the function body; `test_recovery_config.py`,
  `test_resident_placement.py`, `test_ring_host_copies.py`.

## 6. Env-var configuration (user decision D3; do not decide in the cleanup)

| Variable | Read at | Role |
|---|---|---|
| `TLLM_FAULT_TOLERANCE_MODE` | `_utils.py:520` (uncached), shell launcher :172, C++ via main's `mpi::isFaultToleranceModeEnabled()` (moeLoadBalancer.cpp:511, :988, :1207 latched; hostAccessibleDeviceAllocator.cpp:251 static) | **already main's env var** |
| `TLLM_FAULT_TOLERANCE_AUTHORITY` | `_utils.py:536` (lru_cache), client.py, fault_tolerance_status.py, hang_evidence.py, launcher | — |
| `TLLM_FAULT_TOLERANCE_AUTHORITY_KEY` | client.py:62 | a secret |
| `TLLM_FAULT_TOLERANCE_CONFIG` | config.py:143 | YAML/JSON with 10 `FaultToleranceConfig` fields, digest-checked between workers and the authority |
| internal / runtime | `OMPI_MCA_mpi_ft_enable`, `OMPI_MCA_pml`/`btl`, `PMIX_NAMESPACE` | not candidates |

- No test-only env vars are read by product code. `WIDEEP_FT_TEST_*` and `TLLM_REQUIRE_ULFM` are read only in
  tests.
- C++ adds no new `getenv`.

Options:
- **A. Keep env** (cost about 0). P sub-cleanups: centralise the env names (item 27) and, optionally, latch MODE
  once in the C++ `MoeLoadBalancer` constructor (S, with a rebuild).
- **B. Mixed.** Move only the 10 tuning fields into a `TorchLlmArgs` prototype field. MODE, AUTHORITY and KEY stay
  env, because bash, MPI init, C++ statics and MGMN workers read them before LlmArgs exists, and KEY is a secret.
  This is **class B**: it adds a public field, which per AGENTS.md needs the golden manifest regenerated and
  telemetry/privacy CODEOWNER approval, the serve JSON schemas regenerated, the api_stability reference updated,
  and test_recovery_config.py, the integration harness and the docs reworked. The authority still needs the
  values from a file or CLI for its digest check. About +30/-15 code lines plus about 150 lines of tests and docs.
- **C. Everything in LlmArgs:** not workable without changing main's MODE handling.

The monitor's read: A now, with B as a follow-up PR if owners ask. The PR 5 description must state the choice.

## 7. C++ items (all need a rebuild + `moeLoadBalancerTest` gtest + the Python op tests)

| # | file:line | change | class | lines | risk |
|---|---|---|---|---|---|
| C1 | comment items in 3.1 (C++ rows) + moeLoadBalancer.cpp:662 | wording | P | ~30 | none |
| C2 | nanobind/runtime/moeBindings.cpp:24 | unused `#include <nanobind/stl/pair.h>` | P | 1 | low |
| C3 | hostAccessibleDeviceAllocator.cpp:397-401 `registerHostAccessible` NUMA branch | dead for every caller (reached only when `gpuMemoryNumaId() < 0`, ghostSlotMemory.cpp:311/:442); replace with `TLLM_CHECK` | P | 4 | low |
| C4 | hostAccessibleDeviceAllocator.cpp:383-387 `memcpyToDeviceAddress` memcpy branch | dead for current callers; optional | P/S | 4 | low |
| C5 | peerFailureWatch.cpp:56/195 `mComm` | only read in the constructor, so make it a local; optionally reuse `MpiProgressThread::commFromFortranHandle` | P | 2 | none |
| C6 | ghostSlotMemory.cpp:43 `AccessDescs` vector | always one descriptor; use a single `CUmemAccessDesc` like main's `UnicastConfigurator` | S | 8 | low |
| C7 | moeAlltoAllOp.cpp ~:797 and ~:1013; ~:575-586 and ~:807-809 | duplicated mask/abort resolution and expert_to_slot checks become `resolveRankMaskAndAbort` / `resolveExpertToSlot` helpers | S | ~20 | low |
| C8 | moeLoadBalancer.cpp:511, :988, :1207; hostAccessibleDeviceAllocator.cpp:251 | 4 separate MODE env reads become one latch in the constructor (identical unless the env changes mid-process) | S | 3 | low |
| — | moeAlltoAllKernels.cu:1393 `SWITCH_BOOL(USE_EXPERT_TO_SLOT)` doubles fence-dispatch instantiations from 96 to 192; 72 are unreachable | nest the switch under `!EPLB && MASK` + `TORCH_CHECK` | **B** (op rejects inputs it accepts today) | — | follow-up |
| — | A19 `restorePlacement` CPU copy threads idle when `slotsLoaded` | **S/med**; needs restore tests on both copy paths | follow-up |
| — | routing op one-slot kernel (moeLoadBalanceOp.cpp:164-170) | B, needs a perf number | follow-up |
| — | FLAG_VAL 8 B (A18) | needed (the released-round marker); disclose in the PR, no change | keep |

Gap: `PeerFailureWatch` has no unit test, in C++ or Python. Note it in PR 8 and add one in the follow-up.

**Do C1-C5 only if a C++ rebuild happens anyway. C6-C8 only with a planned rebuild and the listed tests.**
Otherwise all go to the follow-up list. Note the series is frozen and benchmarked: a C++ rebuild means a new wheel.

## 8. Follow-up PRs (class B, not in this cleanup)

1. Env tuning fields to LlmArgs (D3 option B).
2. Typed exception for bad HMAC in main's `ipc.py`, instead of `client.unauthenticated()` matching error text
   (client.py:74-82).
3. `_best_effort_flush_streams` reuse in py_executor/mpi_session flush paths.
4. Trim the `USE_EXPERT_TO_SLOT` instantiations; A19 copy threads; the routing-op one-slot kernel (perf-gated).
5. A15 (shm mappings leak per FT-on weight reload, LB:313-315).
6. `split_unservable_requests` on every FT-on step with arrivals (py_executor.py ~6793).
7. RequestLedger composition (section 4, option b), if reviewers ask.
8. `MapConfigurator` folded into main's `UnicastConfigurator` (changes main's behaviour on `cuMemSetAccess` failure).
9. Send PR 1-3 and PR 19 (ADP burst rule) to main, not to the feature branch.

## 9. User decisions needed

- **D1:** validation bar for S items under the code-freeze rule (unit tests only, or a 1-node smoke, or a VAL
  rerun). P items are proven by AST equality and need no rerun.
- **D2:** split peer_loss_recovery.py now (mixin moves inside PRs 20/21) or as a follow-up move-only PR. The
  layering move (section 5) is recommended now either way.
- **D3:** env versus LlmArgs for the 10 tuning fields (A keep / B mixed). MODE, AUTHORITY and KEY must stay env.
- **D4:** R2: merge PR 10 + PR 12 (recommended), or add a throwaway CPU-to-device mask copy to PR 10.
- **D5:** whether C++ P/S items ride along (only if a rebuild is planned).

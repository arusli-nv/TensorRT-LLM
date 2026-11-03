# v5-pr (b27007aca1) vs v5-final (27442bd91d): behaviour equivalence

Source: a read-only hunk-by-hunk review on 2026-10-25, run by an agent and spot-checked by the monitor. Method:
`git diff 27442bd91d v5-pr`, then `git grep` at both refs for the callers of anything removed or changed.

**Verdict: equivalent with FT on and with FT off.** No hunk changes runtime logic, defaults, timeouts, log or
exception text, or wire formats (heartbeat datagram, HMAC, `STALLED_EXIT_CODE`, fence grace).

| File | Change |
|---|---|
| executor/recovery/config.py, _torch/pyexecutor/peer_loss_recovery.py | `failure_domains` / `FailureDomain` moved to config.py (identical body), re-imported |
| _torch/moe/fused_moe/moe_load_balancer.py | lazy import of `failure_domains` -> module import from config.py; docstrings |
| _torch/pyexecutor/hang_evidence.py | local 32-bit mask -> `alltoall_watchdog._COMPLETION_FLAG_MASK` (same value); `AUTHORITY_ENV` alias -> `FAULT_TOLERANCE_AUTHORITY_ENV` (same object) |
| _utils.py | `is_mpi_peer_failure_class` inlined (single caller) |
| executor/recovery/client.py, serve/fault_tolerance_status.py | `AUTHORITY_ENV` alias -> `FAULT_TOLERANCE_AUTHORITY_ENV` |
| executor/recovery/protocol.py | test-only `encode_datagram` moved into the test |
| executor/recovery/hang_monitor.py | unused `count=2` parameter of `live_helpers` dropped; the only caller never passed it (monitor verified) |
| llmapi/mpi_session.py | unused `status` parameter of `exit_fault_tolerant_process` dropped; all callers pass none (monitor verified) |
| C++ heartbeatSender, peerFailureWatch, ghostSlotMemory | license header shortened to SPDX form (339 cpp headers in the repo use it) |
| cpp/tensorrt_llm/runtime/CMakeLists.txt | source list sorted, same set |
| docker/common/install_mpi4py.sh | header restored to main's |
| 12 other product files | docstrings, comments, wrapping only |

Residual: the new module-level import path is not GPU-run. `tools/import_smoke.sh` in the container covers it.
Per-commit tests: test_ring_host_copies.py passes at the folded host-copy commit and at the next commit that touches
those files (V5PR-FOLD-TEST 8275622). The tip tree equals the backup's (v5-pr-backup-19b0e90251).

GPU tests of v5-pr (2026-10-25):
- V5PR-SMOKE 8276387 (8 nodes, overlays/v5pr-b27007aca1 = v5-pr's changed .py over build-freeze-58caa0fde6):
  FT-off online serve PASS; online rank5 kill PASS (resume 1.21 s, throughput ratio 0.95, 31/32 live; v5-final
  measured 1.21/1.20 s, CLAIMS 193).
- V5PR-TESTS (1 node): import smoke PASS; per-commit unit tests [pending, 8276994]. Test infrastructure note:
  tests/integration/defs/*.py and tests/integration/__init__.py are copied from the v5-pr worktree into the
  exports (git archive of tests/integration stops at an LFS object). They are test infrastructure, not under
  test.

Update 2026-10-25 07:55 PT:
- test_mpi_session's one failure is environmental. test_remote_mpi_session[flashinfer_temporary_cleanup] passes with a
  short pytest basetemp on the v5-pr tip and on v5-final (zmq ipc path limit). The whole file passes 76/76 at the
  executor-recovery commit (rerun B 8279014).
- Per-commit testing found split errors, fixed by restacks with the tip tree unchanged (backup v5-pr-backup-b27007aca1):
  ring-copy pieces in the extra-slots commit (moved to the ring commit); the executor-recovery commit calling methods
  from the next commit (the two squashed); an attention-DP test calling a later method (moved).
- Rerun B 8279014 (restacked commits): KV rollback, context hold, executor recovery, /health pass; attention-DP
  routing passes except the moved test.
- Reruns A1 8279013 and A2 8279691 [pending].

# WideEP fault tolerance: PR description drafts (for review; not opened)

Source: `/tmp/claude-161238/wt-split`, branch `v5-final`. The code of record is `7e8dfb37df`, on top of upstream main
`9cc37929e8`, plus the doc-only commit `6e311f2f04`. Target branch: `feat/wide-ep-ft-mvp`.
Inputs: `review/PR_SPLIT.md`, `review/CODE_AUDIT_20261023.md`, `docs/source/features/wide-ep-fault-tolerance.md`
(FD), and the commits' own "FT-off guards" notes. Drafted 2026-10-23. Nothing was pushed, opened or committed.

**The branch moved while this was drafted.** `v5-final` now ends at `65741c820a`, a test-only commit made at
09:42 PT ("Give the multimodal executor fake the step-change helper..."). It is included below (PR 16). It does
not change the shipped code.

"FT off" means `TLLM_FAULT_TOLERANCE_MODE` is unset. All results are `[FINALS: ...]` placeholders, to be filled
only from finalized CLAIMS rows.

---

## A. Split problems: PR_SPLIT.md vs the real series

PR_SPLIT.md was written for `v5-zero` (378 commits on `0829689c52`). The real series is 29 commits on `9cc37929e8`,
one commit per planned PR, with the changes below.

### A1. Commits that do not stand alone (must be fixed before review)
1. **`e74cadb97c` breaks FT-off overlap serving until `ddc2658518`.** `e74cadb97c` adds an unguarded
   `self._note_step_change(request)` call to `_update_generation_requests_that_will_complete_next_iteration`.
   `_note_step_change` then calls `self._step_changes.append(...)`, but `_step_changes` is `None` when recovery
   is off. So, with FT off and the overlap scheduler, the first request that "will complete next iteration"
   raises `AttributeError`. `db6f10dc0b` inherits the bug. `ddc2658518` fixes it, and `65741c820a` fixes the
   multimodal test fake that the fix broke. **Squash `ddc2658518` and `65741c820a` into `e74cadb97c`.**
   Rebasing `db6f10dc0b` over that squash touches neighbouring context in `py_executor.py`, so expect a trivial
   conflict.
2. **`34611ef619` (A2A kernel) breaks main's existing rank-mask path until `2b5005b9a1`.** The op now requires
   `active_rank_mask` to be a CUDA tensor on the workspace device; the check is
   `t.is_cuda() && t.device() == workspace.device()`. Main's `AllToAllWatchdogCoordinator.capture_active_rank_mask`
   still builds the mask with `device="cpu"`, and `nvlink_one_sided.py` passes that tensor through. With
   `EPGroupHealth` or `TLLM_FAULT_TOLERANCE_MODE=1`, every rank-mask dispatch therefore fails at
   `34611ef619` and `6f5f1d080b`. `2b5005b9a1` is the first commit that publishes the mask into a device record.
   FT off (no rank mask) is not affected. Fix: either merge P5a and P5b (PRs 6 and 8 below) into one PR, or have
   P5a copy a CPU mask to the device.
3. **`17a047ea76` (ADP burst rule) cannot go to main as written.** It uses the local `live_ranks` that
   `9cf51e9801` introduces. To send it to main on its own, rebase it onto main and use `self.dist.tp_size`.

The plan's "build-alone" check was done statically, not by building or importing anything. For each commit, every
`tensorrt_llm` `from ... import` in every Python file the series touched was resolved against that commit's tree;
none were missing. C++ ordering looks consistent: `heartbeatSender` comes before `peerFailureWatch`, and the C++
load balancer comes before the Python EPLB code. Nothing was built per commit, so a per-commit C++ build plus a
pytest collection run is still owed.

### A2. Moved or merged relative to PR_SPLIT.md
- **P14 hang detection is not separable or deferrable as planned.** It is spread across five commits:
  - the hang messages in `protocol.py` (`63ae2e1de3`);
  - `hang_monitor.py` and its 661-line test, plus the hang paths in `authority.py` (`2f2d2f6e85`);
  - worker evidence, `heartbeatSender` and `pmix_fence` (`9ff4fe2959`, ordered *before* ULFM);
  - the wake request and `waiting_for_requests` flag (`9cf51e9801`);
  - fence-before-plan and the evidence start-up (`e74cadb97c`).

  Deferring it would mean rewriting four PRs.
- **P7a and P7b are one commit** (`ecfd7d3934`, 2774 lines: 1672 source, 1102 gtest). The host-accessible
  ghost-slot chain (GDRCopy, NUMA probe) that PR_SPLIT suggested splitting off or dropping is kept; FD documents
  it as the online-EPLB `ghost` path.
- **P12a and P12b are one commit** (`e74cadb97c`, 4596 lines: 1714 source, 2882 tests). It also carries the
  `llm_request.py` step-change fields (planned for P10), `test_moe_a2a_recovery.py` (planned for P6) and the
  hang fence-before-plan code (planned for P14b). It is the largest review unit; PR_SPLIT's split options
  (i)/(ii) still apply.
- **P15's kernel half is in P5a.** The `USE_EXPERT_TO_SLOT` template path, `expert_to_slot` /
  `num_routed_experts` and the op argument are all in `34611ef619`. `b0a8905b44` only adds the one-slot-table op,
  the scheduler and the caching. Dropping P15 would leave a dead template dimension in P5a.
- The `cuda_graph_runner.py` change (planned for P5b) is the attention-DP vote's `departed=`, in `9cf51e9801`.
- `failure_domains()` lands as a 24-line `peer_loss_recovery.py` stub in `d12f179f47`.
  `moe_load_balancer.py` still lazily imports `pyexecutor.peer_loss_recovery` (`moe_load_balancer.py:1168`), so
  PR_SPLIT leftover #8 (the layering fix) is still open.
- Fix-ups that should be squashed:
  - `ddc2658518` and `65741c820a` into `e74cadb97c` (see A1);
  - `7e8dfb37df` into `ffc8bc30c3`;
  - `e25e638054` and `6e311f2f04` into `c86566c184`.
- `3839e08829` (`/health`) also adds 5 authority status tests to P4's `test_recovery_authority.py`, and 72 lines
  to P0c's `test_stream_error_events.py`. That is fine, since both depend on the earlier PRs.

### A3. Commits that mix concerns
- `2f2d2f6e85`: the recovery authority plus authority-side hang confirmation and fencing (see A2).
- `9cf51e9801`: attention-DP absent ranks, plus hang-detection plumbing (`WAKE_REQUEST_ID`, `wake()`,
  `waiting_for_requests`), plus `ModelEngine.moe_execution_state` / closing the state on teardown (P5b
  lifecycle).
- `e74cadb97c`: the executor recovery hook, the PeerLossRecovery orchestration, KV-cache-event refusal and the
  hang fence. All are recovery, but the PR is too large.
- `17a047ea76`: not fault tolerance at all. It is kept as its own PR (PR 15).

### A4. PR_SPLIT leftovers, checked against the final tree
| # | Leftover | State at `7e8dfb37df` |
|---|---|---|
| 1 | BAR1 log and module-level `pynvml` import | gone |
| 2 | per-signature dispatch-part info log | gone |
| 3 | host-accessible ghost-slot chain | kept, and documented |
| 4 | `TLLM_FAULT_TOLERANCE_MODE` read in C++ | still env: `mFaultToleranceMode = mpi::isFaultToleranceModeEnabled()` at `finalizeModel` (main's helper), not a constructor flag |
| 5 | env plus side-YAML config instead of `LlmArgs` | unchanged; needs a decision in the PR 5 description |
| 8 | `failure_domains` import cycle | open |
| 9 | history-narrating comments | some remain (for example `request_utils.py` "instead of leaving it queued forever") |

---

## B. PR list (landing order)

| # | Title | Commits | Lines (src/tests) |
|---|---|---|---|
| 1 | [None][fix] Accept the real arguments in the fake moe_a2a op registrations | 1bfd815de0 | 4/0 |
| 2 | [None][fix] Share EPGroupHealth across ConfigurableMoE's model-config deepcopy | 68c6feb920 | 4/28 |
| 3 | [None][fix] End every failed OpenAI stream with one error event and one [DONE] | 7b46bf9bdb | 142/65 |
| 4 | [None][infra] Upgrade mpi4py to 4.1.2 for ULFM | ba94614ca0 | 101/0 |
| 5 | [None][feat] Add the WideEP recovery protocol, config and client | 63ae2e1de3 | 774/106 |
| 6 | [None][feat] Add the external WideEP recovery authority | 2f2d2f6e85 | 1089/1755 |
| 7 | [None][feat] Detect a hung WideEP rank with heartbeats, hang evidence and PMIx fencing | 9ff4fe2959 | 702/314 |
| 8 | [None][feat] Add ULFM survivor communicators for host collectives | 372f2ae908 | 901/780 |
| 9 | [None][feat] Keep the MPI session serving and exit cleanly after a WideEP peer loss | 9e3c0be56d | 410/669 |
| 10 | [None][feat] Read the NVLinkOneSided rank mask through a pointer and add an abort record | 34611ef619 | 459/515 |
| 11 | [None][feat] Remap the MNNVL all-to-all workspaces at the same addresses among the survivors | 6f5f1d080b | 183/311 |
| 12 | [None][feat] Bind NVLinkOneSided layers to a shared MoeExecutionState for WideEP recovery | 2b5005b9a1 | 356/893 |
| 13 | [None][feat] Restore lost experts into EPLB slots backed by ghost slot memory (C++ load balancer) | ecfd7d3934 | 1672/1102 |
| 14 | [None][feat] Append extra EPLB slots and restore lost experts into them (Python) | d12f179f47 | 655/742 |
| 15 | [None][feat] Keep WideEP EPLB host copies in a ring of nodes and stage restores from them | 766b92719f | 275/880 |
| 16 | [None][perf] Tune and key the MoE tactics on the loaded slots under WideEP recovery | e9c313962a | 242/238 |
| 17 | [None][feat] Roll back a discarded step's KV cache allocations in KV Cache Manager V2 | abb38ab298 | 60/313 |
| 18 | [None][feat] Keep attention-DP routing serving with absent ranks | 9cf51e9801 | 198/434 |
| 19 | [None][perf] Release an attention-DP context hold once a burst of arrivals has ended | 17a047ea76 | 30/77 |
| 20 | [None][feat] Recover the PyTorch executor from a WideEP peer loss | e74cadb97c + ddc2658518 + 65741c820a | 1720/2899 |
| 21 | [None][feat] Run a lost rank's undelivered requests again on the survivors | db6f10dc0b | 450/796 |
| 22 | [None][feat] Report WideEP fault-tolerance state on /health and answer peer-loss failures with 503 | 3839e08829 | 199/246 |
| 23 | [None][perf] Look up dispatch slots in a device expert-to-slot table under WideEP recovery | b0a8905b44 | 123/293 |
| 24 | [None][test] Add WideEP peer-loss recovery integration tests | ffc8bc30c3 + 7e8dfb37df | 5/896 |
| 25 | [None][doc] Document WideEP fault tolerance | c86566c184 + e25e638054 + 6e311f2f04 | docs 415 |

PRs 1-3 are fixes to main's own code, and PR 19 changes scheduling for every user. They are better sent to `main`
than to the feature branch; for the MVP they are drafted against `feat/wide-ep-ft-mvp`. If the review prefers it,
PRs 10 and 12 should merge (A1.2).

Mapping to PR_SPLIT ids:
- PR 1-4 = P0a-d; 5 = P3; 6 = P4 (+P14b); 7 = P14a; 8 = P1; 9 = P2; 10 = P5a (+P15 kernel); 11 = P6;
  12 = P5b; 13 = P7a+P7b; 14 = P8a; 15 = P8b; 16 = P9; 17 = P10; 18 = P11; 19 = the ADP burst rule split out of
  P12a; 20 = P12a+P12b; 21 = P12c; 22 = P13; 23 = P15 (rest); 24 = P16; 25 = P17.

---

## C. PR descriptions

Each description below is ready to paste. Delete the "Commits/files" block before posting if it is not wanted.

---

### PR 1: [None][fix] Accept the real arguments in the fake moe_a2a op registrations

**Commits/files:** `1bfd815de0`. `tensorrt_llm/_torch/custom_ops/cpp_custom_ops.py` (+4).

**Summary**
The fake (meta) registrations of `trtllm::moe_a2a_dispatch` and `moe_a2a_combine` lacked arguments that the real
ops in `moeAlltoAllOp.cpp` already declare: `use_cft_counted_writes`, `expert_id_payload_index` and
`invalid_token_expert_id`. A fake-tensor trace that passes these arguments therefore failed. This PR adds them with
the real defaults.

**FT-off impact vs main:** none. These are fake registrations only, matching the real schema.

**Test plan**
- The existing MoE all-to-all fake-tensor coverage (`tests/unittest/_torch/moe/test_moe_comm.py`).
- PR 10 exercises the fakes under `FakeTensorMode` (`tests/unittest/_torch/moe/test_moe_a2a_abort.py`).

**Disclosures:** this is a bug on main and could be sent to main directly.

**Follow-ups:** none.

**Results:** not applicable.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

### PR 2: [None][fix] Share EPGroupHealth across ConfigurableMoE's model-config deepcopy

**Commits/files:** `68c6feb920`.
- `tensorrt_llm/_torch/moe/fused_moe/ep_group_health.py` (+4);
- `tests/unittest/_torch/moe/fused_moe/test_configurable_moe.py` (+28).

**Summary**
`ConfigurableMoE` deep-copies the model config for per-layer quant overrides. Without a `__deepcopy__`, each copy
gets its own `EPGroupHealth`, so a membership change reaches only some layers. `EPGroupHealth.__deepcopy__` now
returns `self`. It is shared membership state on purpose: every layer has to see one health object.

**FT-off impact vs main:** none in practice. Main only reads `EPGroupHealth` in rank-mask mode.

**Test plan**
- `test_configurable_moe.py::test_layer_quant_override_shares_ep_group_health` (in the `l0_cpu`
  `unittest/_torch/moe` entry).

**Disclosures:** the identity-returning `__deepcopy__` is intentional and documented in the class.

**Follow-ups:** none.

**Results:** not applicable.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

### PR 3: [None][fix] End every failed OpenAI stream with one error event and one [DONE]

**Commits/files:** `7b46bf9bdb`.
- `tensorrt_llm/serve/openai_server.py` (`_stream_error_events`, `_end_stream_on_error`);
- `tests/unittest/llmapi/apps/test_stream_error_events.py`;
- `l0_cpu.yml` (+1).

**Summary**
A streaming `/v1/completions`, `/v1/chat/completions` or `/v1/responses` request that fails after HTTP 200 was
previously cut off. It now ends with exactly one SSE `error` event followed by one `[DONE]`, so OpenAI clients raise
an `APIError` instead of seeing a truncated stream. PR 22 builds the retryable peer-loss code on top of this.

**FT-off impact vs main:** **yes, for every user.** A failed stream's ending changes from a truncated stream to an
error event plus `[DONE]`. Streams that complete are unchanged.

**Test plan**
- `test_stream_error_events.py`, 4 tests in this PR:
  - `test_failed_stream_ends_with_an_error_event_then_done`;
  - `test_a_stream_failing_mid_way_ends_with_one_error_event_and_one_done`;
  - `test_a_failure_after_done_sends_no_second_done`;
  - `test_a_stream_that_completes_passes_through_unchanged`.
- Runs in `l0_cpu`.

**Disclosures:** this is a user-visible behaviour change on the serve path and needs serve CODEOWNER review. It is a
candidate for main.

**Follow-ups:** none.

**Results:** not applicable.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

### PR 4: [None][infra] Upgrade mpi4py to 4.1.2 for ULFM

**Commits/files:** `ba94614ca0`. `docker/common/install_mpi4py.sh` (+47/-54).

**Summary**
ULFM survivor communicators need `MPI.Comm.Shrink`, which first appears in mpi4py 4. This PR installs mpi4py 4.1.2
and makes two related changes:
- the `TRTLLM_USE_MPI_KVCACHE` futures patch moves from `mpi4py/futures/_lib.py` to `_core.py`, because mpi4py 4
  moved that code;
- the `setuptools<69` pin is dropped.

**FT-off impact vs main:** **every image and every MPI path** (MPICommExecutor, disaggregated MPI KV cache). This
is not measured against stock main. In the finals every arm, main included, installs mpi4py 4 through
`install_wheel.sh`, so there is no delta inside the finals: `[FINALS: main-parity arm, mpi4py 4 in all arms]`.

**Test plan**
- Rebuild the image and run full CI. There is no dedicated unit test.
- The ULFM paths fail closed on mpi4py < 4 (startup checks for `MPI.Comm.Shrink`; see PR 8).

**Disclosures:** needs infra/CODEOWNER sign-off and an image tag bump.

**Follow-ups:** measure stock main (mpi4py 3.1.5) against mpi4py 4.1.2 if infra asks for it.

**Results:** `[FINALS: none planned for this PR]`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

### PR 5: [None][feat] Add the WideEP recovery protocol, config and client

**Commits/files:** `63ae2e1de3`.
- `tensorrt_llm/executor/recovery/{__init__,config,protocol,client}.py`;
- `_utils.py` (`fault_tolerance_mode`, `peer_loss_recovery_enabled`);
- `executor/utils.py`;
- `executor/ipc.py` (`socket_options`);
- `tests/unittest/executor/test_recovery_config.py`;
- `l0_cpu.yml`.

**Summary**
This PR adds the shared pieces that the workers and the external recovery authority use:
- `FaultToleranceConfig`: Pydantic, loaded from the optional `TLLM_FAULT_TOLERANCE_CONFIG` YAML, and compared by
  digest between the workers and the authority.
- The message protocol: pickled dataclasses over ZeroMQ, with an HMAC and a version check, a bounded message size,
  and the hang-detection messages used by PRs 6/7.
- `RecoveryClient`: a worker's connection to the authority.
- `fault_tolerance_mode()` (true with `TLLM_FAULT_TOLERANCE_MODE=1`) and `peer_loss_recovery_enabled()` (true when
  an authority is also set), which every later PR's FT guard uses.

**FT-off impact vs main:** none.
- New modules only. Both helpers return `False` unless `TLLM_FAULT_TOLERANCE_MODE=1`.
- `RecoveryClient` refuses to start without recovery.
- `ZeroMqQueue(socket_options=None)` sets no options.

**Test plan**
- `tests/unittest/executor/test_recovery_config.py` (7 tests; `l0_cpu`).

**Disclosures**
- **Configuration is env vars plus a side YAML, not an `LlmArgs` field.** The authority is a separate process
  that has to validate the same settings, so this is deployment-level config. A reviewer may ask for
  `TorchLlmArgs`. If so, the golden manifest, the serve schemas and telemetry CODEOWNER approval apply (AGENTS.md).
- The wire format is `pickle` with HMAC-SHA256 and is not encrypted. The key is any hex key of at least 16 bytes.
- The hang messages ship here even though hang detection is off by default.

**Follow-ups:** decide between `LlmArgs` and a documented deployment config before merging to main.

**Results:** not applicable.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

### PR 6: [None][feat] Add the external WideEP recovery authority

**Commits/files:** `2f2d2f6e85`.
- `tensorrt_llm/executor/recovery/authority.py` (876);
- `hang_monitor.py` (213);
- `tests/unittest/executor/test_recovery_authority.py` (1092);
- `test_recovery_hang_monitor.py` (661);
- `l0_cpu.yml`.

**Summary**
This PR adds a standalone process (`python -m tensorrt_llm.executor.recovery.authority`) that orders recoveries:
- workers register at startup and report a loss;
- the authority collects the survivors' stop reports, proposes one plan, and has the survivors install and
  commit it, within `authority_timeout_s`;
- it answers status queries for `/health`.

With `hang_detection`, it also confirms a hung rank from heartbeats and device-wait evidence (`hang_monitor.py`),
and has the rank fenced, or reports a deployment-wide stall as terminal. It holds no GPU.

**FT-off impact vs main:** none. Nothing starts the authority unless an operator runs it.

**Test plan**
- `test_recovery_authority.py` (57 tests).
- `test_recovery_hang_monitor.py` (36 tests).
- Both run in `l0_cpu`.

**Disclosures**
- Operators must run one more process. While it is unreachable, workers keep serving but report `unprotected`,
  and a loss then ends the job.
- An authority restarted under a running job loses the deployment.
- Any host that can reach its port can disrupt recovery even without the key: by connecting with a worker's
  routing id, it takes over the authority's messages to that worker. Bind it to a private interface only.
- `--fence-command` runs an operator template (`{host}`/`{pid}`) without a shell; it needs a security read.
- Flood bounds are per connection.
- Two ranks hung at once are not confirmed.

**Follow-ups:** none specific to this PR.

**Results:** `[FINALS: recovery protocol stage times stop/agree/install, run, scale, n]`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

### PR 7: [None][feat] Detect a hung WideEP rank with heartbeats, hang evidence and PMIx fencing

**Commits/files:** `9ff4fe2959`.
- `cpp/tensorrt_llm/nanobind/runtime/heartbeatSender.{cpp,h}`, plus bindings and CMake;
- `tensorrt_llm/_torch/distributed/pmix_fence.py`;
- `tensorrt_llm/_torch/pyexecutor/hang_evidence.py`;
- `tests/unittest/_torch/executor/test_hang_evidence.py`.

**Summary**
This is the worker side of hang detection (`hang_detection`, default `false`):
- A native heartbeat thread runs without the GIL and sends UDP datagrams with HMAC.
- A Python evidence thread reports host waits and all-to-all device waits that last longer than
  `hang_suspect_s`.
- A rank the authority confirms hung is fenced in this order: it kills itself; else, 1 s later, another rank kills
  it through PMIx job control; else the authority's fence command runs.

**FT-off impact vs main:** none. Everything starts only with recovery and `hang_detection`.

**Test plan**
- `test_hang_evidence.py` (16 tests; needs CUDA). It is picked up by the `unittest/_torch/executor` directory
  entries on the B300 lists.
- The integration test `test_hung_rank_fenced` (PR 24).

**Disclosures**
- A frozen rank is fenced only after about 35 s (`heartbeat_timeout_s` 5 + `hang_confirm_s` 30, the defaults),
  and serving stalls until then.
- Two ranks hung at once are not confirmed.
- A partition between the authority and a node looks like a stop.
- `pmix_fence.py` binds libpmix through ctypes with fixed struct layouts (`_PMIX_MAX_NSLEN=255`), which is fragile
  across PMIx versions.
- Hang evidence may read workspace rows that are being used for ring-restore staging (evidence only).

**Follow-ups:** none specific to this PR.

**Results:** `[FINALS: SIGSTOP-to-serving time with default hang_confirm_s, run, scale, n]`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

### PR 8: [None][feat] Add ULFM survivor communicators for host collectives

**Commits/files:** `372f2ae908`.
- `tensorrt_llm/_torch/distributed/communicator.py` (+582/-35);
- `_utils.py` (MPI peer-failure helpers);
- `cpp/tensorrt_llm/nanobind/runtime/peerFailureWatch.{cpp,h}`, plus bindings and CMake;
- `tests/unittest/_torch/distributed/test_mpi_survivors.py` and `test_mpi_survivors_ulfm.py`.

**Summary**
Under recovery, `MPIDist` runs host collectives on a private world that is set to `ERRORS_RETURN`, and adds:
- `HostCollectiveInterrupted` and `live_ranks`;
- `departed=` / `fill_value=` on `allgather`, `tp_allgather`, `tp_gather` and `tp_allgather_int64`, so that the
  survivors keep exchanging after a loss;
- a native thread (`peerFailureWatch`) that polls MPI for peer failures and raises the abort record.

Startup requires ULFM, `MPI_THREAD_MULTIPLE` and the `ob1` pml.

**FT-off impact vs main**
- `_interruptible` calls the method unchanged until `enable_peer_loss_recovery()` creates the private world, and
  only recovery creates it.
- `ERRORS_RETURN` is set only on the private, survivor and watch communicators.
- `departed=` is a no-op with every rank live.
- **FT-on cost (audit):** recovery forces Open MPI `ob1` over TCP (`communicator.py:960-963`). That makes about
  8 host collectives per step slower than UCX, which counts against FT on vs FT off. The finals separate it with
  an FT-off ULFM arm: `[FINALS: FT-off ob1/TCP arm vs FT-off default, run, scale, n]`.

**Test plan**
- `test_mpi_survivors.py` (30 tests, `cpu_only`; collected by the `l0_cpu` `unittest/_torch/distributed` entry).
- `test_mpi_survivors_ulfm.py`: needs Open MPI with ULFM; it skips unless `TLLM_REQUIRE_ULFM=1`.

**Disclosures**
- `Distributed` gains new keyword arguments; PR 18 updates the call sites.
- Fault tolerance is tied to Open MPI ULFM and `ob1`.
- The native thread's detach-on-destroy logic needs careful review.

**Follow-ups:** `test_mpi_survivors_ulfm.py` has no CI stage that sets `TLLM_REQUIRE_ULFM=1`, so in `l0_cpu` it
skips. Add it to a ULFM stage.

**Results:** `[FINALS: transport arm, as above]`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

### PR 9: [None][feat] Keep the MPI session serving and exit cleanly after a WideEP peer loss

**Commits/files:** `9e3c0be56d`.
- `tensorrt_llm/llmapi/mpi_session.py`, `mgmn_worker_node.py`, `mgmn_leader_node.py`, `trtllm-llmapi-launch`;
- `tests/unittest/llmapi/test_mpi_session.py` (+548);
- `test_trtllm_llmapi_launch_exit.py`;
- `l0_cpu.yml`.

**Summary**
After a peer loss, the MPI session keeps serving: departed-peer handling, and a manager-thread excepthook for the
expected `MPI_ERR_PROC_FAILED`. `MPI_Finalize` is bounded at 30 s, with a SIGALRM backstop, so a stop under ULFM
cannot hang. Under FT, `trtllm-llmapi-launch` `exec`s the worker, so that its death reaches its peers, and defaults
`OMPI_MCA_pml=ob1`, `btl=self,tcp` unless they are set. The launcher's FlashInfer lock is passed as
`--flashinfer-lock-fd`, and the worker marks it non-inheritable before any import.

**FT-off impact vs main**
- Departed-peer handling, the bounded finalize and the excepthook sit behind `peer_loss_recovery_enabled()`.
- The launcher's `exec` and the MCA defaults apply only with both `TLLM_FAULT_TOLERANCE_MODE=1` and an authority.
- `mgmn_worker_node.py` now always goes through `run_worker_rank`, which keeps main's path with FT off. It also
  now imports `tensorrt_llm.llmapi.mpi_session` at worker start.

**Test plan**
- `test_mpi_session.py` (18 new tests; `l0_cpu` and `l0_a100`, ISOLATION).
- `test_trtllm_llmapi_launch_exit.py` (5 tests; `l0_cpu`).

**Disclosures**
- A stop under ULFM can take up to about 30 s longer.
- `threading.excepthook` is overridden while FT is on (global state).

**Follow-ups:** worker `os._exit` may drop a queued shutdown ack (unverified; audit).

**Results:** `[FINALS: none specific]`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

### PR 10: [None][feat] Read the NVLinkOneSided rank mask through a pointer and add an abort record

**Commits/files:** `34611ef619`.
- `cpp/tensorrt_llm/kernels/moe/communication/moeAlltoAllKernels.{cu,h}`;
- `thop/moe/communication/moeAlltoAllOp.cpp`;
- the fakes in `cpp_custom_ops.py`;
- `tests/unittest/_torch/moe/test_moe_a2a_abort.py`, `test_moe_comm.py`;
- `l0_b200.yml`.

**Summary**
The all-to-all kernels read the active-rank mask through a device pointer at execution time instead of a by-value
array, so captured CUDA graphs see a mask rewritten in place. A `MoeA2AAbortState` record lets waits stop on a
peer loss: they poll the record and release sources whose count is invalid while an abort is pending. The op gains:
- `abort_state` and `expert_to_slot` / `num_routed_experts`, which back the `USE_EXPERT_TO_SLOT` path PR 23
  enables;
- `moe_a2a_initialize(barrier=True)`.

**FT-off impact vs main**
- Without `enable_rank_mask` the mask-free kernel instantiations run. Abort polling and the released-round check
  are compiled only into the `ENABLE_RANK_MASK` instantiations.
- **Workspace layout changes for every caller:** `FLAG_VAL` grows from 4 to 8 bytes (`moeAlltoAllOp.cpp:238`,
  audit A18). This is a layout delta with no expected perf effect; disclose it.
- More template instantiations, so longer compile time.

**Test plan**
- `test_moe_a2a_abort.py` (11 tests; `l0_b200`).
- `test_moe_comm.py` (updated; `l0_gb200_multi_gpus`).

**Disclosures**
- Op-schema changes: new optional args, all defaulted.
- **The rank-mask op contract changes from a CPU tensor to a CUDA tensor on the workspace device, and the "must
  mark the local ep_rank" check is removed.** Main's watchdog still passes a CPU mask, so rank-mask mode fails
  between this PR and PR 12 (A1.2). Land them together.
- The `USE_EXPERT_TO_SLOT` kernel path ships here, but nothing uses it before PR 23.

**Follow-ups:** confirm with SASS that the FT-off instantiation is identical to main
(`moeAlltoAllKernels.cu` SWITCH_BOOL, about line 1393; not yet verified).

**Results:** `[FINALS: healthy-path cost of rank mask + abort polling, FT on vs off, run, scale, n]`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

### PR 11: [None][feat] Remap the MNNVL all-to-all workspaces at the same addresses among the survivors

**Commits/files:** `6f5f1d080b`.
- `tensorrt_llm/_torch/distributed/mnnvl_memory.py`, `_torch/mnnvl_alltoall_workspace.py`;
- `tests/unittest/_torch/test_mnnvl_memory_remap.py`, `test_mnnvl_alltoall_workspace.py`;
- `l0_b200.yml`.

**Summary**
A failed rank's kernels can still write to the workspace pages it mapped. On recovery, the survivors therefore
release their workspace backing and map fresh pages at the same virtual addresses, shared only among the
survivors. The CUDA graphs keep their addresses, and late writes land in pages that no survivor maps. The PR adds
`pin`, `unpin` and `release_backing`, and refuses checkpointing after a remap. Workspace re-initialization now
checks for CUDA errors and restarts the watchdog.

**FT-off impact vs main:** none on the healthy path. The remap APIs are called only by recovery. The reinit error
check and watchdog restart act only on reinit.

**Test plan**
- `test_mnnvl_memory_remap.py` (10 tests; `l0_b200`).
- `test_mnnvl_alltoall_workspace.py` (+29; `l0_a10`).
- The multi-GPU remap test `test_moe_a2a_recovery.py` lands with PR 20.

**Disclosures**
- The remap may borrow KV cache memory until the failed processes end (PR 20).
- A survivor that cannot release or map its workspace ends the job.
- A further loss during the remap re-plans the recovery.

**Follow-ups:** none.

**Results:** `[FINALS: remap time and KV bytes lent per recovery, run, scale, n]`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

### PR 12: [None][feat] Bind NVLinkOneSided layers to a shared MoeExecutionState for WideEP recovery

**Commits/files:** `2b5005b9a1`.
- `communication/execution_state.py` (198), `nvlink_one_sided.py`, `alltoall_watchdog.py` (-9), `wide_ep_ft.py`,
  `_utils.py` (+5), `MOE_DEVELOPER_GUIDE.md`;
- tests: `test_moe_execution_state.py`, `multi_gpu/test_nvlink_one_sided.py`, `multi_gpu/test_moe_a2a_late_writes.py`,
  `modules/test_alltoall_watchdog.py`;
- `l0_b200.yml`, `l0_gb200_multi_gpus.yml`.

**Summary**
All NVLinkOneSided layers bind to one `MoeExecutionState`. It owns:
- the device rank-mask and abort records;
- fixed, token-interleaved receive slots per payload signature.

The kernels read the state at execution time, so CUDA graphs captured after binding follow later membership
changes without recapture. The PR replaces main's blanket `reject_rank_mask_cuda_graph_capture` with a refusal for
unbound rank-mask layers only. Combine takes its row order from what the dispatch wrote.

**FT-off impact vs main:** none. The bound-state path runs only when `_rank_mask_enabled` (`EPGroupHealth` or
`TLLM_FAULT_TOLERANCE_MODE=1`); otherwise main's dispatch and combine run. `MoeExecutionState` is created only by
recovery callers.

**Test plan**
- `test_moe_execution_state.py` (11 tests; `l0_b200`).
- `test_moe_a2a_late_writes.py` (`l0_gb200_multi_gpus`, TIMEOUT 10).
- `test_nvlink_one_sided.py` (multi-GPU; `l0_dgx_b200`).
- `test_alltoall_watchdog.py` (updated).

**Disclosures**
- Detection-only mode (`TLLM_FAULT_TOLERANCE_MODE=1` without an authority) refuses CUDA graph capture
  (`nvlink_one_sided.py:1036-1042`).
- A 3D combine input is rejected after a bound dispatch.

**Follow-ups:** none.

**Results:** see PR 10.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

### PR 13: [None][feat] Restore lost experts into EPLB slots backed by ghost slot memory (C++ load balancer)

**Commits/files:** `ecfd7d3934`.
- `cpp/tensorrt_llm/runtime/moeLoadBalancer/{moeLoadBalancer,ghostSlotMemory,hostAccessibleDeviceAllocator}.{cpp,h}`;
- `moeLoadBalanceKernels.{cu,h}` (abandon-aware CPU-stage wait), `moeLoadBalanceOp.cpp`, `moeBindings.cpp`,
  `runtime/CMakeLists.txt`;
- `cpp/tests/unit_tests/runtime/moeLoadBalancerTest.cpp` (+1102).

**Summary**
The C++ load balancer gains:
- `restorePlacement`, `quiesce`, `endIter(interrupted)`, `resetUpdateSchedule`, placement validation and install;
- dead-rank masks, so EPLB updates skip empty (-1) slots.

**Ghost slots** are extra slots whose device memory, in VMM-granularity chunks that lie wholly inside the slot, is
left unbacked while healthy. Boundary chunks stay backed, plus a per-process scratch of 1, 2, 4 and 8 chunks. A
restore backs a ghost slot, taking memory from the KV cache if needed. Where GPUDirect RDMA for VMM allows it, the
ghost memory is host accessible through GDRCopy, so that online EPLB host updates reach it.

**FT-off impact vs main:** none.
- `mFaultToleranceMode` is set at `finalizeModel` from `TLLM_FAULT_TOLERANCE_MODE`. The new entry points check
  it, and `waitCpuStage` and the worker thread keep main's path without it.
- -1 slots are accepted only with FT on.
- The `HostAccessibleDeviceAllocator` refactor keeps main's support decision. It is deliberately leaked only under
  FT.

**Test plan**
- `moeLoadBalancerTest.cpp`: 26 new gtests in the existing `add_gtest(moeLoadBalancerTest ...)` target.

**Disclosures**
- Ghost slots remap device memory under the KV cache.
- Restored experts lower KV capacity until restart.
- GPUs whose memory is a NUMA node fall back to `empty` slots with online EPLB.
- `HostAccessibleDeviceAllocator` is never destroyed under FT, on purpose.
- The FT flag is still read from the environment (main's `isFaultToleranceModeEnabled()`), not passed as a
  constructor argument.

**Follow-ups**
- Pass the FT flag in as a parameter.
- Consider splitting off the host-accessible chain. The PR is about 2.8k lines.

**Results:** `[FINALS: ghost-slot restore cases, KV bytes taken, run, scale, n]`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

### PR 14: [None][feat] Append extra EPLB slots and restore lost experts into them (Python)

**Commits/files:** `d12f179f47`.
- `moe_load_balancer.py` (+544), `impl_blocks.py`, `impl_base.py`, `interface.py`;
- a new `pyexecutor/peer_loss_recovery.py` (`failure_domains`, 24 lines);
- `tests/unittest/_torch/moe/test_resident_placement.py` (26 tests), `test_moe_load_balancer.py` (+6 tests).

**Summary**
Where the survivors of a failure domain's loss cannot hold every expert, fault tolerance appends extra slots to
each rank (`ghost` or `empty`). It adds quiesce, recovery placements, restoring lost experts and installing
placements, plus a placement digest. Startup refuses placements that cannot cover one failure domain.

**FT-off impact vs main:** none.
- Extra slots are appended only with `peer_loss_recovery_enabled()`, so `num_slots` and
  `expert_size_per_partition` are unchanged.
- The slot rows in `impl_blocks` and the hosts gather run only with recovery.

**Test plan**
- `test_resident_placement.py` (`cpu_only`).
- `test_moe_load_balancer.py`.
- Both are collected by the `l0_cpu` `unittest/_torch/moe` entry; `test_moe_load_balancer.py` is also in the
  B200, B300, H100 and GB300 lists.

**Disclosures**
- `moe_load_balancer.py` lazily imports `pyexecutor.peer_loss_recovery` for `failure_domains`, a layering
  inversion.
- `empty` slots hold memory while healthy: about one slot per rank per MoE layer
  (`[FINALS: measured GiB/GPU, DeepSeek-R1 NVFP4]`).

**Follow-ups:** move `failure_domains` to `executor/recovery/config.py`.

**Results:** `[FINALS: as above]`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

### PR 15: [None][feat] Keep WideEP EPLB host copies in a ring of nodes and stage restores from them

**Commits/files:** `766b92719f`.
- `moe_load_balancer.py` (+275);
- `tests/unittest/_torch/moe/test_ring_host_copies.py` (29 tests), `multi_gpu/test_ring_restore_staging.py`;
- `l0_gb200_multi_gpus.yml`.

**Summary**
With a static placement without replicas, each node keeps host copies (`/dev/shm`) of only the next node's home
experts, about 1/N of them. A restore copies each expert from a GPU of the copy node through the restoring rank's
remapped NVLinkOneSided workspace, in chunks of half the workspace. Under FT:
- shm names are per job;
- the reserve is checked and allocated up front;
- names are unlinked once every local rank has mapped them;
- a prefault thread warms the pages.

**FT-off impact vs main:** none. All of it runs only with `peer_loss_recovery_enabled()`, and main's shm lifecycle
is kept otherwise.

**Test plan**
- `test_ring_host_copies.py` (`cpu_only`; `l0_cpu`).
- `test_ring_restore_staging.py` (`l0_gb200_multi_gpus`, TIMEOUT 10).

**Disclosures**
- Losing a node together with its copy node is unrecoverable.
- After a loss, the next node's experts have no host copies.

**Follow-ups:** A15, the shm mappings leaked per FT-on weight reload (`moe_load_balancer.py:313-315`).

**Results:** `[FINALS: ring-restore recovery time, run, scale, n]`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

### PR 16: [None][perf] Tune and key the MoE tactics on the loaded slots under WideEP recovery

**Commits/files:** `e9c313962a`.
- `cute_dsl_custom_ops.py`, `torch_custom_ops.py`, `fused_moe_cute_dsl.py`, `fused_moe_cutlass.py`;
- `tests/unittest/_torch/moe/test_moe_runner.py` (6), `moe/kernels/parallel/test_cute_dsl_moe.py` (+105).

**Summary**
Extra slots are empty while healthy, so the tactics are tuned and keyed on the loaded slots
(`num_extra_local_experts`) rather than on all slots.

**FT-off impact vs main:** none. `num_extra_local_experts` defaults to 0, and the backends pass 0 without recovery,
which keeps main's tactic and autotuner cache keys.

**Test plan**
- `test_moe_runner.py` (`cpu_only`).
- `test_cute_dsl_moe.py` (`l0_b200`).

**Disclosures:** op-schema change; `trtllm::fused_moe` / `MoERunner` gain a defaulted argument.

**Follow-ups:** the loaded-slot key covers only Cutlass and CuteDSL.

**Results:** `[FINALS: FT on vs off healthy throughput includes this]`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

### PR 17: [None][feat] Roll back a discarded step's KV cache allocations in KV Cache Manager V2

**Commits/files:** `abb38ab298`.
- `kv_cache/kv_cache_manager_v2.py` (+60);
- `tests/unittest/_torch/executor/test_kv_cache_revert.py` (7 tests; `l0_b200`).

**Summary**
This PR adds `context_capacities` and `revert_prepare_resources`. Recovery discards an interrupted step, and these
let it return the step's KV allocations, so the discarded step publishes no reusable blocks.

**FT-off impact vs main:** none. The API is called only by recovery's step discard.

**Test plan**
- `test_kv_cache_revert.py`.

**Disclosures:** with speculative decoding under the overlap scheduler, a loss in the step after a committed step
can skip the committed step's KV update for its rejected draft tokens.

**Follow-ups:** none.

**Results:** not applicable.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

### PR 18: [None][feat] Keep attention-DP routing serving with absent ranks

**Commits/files:** `9cf51e9801`.
- `scheduler/adp_router.py`, `request_utils.py`, `executor_request_queue.py`, `py_executor.py` (+33),
  `cuda_graph_runner.py`, `model_engine.py`, `engine/runners/{common,decoder/*}.py`;
- tests: `test_adp_absent_ranks.py` (12), `test_adp_router.py`, `test_adp_host_exchange.py`,
  `test_executor_request_queue.py`, `test_request_utils.py`, `test_py_executor.py`, plus mock-signature updates in
  `disaggregation/*` and `test_modeling_deepseekv41.py`.

**Summary**
Attention-DP routing, balancing, the CUDA-graph vote and the batch exchanges run over the live ranks, using
`departed=` on the gathers and per-rank state keyed by logical rank. A new request strictly pinned to a lost rank
fails with HTTP 400 ("Attention DP rank N is not serving").

It also carries two pieces that belong elsewhere:
- `wake()` / `WAKE_REQUEST_ID` and `waiting_for_requests`, which hang detection reads;
- `ModelEngine.moe_execution_state` (closed on teardown).

**FT-off impact vs main**
- `departed=` has no effect and `live_ranks` is every rank.
- A pinned target is indexed by position, as on main, unless recovery is on.
- The wake request is never broadcast.
- `waiting_for_requests` is set around the blocking queue read; this is negligible.

**Test plan**
- The tests listed above. All are collected by the `l0_cpu` `unittest/_torch/executor` entry.

**Disclosures:** the mixed concerns described in the summary.

**Follow-ups:** move the hang plumbing and `moe_execution_state` into their own PRs (PRs 7 and 12) if reviewers ask.

**Results:** not applicable.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

### PR 19: [None][perf] Release an attention-DP context hold once a burst of arrivals has ended

**Commits/files:** `17a047ea76`.
- `py_executor.py` (+30: `ADP_BURST_QUIET_ITERS`, `adp_recent_arrivals`);
- `tests/unittest/_torch/executor/test_py_executor.py` (6 tests).

**Summary**
With attention-DP balancing, a rank that holds context requests to align with the other ranks waits out
`attention_dp_time_out_iters` even after a burst of arrivals has ended. In a closed loop this keeps its users out
of phase with the next burst. The hold is now released once all of these hold:
- every rank has free context slots;
- the waiting queue is empty;
- the timeout window saw at least one arrival per serving rank;
- `ADP_BURST_QUIET_ITERS` (4) decisions have passed without arrivals.

**FT-off impact vs main:** **yes, for every attention-DP-balance user, with or without FT.** This is not gated on
FT. It changes context scheduling, and therefore TTFT and throughput, against main. The finals main-parity arm
(our FT-off build vs main) measures it as the "gate-fix effect":
`[FINALS: main-parity arm, ADP balance on, TTFT/throughput delta, run, scale, n]`.

**Test plan**
- `test_py_executor.py`:
  - `test_straggler_after_a_burst_is_released_once_arrivals_stop`;
  - `test_sparse_arrivals_wait_out_the_timeout`;
  - `test_queued_requests_keep_the_hold`;
  - `test_a_full_rank_keeps_the_hold`;
  - `test_arrivals_of_fetches_before_one_decision_add_up`;
  - `test_arrivals_leave_the_window_after_the_timeout`.

**Disclosures**
- This is unrelated to fault tolerance and needs a perf justification from the scheduler owners.
- It depends on PR 18's `live_ranks`; to send it to main, use `self.dist.tp_size`.

**Follow-ups:** none.

**Results:** `[FINALS: as above]`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

### PR 20: [None][feat] Recover the PyTorch executor from a WideEP peer loss

**Commits/files:** `e74cadb97c` squashed with `ddc2658518` and `65741c820a`.
- `pyexecutor/peer_loss_recovery.py` (+1375), `py_executor.py` (+283), `llm_request.py` (+53),
  `py_executor_creator.py`;
- tests: `test_peer_loss_recovery.py` (84 tests here), `multi_gpu/test_moe_a2a_recovery.py` (3; `l0_gb200_multi_gpus`,
  TIMEOUT 10), `test_kv_pool_rebalance.py` and `test_multimodal_scheduler.py` (fakes).

**Summary**
`PyExecutor` gets a recovery hook. It is a pass-through `RecoveryHook` unless `PeerLossRecovery.create` enables
recovery. When a peer loss is detected, the survivors:
1. stop the step in flight and discard the interrupted steps (KV rollback, PR 17; step-change snapshots);
2. agree on the failed ranks;
3. have the MPI runtime kill them before any plan;
4. remap the all-to-all workspaces (PR 11);
5. restore lost experts (PRs 13-15);
6. install and commit the authority's plan.

Serving then resumes on the same CUDA graphs. Startup refuses unsupported configurations (listed in FD) and
checks that slots cover a failure domain. KV cache events are refused under recovery.

**FT-off impact vs main**
- The hook is the do-nothing `RecoveryHook`, and the per-step hook calls are guarded by `.enabled`.
- `_note_step_change` is a no-op when `_step_changes is None`. With the squash included, the overlap scheduler's
  completion update runs as on main (A1.1).
- `publish_responses` keeps main's enqueue-then-terminate order.
- A raising forward ends the iteration as on main when no peer failed.
- `split_unservable_requests` runs only with FT on and new requests.
- **FT-on healthy-path cost (rank mask + abort polling + expert_to_slot):**
  `[FINALS: FT on vs off, identical config, both FORCE_CFT=0, run, scale, n]`.

**Test plan**
- `test_peer_loss_recovery.py` (`cpu_only`; `l0_cpu` executor entry). This includes
  `test_fault_tolerance_off_marks_requests_completing_next_iteration_without_recording`.
- `test_moe_a2a_recovery.py` (multi-GPU).
- End to end: PR 24's integration tests, including `test_serving_without_fault_tolerance[sync|overlap]`.

**Disclosures**
- These are not covered and end the job:
  - rank 0;
  - a node lost together with its ring copy node;
  - a loss after install/commit, or a remap mapping failure;
  - a further loss that the remaining slots cannot cover.
- Startup refusals (`peer_loss_recovery.py:416-460`, `communicator.py:941-967`) and per-request rejections
  (`peer_loss_recovery.py:507-520`) are listed in FD.
- Speculative decoding other than one-model MTP is untested and logs a warning.
- A recovery can take up to about 2x `recovery_timeout_s` after agreement.
- KV cache lent to the remap stays lent under the overlap scheduler with steady load.
- KV cache given to restored experts stays given until restart.

**Follow-ups**
- Split `peer_loss_recovery.py` (about 1800 lines, with a long `_recover`) and `test_peer_loss_recovery.py` into
  orchestration, requests, and EPLB/remap modules.
- `split_unservable_requests` runs on every FT-on step with arrivals (`py_executor.py:6793`).

**Results:** `[FINALS: detection-to-resume time, KV lent/taken, post-loss throughput, run, scale, n]`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

### PR 21: [None][feat] Run a lost rank's undelivered requests again on the survivors

**Commits/files:** `db6f10dc0b`.
- `peer_loss_recovery.py` (+421), `py_executor.py` (+29);
- `test_peer_loss_recovery.py` (+38 tests).

**Summary**
The requests a lost rank was serving that had delivered nothing (every non-streaming request, and a streaming one
before its first token) run again from their prompt on a survivor. Rank 0 holds the response gather so that it
knows what was delivered. A request that a second loss interrupts without progress since the first is quarantined
and fails with a retryable error. A lost rank's unanswered cancellations end as cancelled.

**FT-off impact vs main:** none. Holding, quarantine and re-runs run inside the enabled hook only; with recovery off,
rank 0 gathers responses without hook calls.

**Test plan**
- `test_peer_loss_recovery.py`: the re-run, quarantine and delivery cases.

**Disclosures**
- Re-run outputs are not token-exact.
- A lost rank's steps count only once it reported them.
- The server does not recognize a client's retry.
- A crashing request that lands on rank 0 ends the deployment.

**Follow-ups:** see PR 20 (module split).

**Results:** `[FINALS: requests re-run / failed per recovery, accuracy vs no-loss baseline, run, scale, n]`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

### PR 22: [None][feat] Report WideEP fault-tolerance state on /health and answer peer-loss failures with 503

**Commits/files:** `3839e08829`.
- `tensorrt_llm/serve/fault_tolerance_status.py`, `openai_server.py` (+45);
- tests: `test_fault_tolerance_health.py` (3), `test_stream_error_events.py` (+4), `test_recovery_authority.py`
  (+5);
- `l0_cpu.yml`.

**Summary**
`trtllm-serve` asks the authority for its state through one probe thread. `/health` returns:
- 200 with `healthy`, `degraded`, `unprotected` or `unknown`;
- 503 while `recovering` or `terminal`.

The body carries the generation, the live and failed ranks, the recovery count and stage times, and the failed
requests. The abort reason is kept out of this unauthenticated body. Peer-loss failures map to HTTP 503 with
`Retry-After: 1` and the code `peer_loss_retryable`. A failed stream gets that code in its error event (built on
PR 3).

**FT-off impact vs main:** none. The probe exists only with recovery, and the 503 mapping and retryable code apply
only when `fault_tolerance_mode()` is on.

**Test plan**
- `test_fault_tolerance_health.py`.
- `test_stream_error_events.py`:
  - `test_a_peer_loss_failure_streams_a_retryable_code`;
  - `..._is_a_503_with_retry_after`;
  - `test_without_fault_tolerance_a_tagged_message_is_an_ordinary_error`;
  - `test_other_errors_stay_400_without_retry_after`.
- `l0_cpu`.

**Disclosures:** under FT, `/health` returns 503 for non-serving states. Probe guidance is in FD.

**Follow-ups:** none.

**Results:** `[FINALS: /health 503 duration during a recovery, run, scale, n]`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

### PR 23: [None][perf] Look up dispatch slots in a device expert-to-slot table under WideEP recovery

**Commits/files:** `b0a8905b44`.
- `moeLoadBalanceKernels.{cu,h}`, `moeLoadBalanceOp.cpp` (`moe_load_balance_one_slot_table`) and its fake;
- `nvlink_one_sided.py`, `moe_load_balancer.py`, `moe_scheduler.py`;
- tests: `test_nvlink_one_sided.py` (multi-GPU), `test_moe_execution_state.py`, `test_moe_load_balancer.py`.

**Summary**
When every expert routes to exactly one slot, the dispatch looks up slots in a device expert-to-slot table, using
the `USE_EXPERT_TO_SLOT` kernel path from PR 10, instead of running the routing kernel.

**FT-off impact vs main:** none.
- `ExternalCommMoEScheduler` uses the table only with recovery.
- `moe_load_balance_routing` passes `oneSlotPerExpert = routesEachExpertToOneSlot()`, which is false without FT, so
  main's routing kernel runs.

**Test plan**
- The tests listed above (10 new across the three files).

**Disclosures:** a new op and a new template dimension, used only with FT.

**Follow-ups:** fix the `moe_scheduler.py:266-278` docstring, which says the slot-table path is "only with
peer-loss recovery". The actual gate is in C++ (`moeLoadBalancer.cpp:660-664`, `routesEachExpertToOneSlot`). This
is comment-only.

**Results:** `[FINALS: same-node A/B with and without the table, FT on, run, scale, n]`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

### PR 24: [None][test] Add WideEP peer-loss recovery integration tests

**Commits/files:** `ffc8bc30c3` squashed with `7e8dfb37df`.
- `tests/integration/defs/wide_ep_ft/{__init__,peer_loss_driver,test_peer_loss_recovery}.py`;
- `worker_hooks/sitecustomize.py`;
- `l0_gb200_multi_gpus.yml` (+14);
- `jenkins/L0_Test.groovy` (+5).

**Summary**
End-to-end `trtllm-serve` runs on GB200 under `mpirun --with-ft ulfm`, with an authority:
- `test_peer_loss_recovery[idle|inflight]-[sync|overlap]`;
- `test_peer_loss_recovery_under_load`;
- `test_two_ranks_lost_together`;
- `test_rank_lost_during_recovery`;
- `test_restore_into_ghost_slots[static|online]`;
- `test_restore_into_appended_empty_slots`;
- `test_hung_rank_fenced`;
- `test_authority_lost_while_serving`;
- `test_serving_without_fault_tolerance[sync|overlap]`, which serves the FT-off executor end to end under both
  schedulers.

**FT-off impact vs main:** test-only. `TLLM_REQUIRE_ULFM=1` is injected for `GB200-4_GPUs-PyTorch*` stages, so a
missing ULFM fails these tests instead of skipping them.

**Test plan:** the 14 cases above, each with TIMEOUT 45, in `l0_gb200_multi_gpus`.

**Disclosures**
- Fault injection is test-only: a `sitecustomize` on `PYTHONPATH` patches `RecoveryClient.send`
  (`WIDEEP_FT_TEST_KILL_AFTER_STOPPED`).
- Significant CI cost.
- The Groovy change needs infra CODEOWNER review.
- Needs the PR 4 image.

**Follow-ups:** none.

**Results:** `[FINALS: integration pass list on the frozen build]`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

### PR 25: [None][doc] Document WideEP fault tolerance

**Commits/files:** `c86566c184` squashed with `e25e638054` and `6e311f2f04`.
- `docs/source/features/wide-ep-fault-tolerance.md`;
- `feature-combination-matrix.md`;
- `index.rst`;
- `examples/wide_ep/README.md`.

**Summary**
A feature page, marked experimental, covering requirements, running, what is covered, known limits, speculative
decoding, unsupported combinations, what happens during a recovery, monitoring, test maturity and cost. The
combination matrix gains a row. `6e311f2f04` applies the audit's doc fixes:
- ghost-slot backing;
- a key of at least 16 bytes;
- up to about 2x `recovery_timeout_s`;
- detection-only refusing CUDA graphs;
- remap re-plan;
- KV lent;
- a generation sampler.

**FT-off impact vs main:** none (docs only).

**Test plan:** docs build.

**Disclosures:** before merge, every TODO in FD must be filled from the finals or removed. **Two places still carry
pre-finals numbers that must be replaced or dropped:**
- FD:187: SIGSTOP-to-serving times "10.6 / 15.9 / 36.7 s ... under an earlier rule";
- FD:79: the "estimated 1.4 GiB per GPU" for `empty` slots.

FD:216 ("45 s host stalls ... cost nothing") and FD:295 (GSM8K within two standard errors) are also unverified on
the current design.

**Follow-ups:** none beyond filling the TODOs.

**Results:** `[FINALS: FD:66, 79, 187, 189, 216, 278, 295, 332, 353, 366]`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

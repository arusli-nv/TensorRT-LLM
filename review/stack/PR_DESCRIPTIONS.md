# WideEP fault tolerance: pull-request descriptions

Overview of the series (26 PRs, one commit each):
- M1-M6 target `main`. They are independent of each other and stand alone: two fixes in the fake MoE ops and EPGroupHealth, an OpenAI stream fix (M3), an attention-DP scheduling change (M4), a kernel fence fix (M5) and the mpi4py 4.1.2 image upgrade (M6, needs an image rebuild).
- 01-20 target `feat/wide-ep-ft-mvp`, which equals `main` plus M1-M6; each PR header lists the PRs it depends on.
- Order to open: M1-M6 first (all six can be open at once; M6 needs an image rebuild, so open it early). Merge them, then sync `feat/wide-ep-ft-mvp` to `main` before opening 01.
- Then open 01..20 one or two at a time, in dependency order. Roots: 01 (protocol), 04 (ULFM), 06 (all-to-all mask), 10 (EPLB ghost slots), 14 (KV rollback).
- Chains: 01-02-03 (recovery authority and hang detection); 04-05 (MPI survivors, session); 06 then 07, 08, 09 (all-to-all); 10-11 then 12, 13 (EPLB); 04-15 (attention DP).
- 16 (executor recovery) needs all of the above; 17 (`/health`) needs 16 and M3; 18 (dispatch lookup) needs 06 and 11; 19 (integration tests) needs 16 and 17; 20 (docs) comes last.
- Everything in 01-20 is off unless `TLLM_FAULT_TOLERANCE_MODE=1` (recovery also needs `TLLM_FAULT_TOLERANCE_AUTHORITY`). Each PR says what, if anything, changes for users who leave it off.
- Changes that reach every user: M3 (stream errors), M4 (attention-DP release), M5 (fences), M6 (image), the 8-byte round-flag layout in 06, `departed=` on `Distributed` gathers in 04, and one trailing argument on the fused-MoE ops in 13.
- Most of the lines are tests (about 60%). Hashes change with each fixup round, so PR text does not quote them.
- Experimental feature; the guide is `docs/source/features/wide-ep-fault-tolerance.md` (PR 20).

---

## M1: [None][fix] Accept the real arguments in the fake moe_a2a op registrations
Base: main; Depends on: none.
Description:
- Adds `use_cft_counted_writes`, `expert_id_payload_index` and `invalid_token_expert_id` to the fake (meta) registration of `moe_a2a_dispatch`, and `use_cft_counted_writes` to the one of `moe_a2a_combine`.
- A traced call (for example under `torch.compile`) that passes these arguments fails against a fake that lacks them.

With fault tolerance off: applies to every user of these ops under tracing. The new arguments default to `False`/`None`, so a call that omits them is unchanged.

Design notes:
- The fake's argument order follows the real op schema.
- The real ops also take `abort_state` and `expert_to_slot`; the PRs that add them (06, 18) extend the fakes.

Test coverage:
- `tests/unittest/_torch/moe/test_moe_a2a_fake_ops.py` (CPU only, `FakeTensorMode`): calls both fakes with every argument of the real ops.

Testing: `tests/unittest/_torch/moe/test_moe_a2a_fake_ops.py` (new) (CI: pending).

PR Checklist:
- [ ] PR description clearly explains what and why
- [ ] Follows TRT-LLM CODING GUIDELINES
- [ ] Test cases are provided for new code paths
- [ ] API-change label (`api-compatible` / `api-breaking`) added if the PR changes an API
- [ ] New dependencies scanned for license and vulnerabilities
- [ ] CODEOWNERS updated if ownership changes
- [ ] Documentation updated as needed
- [ ] tava architecture diagram updated if the design changes significantly
- [ ] Reviewers assigned are appropriate
- [ ] Please check this after reviewing the above items

## M2: [None][fix] Share EPGroupHealth across ConfigurableMoE's model-config deepcopy
Base: main; Depends on: none.
Description:
- `EPGroupHealth.__deepcopy__` returns the object itself.
- `ConfigurableMoE` deep-copies the model config for a per-layer quantization override; with the change every layer's copy sees one membership state instead of a private copy that misses updates.

With fault tolerance off: nothing changes where nothing reads an `EPGroupHealth`.

Design notes:
- The method applies to every deepcopy of the class, not only the `ConfigurableMoE` path. The object is shared membership state held in the model config's `extra_attrs`, never per-copy state; a caller that deep-copies it as a snapshot would alias it. The docstring says so.

Test coverage:
- `tests/unittest/_torch/moe/fused_moe/test_configurable_moe.py::test_layer_quant_override_shares_ep_group_health`: `deepcopy` returns the same object, and a layer quantization override through `_create_and_sync_backend` keeps the shared `EPGroupHealth` in the model config.

Testing: `tests/unittest/_torch/moe/fused_moe/test_configurable_moe.py` (CI: pending).

PR Checklist:
- [ ] PR description clearly explains what and why
- [ ] Follows TRT-LLM CODING GUIDELINES
- [ ] Test cases are provided for new code paths
- [ ] API-change label (`api-compatible` / `api-breaking`) added if the PR changes an API
- [ ] New dependencies scanned for license and vulnerabilities
- [ ] CODEOWNERS updated if ownership changes
- [ ] Documentation updated as needed
- [ ] tava architecture diagram updated if the design changes significantly
- [ ] Reviewers assigned are appropriate
- [ ] Please check this after reviewing the above items

## M3: [None][fix] End every failed OpenAI stream with its error events and a single [DONE]
Base: main; Depends on: none (17 builds on it).
Description:
- A chat or completions stream that fails after HTTP 200 ends with SSE `error` events and one `[DONE]` (`_stream_error_events`, `_end_stream_on_error`), so clients do not see a truncated stream.
- Chat and harmony chat send one error event. A multi-prompt completions stream sends one error event per failed prompt while the other prompts keep streaming, then a single `[DONE]`.
- Responses streams already end this way through `guard_responses_stream`.
- The Anthropic `/v1/messages` route turns the chat stream's error event into an Anthropic error event.
- A failure after `[DONE]`, such as in a chat stream's metrics extraction, is logged only. A completions stream extracts metrics before `[DONE]`, so a failure there sends that prompt's error event.

With fault tolerance off: this applies to every user of these streaming routes; it does not depend on fault tolerance.

Design notes:
- The error event carries `str(error)`. The completions stream already sent that text, and OpenAI clients need the cause to act on it.
- `completion_generator` yields the error event; the generator wrapper adds the single `[DONE]`, so a multi-prompt stream cannot end twice.
- The Anthropic adapter treats any upstream chunk with an `error` key as fatal; `ChatCompletionStreamResponse` has no `error` field, so a valid chunk never carries one.
- A failure after `[DONE]` cannot be reported to the client (the stream is closed), so it is logged.

Test coverage:
- `test_stream_error_events.py`: error event then `[DONE]`; a stream failing midway ends with one error and one `[DONE]`; no second `[DONE]` after a failure past `[DONE]`; a healthy stream passes through unchanged. Listed in `l0_cpu.yml`.
- `test_anthropic_adapter.py`: the reframer ends a failed upstream stream with its error. `test_anthropic_routes.py`: the chat stream logs a failure and ends with an error event; the `/v1/messages` route ends a failed stream with the upstream error; a two-prompt completions stream with one failing prompt sends both prompts' output, one error event and a single `[DONE]`.

Testing: `tests/unittest/llmapi/apps/test_stream_error_events.py` (new), `test_anthropic_adapter.py`, `test_anthropic_routes.py` (CI: pending).

PR Checklist:
- [ ] PR description clearly explains what and why
- [ ] Follows TRT-LLM CODING GUIDELINES
- [ ] Test cases are provided for new code paths
- [ ] API-change label (`api-compatible` / `api-breaking`) added if the PR changes an API
- [ ] New dependencies scanned for license and vulnerabilities
- [ ] CODEOWNERS updated if ownership changes
- [ ] Documentation updated as needed
- [ ] tava architecture diagram updated if the design changes significantly
- [ ] Reviewers assigned are appropriate
- [ ] Please check this after reviewing the above items

## M4: [None][perf] Release an attention-DP context hold once a burst of arrivals has ended
Base: main; Depends on: none.
Description:
- Attention-DP balancing holds context requests to align them across ranks until a timeout.
- Once a burst of arrivals has been quiet for `ADP_BURST_QUIET_ITERS` (4) decisions with nothing waiting, no more context is coming to align with, so the hold is released.
- The executor counts requests routed since the last balance decision, keeps the counts of the last `attention_dp_time_out_iters` decisions, and counts decisions since the last arrival.
- Measured with DeepSeek-R1 NVFP4 at DEP32 on GB200, balancing on, closed-loop load: timeout releases per wave drop from 1.7-4.5 to 0 at 32 requests per GPU (4 launches); at 8 per GPU they drop from 10-11 to 0.6-2.7. Throughput at 128 and 512 requests per GPU is unchanged within noise (-0.18% / -0.82% whole window, 4 pairs).

With fault tolerance off: this applies to every user with attention-DP balancing on. With balancing off nothing changes: the release rule sits inside the `attention_dp_enable_balance` branch, and the only other state is a per-fetch counter.

Design notes:
- A class constant, not a config field. The release fires only when all ranks have free context slots, the waiting queue is empty, at least one request per serving rank arrived within the timeout window, and 4 decisions passed without arrivals; the measurement above shows throughput unchanged within noise.
- Every rank takes the same decision: the request queue is replicated and requests are broadcast, so the routed counts, the quiet-decision count and the waiting-queue test are equal on every rank, and all ranks release on the same iteration.
- The rule adds a release condition to the existing timeout; it never holds longer than before.
- Only closed-loop load was measured. Under open-loop load whose arrivals pause for 4 decisions and then resume, the hold can be released before the next arrivals; that case is not measured.

Test coverage:
- `tests/unittest/_torch/executor/test_py_executor.py::TestAdpBalanceReleasesAfterBurst`: a straggler after a burst is released once arrivals stop; sparse arrivals wait out the timeout; queued requests keep the hold; a full rank keeps the hold; arrivals of several fetches before one decision add up; arrivals leave the window after the timeout.

Testing: `tests/unittest/_torch/executor/test_py_executor.py` (CI: pending).

PR Checklist:
- [ ] PR description clearly explains what and why
- [ ] Follows TRT-LLM CODING GUIDELINES
- [ ] Test cases are provided for new code paths
- [ ] API-change label (`api-compatible` / `api-breaking`) added if the PR changes an API
- [ ] New dependencies scanned for license and vulnerabilities
- [ ] CODEOWNERS updated if ownership changes
- [ ] Documentation updated as needed
- [ ] tava architecture diagram updated if the design changes significantly
- [ ] Reviewers assigned are appropriate
- [ ] Please check this after reviewing the above items

## M5: [None][fix] Order the MoE all-to-all payload accesses against their fences
Base: main; Depends on: none.
Description:
- The release and acquire fences around the NVLinkOneSided completion flags were `asm volatile` without a `"memory"` clobber, which orders them only against other volatile asm.
- The compiler could sink a payload store past `fence.release`, so a peer could see the flag before the data, or hoist a combine payload load above `fence.acquire`.
- The fences carry the clobber, as the fabric fence already does (4 lines in `moeAlltoAllKernels.cu`).

With fault tolerance off: applies to every NVLinkOneSided user. The PTX instructions are unchanged; the clobber only stops the compiler from moving memory accesses across the fences.

Design notes:
- Scope is the release/acquire fences of the dispatch and combine kernels (both the sm_90 and the `acq_rel` variants). Other `ld.relaxed` sites in the file are not touched.

Test coverage:
- No test change: a compiler barrier has no direct unit test. The existing NVLinkOneSided multi-GPU tests run these kernels.

Testing: no test files changed; existing NVLinkOneSided tests (CI: pending).

PR Checklist:
- [ ] PR description clearly explains what and why
- [ ] Follows TRT-LLM CODING GUIDELINES
- [ ] Test cases are provided for new code paths
- [ ] API-change label (`api-compatible` / `api-breaking`) added if the PR changes an API
- [ ] New dependencies scanned for license and vulnerabilities
- [ ] CODEOWNERS updated if ownership changes
- [ ] Documentation updated as needed
- [ ] tava architecture diagram updated if the design changes significantly
- [ ] Reviewers assigned are appropriate
- [ ] Please check this after reviewing the above items

## M6: [None][infra] Upgrade mpi4py to 4.1.2 for ULFM
Base: main; Depends on: none (04 needs it).
Description:
- `docker/common/install_mpi4py.sh` builds mpi4py 4.1.2 instead of 3.1.5; fault tolerance needs the ULFM bindings of mpi4py 4 (failed-rank queries, revoke, shrink).
- The `TRTLLM_USE_MPI_KVCACHE` futures patch is ported to mpi4py 4's `futures/_core.py`.
- The `setuptools<69` pin is dropped.

With fault tolerance off: this changes the image for every user, a major-version change of the MPI binding. The CI images must be rebuilt to pick it up; until then CI runs mpi4py 3.1.5.

Design notes:
- The patch is applied with `git apply` at image build, so a mismatch with the 4.1.2 sources fails the build instead of the runtime.
- Script check: a container job ran `docker/common/install_mpi4py.sh` in a venv on the devel image; mpi4py 4.1.2 installed and the `TRTLLM_USE_MPI_KVCACHE` patch applied (Slurm job 8481305, 2026-10-30). The patched code path (`TRTLLM_USE_MPI_KVCACHE=1`) was not run there.
- mpi4py is built from the release tarball, so 4.1.2 is fixed by the script alone; a pip install outside the image gets whatever version pip selects.
- The script does not verify a checksum of the tarball (unchanged from the 3.1.5 script).

Test coverage:
- No unit test: the change is an image build step. Regression evidence for the 3.x to 4.x change is the existing MPI tests (`tests/unittest/llmapi` MPI session tests, MPI KV cache tests) on the rebuilt image.

Testing: no test files changed; existing MPI tests on the rebuilt image (CI: pending).

PR Checklist:
- [ ] PR description clearly explains what and why
- [ ] Follows TRT-LLM CODING GUIDELINES
- [ ] Test cases are provided for new code paths
- [ ] API-change label (`api-compatible` / `api-breaking`) added if the PR changes an API
- [ ] New dependencies scanned for license and vulnerabilities
- [ ] CODEOWNERS updated if ownership changes
- [ ] Documentation updated as needed
- [ ] tava architecture diagram updated if the design changes significantly
- [ ] Reviewers assigned are appropriate
- [ ] Please check this after reviewing the above items

## 01: [None][feat] Add the WideEP recovery protocol, config and client
Base: feat/wide-ep-ft-mvp; Depends on: base (M1-M6 merged).
Description:
- Adds the worker-authority messages (`protocol.py`), `FaultToleranceConfig` (settings, validation, digest) and `RecoveryClient`, a worker's authenticated connection to the authority.
- `MessageQueue` (the `ZeroMqQueue` the recovery connections use) raises `Unauthenticated` for a message that fails HMAC and `MalformedMessage` for one that does not unpickle into protocol messages.
- `ZeroMqQueue` gains `socket_options`; `executor.utils` gains a retryable error code for requests a peer loss failed.
- `fault_tolerance_mode()` and `peer_loss_recovery_enabled()` read `TLLM_FAULT_TOLERANCE_MODE` and `TLLM_FAULT_TOLERANCE_AUTHORITY`.

With fault tolerance off:
- Both functions return False unless `TLLM_FAULT_TOLERANCE_MODE=1` (and the authority address for recovery); `RecoveryClient.from_env` returns None without recovery.
- `ZeroMqQueue.socket_options` defaults to None (no options set).

Design notes:
- Messages unpickle protocol classes only and every field must have its declared type, so an authenticated peer cannot instantiate other classes in the authority.
- Authentication and framing errors are typed exceptions (`Unauthenticated`, `MalformedMessage`); the caller decides by class, not by matching error text. The HMAC is checked first; any exception while unpickling an authenticated message becomes `MalformedMessage`, since unpickling malformed data can raise many types and the message is the sender's.
- Server-to-worker messages carry no replay check (only worker-to-authority messages do, PR 02). The authority binds to localhost by default, and the trust model is that of TRT-LLM's HMAC-protected ZMQ IPC.
- `peer_loss_recovery_enabled()` is read once per process and raises `ValueError` if the authority address is set without `TLLM_FAULT_TOLERANCE_MODE=1`.

Test coverage:
- `tests/unittest/executor/test_recovery_config.py` (listed in `l0_cpu.yml`): defaults, file overrides, invalid settings refused, hang-resolution bounds on confirm and fence times, shrink wait for a hung peer, every field has a description, digest is stable and changes with any field.
- Protocol and client wire tests ship with the authority tests in PR 02.

Testing: `tests/unittest/executor/test_recovery_config.py` (new) (CI: pending).

PR Checklist:
- [ ] PR description clearly explains what and why
- [ ] Follows TRT-LLM CODING GUIDELINES
- [ ] Test cases are provided for new code paths
- [ ] API-change label (`api-compatible` / `api-breaking`) added if the PR changes an API
- [ ] New dependencies scanned for license and vulnerabilities
- [ ] CODEOWNERS updated if ownership changes
- [ ] Documentation updated as needed
- [ ] tava architecture diagram updated if the design changes significantly
- [ ] Reviewers assigned are appropriate
- [ ] Please check this after reviewing the above items

## 02: [None][feat] Add the external WideEP recovery authority
Base: feat/wide-ep-ft-mvp; Depends on: 01.
Description:
- Adds the recovery authority, a standalone process (`python -m tensorrt_llm.executor.recovery.authority`) that admits failure domains, agrees with the survivors on each loss and drives the recovery.
- Adds `HangMonitor`, which confirms a hung or stalled rank from heartbeats and device waits.
- The authority refuses a malformed message to its sender and keeps serving, validates its fence command template at startup, keeps the fence command of each rank whose fencing is under way, and cuts an abort reason to fit a message (4096 characters; the log keeps the whole reason).
- An exception while handling a message is logged and aborts (phase `ABORTED`, an `Abort` to every worker and to the sender); the serve loop keeps running, so workers learn of it at once instead of timing out.

With fault tolerance off: nothing starts the authority unless the operator runs it; no worker code path changes.

Design notes:
- `fence_departed` is on in `main()`. A rank declared failed can still hold its GPU and its mappings of the survivors' memory, and the survivors' agreement does not complete until that process ends, so rank 0 has the MPI runtime end every failed rank before a plan is proposed; the recovery aborts if the runtime refuses or does not answer within 2 s.
- The fence command's host and pid come from the worker's heartbeat. Workers that hold the HMAC key are trusted; the values are checked only to be plain (no option injection, pid above 0), the command runs without a shell and never receives the key.
- Command fences are kept per rank, so each rank hung within the grace gets its fence.
- The authority binds to localhost unless `--endpoint` is given; traffic is authenticated, not encrypted (stated in the module docstring).
- Worker-to-authority messages are replay-checked per rank session; an unauthenticated frame is dropped, a malformed one is refused to its sender, and the loop continues.

Test coverage:
- `tests/unittest/executor/test_recovery_authority.py`: admission, idle/node/further/second loss, aborts and deadlines, protocol violations, replay, fencing, status, plus the wire format, HMAC authentication, key loading and a ZMQ end-to-end run; authenticated payloads that do not unpickle, name a non-message class or are ill-typed are refused; a failing handler aborts the recovery and the serve loop keeps running.
- `test_recovery_hang_monitor.py`: fence rules (every peer waiting, any progress restarts the window), starved ranks with native heartbeats, stall of every rank, command fence (host/pid checks, no shell, no key, one per rank). Both listed in `l0_cpu.yml`.

Testing: `tests/unittest/executor/test_recovery_authority.py` (new), `tests/unittest/executor/test_recovery_hang_monitor.py` (new) (CI: pending).

PR Checklist:
- [ ] PR description clearly explains what and why
- [ ] Follows TRT-LLM CODING GUIDELINES
- [ ] Test cases are provided for new code paths
- [ ] API-change label (`api-compatible` / `api-breaking`) added if the PR changes an API
- [ ] New dependencies scanned for license and vulnerabilities
- [ ] CODEOWNERS updated if ownership changes
- [ ] Documentation updated as needed
- [ ] tava architecture diagram updated if the design changes significantly
- [ ] Reviewers assigned are appropriate
- [ ] Please check this after reviewing the above items

## 03: [None][feat] Detect a hung WideEP rank with heartbeats, hang evidence and PMIx fencing
Base: feat/wide-ep-ft-mvp; Depends on: 02.
Description:
- Each worker reports heartbeats and long all-to-all device waits to the authority (`hang_evidence.py`).
- A native `HeartbeatSender` (nanobind, UDP, HMAC-SHA256) keeps beating while Python threads are starved.
- The authority's `Fence` ends a confirmed hung rank: its own evidence thread ends its process, else another rank has the MPI runtime kill it through PMIx (`pmix_fence.py`); its loss then recovers as any other.

With fault tolerance off:
- Hang evidence, the native sender and PMIx fencing start only with peer-loss recovery and `hang_detection` (default False).
- The only change for every build is one more source (`heartbeatSender.cpp`) in the nanobind module.

Design notes:
- The heartbeat is native so that it runs without the Python interpreter lock; a rank that is alive but starved is not mistaken for a stopped one. Without it, the Python heartbeat alone is used.
- The HMAC is computed in C++; the test covers a key within one HMAC block and a longer one (hashed to a block first).
- One fence thread runs at most one PMIx call; a later fence is skipped with a warning while one is outstanding, and the authority's command fence is the next fallback.
- `pmix_fence` loads libpmix through ctypes and reports `PmixUnavailable` where the runtime has none; the worker keeps running and the authority's fence command is the fallback.
- `kill_rank` raises `RuntimeError` for any refusal, naming the status `unknown` when `PMIx_Error_string` returns NULL, so a refusal is never mistaken for a fence that ran (callers that record it fail closed, PR 16).

Test coverage:
- `tests/unittest/_torch/executor/test_hang_evidence.py`: laggard rule, wait reporting and clearing, a fenced rank ends its own process, restart exit code, first heartbeat before serving, the native heartbeat arriving while Python is starved (both key lengths), PMIx fence with a fake libpmix, rank translation, failed fence leaves the worker running. Listed in `l0_b200.yml` (the entry arrives with PR 06).

Testing: `tests/unittest/_torch/executor/test_hang_evidence.py` (new) (CI: pending).

PR Checklist:
- [ ] PR description clearly explains what and why
- [ ] Follows TRT-LLM CODING GUIDELINES
- [ ] Test cases are provided for new code paths
- [ ] API-change label (`api-compatible` / `api-breaking`) added if the PR changes an API
- [ ] New dependencies scanned for license and vulnerabilities
- [ ] CODEOWNERS updated if ownership changes
- [ ] Documentation updated as needed
- [ ] tava architecture diagram updated if the design changes significantly
- [ ] Reviewers assigned are appropriate
- [ ] Please check this after reviewing the above items

## 04: [None][feat] Add ULFM survivor communicators for host collectives
Base: feat/wide-ep-ft-mvp; Depends on: M6 (mpi4py 4.1.2).
Description:
- Under peer-loss recovery `MPIDist` runs host collectives on a private ULFM world; a collective that a peer's loss interrupts raises `HostCollectiveInterrupted` instead of hanging or aborting the job.
- After a recovery the survivors' communicator serves the collectives, with logical ranks kept.
- A native `PeerFailureWatch` thread reports peer failures; `Distributed.allgather`, `tp_allgather` and `tp_gather` take `departed=`/`fill_value=`.

With fault tolerance off:
- `MPIDist._interruptible` runs the method unchanged while no private host world exists, i.e. until `enable_peer_loss_recovery()` (called only with recovery enabled).
- `ERRORS_RETURN` is set only on the private world, survivor and watch communicators.
- `departed=` and `fill_value=` matter only after a recovered loss; `live_ranks` is every rank otherwise.
- The watch thread starts only from recovery code.

Design notes:
- Each wrapped host collective is followed by an `MPI_Comm_get_failed` query (a group allocation and free). It is part of the measured cost of fault tolerance on healthy serving: mean decode step +0.35% to +0.87%, client throughput within noise (DEP32, DeepSeek-R1 NVFP4, 8 pairs per cell; the guide gives the table).
- Replaced communicators are not freed: freeing one with failed members is not guaranteed to be local. The count is bounded by the number of recoveries.
- After a non-ULFM exception the code polls for up to 1 s for failure evidence, because a garbled payload can fail to unpickle with any exception type.
- `departed=` is part of the abstract `Distributed` gathers; `TorchDist` accepts and ignores it. An out-of-tree `Distributed` subclass must accept it once callers pass it (PR 15).

Test coverage:
- `tests/unittest/_torch/distributed/test_mpi_survivors.py` (fake communicators, `cpu_only`): revoke of the private world, interruption classification, result-after-failure, `install_survivors` and `shrink`, unsupported layouts, hcoll/ULFM/mpi4py requirements, watch polling and release. It runs pre-merge in `l0_cpu.yml` through the existing `unittest/_torch/distributed` entry.
- `test_mpi_survivors_ulfm.py`: multi-process ULFM run in which survivors recover from departed ranks (optional second victim, frozen rank fenced through PMIx), driving the native watch. It skips without Open MPI with the ULFM (`ftmpi`) extension and mpi4py>=4, and fails instead with `TLLM_REQUIRE_ULFM=1`. The `l0_gb200_multi_gpus.yml` pre-merge entry arrives with PR 19.

Testing: `tests/unittest/_torch/distributed/test_mpi_survivors.py` (new), `tests/unittest/_torch/distributed/test_mpi_survivors_ulfm.py` (new) (CI: pending).

PR Checklist:
- [ ] PR description clearly explains what and why
- [ ] Follows TRT-LLM CODING GUIDELINES
- [ ] Test cases are provided for new code paths
- [ ] API-change label (`api-compatible` / `api-breaking`) added if the PR changes an API
- [ ] New dependencies scanned for license and vulnerabilities
- [ ] CODEOWNERS updated if ownership changes
- [ ] Documentation updated as needed
- [ ] tava architecture diagram updated if the design changes significantly
- [ ] Reviewers assigned are appropriate
- [ ] Please check this after reviewing the above items

## 05: [None][feat] Keep the MPI session serving and exit cleanly after a WideEP peer loss
Base: feat/wide-ep-ft-mvp; Depends on: 04.
Description:
- A departed peer's failure is left to the survivors' recovery instead of tearing down the MPI session.
- A rank ends through a bounded `MPI_Finalize` (at most 30 s), or SIGKILL on failure, so one rank's exit does not end the job.
- `trtllm-llmapi-launch` execs the worker so the MPI runtime sees its end; `mgmn_worker_node` goes through `run_worker_rank`.
- The startup peer-failure watch is stopped before `MPI_Finalize`; the executor stops its other watches before its serving loop returns (PR 16).

With fault tolerance off:
- `mpi_session`: departed-peer handling, the bounded finalize and the manager-thread excepthook sit behind `peer_loss_recovery_enabled()`.
- `trtllm-llmapi-launch` execs the worker, and defaults `OMPI_MCA_pml=ob1`, `btl=self,tcp`, `coll=^hcoll,ucc`, only when `TLLM_FAULT_TOLERANCE_MODE=1` and `TLLM_FAULT_TOLERANCE_AUTHORITY` are set.
- `mgmn_worker_node` always goes through `run_worker_rank`, which runs the worker unchanged; it now imports `tensorrt_llm.llmapi.mpi_session` at module load, which the worker imported later anyway.

Design notes:
- A survivor whose task fails after a loss ends the world with `MPI_Abort`. Errors that leave a single rank unusable are handled earlier: that rank ends itself (PR 16), so the survivors recover from its loss.
- The launcher execs the worker because each worker must be the process `mpirun` started for its death to reach its peers; a wrapper of the user's must exec too.
- ULFM needs the ob1 messaging layer; the launcher selects it over TCP unless the user set `--mca pml` / `btl`.
- `MPI_Finalize` is called through ctypes so that it releases the GIL and the 30 s timer can run. If the symbol cannot be looked up, mpi4py's `Finalize` runs instead, which holds the GIL; then only the native backstop (stack dump and SIGALRM at about 61 s) bounds it. The tests replace `_mpi_finalize`; the ctypes lookup itself is exercised only by the integration runs (PR 19).

Test coverage:
- `tests/unittest/llmapi/test_mpi_session.py`: server leaves a departed worker to the survivors, degraded-world finalize/abort/exit paths, startup watch stopped before finalize, `run_worker_rank` with and without recovery, no traceback from the futures manager.
- `test_trtllm_llmapi_launch_exit.py` (listed in `l0_cpu.yml`): the launcher with and without recovery, the user's transport choice kept, temporary workspace cleanup.

Testing: `tests/unittest/llmapi/test_mpi_session.py`, `tests/unittest/llmapi/test_trtllm_llmapi_launch_exit.py` (new) (CI: pending).

PR Checklist:
- [ ] PR description clearly explains what and why
- [ ] Follows TRT-LLM CODING GUIDELINES
- [ ] Test cases are provided for new code paths
- [ ] API-change label (`api-compatible` / `api-breaking`) added if the PR changes an API
- [ ] New dependencies scanned for license and vulnerabilities
- [ ] CODEOWNERS updated if ownership changes
- [ ] Documentation updated as needed
- [ ] tava architecture diagram updated if the design changes significantly
- [ ] Reviewers assigned are appropriate
- [ ] Please check this after reviewing the above items

## 06: [None][feat] Read the NVLinkOneSided rank mask through a pointer and add an abort record
Base: feat/wide-ep-ft-mvp; Depends on: M1, M5.
Description:
- The rank-mask all-to-all kernels read the active-rank mask through a device pointer and poll an abort record (`MoeA2AAbortState`), so a recovery can change membership without recapturing CUDA graphs and release kernels that wait on a lost peer.
- The ops take the mask as a CUDA tensor on the workspace device instead of a CPU tensor.
- The callers that check that the local rank is in the mask are the ones that publish it (`NVLinkOneSided._publish_rank_mask`, `MoeExecutionState`); `release_workspaces` checks it too.
- Rank-mask dispatch and combine record a released round when a peer's data never arrived, and the rank then publishes no round flag.

With fault tolerance off:
- Without `enable_rank_mask` the mask-free kernels run; abort polling and the released-round check are compiled only into the `ENABLE_RANK_MASK` instantiations.
- `abort_state` and `expert_to_slot` default to None; `moe_a2a_initialize`'s barrier defaults to True.
- The workspace layout change (8-byte `FLAG_VAL`) applies to every caller.

Design notes:
- Op contract: `moe_a2a_dispatch` and `moe_a2a_combine` require a CUDA `uint64` mask on the workspace device; a CPU mask is rejected. Callers of the ops with `enable_rank_mask` must change.
- The local-rank-bit check lives in the publishers (a mask without the local bit would make the rank skip its own flag and receive count) and in `release_workspaces`, which refuses to release the rank's own workspace.
- `expert_to_slot` needs the rank-mask kernels; the check sits in `moe_a2a_dispatch_launch`, where the instantiation is chosen.
- The mask is read before `cudaGridDependencySynchronize`; `install_mask` and `release_workspaces` synchronize before writing it.
- The abort record lives in pinned host memory; the op looks up its mapped device address and caches the last record's address per thread.
- The fake dispatch registration matches the real shapes (tested).

Test coverage:
- `tests/unittest/_torch/moe/test_moe_a2a_abort.py` (GPU, `l0_b200.yml`): cancel and exclusion on replay, masked sources read as empty, late sources, aborted rounds released, slot table and mask read at replay, receive counts, over-capacity rejection, the abort record needs the rank mask, fake/real shapes, the fence timeout still traps.
- `test_moe_comm.py`: updated for the CUDA mask. `test_hang_evidence.py` is added to the same list.

Testing: `tests/unittest/_torch/moe/test_moe_a2a_abort.py` (new), `tests/unittest/_torch/moe/test_moe_comm.py` (CI: pending).

PR Checklist:
- [ ] PR description clearly explains what and why
- [ ] Follows TRT-LLM CODING GUIDELINES
- [ ] Test cases are provided for new code paths
- [ ] API-change label (`api-compatible` / `api-breaking`) added if the PR changes an API
- [ ] New dependencies scanned for license and vulnerabilities
- [ ] CODEOWNERS updated if ownership changes
- [ ] Documentation updated as needed
- [ ] tava architecture diagram updated if the design changes significantly
- [ ] Reviewers assigned are appropriate
- [ ] Please check this after reviewing the above items

## 07: [None][perf] Test the all-to-all rank mask lane-parallel in the EPLB stats loop and load the round state at once
Base: feat/wide-ep-ft-mvp; Depends on: 06.
Description:
- Cuts two costs the rank mask adds to the NVLinkOneSided fence kernels, measured at EP32 with 64 tokens per rank.
- The dispatch's serial loop that writes EPLB statistics to every peer tested each peer's mask bit in turn (2.1 us of the 3.0 us that fault tolerance adds to a round trip with statistics). The lanes test 32 peers at once and ballot, and the serial walk reads a register bit (1.6 us less).
- The dispatch and the combine read the released-round marker in a second load after the release fence. The round flag value and the marker are read together in one 8-byte load (0.15 us less).

With fault tolerance off: the kernels are unchanged without the rank mask (the old loop stays under `else if constexpr (ENABLE_EPLB)`).

Design notes:
- The same peers receive the statistics and the same rounds publish no flag.
- The 8-byte load relies on the round flag and the marker being adjacent and 8-byte aligned: a `static_assert` fixes the marker index, and the workspace layout (8-byte `FLAG_VAL`, PR 06) aligns them; they already came first.
- `__ballot_sync` with the full mask is reached only through the warp-uniform last-token test.

Test coverage:
- No test file changes. The rank-mask kernels this PR modifies are run by `test_moe_a2a_abort.py` (PR 06), which checks the same peers' statistics, flags and released rounds.
- The timings above are not reproduced by a test or benchmark script in this PR.

Testing: no test files changed; `tests/unittest/_torch/moe/test_moe_a2a_abort.py` (PR 06) (CI: pending).

PR Checklist:
- [ ] PR description clearly explains what and why
- [ ] Follows TRT-LLM CODING GUIDELINES
- [ ] Test cases are provided for new code paths
- [ ] API-change label (`api-compatible` / `api-breaking`) added if the PR changes an API
- [ ] New dependencies scanned for license and vulnerabilities
- [ ] CODEOWNERS updated if ownership changes
- [ ] Documentation updated as needed
- [ ] tava architecture diagram updated if the design changes significantly
- [ ] Reviewers assigned are appropriate
- [ ] Please check this after reviewing the above items

## 08: [None][feat] Remap MNNVL all-to-all workspaces in place among the survivors
Base: feat/wide-ep-ft-mvp; Depends on: 06.
Description:
- After a peer loss, `release_backing` unmaps the MNNVL workspace pages and `remap` maps fresh pages at the same addresses, shared only among the survivors.
- Captured CUDA graphs keep their addresses, and a lost peer's late writes reach no live rank's pages.
- `pin`/`unpin` keep an allocation alive while a recovery holds it; the workspace reinit checks CUDA errors and restarts the watchdog.

With fault tolerance off:
- `remap`, `pin`, `unpin` and `release_backing` are called only by peer-loss recovery; `checkpoint_prepare` shares its unmap-and-release step with `release_backing`, unchanged.
- The workspace reinit CUDA error check and watchdog restart apply on reinit only.

Design notes:
- Remap runs an outcome exchange after the handle exchange: if any survivor failed to map or import, every survivor unmaps, so no rank keeps pages a peer then unmaps.
- A failed remap leaves the allocation unmapped and retryable; a retry must be made by every live rank together, because the handle exchange has no round tag. A failed cleanup keeps the remap's error and marks the allocation broken.
- `_LiveRowsComm` restricts an allocation's communicator to its live members; it implements only the communicator calls the mapping code uses.
- An allocation remapped without departed ranks refuses a checkpoint.

Test coverage:
- `tests/unittest/_torch/test_mnnvl_memory_remap.py` (fake driver, `l0_b200.yml`): fresh pages at the same addresses for live rows only, release after remap, failure ordering and retry, peer failure during the exchanges, cleanup failure, checkpoint refusal.
- `test_mnnvl_alltoall_workspace.py`: the watchdog restarts after the row reset. The multi-GPU late-write test is in PR 09.

Testing: `tests/unittest/_torch/test_mnnvl_memory_remap.py` (new), `tests/unittest/_torch/test_mnnvl_alltoall_workspace.py` (CI: pending).

PR Checklist:
- [ ] PR description clearly explains what and why
- [ ] Follows TRT-LLM CODING GUIDELINES
- [ ] Test cases are provided for new code paths
- [ ] API-change label (`api-compatible` / `api-breaking`) added if the PR changes an API
- [ ] New dependencies scanned for license and vulnerabilities
- [ ] CODEOWNERS updated if ownership changes
- [ ] Documentation updated as needed
- [ ] tava architecture diagram updated if the design changes significantly
- [ ] Reviewers assigned are appropriate
- [ ] Please check this after reviewing the above items

## 09: [None][feat] Bind NVLinkOneSided layers to a shared MoeExecutionState for WideEP recovery
Base: feat/wide-ep-ft-mvp; Depends on: 06 (08 for the workspace remap it drives).
Description:
- `MoeExecutionState` owns the rank mask and abort record that every NVLinkOneSided layer of a model reads, so one recovery updates all layers in place and bound rank-mask layers can run in CUDA graphs.
- A bound layer relies on the abort record and the device timeout instead of the host watchdog.
- It also releases and remaps the layers' workspaces in binding order and rearms the abort record.
- Under peer-loss recovery the all-to-all uses the fence path: CFT counted writes are not selected unless forced on, and binding then refuses.
- Rank-mask mode without recovery refuses a model that uses CUDA graphs when it is configured. The developer guide gains a section on the rank mask.

With fault tolerance off:
- `NVLinkOneSided` takes the bound-state path only when `_rank_mask_enabled` (`EPGroupHealth` given or `TLLM_FAULT_TOLERANCE_MODE=1`); otherwise dispatch and combine are unchanged.
- Unbound rank-mask layers still refuse CUDA graph capture.
- `MoeExecutionState` is created only by recovery callers.

Design notes:
- CFT counted writes would otherwise be auto-selected on supported hardware; under recovery the fence path is used and a forced CFT fails at binding with the variable to set.
- Eager bound calls skip the host watchdog as well; the abort record and the device timeout cover them.
- Detection-only mode (no authority) has no bound state, so graph capture there fails at configuration, not at capture.
- Late writes of a lost peer are tested against orphaned pages, with `remap` and without it as negative controls.
- `install_mask` refuses a membership that excludes one of the state's own ranks or an abort record that is set, and `staging_workspace` raises `RuntimeError` when no workspace is bound.

Test coverage:
- `test_moe_execution_state.py` (CPU, `l0_b200.yml`): mask install, abort/rearm after a peer lost mid-collective, workspace pinning and release order, a rank never releases its own workspace, unbound capture refusal, `for_model` binding.
- `test_alltoall_watchdog.py`: CUDA graph options by mode. Multi-GPU: `test_moe_a2a_late_writes.py` (a lost peer's late writes land only on orphaned pages), `test_nvlink_one_sided.py` (bound layers of two signatures on one workspace); `l0_gb200_multi_gpus.yml`. The guide states the late-write tests pass on one node with 4 GPUs.

Testing: `tests/unittest/_torch/moe/test_moe_execution_state.py` (new), `tests/unittest/_torch/modules/test_alltoall_watchdog.py`, `tests/unittest/_torch/moe/multi_gpu/test_moe_a2a_late_writes.py` (new), `tests/unittest/_torch/moe/multi_gpu/test_nvlink_one_sided.py`, `tests/unittest/_torch/moe/multi_gpu/_mnnvl_recovery_utils.py` (new helper) (CI: pending).

PR Checklist:
- [ ] PR description clearly explains what and why
- [ ] Follows TRT-LLM CODING GUIDELINES
- [ ] Test cases are provided for new code paths
- [ ] API-change label (`api-compatible` / `api-breaking`) added if the PR changes an API
- [ ] New dependencies scanned for license and vulnerabilities
- [ ] CODEOWNERS updated if ownership changes
- [ ] Documentation updated as needed
- [ ] tava architecture diagram updated if the design changes significantly
- [ ] Reviewers assigned are appropriate
- [ ] Please check this after reviewing the above items

## 10: [None][feat] Back empty EPLB slots with ghost memory and restore lost experts into them
Base: feat/wide-ep-ft-mvp; Depends on: base (M1-M6 merged).
Description:
- The C++ load balancer accepts empty (-1) slots backed by ghost slot memory (`GhostSlotMemory`, VMM chunks mapped to scratch until backed).
- `restorePlacement` restores a lost rank's experts into the survivors' slots.
- The balancer quiesces and resets its update schedule around a recovery; `HostAccessibleDeviceAllocator` gains register/unregister calls.

With fault tolerance off:
- These paths need peer-loss recovery (`mpi::isPeerLossRecoveryEnabled()`: `TLLM_FAULT_TOLERANCE_MODE=1` and an authority address, as `peer_loss_recovery_enabled()` in Python). `restorePlacement`, `quiesce` and `resetUpdateSchedule` check it.
- `waitCpuStage` and the worker thread are unchanged without it; -1 slots and ghost slot memory are accepted only with it.
- `moe_ghost_slot_storage` is requested only by WideEP extra slots.

Design notes:
- One PR, not three: a restore maps memory behind a ghost slot, and updates fill a ghost slot only once a restore loaded it, so ghost memory, `restorePlacement` and `quiesce` are one state machine and are tested together.
- `quiesce` abandons the CPU stage by setting the signal owner from the host; this is valid only once the device work of that iteration is done, which the header and the Python docstring state.
- Stray writes to an unbacked chunk land in scratch shared by all ranges; `moe_ghost_slot_storage` copies the whole old tensor, which is write-tolerant for that reason.
- `mOneReplicaPerExpert` is refreshed on placement installs, not on EPLB updates; it is read only with static routing (`layer_updates_per_iter == 0`).

Test coverage:
- `cpp/tests/unit_tests/runtime/moeLoadBalancerTest.cpp`: unusable-slot placement, FT-off rejection of -1 slots and restores, `restorePlacement` (loads, rejections, needs quiesce, needs host weights), quiesce paths, ghost-slot restore and CPU copies. `GhostSlotMemoryTest`: shared scratch, captured graph reads, allocation failure, sleep tags, CPU writes.
- `mpiUtilsTest.cpp`: `isPeerLossRecoveryEnabled` needs mode and authority.

Testing: `cpp/tests/unit_tests/runtime/moeLoadBalancerTest.cpp`, `cpp/tests/unit_tests/runtime/mpiUtilsTest.cpp` (CI: pending).

PR Checklist:
- [ ] PR description clearly explains what and why
- [ ] Follows TRT-LLM CODING GUIDELINES
- [ ] Test cases are provided for new code paths
- [ ] API-change label (`api-compatible` / `api-breaking`) added if the PR changes an API
- [ ] New dependencies scanned for license and vulnerabilities
- [ ] CODEOWNERS updated if ownership changes
- [ ] Documentation updated as needed
- [ ] tava architecture diagram updated if the design changes significantly
- [ ] Reviewers assigned are appropriate
- [ ] Please check this after reviewing the above items

## 11: [None][feat] Add extra EPLB slots for WideEP recovery and restore lost experts into them
Base: feat/wide-ep-ft-mvp; Depends on: 10.
Description:
- Under peer-loss recovery the Python load balancer appends extra (ghost or empty) slots and keeps host copies of the experts.
- After a loss it places every lost expert in the survivors' slots (`recovery_placements`) and restores it there.
- `failure_domains` tells the authority which losses that placement absorbs; `placement_digest` lets survivors check they agree.

With fault tolerance off:
- Extra slots are appended only with `peer_loss_recovery_enabled()`; otherwise `num_slots` and `expert_size_per_partition` are unchanged.
- `impl_blocks` slot rows and host copies with static routing only with recovery.
- `end_iter(interrupted)` only resumes after a raising forward with recovery (the binding receives `interrupted=False` otherwise).
- The hosts gather and `failure_domains` run only with recovery.

Design notes:
- Node identity is the host name the ranks report (`failure_domain: node` is defined that way in the guide). Host copies are shared through the MPI shared-memory communicator, so `_gather_hosts` also gathers each rank's shared-memory communicator and raises `ValueError` at startup unless host names and communicators group the ranks the same way (for example ranks in separate containers with one host name).
- Whether every rank can map ghost memory for the host is agreed in one collective (`_agree_ghost_slot_memory_host_accessible`, an allgather), run once per process by `register_weight_slots_after_to_cuda` on every rank with online EPLB under recovery; `extra_slots_mode()` only reads the agreed value, and raises `RuntimeError` if it needs it before the agreement.
- `placement_digest` hashes int32 placements, so the digest is identical on every rank.
- Online EPLB defaults to `empty` slots where ghost memory is not host accessible, because its weight updates would otherwise copy on the GPU every iteration.
- The hosts gather needs the EP group to be every rank, otherwise recovery is refused.

Test coverage:
- `test_moe_load_balancer.py`: a raising forward ends the iteration interrupted (and as before without recovery), shutdown barrier, slot rows, ghost storage copy ordering. `test_resident_placement.py`: host copies only with recovery, digest, evict-most-replicated and never-last-replica rules, agreement across survivors, online and static placements with replicas, DeepSeek-R1 EP32 node/rank losses.
- `test_recovery_config.py`: failure domains are every rank, then every host but rank 0's.

Testing: `tests/unittest/_torch/moe/test_moe_load_balancer.py`, `tests/unittest/_torch/moe/test_resident_placement.py` (new), `tests/unittest/executor/test_recovery_config.py` (CI: pending).

PR Checklist:
- [ ] PR description clearly explains what and why
- [ ] Follows TRT-LLM CODING GUIDELINES
- [ ] Test cases are provided for new code paths
- [ ] API-change label (`api-compatible` / `api-breaking`) added if the PR changes an API
- [ ] New dependencies scanned for license and vulnerabilities
- [ ] CODEOWNERS updated if ownership changes
- [ ] Documentation updated as needed
- [ ] tava architecture diagram updated if the design changes significantly
- [ ] Reviewers assigned are appropriate
- [ ] Please check this after reviewing the above items

## 12: [None][feat] Keep WideEP EPLB host copies in a ring of nodes and stage restores from them
Base: feat/wide-ep-ft-mvp; Depends on: 06, 11.
Description:
- Each expert's host copies live on a ring of nodes, so the loss of a whole node leaves a copy on a survivor; a restore stages the copies through the all-to-all workspaces into the target slots over NVLink.
- A static placement of one slot per expert keeps ring copies; its restores load only the slots recovery appends (`extra_slots_mode()`: ghost or empty).
- A layer with replicas keeps every expert's host copy on every node, since a restore may evict any replica. A loss that takes an expert's home rank and its copy node is not covered.
- Each rank maps its peers' copies lazily and drops its own page-table entries once written. On DeepSeek-R1 NVFP4 DEP32 GB200, after a page-cache drop on a used node, prefaulting every peer's copies stalled serving for 21-46 s; with lazy mapping and the owner's mappings dropped the stall is 2.7-5.7 s, against 2.2-3.2 s with fault tolerance off.

With fault tolerance off:
- Per-job `/dev/shm` names, the up-front reserve, unlinking once mapped and dropping the owner's mappings run only with `peer_loss_recovery_enabled()`; the existing shm lifecycle is kept without it.
- Ring copies and staging are used only by recovery; the host copies' split over a node's ranks is unchanged.

Design notes:
- Kernel compaction migrates `/dev/shm` pages and each migration interrupts every process that maps the page; hence lazy mapping and dropping the owner's entries (`madvise(MADV_DONTNEED)`; constant 4).
- `/dev/shm` sizing is checked per layer when it is created, and the node's ranks together at the up-front reserve; both fail at startup with the sizes, not at recovery. The sizes of all layers are not totalled before weights load.
- Weight reload (for example the RLHF reload, which finalizes each layer's host copies again) keeps main's segment lifecycle: as with fault tolerance off, the old segments are not closed (`own_shm` is replaced, imported peers' segments are appended to `imported_shms`, closed at shutdown), so `/dev/shm` holds one more copy per reload until shutdown. Under recovery the name is unlinked and created again, and the up-front reserve fails the reload with the sizes if `/dev/shm` cannot hold the new copy. Closing the old segments on reload is a change to main's lifecycle and is left out of this PR; no test covers a second successful finalize.
- Mixing ring and replica layers fails the model build with a `ValueError` (recovery on only).
- The staging plan alternates the two halves of the workspace row between chunks; `restore_copy_plan` raises `ValueError` when no aligned room exists or a placement would stage over a slot that still routes an expert.

Test coverage:
- `test_ring_host_copies.py` (CPU): sharer splits, no prefault, owner mappings dropped, shm sizing failures, per-job names, FT-off lifecycle, ring placement, copy plan (same on every rank, alternating halves, bit-exact staging), DeepSeek-R1 EP32 cases. `test_resident_placement.py`: replicas vs ring.
- `test_ring_restore_staging.py` (multi-GPU, `l0_gb200_multi_gpus.yml`, TIMEOUT 10): staged restore through peer workspaces, bit-exact.

Testing: `tests/unittest/_torch/moe/test_ring_host_copies.py` (new), `tests/unittest/_torch/moe/test_resident_placement.py`, `tests/unittest/_torch/moe/multi_gpu/test_ring_restore_staging.py` (new) (CI: pending).

PR Checklist:
- [ ] PR description clearly explains what and why
- [ ] Follows TRT-LLM CODING GUIDELINES
- [ ] Test cases are provided for new code paths
- [ ] API-change label (`api-compatible` / `api-breaking`) added if the PR changes an API
- [ ] New dependencies scanned for license and vulnerabilities
- [ ] CODEOWNERS updated if ownership changes
- [ ] Documentation updated as needed
- [ ] tava architecture diagram updated if the design changes significantly
- [ ] Reviewers assigned are appropriate
- [ ] Please check this after reviewing the above items

## 13: [None][perf] Tune and key the MoE tactics on the loaded slots under WideEP recovery
Base: feat/wide-ep-ft-mvp; Depends on: 11.
Description:
- The extra slots recovery appends hold no expert until a restore, so the Cutlass and CuTe DSL MoE runners profile and key their tactics on the loaded slots only (`loaded_slot_layout`, `_loaded_slot_tuner_inputs`).
- The Cutlass and CuTe DSL backends pass their extra local slots as `num_extra_local_experts`.

With fault tolerance off: `num_extra_local_experts` defaults to 0 and the backends pass 0, which keeps the existing tactic and autotuner cache keys.

Design notes:
- Interface: `trtllm::fused_moe` and three CuTe DSL ops (and their fakes) take a trailing `num_extra_local_experts: int = 0`; callers that omit it are unchanged.
- A tactic tuned on the loaded layout runs over every local slot: a slot without tokens adds no tile, and the tactics fix only tile shapes. The integration restore tests (PR 19) run the real kernels with extra slots, including restored ones.
- Not every path passes the argument (for example the Rubin gather path and TRTLLM-gen); those keep keying on all slots, which is safe.
- With extra slots, `loaded_slot_layout` raises `ValueError` unless `num_experts` and the local offset are multiples of `num_local_experts`, as the runners lay slots out; without extra slots it returns the layout unchanged.
- Both grouped GEMM calls in `fused_moe` share one `tuner_inputs` list.

Test coverage:
- `tests/unittest/_torch/moe/test_moe_runner.py`: runner keys extra slots as loaded slots; `fused_moe` profiles the weights of the loaded slots; quint4x2 weights sliced by expert.
- `.../kernels/parallel/test_cute_dsl_moe.py`: `loaded_slot_layout`, the grouped-GEMM inputs helper, runner and GEMM keys (metadata-level tests with a mock tuner). The `ValueError` of an uneven layout with extra slots has no test.

Testing: `tests/unittest/_torch/moe/test_moe_runner.py`, `tests/unittest/_torch/moe/kernels/parallel/test_cute_dsl_moe.py` (CI: pending).

PR Checklist:
- [ ] PR description clearly explains what and why
- [ ] Follows TRT-LLM CODING GUIDELINES
- [ ] Test cases are provided for new code paths
- [ ] API-change label (`api-compatible` / `api-breaking`) added if the PR changes an API
- [ ] New dependencies scanned for license and vulnerabilities
- [ ] CODEOWNERS updated if ownership changes
- [ ] Documentation updated as needed
- [ ] tava architecture diagram updated if the design changes significantly
- [ ] Reviewers assigned are appropriate
- [ ] Please check this after reviewing the above items

## 14: [None][feat] Roll back a discarded step's KV cache allocations in KV Cache Manager V2
Base: feat/wide-ep-ft-mvp; Depends on: base (M1-M6 merged).
Description:
- A recovery discards the step that a peer loss interrupted; `context_capacities` and `revert_prepare_resources` undo that step's KV cache allocations so its requests can run again.
- A generation request shrinks by the step's token and draft slots; a context request returns to its capacity before the step's resize; a first chunk that reused a prefix is freed and runs again from scratch.
- `revert_allocate_generation` and the new rollback share one helper (`_give_back_generation_step`).

With fault tolerance off: the rollback API is called only by the step discard of peer-loss recovery.

Design notes:
- `context_capacities` is taken right after scheduling: under the overlap scheduler the next step overwrites `py_ctx_pre_resize_cap` when it schedules the request's next chunk, before this step is accepted.
- Overlapped steps revert newest first.
- A reverted context step is not offered for block reuse, and a discarded later chunk keeps the committed chunks reusable.
- `revert_prepare_resources` must run after the executor has restored `context_current_position` (PR 16's discard does that).

Test coverage:
- `tests/unittest/_torch/executor/test_kv_cache_revert.py` (listed in `l0_b200.yml`): round trip, every shrink reported, reverted context not offered for reuse, a first chunk that reused a prefix runs again unchanged, two overlapped steps newest first, later chunk keeps earlier chunks reusable (sync and overlap), draft KV pool reverts with the target.

Testing: `tests/unittest/_torch/executor/test_kv_cache_revert.py` (new) (CI: pending).

PR Checklist:
- [ ] PR description clearly explains what and why
- [ ] Follows TRT-LLM CODING GUIDELINES
- [ ] Test cases are provided for new code paths
- [ ] API-change label (`api-compatible` / `api-breaking`) added if the PR changes an API
- [ ] New dependencies scanned for license and vulnerabilities
- [ ] CODEOWNERS updated if ownership changes
- [ ] Documentation updated as needed
- [ ] tava architecture diagram updated if the design changes significantly
- [ ] Reviewers assigned are appropriate
- [ ] Please check this after reviewing the above items

## 15: [None][feat] Keep attention-DP routing serving with absent ranks
Base: feat/wide-ep-ft-mvp; Depends on: 04.
Description:
- After a recovered peer loss the attention-DP gathers, the CUDA graph vote included, skip the departed ranks (`live_rows` drops a departed rank's row), and routing places requests only on the serving ranks.
- `DefaultADPRouter` and the admission check use per-rank dictionaries keyed by logical rank.
- A wake request (`ExecutorRequestQueue.wake()`) releases a blocked queue read without a request.
- `DecoderRunner` binds the model's NVLinkOneSided layers to one `MoeExecutionState` (`MoeExecutionState.for_model`, PR 09), which the recovery in PR 16 uses; `ModelEngine.close()` closes it.

With fault tolerance off:
- `departed=` on the gathers, the CUDA graph vote included, only matters after a recovered loss; `live_ranks` is every rank otherwise.
- A pinned target missing from the serving ranks is looked up by position (the existing indexing) unless `peer_loss_recovery_enabled()`.
- The wake request is dropped where it is read and never broadcast.
- `DecoderRunner` creates no `MoeExecutionState` without `peer_loss_recovery_enabled()`.

Design notes:
- Gathers take `departed=` (PR 04). Test doubles of `Distributed` (`fake_dist.py` and the ADP tests' fakes) accept it; an out-of-tree `Distributed` subclass must accept it too.
- The live-row filter is one helper, `live_rows`; it relies on a gather returning one row per logical rank.
- Only `DefaultADPRouter` supports absent ranks; the KV-cache-aware and conversation-aware routers still index lists by position, and PR 16 refuses non-default routers under recovery.
- Without recovery a pin past the last rank or a negative pin keeps its existing behavior ("as on main"), and tests pin it.
- A strict pin to a failed rank is unservable and fails; a relaxed pin is rebalanced.

Test coverage:
- `test_adp_absent_ranks.py`: gathers omit failed ranks, balancing over survivors by logical rank, strict/relaxed pins, graph vote, per-step votes, behavior without recovery as on main. `test_adp_host_exchange.py` (tp/cp gather fills departed rows), `test_executor_request_queue.py` (wake), `test_py_executor.py` (balancing continues after a loss).
- Updated for `departed=`: `test_adp_router.py`, `test_kvcache_aware_router.py`, `test_request_utils.py`, `fake_dist.py`, and other tests that stub the gathers.

Testing: in `tests/unittest/_torch/executor/`: `test_adp_absent_ranks.py` (new), `test_adp_host_exchange.py`, `test_adp_router.py`, `test_executor_request_queue.py`, `test_kvcache_aware_router.py`, `test_py_executor.py`, `test_remote_tail_adp_phase.py`, `test_request_utils.py`, `engine/runners/test_common.py`, `engine/runners/test_no_kv_cache.py`; also `tests/unittest/_torch/disaggregation/fake_dist.py`, `test_benchmark_disagg.py`, `tests/unittest/_torch/modeling/test_modeling_deepseekv41.py`, `.../hw_agnostic/test_dspark_cuda_graph_runtime.py` (CI: pending).

PR Checklist:
- [ ] PR description clearly explains what and why
- [ ] Follows TRT-LLM CODING GUIDELINES
- [ ] Test cases are provided for new code paths
- [ ] API-change label (`api-compatible` / `api-breaking`) added if the PR changes an API
- [ ] New dependencies scanned for license and vulnerabilities
- [ ] CODEOWNERS updated if ownership changes
- [ ] Documentation updated as needed
- [ ] tava architecture diagram updated if the design changes significantly
- [ ] Reviewers assigned are appropriate
- [ ] Please check this after reviewing the above items

## 16: [None][feat] Recover the PyTorch executor from a WideEP peer loss
Base: feat/wide-ep-ft-mvp; Depends on: 01-15 (authority, ULFM, all-to-all, EPLB, KV rollback, attention DP).
Description:
- Adds `PeerLossRecovery`, which drives a recovery through the authority in stages (stop and discard the interrupted steps, plan with the authority, install the survivors and the restored experts, resume), and the executor's recovery hook.
- Configurations it does not support are refused at startup (the guide lists them).
- Requests a lost rank had not finished run again on the survivors: ranks keep their part of each response gather for redelivery, and rank 0 notes which rank serves each request.
- After the commit the survivors exchange held and answered requests, requeue unscheduled ones, re-admit the lost ranks' undelivered requests, and fail only a request that a loss interrupted again without progress.

With fault tolerance off:
- `PyExecutor._peer_loss_recovery` is the do-nothing `RecoveryHook` unless `PeerLossRecovery.create` enables recovery; `create` returns the pass-through hook without `peer_loss_recovery_enabled()`. Every per-step hook call is guarded by `.enabled`.
- `publish_responses` keeps the enqueue-then-terminate order; a raising forward ends the iteration unchanged when no peer failed.
- Response-gather holding, quarantine and re-runs run only inside the enabled hook; rank 0 gathers responses without hook calls.
- The startup peer watch starts only with recovery and an `MPIDist`; `resize_kv_cache_quota` is called only by recovery; KV cache events are refused only with recovery.

Design notes:
- A rank whose loop fails with an error classified immediate-fatal, while admitted and outside a recovery, kills itself (SIGKILL), so the survivors recover from its loss; other errors are not treated this way.
- A forward error that is not a peer loss fails its requests as without fault tolerance; the hook then forgets the prepared steps of that forward (`steps_failed`), so the next step commits in order.
- Rank 0 ends departed ranks through PMIx before a plan (`fence_departed`, PR 02) and fails closed: any exception of `kill_rank` is reported to the authority as the fence's error (`Fenced(error=...)`), and a call that does not return within the time limit is reported as unanswered; only a call that returns reports the rank ended.
- `peer_loss_recovery.py` is one module (about 1,850 lines) because its stages are one state machine that shares the executor and the authority client.
- A request that an earlier loss interrupted and that completed no step since is quarantined and fails with a retryable error, because it may be what crashes the ranks.
- Recovery is not run while the executor gathers every rank's responses (KV cache estimation); a loss before serving ends the job.
- `LogitsStorage` and `PyResult` gain `checkpoint`/`rollback` so a discarded step's logits are dropped; `PyResult.rollback` does not roll back `diff`, which pipeline parallelism would need (PP is refused).
- Repetition, presence and frequency penalties and logits post-processors are rejected per request: their state cannot be rolled back.

Test coverage:
- `tests/unittest/_torch/executor/test_peer_loss_recovery.py`: protocol order and fail-closed transitions, node loss, loss during and after a recovery, fences before the shrink, discard/commit of steps, request re-runs, quarantine and cancellation, gather redelivery, `create` refusals and warnings, EPLB install and ring staging, `exits_alone` classification, startup watch, a failed forward that is no peer loss followed by a step that commits. Also `test_adp_absent_ranks.py`, `test_kv_pool_rebalance.py`, `test_multimodal_scheduler.py`; `test_py_executor.py` fakes gain the recovery hook.
- `tests/unittest/_torch/moe/multi_gpu/test_moe_a2a_recovery.py` (`l0_gb200_multi_gpus.yml`, TIMEOUT 10): remap orphans a late writer, second loss, aborted round with a run-ahead rank. The guide reports DEP32 runs on 32 GB200 GPUs (see PR 20).

Testing: `tests/unittest/_torch/executor/test_peer_loss_recovery.py` (new), `test_adp_absent_ranks.py`, `test_multimodal_scheduler.py`, `test_py_executor.py`, `kv_cache/test_kv_pool_rebalance.py` (same directory), `tests/unittest/_torch/moe/multi_gpu/test_moe_a2a_recovery.py` (new) (CI: pending).

PR Checklist:
- [ ] PR description clearly explains what and why
- [ ] Follows TRT-LLM CODING GUIDELINES
- [ ] Test cases are provided for new code paths
- [ ] API-change label (`api-compatible` / `api-breaking`) added if the PR changes an API
- [ ] New dependencies scanned for license and vulnerabilities
- [ ] CODEOWNERS updated if ownership changes
- [ ] Documentation updated as needed
- [ ] tava architecture diagram updated if the design changes significantly
- [ ] Reviewers assigned are appropriate
- [ ] Please check this after reviewing the above items

## 17: [None][feat] Report WideEP fault-tolerance state on /health and return 503 on peer loss
Base: feat/wide-ep-ft-mvp; Depends on: 16, M3.
Description:
- `/health` reports the fault-tolerance state the recovery authority holds, at most 0.1 s old (`healthy`, `degraded`, `recovering`, `unprotected`, `terminal`).
- A request that a peer loss failed gets HTTP 503 with `Retry-After`, and a retryable code in a stream's error event, since a retry can succeed.
- The probe (`fault_tolerance_status.py`) answers on its own thread, which `/health` awaits; if the authority does not answer, the state is `unprotected`, never `healthy`.

With fault tolerance off:
- The `/health` probe exists only with peer-loss recovery.
- The 503 + `Retry-After` mapping and the retryable code of a stream's error event apply only when `fault_tolerance_mode()` is on; with the mode on and no authority `/health` returns `{"state": "unprotected"}`.

Design notes:
- The body carries the state, the generation, live and failed ranks, recovery counters, the last recovery's stage durations and the number of failed requests: what operators alert on. Its fields become an interface monitoring depends on. The abort reason stays in the authority's log, because `/health` is not authenticated, so the body exposes rank ids and counts only.
- `/health` returns 503 only while `recovering` or `terminal`; a probe that polls less often than a recovery lasts (about 0.7-0.9 s) can miss it. The guide gives probe settings.
- The probe queries the authority on one thread (0.5 s timeout) and keeps the answer for 0.1 s, stamped when the answer arrives, so polls queued behind a slow or unanswered query reuse its result instead of querying again.
- Querying the authority changes nothing on the authority.
- The retryable decision matches a tag in the error message; a message without it stays an ordinary error (400, no `Retry-After`).

Test coverage:
- `test_fault_tolerance_health.py` (`l0_cpu.yml`): `unprotected` with detection only, 503 only while recovering or terminal, probe closed at lifespan teardown.
- `test_recovery_authority.py`: the probe reads status, answers on any thread and within its timeout while the authority is down, drops bad answers; health states follow the authority phase; status body.
- `test_stream_error_events.py`: peer-loss failure streams a retryable code, is a 503 with `Retry-After`, an ordinary error without fault tolerance, other errors stay 400.

Testing: `tests/unittest/llmapi/apps/test_fault_tolerance_health.py` (new), `tests/unittest/llmapi/apps/test_stream_error_events.py`, `tests/unittest/executor/test_recovery_authority.py` (CI: pending).

PR Checklist:
- [ ] PR description clearly explains what and why
- [ ] Follows TRT-LLM CODING GUIDELINES
- [ ] Test cases are provided for new code paths
- [ ] API-change label (`api-compatible` / `api-breaking`) added if the PR changes an API
- [ ] New dependencies scanned for license and vulnerabilities
- [ ] CODEOWNERS updated if ownership changes
- [ ] Documentation updated as needed
- [ ] tava architecture diagram updated if the design changes significantly
- [ ] Reviewers assigned are appropriate
- [ ] Please check this after reviewing the above items

## 18: [None][perf] Look up dispatch slots in a device expert-to-slot table under WideEP recovery
Base: feat/wide-ep-ft-mvp; Depends on: 06, 11.
Description:
- When every expert has one slot, as under WideEP recovery, the all-to-all dispatch looks up each token's slot in a device expert-to-slot table instead of running the replica-choosing routing kernel.
- The table is a view of the load balancer's global slot ids, rewritten in place by a restore, so a replayed CUDA graph reads the current placement.
- `moe_load_balance_routing` passes `oneSlotPerExpert = routesEachExpertToOneSlot()`.

With fault tolerance off:
- `ExternalCommMoEScheduler` uses the table only with `peer_loss_recovery_enabled()`; otherwise `route()` is unchanged and NVLinkOneSided passes no `expert_to_slot`.
- `routesEachExpertToOneSlot()` is false without fault-tolerance mode, so the existing routing kernel runs.
- A direct call of `moe_a2a_dispatch` with `expert_to_slot` but without `enable_rank_mask` raises (`TORCH_CHECK`: "expert_to_slot requires enable_rank_mask=True").

Design notes:
- The table kernel is instantiated only with the rank mask; the op (`TORCH_CHECK`) and `moe_a2a_dispatch_launch` (`TLLM_CHECK`) both refuse a table without it, so it cannot silently run the non-table kernel.
- `one_slot_table` caches the first eager result, including None. It is safe because the committed one-slot routing is fixed once chosen, and the routing kernel serves every placement.
- The table is valid while the layer's balancer exists; a captured graph must not outlive it.

Test coverage:
- `tests/unittest/_torch/moe/multi_gpu/test_nvlink_one_sided.py::test_dispatch_through_slot_table`: slot ids match the lookup-free dispatch bit for bit, eager and with the table rewritten in place under a replayed graph; the layer is bound to a `MoeExecutionState`, as under recovery, since only the rank-mask kernels read the table.
- `test_moe_execution_state.py`: the scheduler records recovery and finds the table; only recovery dispatches by it. `test_moe_load_balancer.py`: the table is what routing reads; a None result is kept.

Testing: `tests/unittest/_torch/moe/multi_gpu/test_nvlink_one_sided.py`, `tests/unittest/_torch/moe/test_moe_execution_state.py`, `tests/unittest/_torch/moe/test_moe_load_balancer.py` (CI: pending).

PR Checklist:
- [ ] PR description clearly explains what and why
- [ ] Follows TRT-LLM CODING GUIDELINES
- [ ] Test cases are provided for new code paths
- [ ] API-change label (`api-compatible` / `api-breaking`) added if the PR changes an API
- [ ] New dependencies scanned for license and vulnerabilities
- [ ] CODEOWNERS updated if ownership changes
- [ ] Documentation updated as needed
- [ ] tava architecture diagram updated if the design changes significantly
- [ ] Reviewers assigned are appropriate
- [ ] Please check this after reviewing the above items

## 19: [None][test] Add WideEP peer-loss recovery integration tests
Base: feat/wide-ep-ft-mvp; Depends on: 16, 17.
Description:
- End-to-end tests: an EP4 deployment (DeepSeek-V3-Lite, attention DP) survives the loss of workers and of its recovery authority.
- A driver runs on MPI rank 0 under `trtllm-llmapi-launch`, SIGKILLs (or SIGSTOPs) the victim ranks' worker processes and records what every request returned.
- The tests need Open MPI with ULFM and mpi4py>=4 and skip without them; `TLLM_REQUIRE_ULFM=1` makes them fail instead.
- One case runs pre-merge on GB200, the rest post-merge.

With fault tolerance off: tests only. `test_serving_without_fault_tolerance` runs the same launch with the fault-tolerance variables unset and checks every request is answered (it keeps the ULFM transport settings of the helper).

Design notes:
- Only worker processes that carry the run's token are faulted.
- After a loss the tests do not assert exact tokens: lost experts are served from other GPUs, which changes the reduction order and may change greedy tokens. Each answer must carry its expected word and exactly `max_tokens` tokens (requests ignore EOS, so a duplicated or dropped token changes the count), and a survivor's first tokens match the healthy round.
- Single-node EP4 runs with `failure_domain: rank`; the guide states that whole-host loss and `failure_domain: node` are covered only by the DEP32 runs.
- A `sitecustomize` hook injects the loss at a recovery stage.

Test coverage:
- `tests/integration/defs/wide_ep_ft/test_peer_loss_recovery.py`: `test_peer_loss_recovery[idle|inflight x sync|overlap]`, `test_serving_without_fault_tolerance`, `..._under_load`, `test_two_ranks_lost_together`, `test_rank_lost_during_recovery`, `test_restore_into_ghost_slots[static|online]`, `test_restore_into_appended_empty_slots`, `test_hung_rank_fenced`, `test_authority_lost_while_serving`.
- The guide reports 52 of 52 DEP32 recovery runs passed.

CI coverage:
- `l0_gb200_multi_gpus.yml` (GB200, 4 GPUs) pre-merge: `test_peer_loss_recovery[idle-sync]` (TIMEOUT 45) and `unittest/_torch/distributed/test_mpi_survivors_ulfm.py`. Post-merge: the other 13 cases of `test_peer_loss_recovery.py` (TIMEOUT 45 each).
- These tests skip unless `mpirun` and `ompi_info` exist, mpi4py is 4 or later and `ompi_info` lists the ULFM `ftmpi` extension (`_ulfm_unavailable_reason` / `_ulfm_available`); with `TLLM_REQUIRE_ULFM=1` they fail instead. The integration cases also skip without `DeepSeek-V3-Lite/nvfp4_moe_only` under `LLM_MODELS_ROOT`. The CI test images do not have ULFM Open MPI and no CI stage sets `TLLM_REQUIRE_ULFM`, so in CI these entries skip and a green stage does not show they ran. Making them effective needs an image with ULFM Open MPI and `TLLM_REQUIRE_ULFM=1` on that stage (follow-up).
- What runs in CI without ULFM: the unit tests of PRs 01-18 on their lists, for example `tests/unittest/_torch/distributed/test_mpi_survivors.py` (fake communicators) in `l0_cpu.yml` through its `unittest/_torch/distributed` entry, and the multi-GPU tests `test_moe_a2a_late_writes.py`, `test_moe_a2a_recovery.py` and `test_ring_restore_staging.py` in `l0_gb200_multi_gpus.yml` pre-merge.
- The pre-merge case's TIMEOUT (45) is to be re-measured once an image provides ULFM.

Testing: `tests/integration/defs/wide_ep_ft/test_peer_loss_recovery.py` (new), with `peer_loss_driver.py` and `worker_hooks/sitecustomize.py`; local run on GB200 (1 node, 4 GPUs, DeepSeek-V3-Lite NVFP4 EP4, ULFM Open MPI): 14 passed (job 8481305). CI: pending.

PR Checklist:
- [ ] PR description clearly explains what and why
- [ ] Follows TRT-LLM CODING GUIDELINES
- [ ] Test cases are provided for new code paths
- [ ] API-change label (`api-compatible` / `api-breaking`) added if the PR changes an API
- [ ] New dependencies scanned for license and vulnerabilities
- [ ] CODEOWNERS updated if ownership changes
- [ ] Documentation updated as needed
- [ ] tava architecture diagram updated if the design changes significantly
- [ ] Reviewers assigned are appropriate
- [ ] Please check this after reviewing the above items

## 20: [None][doc] Document WideEP fault tolerance
Base: feat/wide-ep-ft-mvp; Depends on: 19 (documents 01-19).
Description:
- Adds `docs/source/features/wide-ep-fault-tolerance.md`: requirements, running (authority, launch, variables and settings), what is covered and not, timeouts, known limits, monitoring, how well each part is tested, cost.
- Adds the page to the docs index, a fault-tolerance table in the feature combination matrix, and an entry in the Wide-EP example README.

With fault tolerance off: documentation only.

Design notes:
- Measured numbers in the guide, all on DEP32 with DeepSeek-R1 NVFP4 on GB200: first step after a recovery 0.8-1.6 s for a rank kill and 1.4-2.8 s for a node kill; a frozen rank fenced 29.5-34.1 s after the freeze; GSM8K 0.9477 after a rank kill against 0.9469 with fault tolerance off; mean decode step +0.35% to +0.87% with fault tolerance on, client throughput within noise.
- The guide states the CI scope: multi-GPU tests on one 4-GPU GB200 node with `failure_domain: rank`; whole-host loss and `failure_domain: node` are covered only by DEP32 runs.
- The guide states that CFT counted writes are not measured with fault tolerance, and that the feature is experimental.
- The feature matrix lists the refused or limited combinations: pure attention DP only, PP/CP/expert TP refused, detection-only mode refuses CUDA graph capture, disaggregated serving/guided decoding/beam search/LoRA refused, per-request rejections.

Test coverage:
- No code; no tests. The documented behavior is covered by PRs 01-19; the guide's "How well each part is tested" section lists what is and is not covered. Which tests CI runs, and that the ULFM tests skip on the current CI images, is stated in PR 19.

Testing: no test files changed; docs build (CI: pending).

PR Checklist:
- [ ] PR description clearly explains what and why
- [ ] Follows TRT-LLM CODING GUIDELINES
- [ ] Test cases are provided for new code paths
- [ ] API-change label (`api-compatible` / `api-breaking`) added if the PR changes an API
- [ ] New dependencies scanned for license and vulnerabilities
- [ ] CODEOWNERS updated if ownership changes
- [ ] Documentation updated as needed
- [ ] tava architecture diagram updated if the design changes significantly
- [ ] Reviewers assigned are appropriate
- [ ] Please check this after reviewing the above items

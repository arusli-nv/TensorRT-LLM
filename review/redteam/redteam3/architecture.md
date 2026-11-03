# Red team: architecture and layering (staff/principal lens)

Tree: v2 `cef4670789` vs upstream/main merge-base `0829689c52`: 121 files, +23.5k/-0.7k lines. I read the in-tree
`docs/source/features/wide-ep-fault-tolerance.md`, `mvp/DESIGN_DOC.md`, the MoE guide, and the diffs of
py_executor, peer_loss_recovery, communicator, mpi_session, moe_scheduler, recovery/*, openai_server, the launcher,
the custom ops and install_mpi4py.sh. I ran nothing on a GPU.

## Verdict
The core mechanism fits TRT-LLM well and a principal engineer would likely accept it: fixed slots and graphs; tables
the graphs read; an abort word polled by the a2a kernels; an EPLB-owned placement; a null-object `RecoveryHook` in
`PyExecutor`; and recovery behind the `Distributed` abstraction (`enable_peer_loss_recovery`, which the base class
leaves `NotImplementedError`). It reuses main's seams (`wide_ep_ft.py`, `EPGroupHealth`, `alltoall_watchdog`, EPLB
GDRCopy, KV Cache Manager V2 resize). The packaging around it is what draws pushback: process-env configuration
read in ~15 modules, an operator-run network service, an Open MPI-only transport dependency, and one 23k-line
change. None of that is a design flaw. All of it gets a reviewer to stop before the code itself.

## Ranked pushbacks (most likely to block first), each with the cheapest acceptable resolution

1. **Config comes from env vars and a side YAML instead of LlmArgs (blocker).**
   - Current state: `TLLM_FAULT_TOLERANCE_MODE`, `_AUTHORITY`, `_AUTHORITY_KEY` and `_CONFIG` are read through
     `functools.cache`/`lru_cache` process globals in `_utils.py` and `recovery/config.py`. They are consulted from
     moe_load_balancer (13 times), moe_scheduler, impl_blocks, the decoder runner, adp_router, request_utils,
     mpi_session and openai_server.
   - Main's own `wide_ep_ft.py` says FT "remains opt-in until the integration PR wires a public model option". This
     is that integration PR.
   - The MoE guide makes `impl_environment.py` the only place that probes env; `moe_scheduler` now reads it directly.
   - Going around LlmArgs also skips the golden manifest, the serve schema and telemetry CODEOWNER review. Reviewers
     will read that as getting around the governance process, not as a convenience.
   - Cheapest fix: add a prototype-status `fault_tolerance: Optional[WideEPFaultToleranceConfig]` on
     `TorchLlmArgs`, reusing the existing `FaultToleranceConfig` pydantic model as is. Thread it through
     `ModelConfig`/`extra_attrs`, the path `wide_ep_ft.py` already uses. Keep env only for the secret key and the
     authority address (or set them through the launcher). Regenerate the manifest and schemas.
2. **The compatibility checks run after model load, in `PeerLossRecovery.create` (blocker, cheap).**
   - About 18 hand-coded requirements (MPI executor, ADP router, no beam/guided/PEFT/transceiver/connector/LoRA-TP,
     KV v2, sleep, hang timeout ...) fail only after weights load and CUDA graphs are captured, which costs minutes
     at DEP32.
   - This list duplicates `feature-combination-matrix.md`.
   - Fix: move the static checks into the LlmArgs `model_validator` of item 1. Keep only the runtime-only checks
     (EPLB slot coverage, ULFM probing) in `create`.
3. **The recovery authority is a new, operator-run network process (blocker as shipped; the cheap version is
   fine).**
   - Operators must start a separate ZMQ+UDP service, generate and distribute an HMAC key, bind it to a private
     interface, and accept that "any host that can reach its port can disrupt recovery". A `--fence-command`
     template runs subprocesses.
   - No other TRT-LLM feature needs a sidecar the operator launches by hand, and it fails open: the deployment goes
     `unprotected`.
   - Rank 0 and its node are unrecoverable by design, so co-locating the authority with rank 0 costs no coverage.
   - Fix: have the launcher (`trtllm-llmapi-launch`/`trtllm-serve` on global rank 0) spawn the authority as a child
     process outside the MPI process group. Generate the key in-process and broadcast it once over MPI at
     registration. Keep `--endpoint` for advanced users.
   - Accept the remaining items as follow-ups: TLS/CURVE, authority restart, and a k8s sidecar form.
4. **Mandatory ULFM / Open MPI ob1-TCP, plus a global mpi4py 3.1.5 to 4.1.2 bump (blocker only for the bump).**
   - ULFM is Open MPI-only. It needs ob1 over TCP, even within a node (no shm, no UCX), an mpirun flag set, and an
     experimental `mpi_ft_detector`. The CI container does not ship it, so all 12 GB200 integration cases skip
     unless `TLLM_REQUIRE_ULFM=1`.
   - It moves every FT-on rank's executor host collectives onto a different transport. Its cost has not been
     separated from the mechanism's cost.
   - It cannot serve MPI-less launches (`TLLM_DISABLE_MPI`, Ray, RPC, Dynamo/K8s), which are where WideEP is
     heading.
   - `MPI_Finalize` hangs, so it is bounded at 30 s by ctypes plus `faulthandler`.
   - `pmix_fence.py` finds libpmix through `/proc/self/maps` and defines PMIx structs by hand. That is ABI coupling
     with no version check.
   - The mpi4py 4 bump changes every user's container and re-ports an mpi4py.futures patch
     (`TRTLLM_USE_MPI_KVCACHE`).
   - Acceptable for an experimental MVP if (a), (b) and (c) are met:
     - (a) The mpi4py bump lands first as its own PR with full CI.
     - (b) ULFM stays strictly behind the `Distributed` seam it already uses.
     - (c) A ULFM CI image or stage exists, so the FT tests run rather than skip.
   - Alternatives, in rough order of merit:
     - **Authority-owned membership, no ULFM:** the authority already commits membership. Detection can come from
       the native heartbeat (already built) plus the a2a abort/watchdog (already built). Host collectives can move
       to a communicator the survivors rebuild after a commit (gloo, or a fresh MPI/ZMQ group). This removes the
       ob1/TCP penalty and the Finalize hang, and works without mpirun.
     - **NCCL `ncclCommShrink`/`ncclCommAbort` (NCCL >= 2.27) via `TorchDist`:** shrink and abort are built in.
       This is the natural path for the MPI-less `TorchDist` backend.
     - **NVSHMEM/MNNVL-level detection:** this is only detection. The a2a abort word already covers the device
       side, and it still needs an agreement layer.
     - **Orchestrator restart (Dynamo shadow engines, WARP fast restart):** zero code in the engine, but minutes
       per loss, versus about 1 s here. It complements FT rather than replacing it.
     - **Disagg:** a lost gen worker is replaced by the router. This does not help an aggregated DEP32 world or a
       wide-EP gen worker internally.
5. **Too big to review as one change (blocker on process).**
   - Fix: split along the layers already present.
     - (i) mpi4py 4 bump.
     - (ii) The OpenAI stream SSE error-event change. This changes FT-off behaviour for every stream and should
       stand alone.
     - (iii) a2a fixed per-sender mailboxes, abort and mask kernels, and the new op args.
     - (iv) EPLB ghost slots and `ghostSlotMemory`, plus host copies and ring staging.
     - (v) `Distributed` peer-loss seam (ULFM).
     - (vi) Executor step commit/discard and KV v2 revert.
     - (vii) Authority, protocol and hang monitor.
     - (viii) `/health`, docs and integration tests.
   - Each slice must keep FT-off byte-identical in behaviour and show FT-off perf parity.
6. **Op and kernel surface creep that FT-off pays for.**
   - `moe_a2a_dispatch` gains about 9 optional params. Cutlass fused MoE gains `input_source_tokens`/
     `input_slot_tokens`. `moe_a2a_initialize` gains `barrier`. Kernel signatures change.
   - `moe_scheduler` adds concrete `isinstance(moe.comm, NVLinkOneSided)` checks, 8 in all against main's 5. The
     MoE guide says to use protocol discovery and not add concrete strategy checks.
   - Fix: bundle the FT inputs into one optional struct/tensor-list argument (`FtDispatchState`). Expose the receive
     slots through a `Communication` capability (`dispatch_source_slots()` on the ABC, returning None by default)
     instead of isinstance checks.
   - Require a microbenchmark showing FT-off kernels unchanged; follow-up acceptable if numbers show parity.
7. **Third hang detector and process-killing policy spread across layers.**
   - Main already has the executor `HangDetector` (300 s) and `alltoall_watchdog`. FT adds `hang_evidence`
     (worker), `hang_monitor` (authority), a C++ `HeartbeatSender`, PMIx fencing, a self-`SIGKILL` in
     `PyExecutor._event_loop_wrapper`, a `SIGKILL` of the launcher, and exit code 75.
   - The doc lists seven process-ending paths. The timeouts are coupled by hand: FT checks `hang_detector.timeout`
     against its own.
   - Fix: hang detection is already off by default, so mark it experimental and ship it in a separate slice (vii).
     Route the self-kill decision through one `FaultPolicy` object that the executor asks. Make the executor
     `HangDetector` the single source of "no step progress", and have FT subscribe to it.
8. **Implementation code in the nanobind layer.**
   - `peerFailureWatch.cpp` and `heartbeatSender.cpp`, about 440 lines with threads, MPI and UDP, live in
     `cpp/tensorrt_llm/nanobind/runtime/`. They call `PyGILState_Check` in destructors and cannot be tested with
     gtest.
   - Fix: move the logic to `cpp/tensorrt_llm/runtime/` and keep thin bindings. Follow-up acceptable.
9. **Global and module state.**
   - `_ghost_host_accessible` (moe_load_balancer), `_startup_watch` (peer_loss_recovery), cached env config, class
     attributes on `PyExecutor` used as defaults, and `mpi_session` reaching into
     `sys.modules["...peer_loss_recovery"]` to stop a watch.
   - Fix: own these on the executor/LoadBalancer instance and pass them down. Replace the `sys.modules` lookup with
     a registered shutdown callback. Cheap, so do it before merge.
10. **The executor hook is FT-shaped where a general capability is called for.**
    - About 12 callbacks are threaded through both event loops: `step_prepared`, `commit_step`,
      `broadcast_requests`, `requests_routed`, `gathering_responses`, and so on. Step commit/discard with a KV v2
      revert is generic: sleep/wakeup, PP rebalance and preemption could use it.
    - `peer_loss_recovery.py` (1.66k lines) touches about 35 executor attributes.
    - Fix: accept for now, as the null-object hook keeps FT-off paths clean. Follow-up: promote "transactional
      step" into the executor with a narrow interface, and shrink the attribute surface to the public methods
      already added (`publish_responses`, `readmit_requests`, `discard_interrupted_steps`, ...).
11. **The serving layer talks to the authority directly.**
    - `openai_server` builds a `RecoveryStatusProbe.from_env()` and queries the authority over ZMQ on every
      `/health`, bypassing `LLM` → executor.
    - The gRPC frontend and the disagg server get no FT state.
    - Fix (follow-up): expose `llm.fault_tolerance_status()` (or put the status in executor stats) and have all
      frontends use it.
12. **Strategic fit.**
    - WideEP at scale in production is mostly disagg (gen workers), and FT refuses KV transceivers. Aggregated
      DEP32 is the less common deployment.
    - Host copies take 368 GB of `/dev/shm` per node with replicas.
    - Fix: no code change. State in the PR that disagg-gen FT is the next target, and show that the KV-transceiver
      refusal is the only blocker there.

## Require before merge vs accept as follow-up
- **Before merge:**
  - Items 1 and 2 (LlmArgs config plus early validation).
  - Item 3 (the launcher spawns the authority, with the key auto-generated and broadcast).
  - Item 4(a)(c) (separate mpi4py PR, and a ULFM CI stage so the FT tests run).
  - Item 5 (split).
  - Item 6's FT-off perf parity evidence, and the `isinstance` to capability change.
  - Item 9 (no module globals).
  - The SSE error-event change as its own PR.
- **Follow-up (track in the PR description):**
  - Item 4's non-ULFM backend (authority-owned membership, or NCCL shrink via `TorchDist`).
  - Items 7, 8, 10 and 11.
  - Encryption and restart for the authority.
  - Disagg-gen support.
  - In-place mailbox reads for DeepGEMM/TRTLLM-Gen.

## What a principal would not push back on
- One set of CUDA graphs plus table rewrites, instead of recapture. This is the right call for recovery time.
- Per-sender fixed mailboxes, so a late NVLink write is contained without a per-step copy.
- Placement as a deterministic function every survivor computes, checked by hash at the authority.
- Fail-closed startup, and a clear list of unsupported combinations.
- A null-object hook, so FT-off executor branches are just `enabled` checks.
- Rank 0 declared unrecoverable. It is honest and keeps the llmapi/serve layer untouched.

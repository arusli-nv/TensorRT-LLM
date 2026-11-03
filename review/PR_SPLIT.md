# WideEP FT (v5-zero) -> `feat/wide-ep-ft-mvp`: proposed PR split

Source: `/tmp/claude-161238/wt-zero`, branch `v5-zero` @ `7a03252e92`, 378 commits on upstream main `0829689c52`.
Net diff: 121 files, +24798 / -708 (about 9.3k source, 13.5k tests, 0.35k docs, 0.9k integration/CI).
This is a read-only plan: nothing was built or run. Line counts come from `git diff --numstat 0829689c52 HEAD`.
Where one file is split across PRs, its count is marked `~` as an estimate.

Main already has the rank-mask A2A (mask passed by value), `EPGroupHealth`, `alltoall_watchdog.py` and
`wide_ep_ft.py`. The branch builds on those, so several PRs change main's existing interfaces rather than
adding new ones. Those changes are the main review risk.

## Ground rules applied
- PR title `[None][type] ...` (swap in a JIRA ID when one exists). One concern per PR, aiming at <1500 changed
  lines. Test files are allowed to push a PR over the limit where the source stays small.
- Land each PR as a squashed, rewritten change, not a cherry-pick of the history. The history contains
  5 revert pairs (eplb_recovery.py, authority_server.py, mpi_exit.py and failure_domains moves; the
  Cutlass expand-rows remap; the CuteDSL gather FC1 bake) and many fix-on-fix chains, all of which are
  net-zero or net-small.
- Each PR has to be a no-op when `TLLM_FAULT_TOLERANCE_MODE` is unset. Many late commits restore main's
  behaviour when FT is off ("as main does without peer-loss recovery", e.g. 86a7ab7b0b, c8df55f5df,
  e8001b1be0, a1a7c2a687, ae76f7b9fc, d373d8e514, c6d48b7eba, c31ce145fd, b5cec98430, 88860b1d6b). Each PR
  description should list its FT-off guards so reviewers can check them.
- CODING_GUIDELINES points that recur: specific exceptions only; `ValueError`, not `assert`; Pydantic for
  user-facing config; no dead or commented-out code; comments describe the code as it stands, not its
  history. Many branch docstrings narrate history ("now", "as main does", "instead of"), so trim them per
  PR.

## Dependency graph (landing order)

```
P0a fake-op fix ─┐
P0b EPGroupHealth deepcopy ─┤ (independent; better sent to main)
P0c OpenAI stream error events ─┘
P0d mpi4py 4.1.2 ──> P1 ULFM communicators ──> P2 MPI session/launcher
                                   │
P3 recovery config/protocol/client ┴──> P4 recovery authority
P5a A2A kernel/op: mask ptr + abort ──> P5b MoeExecutionState + NVLinkOneSided binding ──> P6 workspace remap
P7a EPLB C++ restore/quiesce ──> P7b ghost slot memory (device VMM) ──> P8a Python EPLB extra slots + restore
                                                                  └──> P8b ring host copies + staging
P8a ──> P9 autotuner loaded-slot keys
P10 KV cache v2 rollback (independent)
P1 ──> P11 attention-DP absent ranks + departed= call sites
P1,P3 ──> P12a executor RecoveryHook + step discard/commit
P2,P4,P5b,P6,P8a/b,P10,P11,P12a ──> P12b PeerLossRecovery orchestration ──> P12c request re-run/delivery
P12b ──> P13 serve /health + retryable 503
P12b ──> P14a/P14b hang detection (optional, default off; may be deferred)
P8a,P5a ──> P15 expert_to_slot dispatch lookup (optional perf)
P12c,P13 ──> P16 integration tests + CI lists, P17 docs
```

## PRs

### P0a `[None][fix] Accept the real arguments in the fake moe_a2a op registrations`
- Files: `tensorrt_llm/_torch/custom_ops/cpp_custom_ops.py` (about 5 of the 15 lines: `use_cft_counted_writes`,
  `expert_id_payload_index`, `invalid_token_expert_id`), from commit 9eadb87a8f.
- Tests: the existing torch.compile/fake-tensor MoE comm tests.
- Risk: none. This is a main bug, so it should go to main.

### P0b `[None][fix] Share EPGroupHealth across ConfigurableMoE's model-config deepcopy`
- Files: `ep_group_health.py` (+4), `tests/unittest/_torch/moe/fused_moe/test_configurable_moe.py` (+28). Commit a4886841b3.
- Risk: `__deepcopy__` that returns `self` is intentional shared state, so the PR should say so. This is also a
  main fix.

### P0c `[None][fix] End every failed OpenAI stream with one error event and one [DONE]`
- Files: the generic part of `tensorrt_llm/serve/openai_server.py` (`_stream_error_events`, `_end_stream_on_error`,
  ~60), `tests/unittest/llmapi/apps/test_stream_error_events.py` (123), plus 1 line in `l0_cpu.yml`. Commits
  e1d6b9752a and 76606c3404.
- Risk: this changes stream behaviour for every user, not only FT, so it needs serve CODEOWNER review. Send it
  to main, not to the feature branch.

### P0d `[None][infra] Upgrade mpi4py to 4.1.2 for ULFM`
- Files: `docker/common/install_mpi4py.sh` (+47/-54), from commits 4455b18d48 and 3bb877e24f.
- Tests: full CI on a rebuilt image. mpi4py 4 changes `mpi4py.futures` internals: the TRTLLM_USE_MPI_KVCACHE
  patch moves from `_lib.py` to `_core.py`, and the setuptools<69 pin is dropped.
- Risks: this affects every container and every MPI path (MPICommExecutor, disagg MPI KV cache), so it needs
  infra/CODEOWNER sign-off and an image tag bump. Everything ULFM-related depends on it, and P1 has to fail
  closed on mpi4py<4 (168e3982da).

### P1 `[None][feat] Add ULFM survivor communicators for host collectives`
- Files: `tensorrt_llm/_torch/distributed/communicator.py` (+556/-35), and from `tensorrt_llm/_utils.py` the MPI
  helpers (`is_mpi_peer_failure*`, `mpi_failed_ranks`, `mpi_world_ranks`, ~70). Native peer-failure watch:
  `cpp/tensorrt_llm/nanobind/runtime/peerFailureWatch.{cpp,h}` (256), `nanobind/CMakeLists.txt` (+2) and
  `runtime/bindings.cpp` (+2).
- Tests: `tests/unittest/_torch/distributed/test_mpi_survivors.py` (586) and `test_mpi_survivors_ulfm.py` (182);
  the ULFM ones need Open MPI with ULFM. Total ~1620, about 890 of it source.
- Risks:
  - The public-ish `Distributed` API changes: `allgather` / `tp_allgather` / `tp_gather` / `tp_allgather_int64`
    gain `departed=` and `fill_value=`, and `live_ranks` is new. Call sites are updated in P11.
  - `HostCollectiveInterrupted` wraps every host collective through `_interruptible`. That needs a per-call cost
    check with FT off.
  - Startup checks `OMPI_MCA_mpi_ft_enable` and `OMPI_MCA_pml == ob1`, which ties FT to Open MPI.
  - `ERRORS_RETURN` is set on the communicators.
  - A native thread polls MPI. Its detach-on-self-destroy logic (1d050d20d4, fd5b37ba54) needs careful review.

### P2 `[None][feat] Keep the MPI session serving and exit cleanly after a WideEP peer loss`
- Files: `tensorrt_llm/llmapi/mpi_session.py` (+323/-9), `mgmn_worker_node.py` (+46/-8), `mgmn_leader_node.py`
  (+17/-1) and `trtllm-llmapi-launch` (+26).
- Tests: `tests/unittest/llmapi/test_mpi_session.py` (541) and `test_trtllm_llmapi_launch_exit.py` (120).
  Total ~1090.
- Risks:
  - Under FT the launcher `exec`s the worker and passes the FlashInfer lock as **fd 200**. The worker node calls
    `os.set_inheritable(200, False)` before its imports, which is a magic fd.
  - `mgmn_worker_node.py` now always goes through `run_worker_rank`, even with FT off. The non-FT behaviour must
    be shown to be identical.
  - The launcher defaults `OMPI_MCA_pml=ob1` and `OMPI_MCA_btl=self,tcp` under FT. Running over TCP within a
    node costs performance and needs a measured healthy-cost number.
  - `MPI_Finalize` is bounded by a GIL-free timer with a SIGALRM backstop. About 12 fix commits went into this
    (f66fea36dd .. 409d2bd1f8, b075b66bb4), which marks it as fragile.
  - Departed-peer manager threads have their tracebacks suppressed by a `threading.excepthook` override (global
    state).
  - The copyright header added to the mgmn scripts (29699bc43e) is fine.

### P3 `[None][feat] Add the WideEP recovery protocol, config and client`
- Files:
  - `tensorrt_llm/executor/recovery/{__init__,config,protocol,client}.py` (3+149+332+248). Leave the
    hang-detection messages out of `protocol.py` (Heartbeat, NativeHeartbeat, DeviceWait, Fence, Fenced and the
    datagram codec, ~110) and add them in P14.
  - `_utils.py` `fault_tolerance_mode` / `peer_loss_recovery_enabled` (~25), `executor/utils.py` (+21) and
    `executor/ipc.py` `socket_options` (+5/-1).
  - Fold `wide_ep_ft.py` (+5/-13), which switches to the shared mode helper, in here.
- Tests: `tests/unittest/executor/test_recovery_config.py` (105). Total ~860.
- Risks:
  - **Configuration is env vars plus a side YAML file** (`TLLM_FAULT_TOLERANCE_MODE`, `_AUTHORITY`,
    `_AUTHORITY_KEY`, `_CONFIG`), not an `LlmArgs` field. AGENTS.md expects user-facing config in Pydantic
    LlmArgs, with the golden manifest and serve-schema regeneration plus telemetry CODEOWNER approval.
    Decide this up front:
    - either move `FaultToleranceConfig` into `TorchLlmArgs` (the authority would then need its own copy, since
      it validates by digest),
    - or document why it is deployment-level. The worker and the authority must agree, and the authority is a
      separate process.
  - The wire protocol is pickled messages over ZeroMQ with HMAC and a version check (33f8f55a6f). It needs a
    security review of the unauthenticated surfaces (117cf4f877).

### P4 `[None][feat] Add the external WideEP recovery authority`
- Files: `tensorrt_llm/executor/recovery/authority.py` (909, of which ~200 for hang confirmation and fencing can
  move to P14b).
- Tests: `tests/unittest/executor/test_recovery_authority.py` (1198; the hang cases ~300 move with P14b).
  Total ~1600, or ~2100 without the hang split.
- Risks:
  - New standalone process and CLI (`main`) that operators must run, and a single point of failure. The design
    reports "unprotected" while the authority is unreachable (851093b78b, b5e76fb217).
  - `command_fence` runs a shell template restricted to a plain host and pid (e9986cc3f0), which needs a
    security read.
  - Flood bounds are per connection, with residual risk acknowledged (e327247d9f).

### P5a `[None][feat] Read the NVLinkOneSided rank mask through a pointer and add an abort record`
- Files: `cpp/tensorrt_llm/kernels/moe/communication/moeAlltoAllKernels.{cu,h}` (+260/-83, minus the
  `USE_EXPERT_TO_SLOT` parts, which go to P15), `thop/moe/communication/moeAlltoAllOp.cpp` (+80/-32), and the
  matching fakes in `cpp_custom_ops.py` (~10).
- Tests: `tests/unittest/_torch/moe/test_moe_a2a_abort.py` (483) and `test_moe_comm.py` (+24/-7). Total ~950.
- Risks, all changes to main's kernel interface:
  - **Param structs**: `active_rank_mask` goes from `uint64_t[kRankMaskWords]` held by value (default all ones)
    to `uint64_t const*` read at execution time. `MoeA2AAbortState* abort` is added.
  - **Op contract**: `active_rank_mask` was a CPU tensor and is now a CUDA tensor on the workspace device. The
    "must mark the local ep_rank" check is removed. Any main caller that passes a CPU mask will now fail.
  - **FLAG_VAL grows from 4 to 8 bytes** (`reserve(FLAG_VAL_OFFSET_INDEX, 2*sizeof(uint32_t))`, plus
    `kReleasedRoundIndex`). This changes the metainfo offsets and workspace layout, so every workspace-size
    computation must agree, Python and C++ alike.
  - **Op schema**: `moe_a2a_dispatch` and `moe_a2a_combine` gain `abort_state`, and `moe_a2a_initialize` gains
    `barrier: bool`. These are torch.library schema changes that need fake/meta updates and a BC note.
  - The wait loop polls the abort record (`kAbortPollCycles`) and traps on timeout, so the hot path needs a
    healthy-cost measurement. Releasing a source on an invalid or unset count (7c7a0a901a, ddf3261fb7) changes
    failure semantics.
  - New template parameters (`ENABLE_RANK_MASK`, `USE_RANK_COMPACT_ROUTING`) add instantiations and increase
    compile time.
  - The fence-asm memory clobber (4184572c12) was reverted back to main's asm (c0b36968a4). Confirm the net diff
    keeps main's asm.

### P5b `[None][feat] Bind NVLinkOneSided to a MoeExecutionState with fixed, token-interleaved receive slots`
- Files: `tensorrt_llm/_torch/moe/fused_moe/communication/execution_state.py` (205), `nvlink_one_sided.py`
  (+157/-15), `alltoall_watchdog.py` (+16/-8), `cuda_graph_runner.py` (+9/-3), `MOE_DEVELOPER_GUIDE.md` (+6) and
  `wide_ep_ft.py`. Lifting the CUDA-graph refusal is part of this PR.
- Tests: `test_moe_execution_state.py` (337), `multi_gpu/test_nvlink_one_sided.py` (346),
  `modules/test_alltoall_watchdog.py` (+96/-15) and `multi_gpu/test_moe_a2a_late_writes.py` (340).
  Total ~1560.
- Risks:
  - CUDA graphs are allowed under rank-mask mode only when the layers are bound; unbound layers refuse capture
    (a0851f0458).
  - Each payload signature gets its own fixed workspace region (2ccc3e8e1c).
  - Combine takes its row order from what the dispatch wrote. A 3D combine input is rejected after a bound
    dispatch (87ed370f67).
- Drop: the "Log each payload signature's dispatch part" info log (c7b13ceb85), or make it debug.

### P6 `[None][feat] Remap the MNNVL all-to-all workspaces at the same addresses among the survivors`
- Files: `tensorrt_llm/_torch/distributed/mnnvl_memory.py` (+148/-10) and `mnnvl_alltoall_workspace.py`
  (+23/-1). The latter checks CUDA errors on reinit and restarts the watchdog.
- Tests: `tests/unittest/_torch/test_mnnvl_memory_remap.py` (260), `test_mnnvl_alltoall_workspace.py` (29)
  and `multi_gpu/test_moe_a2a_recovery.py` (611, multi-GPU). Total ~1080.
- Risks:
  - VA-stable remap with `pin` / `unpin` / `release_backing`, and checkpointing is refused after a remap
    (0762352bd8).
  - It borrows KV-cache quota during the remap; that hook is in P12b/P10.
  - All survivors must map or fail together.

### P7a `[None][feat] Restore a lost rank's experts into EPLB slots (C++ load balancer)`
- Files: `cpp/tensorrt_llm/runtime/moeLoadBalancer/moeLoadBalancer.{cpp,h}` (+484/-82), restricted to
  `restorePlacement`, `quiesce`, `endIter(interrupted)`, `resetUpdateSchedule`, `validatePlacement`,
  `installPlacement` and the dead-rank masks. Also `nanobind/runtime/moeBindings.cpp` (+28/-1) and
  `thop/moe/loadBalance/moeLoadBalanceOp.cpp`, without the one-slot table, which goes to P15.
- Tests: the restore, quiesce and interrupted parts of `cpp/tests/unit_tests/runtime/moeLoadBalancerTest.cpp`
  (~600 of 1094). Total ~1200.
- Risks:
  - The C++ reads `TLLM_FAULT_TOLERANCE_MODE` directly (`moeLoadBalancer.h/.cpp`). Pass it in as a constructor
    flag instead, since the guidelines prefer config to env.
  - EPLB updates must skip slots marked -1. Main's checks are kept when FT is off (ae76f7b9fc, c6d48b7eba).
  - Follow AGENTS.md's `add_tllm_gtest` / labels rule for any new gtest target.

### P7b `[None][feat] Back WideEP extra EPLB slots with ghost slot memory`
- Files: `cpp/tensorrt_llm/runtime/moeLoadBalancer/ghostSlotMemory.{cpp,h}` (881),
  `hostAccessibleDeviceAllocator.{cpp,h}` (+143/-21), `runtime/CMakeLists.txt` (+1), and the backing APIs in
  `moeLoadBalancer.cpp` (`restoreBackingBytes`, `backRestoredSlots`, `releaseUninstalledBacking`, ~150).
- Tests: the ghost-slot parts of `moeLoadBalancerTest.cpp` (~490). Total ~1650; see below to shrink it.
- Risks:
  - The VMM ghost slots unmap and remap device memory under the KV cache.
  - `HostAccessibleDeviceAllocator` is never destroyed (6f9008e3f9, a deliberate leak).
  - The registry's shared lock is held across copies.
- **Dead code / experiment candidate:** the host-accessible (GDRCopy / HOST_NUMA) ghost-slot chain 2a11912a09,
  a14f60bcde, 4e7b0a9da1, edcb7db48c, f30a209904, 6e778208fb, e2953b232a, e5ec73ab32, 42978bac5e and
  7a03252e92 is a perf optimisation for online EPLB. It accounts for most of ghostSlotMemory.cpp's complexity:
  a NUMA probe, a CPU-write/GPU-read startup proof, and a cross-rank vote. Recommend splitting it into a
  separate follow-up perf PR, or dropping it. With it removed, ghost slots use plain device VMM and online EPLB
  updates them by GPU copy, as the docs already allow (e8e6fbb8e7). P7b then fits under 1500.
- Drop: the BAR1-size log (0a96717e25) in `moe_load_balancer.py`, which also adds a `pynvml` import at
  module scope.

### P8a `[None][feat] Append extra EPLB slots and restore lost experts into them (Python)`
- Files:
  - `tensorrt_llm/_torch/moe/fused_moe/moe_load_balancer.py`, the restore and placement part (~450 of +836/-69):
    `with_ghost_slots`, `extra_slots_mode`, `covers_loss_of`, `quiesce`, `recovery_placements`,
    `restore_lost_experts`, `install_recovery_placement`, `placement_digest` and `_gather_hosts`.
  - `impl_blocks.py` (+61/-14), `impl_base.py` (+7), `interface.py` (+5) and `fused_moe_cutlass.py` (+2/-1).
- Tests: `tests/unittest/_torch/moe/test_resident_placement.py` (597) and `test_moe_load_balancer.py`
  (+186/-1). Total ~1300.
- Risks:
  - `expert_size_per_partition` and `num_slots` now include the extra slots. That touches every MoE backend's
    weight shapes; the slots are empty while healthy.
  - Startup refuses EPLB placements that cannot cover a failure domain.
  - `failure_domains()` lives in `peer_loss_recovery.py`, so `moe_load_balancer.py` imports pyexecutor. That
    layering problem was fixed in 820e66b4ff and then reverted (8d0ec8cf1c). Redo the fix (move it to
    `recovery/config.py`).

### P8b `[None][feat] Keep WideEP EPLB host copies in a ring of nodes and stage restores from them`
- Files: `moe_load_balancer.py`, the host-copy and staging part (~390): `/dev/shm` reserve, prefault,
  per-job names, `keep_experts`, `ring_copy_nodes`, `staging_halves`, `restore_copy_plan` and
  `stage_restored_slots`.
- Tests: `tests/unittest/_torch/moe/test_ring_host_copies.py` (805) and
  `multi_gpu/test_ring_restore_staging.py` (243). Total ~1440.
- Risks:
  - `/dev/shm` lifecycle: per-job naming, unlinking once every local rank maps it, and a reserve at startup
    (1bd9142f67, ecdde3472b, 1d84dfc141). Main's lifecycle is kept with FT off (b5cec98430).
  - Background prefault threads run only under FT.
  - Staging must stay off workspace bytes that a lost rank may still write (899af7a7de, 1d22e8e1b1).

### P9 `[None][perf] Tune and key the MoE tactics on the loaded slots under WideEP recovery`
- Files: `torch_custom_ops.py` (+30/-9), `cute_dsl_custom_ops.py` (+116/-31) and `fused_moe_cute_dsl.py`
  (+38/-16).
- Tests: `tests/unittest/_torch/moe/test_moe_runner.py` (133) and `thop/parallel/test_cute_dsl_moe.py`
  (105). Total ~480.
- Risks:
  - **Op schema**: `trtllm::fused_moe` / `MoERunner` gain `num_extra_local_experts` (default 0).
  - The CuteDSL tuning key changes to a power of 2 of the source tokens (1c0ec5d217). Show that the autotuner
    cache keys are unchanged when `num_extra_local_experts == 0`.

### P10 `[None][feat] Roll back a discarded step's KV cache allocations in KV Cache Manager V2`
- Files: `tensorrt_llm/_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py` (+59) and `llm_request.py`
  (+49, step-change snapshot fields).
- Tests: `tests/unittest/_torch/executor/test_kv_cache_revert.py` (294). Total ~400.
- Risks: this is the KV cache owners' area. The quota-resize API is also used by P6 and P12b.

### P11 `[None][feat] Keep attention-DP routing serving with absent ranks`
- Files: `scheduler/adp_router.py` (+56/-16), `request_utils.py` (+17/-5) and `executor_request_queue.py`
  (+24/-3). The `departed=` call sites are in `model_engine.py` (+12), `engine/runners/common.py` and
  `decoder/{inputs,metadata,runner}.py` (~15).
- Tests: `test_adp_absent_ranks.py` (299), `test_adp_router.py` (+23/-21), `test_adp_host_exchange.py`
  (+14/-2), `test_request_utils.py` (+9/-5), `test_executor_request_queue.py` (26),
  `disaggregation/{fake_dist,test_benchmark_disagg}.py` (+5/-1) and `modeling/test_modeling_deepseekv41.py`
  (+2/-2). Total ~560.
- Risks:
  - Main's pin indexing is kept with FT off (a1a7c2a687, c31ce145fd, cef4670789).
  - A strictly pinned request to a lost rank becomes a client error (135ad867c7).
  - The disagg and deepseekv41 tests change only because the `departed=` kwarg appears in their mocks.

### P12a `[None][feat] Add an executor recovery hook with step discard and commit`
- Files: the `RecoveryHook` base, `PreparedStep` and `discard_interrupted_steps` / `_note_step_change` from
  `peer_loss_recovery.py` (~100). From `py_executor.py`: the hook call sites, the discard/commit and
  `fail_requests` (~200 of +341/-31). Also `py_executor_creator.py` (+5/-1).
- Tests: the no-recovery parts of `test_py_executor.py` (~30). Total ~350.
- Risks:
  - The hook calls sit on the per-iteration path and are guarded by `.enabled`. A per-step host cost check is
    needed (82c6711fbe, e3edd12bf1).
  - With FT off, an iteration interrupted by a raising forward must still end as on main (c8df55f5df).
- **Unrelated to FT: drop or send to main separately.** 2e8d6d7da1 ("Release an attention-DP context hold once
  a burst of arrivals has ended") changes attention-DP balancing **unconditionally**: `ADP_BURST_QUIET_ITERS`
  and `adp_recent_arrivals` in `py_executor.py` (~30), plus its 77-line test in `test_py_executor.py`. It
  changes main's scheduling behaviour and needs its own perf justification.

### P12b `[None][feat] Recover the PyTorch executor from a WideEP peer loss`
- Files: `peer_loss_recovery.py` core (~1000): `PeerLossRecovery.__init__`, `create`, `run`, `_recover`, the
  shrink/watch/check-recoverable paths, `_end_ranks`, `_remap_workspaces`, `_make_room`,
  `_return_remap_room`, the placement restore/back/stage, `install_recovery_placement`, the deadlines and the
  startup peer watch. Also the remaining `py_executor.py` (~140).
- Tests: the orchestration parts of `tests/unittest/_torch/executor/test_peer_loss_recovery.py` (~1600 of
  2894).
- Size: ~2700. Further split options:
  - (i) startup watch + detection + shrink + commit, with no EPLB or remap;
  - (ii) wiring in the remap, EPLB restore and ring staging.
  - `test_peer_loss_recovery.py` should become several files along the same lines.
- Risks:
  - `_recover` is a ~200-line function, which reviewers will ask to break up.
  - Spec-decoding acceptance (`_tested_spec_dec`), and a generation sampler is required (b524128581).
  - The DSA KV manager and block reuse are allowed.
  - Recovery phase-time logging (7f1811c819) is fine, but keep it at info level, once per recovery.

### P12c `[None][feat] Run a lost rank's undelivered requests again on the survivors`
- Files: from `peer_loss_recovery.py` (~450): `_deliver_responses`, `_readmit_undelivered_requests`,
  `_requeue_unscheduled_requests`, `_departed_requests`, response-gather holding and quarantine, and
  `_cancelled_responses` / `_relax_attention_dp_pin`. Plus the `llm_request.py` progress fields.
- Tests: the re-run and quarantine parts of `test_peer_loss_recovery.py` (~1300). Total ~1750 (tests ~1300).
- Risks:
  - Every rank 0 response-gather changes (47dc1ca616, 6726581748, 130c9a0e0e) touches main's response path.
    Show it is unchanged with FT off.
  - Retried outputs are not token-exact (c52f327022), so the docs must qualify the accuracy statement.

### P13 `[None][feat] Report WideEP fault-tolerance state on /health and answer peer-loss failures with 503`
- Files: `tensorrt_llm/serve/fault_tolerance_status.py` (154) and the FT part of `openai_server.py` (~50).
- Tests: `tests/unittest/llmapi/apps/test_fault_tolerance_health.py` (71). Total ~280.
- Risks:
  - The `/health` status code changes under FT: 503 for non-serving states.
  - The abort reason is kept out of the unauthenticated body (ae90d16112).
  - A single probe thread queries the authority (5676242aaa).

### P14a / P14b `[None][feat] Detect a hung WideEP rank ...` (optional; `hang_detection` defaults to False)
- P14a, worker evidence: `tensorrt_llm/_torch/pyexecutor/hang_evidence.py` (348),
  `nanobind/runtime/heartbeatSender.{cpp,h}` (239) plus bindings, and the hang protocol messages (~110). Tests:
  `test_hang_evidence.py` (314). Total ~1000.
- P14b, authority confirmation and fencing: `executor/recovery/hang_monitor.py` (213),
  `_torch/distributed/pmix_fence.py` (112), the hang paths in `authority.py` (~200) and the fence-before-plan
  paths in P12b. Tests: `test_recovery_hang_monitor.py` (661) and the authority hang cases (~300).
  Total ~1500.
- Recommendation: defer to after the MVP, since it is off by default. If it ships, note the risks:
  - `pmix_fence.py` calls libpmix via ctypes with hard-coded struct layouts (`_PMIX_MAX_NSLEN=255`), which is
    fragile across PMIx versions.
  - The UDP heartbeat datagrams use HMAC.
  - The fence policy changed several times (37551d6035 to fa884734ad), so keep only the final rule:
    "fence every failed rank before every plan".

### P15 `[None][perf] Look up dispatch slots in a device expert-to-slot table under WideEP recovery` (optional)
- Files:
  - the `USE_EXPERT_TO_SLOT` template path and the `expert_to_slot` / `num_routed_experts` params in
    `moeAlltoAllKernels.{cu,h}` (~80), and the `expert_to_slot` op arg (~20);
  - `moeLoadBalanceKernels.{cu,h}` (+31/-3), `moeLoadBalanceOp.cpp` (the `moe_load_balance_one_slot_table` op,
    ~40) and its fake in `cpp_custom_ops.py`;
  - `moe_scheduler.py` (+32/-1), and `one_slot_table` with its caching in `moe_load_balancer.py` (~40).
- Tests: the slot-table cases in `test_moe_a2a_abort.py` and `test_moe_comm.py` (~80). Total ~350.
- Risks: a new op, an op-schema argument, and another template dimension. It is FT-only, so justify it with a
  same-node A/B or drop it.

### P16 `[None][test] Add WideEP peer-loss recovery integration tests`
- Files: `tests/integration/defs/wide_ep_ft/{__init__,peer_loss_driver,test_peer_loss_recovery}.py`
  (0+215+570) and `worker_hooks/sitecustomize.py` (61). Test lists: `l0_b200.yml` (+4), `l0_cpu.yml` (+6) and
  `l0_gb200_multi_gpus.yml` (+15). CI: `jenkins/L0_Test.groovy` (+5). Total ~880.
- Risks:
  - **Test-only fault injection via a `sitecustomize` on `PYTHONPATH`** that monkeypatches
    `RecoveryClient.send` (`WIDEEP_FT_TEST_KILL_AFTER_STOPPED`). It is acceptable in `tests/`, but reviewers
    may prefer a documented test seam.
  - 12 GB200 cases with `TIMEOUT (45)` add significant CI cost.
  - `TLLM_REQUIRE_ULFM=1` is injected by stage-name prefix in Groovy, so it needs infra CODEOWNER review.
  - It depends on the P0d image.
- Put each PR's unit-test list entries (`l0_cpu.yml`, `l0_b200.yml`) in that PR rather than here.

### P17 `[None][doc] Document WideEP fault tolerance`
- Files: `docs/source/features/wide-ep-fault-tolerance.md` (338), `feature-combination-matrix.md` (+12),
  `index.rst` (+1) and `examples/wide_ep/README.md` (+2). Total ~350.
- Risks: the doc must match the final defaults: `extra_slots` (whether ghost needs host access), the bounded
  finalize, the 5 s detector, and the required ob1/TCP setting. If P7b's host-accessible chain or P14 is
  dropped, remove their sections. Mark the feature experimental.

## Size summary (from numstat; `~` = split estimate)

| PR | src | tests | total |
|----|-----|-------|-------|
| P0a-c | ~70 | ~150 | ~230 |
| P0d | 101 | - | 101 |
| P1 | ~900 | 768 | ~1620 |
| P2 | ~430 | 661 | ~1090 |
| P3 | ~750 | 105 | ~860 |
| P4 | ~700 | ~900 | ~1600 |
| P5a | ~470 | ~510 | ~980 |
| P5b | ~420 | ~1130 | ~1560 |
| P6 | ~180 | ~900 | ~1080 |
| P7a | ~600 | ~600 | ~1200 |
| P7b (no host-access chain) | ~700 | ~350 | ~1050 |
| P8a | ~540 | ~780 | ~1320 |
| P8b | ~390 | ~1050 | ~1440 |
| P9 | ~240 | ~240 | ~480 |
| P10 | 108 | 294 | ~400 |
| P11 | ~140 | ~420 | ~560 |
| P12a | ~300 | ~30 | ~330 |
| P12b | ~1140 | ~1600 | ~2740 (split in two) |
| P12c | ~450 | ~1300 | ~1750 |
| P13 | ~200 | 71 | ~280 |
| P14a+b (optional) | ~1220 | ~1280 | ~2500 |
| P15 (optional) | ~270 | ~80 | ~350 |
| P16 | 851 int. + 30 CI | | ~880 |
| P17 | 353 | | 353 |

The source in P12b and P12c is the one place that still exceeds the target. `peer_loss_recovery.py`
(1812 lines, with a ~200-line `_recover`) and `test_peer_loss_recovery.py` (2894 lines) should each be split
into modules (orchestration / requests / EPLB+remap) before upstreaming. Any split will also make the PR
series line up.

## Commits touching files unrelated to FT
- **2e8d6d7da1**: attention-DP context-hold "burst ended" release in `py_executor.py`. It is unconditional and
  changes main's scheduling. Drop it, or send it to main with data.
- **e1d6b9752a, 76606c3404**: OpenAI streaming error and `[DONE]` handling for all users. Send to main (P0c).
- **9eadb87a8f**: fake moe_a2a registrations, a main bug (P0a). **a4886841b3**: EPGroupHealth deepcopy, a main
  bug (P0b).
- **4455b18d48, 3bb877e24f**: the mpi4py upgrade is global infra (P0d). 3bb877e24f ("Take the rewritten prmain
  test and install-script wording") is a 2-line wording sync and can be squashed.
- **29699bc43e**: copyright headers on the mgmn scripts. Fine, and it travels with P2.
- **6e778208fb, 957ac92c5f**: mpi4py futures/Finalize fallbacks and the manager-thread excepthook. These sit on
  the shared mpi_session path, so check that FT-off behaviour is unchanged.
- `tests/unittest/_torch/disaggregation/*` and `modeling/test_modeling_deepseekv41.py` change only through
  mock signatures (`departed=`). They are not behaviour changes.

## Leftovers / experiment hooks to drop or rework before upstreaming
1. The BAR1 log plus the module-level `pynvml` import (0a96717e25): drop.
2. The per-signature dispatch-part info log (c7b13ceb85): drop, or move to debug.
3. The host-accessible ghost-slot chain, which leans on main's `TRTLLM_EPLB_FORCE_GDRCOPY` /
   `TLLM_HOST_ACCESSIBLE_ALLOW_MANAGED_FALLBACK` env knobs: split off or drop (see P7b).
4. `TLLM_FAULT_TOLERANCE_MODE` is read in C++ (`moeLoadBalancer.h/.cpp`): pass it as a parameter instead.
5. The env-var plus side-YAML configuration: decide between LlmArgs and a documented deployment config (P3).
6. Launcher fd 200 / `exec` / `OMPI_MCA_pml=ob1,btl=self,tcp` defaults: needs CODEOWNER sign-off and a
   healthy-cost number.
7. CFT interplay: both "Drop the unreachable expert_to_slot path from CFT" (c3601e76c0) and dbeab13f9b
   ("Name the setting that selects the fence path when FT meets CFT") exist. CFT is out of scope, so keep only
   the refusal check (`expert_to_slot` / abort ⇒ fence path) and no CFT code paths.
8. The `failure_domains` import cycle (moe_load_balancer imports pyexecutor), whose fix was reverted: redo it.
9. History-narrating docstrings and comments ("as main does", "now", "instead of"): rewrite them as contracts.
10. `sitecustomize.py` test fault injection: acceptable, but document it in the integration test README or the
    PR description.
11. The automated scan for new Python functions with no other reference did not finish within budget; it found
    only `_check_order` (a Pydantic validator, so not dead). Run a vulture-style pass before each PR.

## TODO: history-narrating comments and docstrings per PR (not rewritten in v5-split)
v5-split (built from v5-zero `65a39b411d`) keeps v5-zero's tree byte for byte, so these comments are listed here and not
edited in the code. Line numbers are for `65a39b411d`. A regex scan of the added comment and docstring lines ("main",
"now", "no longer", "instead of", "any more", "from now on", ...) found few real narrations. Most hits are present-tense
runtime descriptions ("a rank that died no longer uses ...") and are not listed. Rewrite these as contracts:
- P1: `communicator.py:995-996` "otherwise built on first use, exists from now on".
- P2: `mpi_session.py:97-98` "left to the surviving ranks' recovery path instead of tearing down the session"
  (state what happens, not what would have happened).
- P7: `cpp/tests/unit_tests/runtime/moeLoadBalancerTest.cpp:484` "The placement main's planner makes without a mask";
  `moeLoadBalancer.cpp:729` "may fill it from now on".
- P11: `scheduler/adp_router.py:238` "looked up by position, as in a per-rank list" (refers to the old `List[int]`);
  `request_utils.py:122-124` "instead of leaving it queued forever".
- P12b: `peer_loss_recovery.py:161-163` "from now until ..." and "the MPI runtime no longer ends the job"
  (contrasts ULFM with default MPI; say what ULFM does).
- P13: `serve/fault_tolerance_status.py:102` "later ones fail at once instead of piling up" (borderline).

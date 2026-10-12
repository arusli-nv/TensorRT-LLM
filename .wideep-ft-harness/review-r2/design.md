# WideEP FT Phase 1 — high-level design review (round 2)

Scope: `git diff 55279c109f` in `.wideep-ft-runs/claude/wt` (committed + uncommitted), excluding
4dc8d8803f / e13dab7bcd / 952dcac056 / e80f9cbed6. Line refs are to the working tree.
Paths are relative to the worktree. "PE" = `tensorrt_llm/_torch/pyexecutor/py_executor.py`,
"PLR" = `tensorrt_llm/_torch/pyexecutor/peer_loss_recovery.py`,
"COMM" = `tensorrt_llm/_torch/distributed/communicator.py`.

Overall verdict: the architecture is sound and fits main. Survivors keep their logical ranks, a
graph-latched descriptor carries the mask and abort state, a local commit gate is sound under
ADP, and an external authority acts as the commit barrier. The MoE data-path argument holds:
combine is pull-based and a readiness flag implies the signaller's data for that round is
complete. So a survivor that finishes a step without aborting computed valid results, and
local commit decisions are safe even when other survivors discard the same step.
I found **two merge blockers**. Both are fixes inside the current design, not redesigns:
(B1) recovery waits that are not bounded (Shrink and the device syncs inside Install) and that
start before the authority's deadline is armed, and (B2) watcher double-stop that masks the
root-cause error and can raise from `finally`. Most of the remaining items are should-fix
hardening or PR hygiene.

---------------------------------------------------------------------------------------------
## Blockers

**B1. Recovery is not bounded end-to-end; the authority deadline starts too late.**
`PLR:_recover` calls `dist.shrink()` (COMM:933 `comm.Shrink()`) *before* sending `Stopped`.
ULFM Shrink blocks until every live process joins. The authority only arms its deadline on the
first `Stopped` (`authority.py:_on_stopped`), and the worker `deadline` computed at PLR:278 is
not applied to Shrink. Some survivors may never reach Shrink. Examples: a rank blocked in a
communicator nobody revoked (EPLB shared comm, `_tp_comm`-derived comms, mpi4py.futures
traffic), a rank whose watcher never fires, or a GPU wait not covered by cooperative abort.
In those cases every survivor hangs forever, with the hang detector already exited
(`_run_event_loop` docstring, PE:1494). `MoeExecutionState.rearm/install_mask`
(`execution_state.py:110,150,155`) also use unbounded `torch.cuda.synchronize`. This breaks
the design invariant "every recovery wait is bounded".
*Fix:* add a worker-side recovery watchdog, armed at detection, that takes the terminal path
when `deadline` passes. A `threading.Timer` that logs and calls `MPI.COMM_WORLD.Abort(1)` is
enough, since the process cannot return from Shrink. Also make the authority aware earlier:
either send a `Detected(rank)` before Shrink, or rely on the worker watchdog alone and
document it. Use one deadline for Shrink, join, install and commit.

**B2. `PeerFailureWatch.stop()` is not idempotent, so a failed recovery masks its cause.**
`_recover` stops the watch at PLR:285. If anything after that raises (shrink count != 1,
`RecoveryAborted`, digest mismatch, timeout), `run()`'s `finally` (PLR:193) calls
`self._watch.stop()` again. That calls `Cancel()` on `MPI.REQUEST_NULL` and `Free()` on a
freed comm (COMM:830-840), which raises a non-peer MPI error from `finally` and replaces the
real `RecoveryError` in the terminal path.
*Fix:* make `stop()` idempotent (guard on a `_stopped` flag, or null the request and comm), or
set `self._watch = None` after stopping. Add a unit test where the authority aborts.

---------------------------------------------------------------------------------------------
## Q1 — External recovery authority (per the design: a separate CPU coordinator is required)

The design doc mandates the coordinator, so I review its implementation rather than its
existence. **Verdict: acceptable for Phase 1 and reasonably lean (~730 LOC across authority,
protocol and client). The ULFM-Shrink / authority split is mostly clean. Not blocking apart
from B1.**

What the split does well:
- Shrink builds the survivor comm but does not activate it. `install_survivors` activates it
  only after `Install` and cross-checks the committed failed set against Shrink (COMM:938-969).
  This is the right separation: MPI provides agreement and transport, and the authority owns
  the decision and commit.
- The Install→Installed→Commit barrier is necessary, not ceremonial. `rearm()` resets each
  survivor's own workspace row and flag_val. No survivor may run a collective until every
  survivor has re-armed, and the authority's barrier guarantees that. This is the best
  justification for the authority in the MVP.
- Admission at registration (intersection of `covers_loss_of` across workers,
  `authority.py:_admit`) fails closed for uncovered ranks and rank 0. This is good.
- The protocol is torch-free and type-checked on decode. HMAC with MAXMSGSIZE, stable
  ROUTING_ID with ROUTER_HANDOVER, and terminal Abort are all fine for a trusted network.

Should-fix / consider:
- **S-A1 One round trip is redundant.** After Shrink, every survivor already knows the failed
  set, the survivor set (world − failed) and the next generation, so `Propose` carries no new
  information in Phase 1. You could fold `Ready(digest)` into `Stopped` and drop
  `Propose`/PREPARING, leaving `Stopped(failed, digest) → Install → Installed → Commit`. That
  saves one RTT plus a state and its tests. Keep `Propose` only if Phase 2 lets the authority
  choose the failed set (for example, evicting a hung but alive rank); if so, write that down
  in the design.
- **S-A2 Three generation counters.** These are the authority `generation`, `EPGroupHealth`
  generation (`model_engine.install_resident_placement`), and
  `MoeExecutionState.installed_generation`, which is written and never read
  (`execution_state.py:68,156`). Drop the last one, and either assert that the
  health generation equals the authority generation or document that they are independent.
- **S-A3 No authority liveness while serving.** If the authority dies, FT silently becomes
  terminal at the next failure: the worker times out after 120 s and then takes the terminal
  path. That matches "coordinator loss fails closed", but operators get no signal. At minimum,
  log a warning or metric. A cheap DEALER heartbeat checked on rank 0's idle poll would also
  work. Not blocking.
- **S-A4 Registration timeout is 600 s** (PLR:42). A mis-set endpoint looks like a 10-minute
  startup hang. Use a shorter connect probe, or log on rank 0 every N seconds while waiting.
- **S-A5 Layering.** `wide_ep_ft.recovery_enabled()` (`wide_ep_ft.py:46`) and
  `mpi_session._recovery_enabled()` import `tensorrt_llm.executor.recovery.client`, so
  `_torch/moe` and `llmapi` depend on the executor package for an env check. Move
  `recovery_requested()` and the env names into a leaf, torch-free module (for example
  `tensorrt_llm/_ft_env.py` or `wide_ep_ft.py`) and have the client import from there.
- **S-A6 Deployment burden.** There is no launcher or `trtllm-serve` integration, and the key
  travels in the environment. This is OK for an experimental MVP behind an env var, but the
  final PR needs: a console entry point (or a documented `python -m`), a doc page covering
  env vars, how to launch the authority outside `mpirun`, key generation, and timeout
  ordering, plus an explicit "experimental" label. Note that the authority is per deployment
  instance: restarted workers cannot re-register (`_on_register` rejects them after
  REGISTERING), so it must be restarted together with the workers. Document this.

---------------------------------------------------------------------------------------------
## Q2 — Layering / PyExecutor hooks

**Verdict: acceptable for an MVP but heavier than needed. Should-fix before the executor PR,
not blocking.**

PyExecutor gets about 14 touch points. These are: the `_peer_loss_recovery` attribute plus
`create` (PE:1117); `_run_event_loop` (PE:1494); `step_prepared` and `commit_step`
(PE:4740, 4858); the idle poll (PE:6290); the unbroadcast capture (PE:6326);
`requests_routed` (PE:6563); `_fail_unservable_requests`; three
`except HostCollectiveInterrupted` / `_raise_if_peer_failed` pairs (PE:8284, 8460, 8492);
`_publish_responses` (PE:8890); `gathering_responses` and `responses_gathered`
(PE:8918, 8952); and fill values in three gathers. In the other direction, PLR reaches into
executor privates: `_discard_uncommitted_step`, `_requeue_unscheduled_requests`,
`_publish_responses`, `_disable_mpi`, `_is_kv_manager_v2`, `execution_stream`,
`gather_all_responses`, `adp_router`, `kv_connector_manager` and `drafter`.

Better seam (concrete):
1. **Validate early, not at the end of `PyExecutor.__init__`.** `create()` (PLR:111-164) runs
   after model load and CUDA-graph capture, which takes minutes for R1 EP32, and after
   `MoeExecutionState.for_model` has already changed the workspace layout. Move the
   requirement checks to `create_py_executor` / llm_args validation, where `llm_args`,
   `mapping` and the spec/guided/kv-connector configs are known. `PeerLossRecovery.__init__`
   then only receives what it uses.
2. **A narrow host protocol.** PLR should depend on a 4-method `Protocol`
   (`discard_uncommitted_step`, `requeue_unscheduled_requests`, `publish_responses`,
   `join_device_work`) that PyExecutor implements as non-underscore methods. This makes the
   coupling explicit and testable without a fake PyExecutor.
3. **Own the step transaction in PyExecutor.** "Uncommitted step" is an executor concept
   (V2 already has `_revert_gen_alloc`). Track it in PyExecutor and let recovery be one
   optional validator called at the commit point. Then `step_prepared` disappears.
4. **Response numbering belongs to the gather, not to hooks around it.** Wrap the
   `tp_gather` in `_enqueue_responses` (PE:8907-8952) in a small `ResponseGather` object that
   numbers, retains and redelivers. That replaces two hooks and the `_last_processed_gather`
   bookkeeping.
5. The `nvtx_range("_enqueue_responses")` decorator moved onto `_publish_responses`
   (PE:8889-8890). `_enqueue_responses` lost its range, and direct callers
   (`_fail_unservable_requests`) are no longer annotated. Restore it and give
   `_publish_responses` its own range. This is an unconditional change.

---------------------------------------------------------------------------------------------
## Q3 — Is FT-off strictly zero-change?

**Verdict: nearly. There is no measurable perf change on hot paths, but several unconditional
*behavioral* or *structural* changes need to be gated or moved into their own PRs.**

Unconditional changes (FT off):
1. **ADP routing list→dict** (`adp_router.py:470-471`, `request_utils.py:134`,
   PE:6515). Semantics change for a strict `attention_dp_rank` outside `[0, tp_size)`. Before,
   this raised an IndexError. Now it is routed under a missing key and failed with
   "Attention DP rank X is not serving" via `_fail_unservable_requests` (a new response-gather
   collective). It is arguably a fix, but it is a user-visible change. Put it in its own
   commit or PR with a test.
2. **`unservable_requests` every ADP schedule** (PE:6558). It is cheap (early return at
   adp_router.py:375), but it is new code on the scheduling path.
3. **`_gather_rank_status` None-filter** (PE:~4040), **`num_admitted_requests`** (PE:6524),
   **`_can_queue fill_value=1`** (PE:3876), **balance fill row** (PE:6726), **KV-rebalance
   None-filter** (PE:~5346), **`cuda_graph_runner.live_ranks`** (`cuda_graph_runner.py:301`).
   These are all no-ops when every rank is live; a tuple is built per step. Acceptable, but they
   should land with the logical-length-gather PR so reviewers can see they are neutral.
4. **`_interruptible` wrapper on every MPIDist collective** (COMM:67) adds one frame plus a
   try. The conversion only happens when `_host_world` is set (gated). Fine.
5. **`MoeLoadBalancer._shared_barrier`** (`moe_load_balancer.py:1067`) swallows ULFM
   peer-failure errors whether or not recovery is enabled. With mpi4py≥4 on a ULFM-capable
   Open MPI, a real failure at shutdown is now silently ignored. Gate it on
   `recovery_enabled()`.
6. **`get_wide_ep_ft_options` dropped the legacy `extra_attrs["ep_group_health"]` key**
   (`wide_ep_ft.py:83`). No in-tree users, but it is an API change; mention it in the PR.
7. **`ConfigurableMoE` deepcopy memo** (`configurable_moe.py:346`). This is a bug fix for
   detection-only mode, where the backend copy forked `EPGroupHealth`. It is a behavior change
   whenever health exists. Make it a separate small fix PR, coordinated with the #15788
   `__deepcopy__` fix.
8. **`MnnvlMemory.pin_count`** (`_mnnvl_utils.py:264,329,829,886`). This is a new field and new
   checks for all users; behavior is unchanged at count 0.
9. **Kernel ABI.** `DispatchKernelPointers` grows by 16 B and `CombineKernelPointers` by 8 B
   (kernel params for all variants). `vectorized_dispatch_impl` gains a template parameter
   (more instantiations, binary size and compile time). The non-rank-mask variant compiles the
   same code (descriptor is constexpr nullptr). The detection-only rank-mask variant now uses
   `__syncthreads_or` instead of `__syncthreads` in combine (tiny). Op schema adds
   `Tensor(b!)? execution_descriptor` to both ops, and the fake now matches the real
   signature (good). Run the existing moe_a2a perf sanity before and after to show no change.
10. **mpi4py 3.1.5 → 4.1.2** (`docker/common/install_mpi4py.sh`). This affects everyone (see
    Q6). `requirements.txt` leaves `mpi4py` unpinned, so pip users already get 4.x; that is
    the argument for the bump.
11. **`_update_requests(None)` early return** (PE:~8485) comes from the unrelated 4dc8d8803f.
    Keep it out of the FT PRs.

FT-on costs worth stating in the PR. They are not blockers.
- Descriptor dispatch no longer returns workspace views. It allocates dense recv tensors
  (`moeAlltoAllOp.cpp:816,829`) and launches an extra gather kernel that copies
  `ep × runtime_max_tokens × bytes` per payload per layer (`moeAlltoAllKernels.cu`
  `moeA2AGatherLocalKernel`). Measure decode TPOT with FT on versus off on EP32.
- The watcher thread polls `Irecv.Test()` every 10 ms under `MPI_THREAD_MULTIPLE`, which can
  contend with main-thread collectives. Rank 0 also does a 1 s idle broadcast.
- The commit gate syncs before `_update_request_states`, so host work no longer overlaps the
  tail of GPU work in the non-overlap loop.

---------------------------------------------------------------------------------------------
## Q4 — In-flight correctness

**Verdict: the core model is sound** (see the overall verdict on pull-based combine plus
sticky abort). The holes below are should-fix. Item H1 is the most important.

- **H1. Post-commit tail is skipped and not resumed.** After `commit_step` (PE:4858),
  responses are gathered inside `_handle_responses`. If that gather is interrupted, the rest of
  the iteration is skipped: `perf_manager.compute_batch_gpu_times`,
  `resource_manager.update_resources` (PE:4886), `_add_kv_cache_events`,
  `disagg.handle_timeouts_synced` and `_kv_connector_terminate_requests`. The step is
  committed, so these must not be dropped. For the supported profile (V1, no spec, no
  connector) `update_resources` is probably a no-op, but this is unverified and fragile.
  Either audit and assert (with a test that interrupts the gather and checks KV block counts
  afterwards) or record a "post-commit pending" flag that recovery completes.
- **H2. Pre-commit side effects not rolled back.** `_commit_kv_cache_stats` (PE:4793) and
  `perf_manager.save_timing_to_requests` (PE:4834) run before the gate. Borrowed perf timing
  events (`gpu_forward_events_from_perf_pool`) are not released by
  `_discard_uncommitted_step` (PE:8329), even though INFLIGHT_PLAN lists this. These are stats
  only, but say so or fix them.
- **H3. Rollback reaches only the KV manager.** `_discard_uncommitted_step` calls
  `kv_cache_manager.revert_prepare_resources` directly (`resource_manager.py:1240`). Other
  managers in `resource_manager.prepare_resources` (PEFT/LoRA cache, seq-slot, spec) are not
  reverted, and `create()` does not reject LoRA. Either reject PEFT in `create()` or add a
  `ResourceManager.revert_prepare_resources` that fans out (no-op default) so V2 can plug in
  later.
- **H4. Sampler state is enforced only by documentation.** PLR:116-118 says that penalties,
  seeded sampling and multi-token stop words are not rolled back. `penalties.py` updates
  `has_previous_token_cuda` and counts on device in `sample_async`, which runs before the
  gate, so a re-run double-applies them. Seeded generators advance. Per-request parameters
  arrive at runtime, so `create()` cannot reject them. Fail such requests at admission when
  recovery is enabled, or document the semantic drift precisely in the user doc.
- **H5. n>1 / child requests (E-S9).** `_request_owners` is keyed by the parent id
  (PLR:241). Children of a request owned by the dead rank never get an error response, so
  their clients hang. Either fan the error out to `child_req_ids` or reject n>1 under FT.
- **H6. Cancellation and special items in a torn broadcast.** If the request broadcast
  (PE:6320-6327) completes on some ranks and not others, receivers have already run
  `_handle_special_queue_items`. That processes cancels, and possibly the shutdown sentinel or
  control items, before rank 0 re-broadcasts the same items after recovery
  (`_requeue_unscheduled_requests`, PE:8342). Cancels are idempotent. Shutdown or control
  items could diverge (`is_shutdown` set on some ranks only). Either filter special items from
  the re-broadcast on ranks that already applied them, or make `_requeue_unscheduled_requests`
  also reset `is_shutdown` / `canceled_req_ids` consistently. This is a low-probability edge
  case.
- **H7. Watcher does not revoke.** `_on_peer_failure` (PLR:195) only sets the flag and
  requests abort. A survivor blocked in a host collective relies on some other rank's
  collective raising and revoking. In practice ULFM's error on a send to the dead peer does
  that, but whether it happens depends on the collective algorithm. Calling
  `dist.revoke()` from the watcher callback (legal under `THREAD_MULTIPLE`; `revoke()` is
  idempotent) makes detection-to-interrupt latency independent of the topology. This
  complements B1.
- **H8. Device timeout vs ULFM detection.** In descriptor mode, a kernel that times out still
  traps (`moeAlltoAllKernels.cu:704,1966`), which kills a *survivor* and causes a second loss,
  which is terminal. `timeout_cycles` must stay well above the ULFM detector timeout plus the
  10 ms watcher poll plus the abort-stream latency. Document the required ordering (ULFM
  `mpi_ft_detector_timeout` < device timeout) and check it at startup if you can.
- **H9. Response redelivery numbering** (PLR:96-100, 243-257, 340-374) is correct for one
  recovery. A non-root can finish at most one gather ahead (eager send), and the next
  all-gather pins it, so the `deque(maxlen=2)` window suffices. After recovery,
  `_gather_count` can differ by one between ranks. That is harmless because a second loss is
  terminal, but add an assert or comment so Phase 2 does not inherit a latent misnumbering.
- **H10. KV V1 rollback assumptions are OK for the profile.** Generation requests:
  `rewind_kv_cache(1 + max(draft, reserve))` mirrors `add_token` exactly
  (`resource_manager.py:1213-1216`). First chunks: `free_resources` mirrors
  `add_sequence_batch`; with no reuse, re-adding is clean; pre-prepared dummies are skipped in
  both directions. The ADP dummy: finalize(False), then rewind, then retire. Correct. Ask for
  a GPU unit test that runs prepare→revert→prepare and compares block tables. INFLIGHT_PLAN
  lists it; I did not find it.
- **H11. Re-entering `event_loop`** re-runs the `_executor_loop` prologue (PE:4652-4660):
  set_device, `_profiler()` and the hang-detector contexts, and `can_forward`. That is fine for
  the supported profile, but the profiler restart can re-trigger a capture window if
  `TLLM_TORCH_PROFILE_TRACE` / start-stop iterations are set. Reject that, or document it.

---------------------------------------------------------------------------------------------
## Q5 — Scope creep

**Verdict: modest. Most code is load-bearing. Move these out or defer them:**
- Commit 44eded2a5c (frontend admission pause) is fully reverted in the working tree
  (`admission.py` deleted, `proxy.py` back to base). Drop it from the series; do not ship an
  add-then-remove.
- ADP strict-rank failure path (`_fail_unservable_requests`, dict-based routing): needed only
  for requests pinned to the dead rank. A smaller MVP option is to treat strict pins to a
  departed rank as relaxed, or to reject strict `attention_dp_rank` under FT. If you keep it,
  give it its own PR because it changes FT-off behavior (Q3.1).
- `MnnvlMemory.pin/unpin`, plus `MoeExecutionState.close` and "retain pinned allocation"
  (`_mnnvl_utils.py:829`, which leaks by design): this guards against sleep/checkpoint
  unmapping. `create()` already rejects sleep_config, so this is defensive. Keep it only if
  cheap to review; otherwise defer it.
- `check_routing_excludes` (`moe_load_balancer.py:968`) routes about
  experts × ep × slots × top_k ids per layer during recovery. It is a valid defensive check,
  but it could be debug-gated or sampled. Not creep exactly, but consider the recovery-latency
  cost on R1 (58 layers).
- `ConfigurableMoE` memo fix, `_shared_barrier`, and `EPGroupHealth.commit_failed` are fine,
  but the memo fix is an independent bug fix.
- Not creep but missing: no integration/e2e test in-tree (the EP32 harness lives in
  `.wideep-ft-runs`), no user doc, and several new unit tests are not in any test-db list
  (`unittest/executor/test_recovery_authority.py`,
  `unittest/_torch/distributed/test_mpi_survivors*.py`,
  `unittest/_torch/moe/test_moe_a2a_abort.py`,
  `unittest/_torch/moe/test_moe_execution_state.py`). `l0_b300` picks up
  `unittest/_torch/executor` as a directory, so `test_peer_loss_recovery.py` does run there.
  Add the CPU ones to `l0_cpu.yml` and the GPU ones to a GB200/B200 list.

---------------------------------------------------------------------------------------------
## Q6 — Proposed PR split (each builds and tests on its own)

0. **Unrelated fixes, each separate, can land any time:** 4dc8d8803f (skip update without a
   sample result), e13dab7bcd (DLPack alias lifetime), 952dcac056 (bounded RPC socket
   teardown), e80f9cbed6 (RPC fatal error propagation plus `l0_cpu.yml` entry).
1. **[infra] mpi4py 4.1.2** — `install_mpi4py.sh` (re-ported futures patch, setuptools pin
   removed), `test_mpi_session.py` startup test. Needs an image rebuild and a full
   pre-merge run, since mpi4py 4 changes futures internals and error classes. It must land
   before PR4's real-ULFM test is enabled. Mention that pip users already get 4.x.
2. **[feat] MoE A2A cooperative abort + execution descriptor + fixed per-source slots** —
   `moeAlltoAllKernels.{cu,h}`, `moeAlltoAllOp.cpp`, `cpp_custom_ops.py` fake, and
   `test_moe_a2a_abort.py`. Squash 273d31cfbe/e4c6cefa41/a0311cb56a with the uncommitted
   edits into one ABI. It has no Python caller yet (op tests only). Coordinate with #19610
   (rebase target) and #15895 (abort ABI).
3. **[feat] MoeExecutionState + NVLinkOneSided binding** — `execution_state.py`,
   `nvlink_one_sided.py` (bind/reset/fixed-slot sizing), `wide_ep_ft.recovery_enabled` (with
   the env check moved to a leaf module, S-A5), `EPGroupHealth.commit_failed`, `MnnvlMemory`
   pin, DecoderRunner/model_engine `moe_execution_state`, and `test_moe_execution_state.py`.
   Depends on 2. Ship the `ConfigurableMoE` memo fix as **3a**, a standalone fix.
4. **[feat] MPIDist ULFM survivor communicators** — COMM changes, `_utils.is_mpi_peer_failure`,
   logical-length gathers in the `Distributed` base, `test_mpi_survivors.py`, and
   `test_mpi_survivors_ulfm.py` (skip if no ULFM; enable after 1). Independent of 2 and 3.
   Include B2 and H7 here.
5. **[feat] Recovery authority + protocol + client** — `executor/recovery/*` and
   `test_recovery_authority.py` (add to `l0_cpu.yml`). Torch-free and independent; it can land
   first. Apply the S-A1 simplification here if you take it.
6. **[feat] MPI session survives a departed worker** — `mpi_session.py` and its tests.
   Depends on 4 (classifier) and the env helper (5 or the leaf module).
7. **[feat] EPLB resident-replica masking** — `moe_load_balancer.py`
   (digest/covers/reconfigure-with-timeout/check_routing/_shared_barrier gated) plus
   `model_engine.prepare/install_resident_placement`. Depends on 3.
8. **[refactor] ADP scheduling tolerates absent ranks** — the FT-neutral parts of the executor
   (`adp_router` dicts, `request_utils`, `_gather_rank_status`, fill values,
   `cuda_graph_runner.live_ranks`, `num_admitted_requests`, `_fail_unservable_requests`,
   `_publish_responses` refactor with the nvtx fix) plus their tests. Depends on 4 for
   `live_ranks`. This isolates every FT-off-visible change (Q3) for review.
9. **[feat] Executor peer-loss recovery** — `peer_loss_recovery.py`, the remaining PE hooks,
   `KVCacheManager.revert_prepare_resources`, `test_peer_loss_recovery.py`, the user doc, and
   an e2e test definition (manual/nightly GB200 multi-node list is acceptable). Depends on
   all of the above. B1 and H1–H5 land here.

If you are limited to about 7 PRs, merge 6 into 4, and 8 into 9. Keep 8 separate if you can,
because it is the one that touches default behavior.

---------------------------------------------------------------------------------------------
## Nits
- `from .peer_loss_recovery import PeerLossRecovery` is out of isort order (PE:106, between
  `disagg_adapter` and `dwdp`).
- `MoeA2AAbortState.status` is written by kernels and never read on the host except
  `abort.any()`. Either use it in the error message or drop it in favor of `requested` only.
- `mpi_session` degraded shutdown calls `COMM_WORLD.Abort(0)` (`mpi_session.py:1054`). Exit
  code 0 hides the degraded state from schedulers; consider a distinct non-zero code, or log a
  clear line that orchestrators can grep.
- `_ADP_BALANCE_DEPARTED_RANK_COUNTS` uses `sys.maxsize` as a sentinel in an int64 vector.
  Fine, but name the column semantics at the use site.
- `RecoveryClient` is never closed; `_registration` is module-global. Acceptable, but add a
  `close()` at executor shutdown, or say it lives for the process lifetime.
- Copyright headers: `communicator.py` got a 2022-2026 SPDX header added; check that the
  repo's header style for previously header-less files matches.

# WideEP FT — Python review, round 2

Scope: Python side of `git diff 55279c109f` (recovery package, pyexecutor, communicator,
mpi_session, _utils, EPLB, wide_ep_ft, install_mpi4py.sh) plus the untracked
`peer_loss_recovery.py` / `test_mpi_survivors_ulfm.py`. Unrelated commits 4dc8d8803f,
e13dab7bcd, 952dcac056, e80f9cbed6 were ignored. The worktree was being edited while this
review ran (import order in py_executor.py was already fixed). Line numbers are for the
working tree at review time.

## Blockers

B1. **`Shrink` is unbounded and nothing watches it.** `peer_loss_recovery.py:289`.
`_recover` runs outside the event loop's `with self.hang_detector`, and `dist.shrink()` comes
before `Stopped`, so the authority deadline has not started yet either. Suppose one survivor
never reaches Shrink. It might be stuck in a GPU wait the MoE abort does not cover (any
non-MoE device collective), or in a collective on a communicator that `revoke()` does not
touch (EPLB `shared_mpi_comm`, a KV connector `mpi_comm()` user, the session `COMM_WORLD`).
Then every other survivor blocks in `MPI_Comm_shrink` forever. That breaks the DESIGN
invariant "every recovery wait is bounded; any failure ⇒ terminal path". The same applies
to the `watch_peer_failures` `Dup` (`:322`) and the `dist.broadcast` in `_deliver_responses`
(`:348`), though ULFM does raise on a *new* failure there. Fix: run `_recover` under a
deadline-armed watchdog that takes the existing hard-kill path, e.g. keep a `HangDetector` /
`start_rank_crash_kill_watchdog` armed for `self._timeout_s`. Add a unit test showing that a
stuck `shrink` ends in the terminal path.

B2. **A failed recovery reports the wrong error, because `PeerFailureWatch.stop()` is not
idempotent.** `communicator.py:830-840`, `peer_loss_recovery.py:193,285`. `_recover` stops
the watch first (`:285`). If anything after that raises (`RecoveryAborted`, `TimeoutError`,
"exactly one departed rank", a second loss), `run()`'s `finally` calls `self._watch.stop()`
again on the same object. `self._comm.Free()` on an already freed (`COMM_NULL`) communicator
raises `MPI.Exception(ERR_COMM)`. That error is not a peer failure, so it propagates and
replaces the real cause on the way into `_event_loop_wrapper`. The client then sees an MPI
handle error instead of "Recovery aborted by authority: …". Fix: make `stop()` idempotent
(return if already stopped), or set `self._watch = None` after stopping and guard the
`finally`. Unit tests miss this because `_watch` is a `Mock`, which tolerates repeated calls.

B3. **The real-ULFM test fails instead of skipping on non-ULFM MPI.**
`test_mpi_survivors_ulfm.py:38-42`. It is collected by the `unittest/_torch/distributed`
entry in `l0_cpu.yml` and is marked `cpu_only`. `_ulfm_available()` only checks for
`mpirun` and mpi4py>=4. The CI image's MPI comes from `install_base.sh`
(`openmpi-bin`/`libopenmpi-dev`, i.e. Open MPI 4.1.x on Ubuntu 24.04). Open MPI 4.1.x does
not know `mpirun --with-ft ulfm`, so after the mpi4py 4.1.2 upgrade this test **fails**
pre-merge. Gate on real ULFM: Open MPI 5 plus `ompi_info --parsable | grep -i ft`, or a
probe run that skips on "unrecognized option". Otherwise move the test to a list whose image
has ULFM.

B4. **`test_recovery_authority.py` is not in any CI list** (A-B3, still open). `l0_cpu.yml`
lists `unittest/executor/*` file by file and does not include it, so the authority's whole
test suite never runs in CI. Add `unittest/executor/test_recovery_authority.py` to
`l0_cpu.yml`.

## Should-fix

S1. **The ULFM capability gate never triggers.** `communicator.py:880-883`. With mpi4py 4.x,
`MPI.Comm.Revoke/Shrink/Get_failed` always exist as attributes, even when the MPI library
lacks ULFM. Two things go wrong:
- The gate passes on a non-ULFM MPI. The deployment registers, and only fails (or the whole
  job aborts) at the first loss.
- Without `--with-ft ulfm`, Open MPI 5 aborts the job on any process failure anyway.

Use a real check: `_peer_failure_error_classes()` must be non-empty, `mpi_comm().Get_failed()`
must not raise `NotImplementedError`, and FT must be enabled at runtime (the `mpi_ft_enable`
MCA parameter, or the `MPI_FT` attribute if mpi4py exposes it). M-B2's "ULFM capability
gate" is therefore not actually done.

S2. **Requests with n>1 drop out of the owner map too early** (E-S9, still open).
`peer_loss_recovery.py:235-257` together with `py_executor.py` (`_handle_responses` keys
child responses by `parent_request_id`, around 8865). `requests_routed` records the parent
id. `responses_gathered` pops it at the *first* child's final response. If the owning rank
dies after that, the remaining children are never failed and the client hangs. Either track
the outstanding child count, or reject `n>1` / `best_of>1` in FT mode.

S3. **`create()` does not enforce all of DESIGN's rejections.** `peer_loss_recovery.py:131-149`.
DESIGN says recurrent state (Mamba/linear attention), sampler RNG/penalties and spec decode
"are rejected". `create()` only documents that "requests are expected to sample greedily".
Missing items:
- A check against a hybrid/Mamba cache manager. `MambaHybridCacheManager` subclasses the V1
  `KVCacheManager` and passes the "KV cache manager v1" check.
- A request-level greedy check (reject non-greedy requests at admission while FT is on), or
  a DESIGN update that says this is documentation only.
- `enable_lm_head_tp_in_adp`, which adds a device collective outside the MoE abort path
  (see B1).

S4. **The degraded session server leaves a sync submit hanging.** `mpi_session.py:1057-1059`.
Once a rank has departed, a task gets only a log line ("dropped a task"). No reply goes to
the client, so `RemoteMpiCommSessionClient.submit_sync` (`:831-846`) loops until shutdown.
Put an error onto `self.queue` (for example `RemoteWorkerDeath` or a `RuntimeError`) so the
client fails fast.

S5. **The `Abort(0)` teardown of a degraded world** (`mpi_session.py:1043-1055`) is
defensible: mpi4py.futures cannot disconnect a world with a failed member (see
MPI4PY_ISSUE.md). It still needs work before main:
- It skips every `atexit` handler on every survivor (EPLB `/dev/shm`, MNNVL, the PP progress
  pump), and `_close_global_comm_executor` is never reached.
- `logging.getLogger().handlers` flushes the root logger, not the TRT-LLM logger's handlers.
- Exit code 0 hides the fact that a rank was lost. Consider a distinct non-zero code, or at
  least a documented choice.
- The behavior needs user-facing documentation, and the `serve()` shutdown / drop branches
  have no test (only `_live_worker_futures` is tested, with stubs).

S6. **Revoked communicators are never freed** (M-S2, partly done). After
`install_survivors`, `_host_world` and `_tp_comm` stay revoked and referenced forever, and
`enable_peer_loss_recovery` on a second `Distributed` instance (if the KV-estimation and
serving executors do not share one) leaks a `Dup`. Free them in `install_survivors`
(`Free` is local in ULFM), or document why they must stay alive.

S7. **The `@nvtx_range("_enqueue_responses")` decorator is now on `_publish_responses`.**
`py_executor.py:8889-8890`. `_publish_responses` was inserted between the decorator and
`_enqueue_responses`. As a result `_enqueue_responses` lost its NVTX range and
`_publish_responses` is mislabeled. This is an FT-off profiling regression. Move the
decorator back and give `_publish_responses` its own range or none.

S8. **FT is silently off while `gather_all_responses` is set** (E-S6). `peer_loss_recovery.py:179`.
`gather_all_responses` is used as a stand-in for "KV-estimation executor". `set_gather_responses`
is public, though, and any caller that sets it turns off the watcher and recovery for
serving without warning. Make the condition explicit: a flag passed by `_util.py`, or
`is_warmup`. At minimum, log at WARNING when `run()` falls back.

S9. **`wide_ep_ft.py:83` drops the `"ep_group_health"` `extra_attrs` fallback.** That fallback
exists on main (`55279c109f`). Removing it is an unrelated FT-off behavior change; restore it
or split it into its own PR. In the same file, `recovery_enabled()` is `functools.cache`d,
but `mpi_session._recovery_enabled()` re-reads the environment on every call. The two can
disagree, and tests must call `cache_clear()`. Pick one.

S10. **Interruptions after the commit gate skip that step's post-gate bookkeeping.** After
`_handle_responses`, the following do not run for that step: `update_resources`,
`disagg.handle_timeouts_synced`, KV events, `iter_counter`, and iteration stats. For the
supported configuration (no spec decode, no block reuse) `update_resources` only does
`store_context_blocks`, which is a no-op there, so this is safe today. It is undocumented,
though: add a comment at the gate, or an assertion in `create()` that the configuration has
no post-gate obligations.

S11. **Executor coupling** (E-S8). `PeerLossRecovery` reads and calls executor privates:
`_discard_uncommitted_step`, `_requeue_unscheduled_requests`, `_publish_responses`,
`_disable_mpi`, `_is_kv_manager_v2`, `execution_stream`, `request_accumulated`, and
`waiting_queue` (indirectly). A small adapter protocol, or public hook methods on
`PyExecutor`, would make the boundary reviewable and the fakes honest.

S12. **Registration timeout is a hard-coded 600 s** (`peer_loss_recovery.py:42`). It also
assumes every rank constructs its first executor within 10 minutes of the others. With
weight loading from a shared FS at EP32 the skew can be larger. Make it configurable, the
same way `TLLM_FAULT_TOLERANCE_RECOVERY_TIMEOUT_S` is. Also validate that variable at parse
time (it must be >0 and greater than the authority's timeout).

## Nits

- `peer_loss_recovery.py:21,44,46,100,106`: new code uses `typing.Dict/List/Tuple/Deque/Optional`.
  CODING_GUIDELINES §Static Typing #6 asks for builtins and `|`. `Responses` should be a
  `TypeAlias`. `_expect_generation(message, …)` and `requests_routed(routed: Dict[int, list])`
  are under-typed.
- `py_executor.py:6441`: `_pop_from_waiting_queue` is still typed `Optional[List[int]]` but
  now receives a `dict`.
- `py_executor.py` / `request_utils.py` FT-off change: a strict `attention_dp_rank` that is
  out of range used to raise `IndexError`; it is now failed as "not serving". This is
  arguably better, but call it out in the PR.
- `communicator.py:1232-1244`: `tp_cp_broadcast_int64` compacts the TP root but not the CP
  root. Harmless while CP=1 is enforced; add a comment.
- `communicator.py` `_watch`: Test() at a 10 ms poll under `THREAD_MULTIPLE` contends for
  the progress lock (M-N8). Measure the overhead or raise the interval.
- `communicator.py` `shrink()`: when `failed_ranks` is empty after a non-peer watcher error
  (M-S4), the terminal message says "requires exactly one departed rank". That misleads;
  say "no departed rank: detection failed".
- `authority.py`: the `--admitted-ranks` parse error surfaces as a raw `ValueError`. The
  module docstring should say the authority keeps no persistent state (a restart loses the
  recovery) and that it imports all of `tensorrt_llm`, so it needs the full wheel and image
  (DESIGN.md still calls it "torch-free").
- `peer_loss_recovery.py:92`: `float(os.environ.get(...))` gets no validation and no
  error context.
- `test_mpi_survivors_ulfm.py:110`: `watch_peer_failures(failure_seen.clear)` never asserts
  that no callback fired. Use a fresh `Event` and `assert not fired.is_set()`.
- REVIEW_FINDINGS.md is stale. M-B1, M-S1, M-S3..M-S7, M-S9..M-S11, A-S13, most of A-S16,
  E-B1, E-B2, E-S3..E-S5 and E-S7 are implemented but still unchecked. M-B2 (capability
  gate), M-S2, E-S6, E-S8, E-S9 and A-B3 remain open (see above).

## Verified OK (items flagged last round or probed now)

- Watcher ordering is correct. `_on_peer_failure` sets the flag before `request_abort()`.
  `commit_step` waits on `sampler_event` (or the stream) and only then checks the flag. A
  flag set after the check is caught by the next step's gate, and `rearm()` precedes
  `install_mask`. The watch is stopped (joined) before `revoke`/`shrink` and re-armed only
  after Commit.
- Timeouts are consistent: authority 60 s from the first `Stopped`, worker 120 s from
  detection, registration 600 s. All `client.wait`, `_join_device_work` and the EPLB
  reconfigure use the same deadline. The only unbounded waits are those in B1.
- A second failure fails closed. During recovery the result is an authority deadline or
  violation, then Abort. After a recovery, `shrink()` raises "second peer loss". During the
  `_deliver_responses` broadcast, `HostCollectiveInterrupted` escapes `run()` because the
  `except` only wraps `event_loop()`.
- A failure in the response gather is handled: contributions are numbered, two are kept per
  rank, terminations are deferred, and rank 0's processed count is broadcast. The
  `test_recovery_resends_gathers_rank0_did_not_process` logic matches the code.
- A failure in the KV-estimation executor is terminal (no watcher). Registration happens
  once per process, with a rank check.
- The KV V1 rollback math matches `prepare_resources`. A first chunk is undone with
  `free_resources`/`remove_sequence`, which also covers extra and draft tokens. Generation
  requests are rewound by `1 + max(draft_len, _kv_reserve_draft_tokens)`. Preprepared
  dummies are skipped. Cross-attention and helix are rejected. The ADP dummy is rolled back
  correctly both before `prepare_resources` (`_finalize_adp_dummy_allocation(False)`) and
  after it (rewind, then retire). CUDA-graph padding dummies are popped in `pad_batch`'s
  `finally`, so they are never rewound.
- `_interruptible` converts errors only when `_host_world` is set (M-B1). FT-off gathers are
  unchanged: `fill_value` defaults, `live_ranks == range(tp_size)`, admission adds +0.
  `RequestBroadcaster` uses only decorated methods.
- Only `_executor_loop` carries hooks. The overlap, PP and disagg loops are excluded by
  `create()`. The broad handlers in `_forward_step`, `_sample_async` and `_update_requests`
  call `_raise_if_peer_failed`. The other broad handlers in the loop path (validation,
  encoder, MM prefetch) run no host collectives.
- Gathers that cross a departed rank are covered: `all_rank_num_tokens` and ctx counts
  (fill 0), `_can_queue` (fill 1), ADP balance (departed row), CUDA-graph votes (live
  rows), rank-state gather (Nones dropped), and the iteration-stats dict gather (handles
  None).

## Test adequacy

What exists covers the units well: the authority state machine, the wire format, the fake-MPI
communicator (`test_mpi_survivors.py`), the recovery protocol order, the gate/discard/defer
hooks, and the ADP routing over survivors. The gaps:

1. **Fakes that are too permissive.** `_recovery()` uses `Mock()` for `_watch`, which hides
   B2. `_executor()`'s `dist.shrink` returns instantly, which hides B1. `revert_prepare_resources`
   is tested only against a mocked `impl`, so nothing proves the C++ block accounting
   returns to the same free-block count (E-N "KV round-trip GPU test" is still open).
2. **No test that the hooks sit in the right place in the real loop.** Nothing asserts that
   `_executor_loop` calls `step_prepared` after `prepare_resources` and `commit_step` before
   `_update_request_states`. A loop-level test with a stub engine/sampler is enough (E-N
   "gate-in-loop test").
3. **No real-ULFM test of revocation unblocking a peer**, which is the core liveness
   mechanism. One survivor should block in `tp_broadcast` from a live root while another
   calls `revoke()`.
4. **No real-ULFM end-to-end recovery test.** Nothing runs real `MPIDist`, a real
   `RecoveryAuthority` over ZMQ, and `PeerLossRecovery._recover`, with only the engine
   stubbed. That would be CPU-only, cheap, and the highest-value missing test.
5. **mpi_session.** The `serve()` drop and Abort paths are untested (S4/S5).
6. **No GPU end-to-end test in any list.** A small multi-GPU test would cover it, in
   `l0_dgx_b200.yml` / `l0_gb300_multi_gpus.yml` with a tiny MoE, EP4, static EPLB with
   replicas, killing rank 2 mid-decode. It needs a ULFM-capable image, so it can start as a
   QA / waived test, but that should be a deliberate choice.

Minimal set of tests to add:
- (a) `PeerFailureWatch.stop` is idempotent, and `run()` surfaces `RecoveryAborted` after a
  failed `_recover` (fake MPI). This also covers B2.
- (b) A stuck `shrink` reaches the terminal path within the deadline (B1).
- (c) A loop-level test of gate placement with a stub engine/sampler.
- (d) In the ULFM script: revoke unblocks a blocked peer, plus an end-to-end
  `PeerLossRecovery._recover` against a real `RecoveryAuthority` (same mpirun, `cpu_only`,
  with a correct ULFM skip).
- (e) A GPU round-trip for `revert_prepare_resources` (real V1 `KVCacheManager`: free block
  count before prepare equals the count after revert). Belongs in `l0_a10`/`l0_h100` via
  `unittest/_torch/executor`.
- (f) `serve()` answers a task submitted after a departure with an error.
- (g) A test for n>1 owner tracking (S2).

CI list placement:
- `test_recovery_authority.py` → `l0_cpu.yml` (explicit entry).
- `test_peer_loss_recovery.py`, `test_mpi_survivors.py`, `test_mpi_survivors_ulfm.py` are
  already collected by the `_torch/executor` and `_torch/distributed` directory entries in
  `l0_cpu.yml` (all `cpu_only`). The `requires_cuda` cases in `test_peer_loss_recovery.py`
  run only on GPU lists that include `unittest/_torch/executor` (h100, b300, …).
- The new `test_mpi_session.py` cases run via the existing `l0_cpu.yml` ISOLATION entry.

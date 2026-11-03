Review of PRs 20–23 as a code owner. PRs 20, 21 and 23 are fully read. PR 22 is only partly read: I covered its `py_executor.py` and `py_executor_creator.py` hunks, `llm_request.py`, and `PeerLossRecovery.create`/`run`/`_on_peer_failure`/`step_*`/`commit_step`. I did not read the rest of the 1846-line `peer_loss_recovery.py`, the ~3000-line `test_peer_loss_recovery.py`, or `test_moe_a2a_recovery.py`, so the PR 22 verdict is provisional.

I did not find a behaviour change for users with `TLLM_FAULT_TOLERANCE_MODE` unset. The PR 22 `publish_responses` rewrite of `_flush_pending_responses` is equivalent to main's enqueue-then-terminate.

## PR 20 – KV Cache Manager V2 step revert

**1. minor – `kv_cache_manager_v2.py:5663-5681` duplicates `revert_allocate_generation` (4783-4821).**
- The generation branch of `revert_prepare_resources` copies the other method's body, and the two have drifted:
  - `revert_allocate_generation` is keyed on `_allocated_draft_lens` and decrements `py_helix_decode_group_index`. The new branch is keyed on `_inflight_draft_lens` and never touches the helix index.
  - `revert_allocate_generation` returns when `reverted_cap < 0`. The new branch (5676-5681) still calls `_after_shrink` in that case.
- Failure scenario: a later change to one copy (for example the helix rollback) silently misses the other.
- Fix: extract one `_shrink_generation_allocation(req, draft_slots)` helper. If helix can run under recovery, revert the index too.
- Commit message: it says nothing on this, so it is not justified.

**2. minor – coverage gaps in `test_kv_cache_revert.py`.**
- The tests cover context and draft-pool reverts, but none exercises a generation request with `draft_slots == 0` that has no `_inflight_draft_lens` entry (the `if not inflight: continue` path).
- The test for the `kv_cache.resize` failure raising `RuntimeError` is also missing.
- Fix: add both. They are cheap.

**3. minor – `if req.is_first_context_chunk: free_resources(req)` (5647).**
- This silently drops a reused prefix and its blocks, and the first chunk reruns from scratch. The test covers the reuse-prefix rewind.
- It relies on the executor having already restored `context_current_position`. `revert_prepare_resources` is order-sensitive against `discard_interrupted_steps` in PR 22, which restores positions before calling it. The docstring should say so.

**Verdict PR 20: approve with changes** (items 1–2).

## PR 21 – attention-DP with absent ranks

**1. major – the `live_ranks` filtering is copy-pasted and relies on an unstated invariant.**
- Three sites repeat `if len(dist.live_ranks) != len(gathered): gathered = gathered[list(live_ranks)]`:
  - `cuda_graph_runner.py:~395`
  - `cuda_graph_runner.py:~1619`
  - `py_executor.py:~7640`
- Each depends on gather row index equalling logical rank, which holds only if `tp_allgather_int64` returns exactly `tp_size` rows.
- Other sites use `departed=0` and keep the departed rows. `get_all_rank_num_tokens` passes `departed=0` and feeds rows including zero rows into `set_adp_token_counts`. The MoE path tolerates this only because zero-token ranks are valid.
- Fix: add a `Distributed.tp_allgather_int64_live(values)` helper, or a module helper that returns survivor rows. That removes the invariant from the call sites and the three-fold duplication.
- Commit message: not addressed.

**2. major – fakes and other `Distributed` stand-ins were not updated for the new keyword.**
- Production now calls `tp_allgather(local_dict, departed=None)`, `tp_gather(..., departed=None)` and `allgather(..., departed=None)`.
- These fakes lack the parameter:
  - `tests/unittest/_torch/disaggregation/fake_dist.py:133` (`tp_allgather`)
  - `tests/unittest/_torch/executor/test_adp_host_exchange.py:278` (`tp_allgather`) and `:263` (`allgather`)
- The PR patched only `tp_allgather_int64` in `fake_dist.py`.
- Failure scenario: any test or out-of-tree `Distributed` implementation that drives `gather_all_rank_states`, `_flush_iter_stats_synced` or `_enqueue_responses` through these fakes raises `TypeError: unexpected keyword 'departed'`. This breaks users with fault tolerance off.
- The same pattern is likely in out-of-tree subclasses of `Distributed`.
- Fix: update the fakes. Consider documenting that `departed` is part of the `Distributed` ABC contract.
- Commit message: not addressed.

**2b. minor – existing tests were rewired to require `departed`.**
- Many tests changed `lambda payload:` to `lambda payload, departed:` (`test_adp_router.py`, `test_kvcache_aware_router.py`). They now fail if the keyword is ever dropped. That is arguably intended, but the churn is large and the lambdas are positional-required.
- Prefer `departed=None` as a default in those mocks.

**3. minor – bug-compat code and tests (`request_utils.py:140-143`, `adp_router.py:462-477`).**
- Without recovery the PR preserves main's quirks. A pin past the last rank raises `IndexError`. A negative pin is looked up by position via `list(all_ranks_num_active_requests)[target_dp_rank]`, then fails with `KeyError` in the router.
- The tests pin this behaviour ("as on main").
- Commit message states a reason: "A pinned target missing from the serving ranks is looked up by position (the existing indexing) unless peer_loss_recovery_enabled()". That justifies the behaviour but not the maintainability cost: two behaviours, gated twice in two files, and tests asserting that a `KeyError` is correct.
- Fix: validate the pin once at admission and fail the request in both modes. If the compat is deliberate, keep it but put the lookup in one helper.

**4. minor – inconsistent data structures in `adp_router.py`.**
- `DefaultADPRouter` and `can_process_attention_dp_request` now use `Dict[rank, count]`.
- `KVCacheAwareADPRouter` (lines 966, 996, 1091) and `ConversationAwareADPRouter` (1267, 1372) still use lists indexed by `range(len(all_rank_states))`. They would be wrong after a loss.
- This is currently safe only because `_requirements` in PR 22 insists on `type(executor.adp_router) is DefaultADPRouter`.
- The docstring of `_assign_explicit_dp_ranks` still advertises it as shared with ConversationAware. Update it, and add a guard or comment on the other routers.

**5. minor – `py_executor.py:7451` still seeds `_fill_admit_cap` from `self.dist.tp_size`.** The PR converted admission capacity to `len(live_ranks)` but left this. It only matters for the benchmark fill phase after a loss.

**6. minor – `ExecutorRequestQueue.wake()` (`executor_request_queue.py:~319`).**
- Other queue consumers still see the wake item: `qsize()` (`:395`), `request_utils.py:57`, and the `raw_queue` drain at `py_executor.py:9828`. They are safe, since the drain skips items with `request is None`, and `is_normal_request` is updated.
- `qsize()` includes the wake item, so it over-reports queue depth in stats.
- `waiting_for_requests` is a bare attribute written by the executor thread and read by the hang-evidence thread. It is benign but undocumented as such.

**Verdict PR 21: request changes** (item 2 breaks fakes and subclasses with fault tolerance off; item 1 is a maintainability block).

## PR 22 – PeerLossRecovery and executor hooks (partial review)

**1. major – `RecoveryHook.run` / `gather_all_responses` silently disables recovery (`peer_loss_recovery.py:545`).**
- `run()` returns after a plain `event_loop()` when `gather_all_responses` is true. `create()` has already called `_register` and `dist.enable_peer_loss_recovery()`, and `step_prepared` and `commit_step` hooks are live.
- The docstring ties this to KV cache estimation, but nothing enforces that. `set_gather_responses(True)` is a public method (`py_executor.py:2543`).
- Failure scenario: a deployment that calls `set_gather_responses(True)` registers with the authority and reports `healthy` on `/health` while a peer loss is unrecoverable.
- Fix: either add `"not gather_all_responses"` to `_requirements`, or skip registration in `create()` when it is set.
- Commit message: not addressed.

**2. major – size and structure.**
- `peer_loss_recovery.py` is 1846 lines in `pyexecutor/` and mixes registration, authority protocol, MPI fencing, KV quota, EPLB and request redelivery.
- The executor takes roughly 20 hook call sites plus public methods (`discard_interrupted_steps`, `readmit_requests`, `take_buffered_responses`, `served_request_ids`, `fail_requests`, `publish_responses`).
- A code owner would want it split: authority/fence client, request redelivery, and step bookkeeping.
- Hooks are gated inconsistently. Some call sites use `if self._peer_loss_recovery.enabled`, others call a no-op base method unconditionally (`step_released`, `requests_cancelled`, `defer_terminations`). The commit message acknowledges this ("except step_released() … requests_cancelled(), which are no-ops").
- Fix: make it uniform. Either always call hooks through the null object, or always guard.

**3. major – `os.kill(os.getpid(), signal.SIGKILL)` (`py_executor.py:~1738`).**
- It is gated by `crashed and exits_alone(e)`, which requires `classify_error(str(error)) == "immediate_fatal"`.
- This is a string-match classifier deciding a process-kill path, so misclassification kills a healthy rank.
- No guard checks that at least one survivor can recover (`survives_own_loss` is `self._admitted`, updated after each recovery).
- I did not see a dedicated test of this exit path in what I read.
- Fix: add a unit test for `exits_alone` over classified and unclassified errors, and log which classification triggered the kill.

**4. minor – redundant change recording.**
- `step_prepared` pre-records the context requests' state (`peer_loss_recovery.py:~665`) and `_update_request_states_tp` also calls `_note_step_change` for the same requests (`py_executor.py:9541`). Restore walks the list in reverse, so the earliest entry wins and it is harmless, but it is redundant.
- Dead state: `_step_changes` is set only in the overlap loop (`py_executor.py:6564`). In the non-overlap loop the return value of `step_prepared` is discarded and `_step_changes` stays `None`. Make that asymmetry explicit.

**5. minor – class-level mutable default `_peer_loss_recovery: RecoveryHook = RecoveryHook()` (`py_executor.py:~526`).** A shared instance as a class attribute is fragile. Tests (`test_kv_pool_rebalance.py`) had to inject it on `MagicMock(spec=PyExecutor)`.

**6. minor – `LogitsStorage.checkpoint` / `PyResult.checkpoint` in `llm_request.py` hot types.**
- They are called only from recovery, but they add public API to `PyResult`, and `rollback` leaves `diff` stale for pipeline parallelism (documented in the docstring).
- Failure scenario: PP recovery sends stale diffs. `_requirements` does not exclude PP, though `enable_peer_loss_recovery` is said to check the layout.
- I did not verify that the layout check excludes PP.

**7. minor – `watch_startup_peer_failures` registers an `atexit` handler and a module-global `_startup_watch`.** Reasonable, but it is global state that is not reset on executor re-creation (KV estimation then serving).

**Verdict PR 22: request changes (provisional)** for items 1–3 and the file split. Re-review once the unread 1846-line module and its tests are covered.

## PR 23 – `/health` and 503 on peer loss

**1. major – unauthenticated `/health` leaks topology (`fault_tolerance_status.py:84-102`).**
- The docstring says an abort reason is withheld "because the endpoint is unauthenticated". But the body still returns `live_ranks`, `failed_ranks`, `generation`, `recoveries`, per-stage recovery milliseconds and `failed_requests`.
- This is an information disclosure about the cluster that the authority's log does not make public. It is also a stable JSON schema that load balancers and alerts will depend on.
- Fix: return only `state` on `/health`, and move details to an authenticated route (or `/metrics`), or gate them behind a flag.
- Commit message: it says "/health reports the fault-tolerance state", and the module docstring justifies hiding only the abort reason. The rest is not justified.

**2. major – blocking probe on the default thread pool (`openai_server.py:~1780`, `fault_tolerance_status.py:144-153`).**
- `health()` runs `asyncio.to_thread(probe.status)`. That call waits on a single-worker `ThreadPoolExecutor` for up to `timeout_s = 0.5 s` per cache miss.
- `max_age_s` (0.1 s) bounds authority queries, not threads. Every concurrent `/health` poll occupies a default executor thread while it waits on the single probe worker.
- Failure scenario: an authority outage plus a burst of load-balancer polls ties up the default thread pool for 0.5 s per queued call, starving other `asyncio.to_thread` users.
- Fix: do the refresh in a background task and have `/health` read the cached value without waiting.

**3. minor – retry semantics from message text (`openai_server.py:~1210`, `_stream_error_events`).**
- A 503 plus `Retry-After: 1` is chosen by `is_peer_loss_retryable_message(message)`, a substring check on an error message that can echo user text. The test file acknowledges a "user text" case.
- It is gated by `fault_tolerance_mode()` and not by `peer_loss_recovery_enabled()`, so a message with the tag turns into 503 in detection-only mode. The test claims detection-only mode is handled, but only for authority unset and mode unset.
- Fix: gate on `peer_loss_recovery_enabled()` and match a prefix rather than a substring.

**4. minor – `/health` body change under mode 1 only.**
- `Response(status_code=200)` becomes a JSON `{"state":"unprotected"}` when `fault_tolerance_mode()` is on without an authority. That is a visible change but gated, and the commit message states it.

**5. minor – `/health` returns 503 for `recovering` and `terminal`.**
- Kubernetes liveness probes pointed at `/health` will restart pods during a roughly 1 s recovery. The commit says "not serving, so load balancers move on" and the comment states the intent, so this is a documentation point.
- Operators need guidance to use `/health` for readiness only. No doc is added in this PR.

**Verdict PR 23: request changes** (items 1–2).

## Summary

| PR | Verdict |
|----|---------|
| 20 | approve with changes |
| 21 | request changes |
| 22 | request changes (provisional) |
| 23 | request changes |

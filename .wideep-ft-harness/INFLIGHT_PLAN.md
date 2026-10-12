# In-flight recovery: discard the aborted step, roll back, re-run

Profile: DeepSeek-R1, ADP, non-overlap `_executor_loop`, KV manager V1, no block reuse, no spec
decode, greedy. Line refs are to the worktree at the time of writing (PE = py_executor.py,
RM = resource_manager.py, S = sampler/sampler.py).

## Commit point
`_update_request_states` (PE ~5006). Before it, no request token/position/state has changed.
Today the host first blocks in `TorchSampler.update_requests` -> `sampler_event.synchronize()`
(S ~1481), i.e. after `_update_request_states`, right before garbage tokens get appended.

## Mutations before the commit point (V1)
- `prepare_resources` (RM ~1179): generation: `add_token` x (1 + reserved draft); first context
  chunk: `add_sequence_batch(prompt)` (+extra KV tokens). Seq slots: assigned once, idempotent.
- ADP padding dummy appended to `active_requests` (PE ~7784), normally removed at
  `_update_request_states_tp` (PE ~8471).
- Sampler (only with features): seeded offsets, penalty `has_previous_token`, multi-token stop
  words. Greedy without them: nothing.
- `_revert_gen_alloc` is V2-only (no-op for V1).

## Rollback (after forward, before commit)
- generation: `kv_cache_manager.rewind_kv_cache(req, 1 + reserved draft)` (inverse of add_token)
- first-chunk context: `kv_cache_manager.free_resources(req)` (keep seq slot)
- later context chunks: nothing (position not advanced)
- ADP dummy: terminate like PE ~8471
- borrowed perf timing events: release
- sampler side state only if penalties/seeds/stop words are enabled

## Design
1. Record the uncommitted step after `prepare_resources`; clear it right before
   `_update_request_states`.
2. Commit gate (recovery enabled): `sampler_event.synchronize()`, then check a host-side
   "abort requested" flag that `request_abort` sets before touching the device; if set raise
   `StepAborted(HostCollectiveInterrupted)`.
3. Re-raise `HostCollectiveInterrupted` in the broad `except Exception` handlers of
   `_forward_step` (PE ~8424), `_sample_async` (~8562), `_update_requests` (~8591); they
   otherwise fail every request via `_handle_errors` (which itself runs collectives).
4. `_run_event_loop`: roll back the uncommitted step before `_recover_from_peer_loss`.
5. `_require_idle_for_recovery` -> require no uncommitted step (active requests survive).
6. Post-commit window: if `_enqueue_responses`' `tp_gather` is interrupted, requeue responses and
   terminations into the pending-transfer path so they are re-sent after recovery.

## Also required
- Re-arm the all-to-all during Install: abort record, `flag_val`, completion flags/counters of
  each survivor's own workspace, agreed across survivors.
- Requests owned by the dead rank: fail individually (rank 0 tracks request -> ADP rank).
- Check every device collective in the step is covered by cooperative abort (LM-head TP is off).
- Hang detector must not fire while the host waits for the abort to land.
- Manager-neutral rollback hook; V2 later via its revert APIs; reject unsupported configs at init.

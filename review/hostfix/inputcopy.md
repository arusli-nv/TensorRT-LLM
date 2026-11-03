# FT deferred input-copy wait: what it is, what it costs, how to drop it

Code: worktree `/tmp/claude-161238/wt-v5` (branch v5, HEAD e452975950); main = merge base
`0829689c52`. Paths below are under `tensorrt_llm/_torch/pyexecutor/`.

## Verdict up front

The wait exists and only FT adds it, but it is **not a plausible cause of either open cost**.
It is main's existing input-copy event, waited earlier in the same iteration, and it runs
only on iterations whose *previous* batch had context requests on this rank. Its extra cost
is bounded by the GPU time of step N+1's pre-forward stream work (H2D input copies plus
small prep kernels), so about 10-50 us. The cost is an estimate from code structure, not a
measurement; there is no CLAIMS row for it. It can be removed cleanly (section 4), but
(a) and (b) need another culprit (section 6).

## 1. What FT adds (diff of `_executor_loop_overlap` vs main)

The full diff of the loop is four hunks: `step_prepared` (py_executor.py:5843-5849),
`step_released` (:5911-5912), the commit block below, and gating main's
`_update_v2_context_resources` off under FT (:6078-6081). The block in question is
py_executor.py:5994-6004:

```python
if self._peer_loss_recovery.enabled:
    self._peer_loss_recovery.commit_step(prev.scheduled_requests, prev.sample_state)
    if prev.scheduled_requests.context_requests:
        self._wait_for_model_engine_input_copy()        # FT-only wait
        self._update_v2_context_resources(prev.scheduled_requests)
```

Both waits block the host. Neither is a stream wait:
- `commit_step` (peer_loss_recovery.py:681-714) calls `sample_state.sampler_event.synchronize()`,
  which is `cudaEventSynchronize` plus a wait on the worker futures (sampler_features.py:565-570).
  This is **not new**: main's `_update_requests` makes the same call on the same event two
  lines later (sampler.py:1353-1354), so FT only moves that sync ahead of it. The fallback
  `torch.cuda.current_stream().synchronize()` (:704) would serialize the overlap, but it is
  reached only when `sampler_event` is None, which does not happen with TorchSampler or the
  spec sampler (sampler.py:1748, spec_sampler_base.py:546). The abort check `abort.any()`
  (execution_state.py:186) reads a pinned host tensor, so it costs a few us and no sync.
- `_wait_for_model_engine_input_copy` -> `DecoderInputs.wait_for_input_copy`
  (engine/runners/decoder/inputs.py:2190-2192) calls `_prepare_inputs_event.synchronize()`.
  The event is recorded on the current stream right after step N+1's `prepare_inputs`, before
  the model or graph launch (runner.py:650, inputs.py:2185-2188). At :6002 it is therefore
  **E_{N+1}**: the end of the current step's H2D copies.

Main already blocks on that same E_{N+1} every iteration, at :6069 (before
`_update_request_states`) and :6089 (before `_process_previous_batch`), plus :5815 at the top
of the next loop. FT only adds an earlier sync point on an event the host waits for anyway.

## 2. Why FT needs it

- Main runs `_update_v2_context_resources(N)` in iteration N (:6078-6081). That call resizes
  the KV history and commits the chunk's blocks for prefix reuse
  (kv_cache_manager_v2.py:6516-6556, `try_commit_blocks` :5729-5755).
- Under FT, step N can still be discarded after a peer loss. `discard_interrupted_steps`
  (py_executor.py:8678-8714) restores `context_current_position` and calls
  `revert_prepare_resources(step.batch, step.context_capacities)`. Those capacities are
  snapshotted in `step_prepared` (peer_loss_recovery.py:643-658).
- If the step's blocks were already committed, garbage KV from an aborted step becomes
  reusable by other requests, and the history marker no longer matches the reverted capacity.
  So FT defers the update until `commit_step` accepts the step (the comment at :6075-6077).
- The input-copy wait guards a host-buffer hazard, not the rollback. `update_context_resources`
  can rewrite host page-index rows that step N+1's non-blocking H2D copy is still reading from
  pinned memory. Main guards the identical hazard with :6069 and :6089.

## 3. Cost estimate

- **When it fires.** Only when the previous batch on this rank had context requests.
  Pure-decode iterations skip it entirely.
- **Extra blocking per firing.** After `commit_step` returns, sampler(N) has completed.
  E_{N+1} trails sampler(N) on the stream only by step N+1's pre-forward work, which is
  KB-scale pinned H2D copies plus metadata prep kernels: an estimated 10-50 us including
  event wake latency. In main, the host would spend that window on `_update_requests(N)`,
  `_send_kv_async` and the `_sample_async(N+1)` enqueue, then block on E_{N+1} at :6069.
- **Decode, host-bound rank 0 (a).** At 1k/1k with 32 requests/GPU, about 3% of a rank's
  iterations follow a context step. Across 32 ranks gated by the MoE all-to-all, about 60% of
  steps have some rank paying it, which amortizes to about 20 us per step at most. That is
  well under the 0.35 ms gap and does not touch pure-decode steps. These are estimates, not
  measured numbers.
- **Context-heavy with EPLB (b).** It fires on almost every iteration, but adds 10-50 us to
  iterations that take tens of ms, so it is far below 1%. It cannot explain a 15% slowdown.

## 4. Non-blocking design: apply the deferred update at main's existing wait

No new event or stream wait is needed. Commit at the same point, but apply the deferred
context update right after main's own E_{N+1} wait. That wait has to come before
`_update_request_states(N+1)` advances `context_current_position` for a request whose next
chunk is in N+1; otherwise we would commit uncommitted N+1 blocks.

```python
# :5994 block becomes
if self._peer_loss_recovery.enabled:
    self._peer_loss_recovery.commit_step(prev.scheduled_requests, prev.sample_state)
    if prev.scheduled_requests.context_requests:
        self._committed_context_batch = prev.scheduled_requests   # no wait here

def _apply_committed_context_resources(self):        # call only after an E_{N+1} wait
    if self._committed_context_batch is not None:
        self._update_v2_context_resources(self._committed_context_batch)
        self._committed_context_batch = None

# can_queue branch, :6069
self._wait_for_model_engine_input_copy()
self._apply_committed_context_resources()            # before _update_request_states(N+1)
self._update_request_states(scheduled_batch)
# previous-batch branch, :6089 (covers can_queue=False, not can_queue_this_rank)
self._wait_for_model_engine_input_copy()
self._apply_committed_context_resources()            # no-op if already applied
self._process_previous_batch()
```

Also clear `_committed_context_batch` in `discard_interrupted_steps`. It holds an accepted
step, so on an interrupt either apply it (it was accepted) or assert that it is None.
Applying it is correct, because the accepted step is no longer in `_uncommitted_steps`.

Order change to check: the update now runs after `_update_requests(N)` and `_send_kv_async(N)`.
- `_update_requests` appends tokens past `context_current_position`, but the commit slices
  tokens only up to that position, and it terminates nothing.
- Requests are terminated later, in `_process_previous_batch`, which is still after the update.
- `_send_kv_async` on a disaggregated context server could need committed blocks. If FT ever
  supports disaggregated serving, keep the old placement whenever `kv_cache_transceiver` is set.

Unit tests (CPU, fake model engine with a recorded call order, following the existing fakes
in tests/unittest/_torch/executor):
1. With FT on and a context batch at N, `update_context_resources(N)` is called after
   `wait_for_input_copy` and before `_update_request_states(N+1)`, using N's positions.
   Use a chunked request whose chunk N+1 is in flight, and assert the history equals the end
   of chunk N.
2. With `can_queue=False` and `not can_queue_this_rank`, the update is applied at the :6089 path.
3. On a peer loss between commit and apply, the accepted step's blocks are committed exactly
   once, and the discarded N+1 is reverted.
4. With FT off, the call order is identical to main (regression guard).

GPU proof (one packed job, fixed nodes, FT on, 3 repeats, warm restart, standard 1k/1k):
- Cost: compare nsys runs with and without the change. The `cudaEventSynchronize` range
  between `commit_step` and `_update_requests` disappears. Expect about a 0-50 us gap
  reduction on post-context iterations and no change on pure-decode iterations.
- Exactness: run the existing kill-one-rank recovery case with a prefix-reuse dataset and
  chunked prefill. Outputs must be token-identical to a no-kill reference, and no reused block
  may come from a discarded step.

## 5. Red team

- **Recovery race.** `commit_step` runs before the apply, so the accepted batch sits outside
  `_uncommitted_steps` for a few lines. A `HostCollectiveInterrupted` raised in between, for
  example from `_update_requests` or the ADP gathers, enters the discard while the commit has
  not been applied yet. Handle it as in section 4. Without that, block reuse misses the commit
  and capacity is not reverted twice: the result is wasted reuse, not corruption, but it is
  still a bug.
- **CUDA graphs.** The event is recorded outside graph capture and replay (runner.py:650,
  before `_graphs.run`), and the design records no new event, so capture and replay are
  unchanged. The hazard concerns host pinned buffers that `prepare_inputs` copies from. Graph
  static inputs are filled by those copies, so the same E_{N+1} covers them.
- **Overlap previous-batch handling.** `previous_batch` is set to None on discard (:8710), so
  no stale apply happens. With `should_process_previous_batch` False, nothing commits and
  nothing is pending. The :6089 fallback covers the branch where :6069 does not run.
- **Main's semantics.** With FT off, `_committed_context_batch` is never set, so the change
  has zero effect on main's path.

## 6. Where (a) and (b) probably come from instead

- **Per-iteration FT host work on the decode path:** `commit_step` bookkeeping, the heartbeat
  publish, `step_prepared` snapshots, and `_note_step_change`. Profile rank 0's iteration with
  nsys NVTX and compare against FT off.
- **Online EPLB under FT:** `end_iter(interrupted=...)` (moe_load_balancer.py:1876) and the
  MoE abort/watchdog paths inside the forward.
- **Deferral side effect:** under ALL_REUSABLE, the scheduler's duplicate-prefix skip assumes
  blocks are committed after one iteration (scheduler_v2.py:702-735). FT commits one iteration
  later, so duplicates wait one more step. This is relevant only for prefix-sharing workloads.

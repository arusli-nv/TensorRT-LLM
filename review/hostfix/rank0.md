# Rank-0 host cost of FT's serving-loop taps (design, no code changed)

Code: worktree /tmp/claude-161238/wt-v5 (branch v5, e452975950). `plr` = tensorrt_llm/_torch/pyexecutor/peer_loss_recovery.py,
`pe` = .../pyexecutor/py_executor.py, `comm` = tensorrt_llm/_torch/distributed/communicator.py.
Target: ~0.35 ms/step at 32/GPU w/o EPLB (CLAIMS #156, stall-free mean step +2.41%, 11 pairs).

## Bottom line

The taps themselves are cheap. With 32 requests/rank and 1024 in total, each per-step Python loop covers a
rank's own ~32 responses or rank 0's 32 contributions, never the 1024 requests. My estimate for all taps
together is **~40-120 us/step on rank 0**, at most a third of the 0.35 ms. These figures are estimates, not
measurements. The monitor's proposal (skip rank-0 bookkeeping on steps without a final response) is mostly
in the code already: an empty contribution is `()` and is skipped. Taken literally, it is unsafe (section 5).
None of the taps adds a host-GPU serialization (section 2), and the watch thread does not hold the GIL (section 3).
**Measure before coding**: the per-piece bisection exists, but it was only run at 128/GPU, where it did not resolve
(CLAIMS #108: host pieces saved 0.44%, CI -3.3..4.2). Run it at 32/GPU (F0). The fixes F1-F4 are lean and safe,
but together they are unlikely to recover more than ~0.1 ms.

## 1. Per-tap work per decode step (ranked by expected rank-0 cost)

| # | Tap (code) | Rank 0, per decode step | Other ranks | Est. rank-0 cost |
|---|---|---|---|---|
| 1 | `_interruptible` around every MPIDist host collective (comm:117-157; the `coll` piece, not a plr tap) | After each collective: `mpi_failed_ranks(world)` (comm:151 -> _utils.py:635): `Get_failed()` makes an MPI Group object (ULFM intersects the comm group with the failed-proc list), then `Get_size`, then `Free`. Also two attribute writes. About 5-7 collectives/step (tp_allgather of rank states, request broadcast, 2x tp_gather of responses at pe:9318 via pe:9598 and pe:1432, tp_allgather_int64 at decoder/inputs.py:479, ...) | same | ~3-8 us each, **~20-50 us** |
| 2 | `gathering_responses` (plr:787-810, called at pe:9316) on each of 2 gathers | `list(responses)` copy; a loop over this rank's ~32 responses (streaming) checking error/is_final and set membership in `_answering`; deque append to `_gathers`; returns a 4-tuple, so pickling adds 3 short lists | same, on its own 32 | ~10-20 us |
| 3 | `responses_gathered` (plr:812-838, pe:9358) | For each of 32 contributions: an isinstance check, then `_note_responses(finished, streaming, progressed)` (plr:840-851). That is ~32 Python calls/gather with mostly empty lists. Also `_retire` per finished request (dict pop, 3 set discards). The executor's unwrap loop (pe:9327-9334) also adds per-contribution isinstance/len checks | only the counter update | ~15-40 us (2 gathers) |
| 4 | `commit_step` (plr:681-716, pe:5994) | `sampler_event.synchronize()` (shared with FT off, section 2); `peer_failed` (Event.is_set plus a nanobind read); `moe_execution_state.aborted()` = `bool(self.abort.any())`, a **torch op on a 2-int pinned CPU tensor** (execution_state.py:180-186, ~5-10 us of dispatcher overhead); popleft; `_note_progress` only after a recovery; native heartbeat publish | same | ~8-15 us |
| 5 | `step_prepared` (plr:643-658, pe:5843) | A list comprehension over context requests (0 in decode), `context_capacities` (a dict over 0 context requests, kv_cache_manager_v2.py:4316), a `PreparedStep` dataclass, a deque append | same | ~3-5 us |
| 6 | `fetching_requests` / `broadcast_requests` / `requests_routed` (plr:729-781; pe:6586, 6638, 6896) | clear(); a try/except wrapper; on rank 0 a loop over **new** items only (owner dict insert and cancel check); `split_unservable_requests` (adp_router.py:263) also loops over new items only | `_rerouting` is empty when healthy, so the early exit applies | ~2-5 us, plus ~1 us per new request |
| 7 | `step_released`, `step_outputs_appending` (plr:660-679) | released steps only; an early return unless logits/additional outputs were requested | same | ~0 |
| 8 | `_note_step_change` (pe:8716) | only for context requests and requests that complete next iteration | same | ~0 |

Only rank 0's cost matters. The other ranks have less host work and wait for rank 0 in the step's
collectives, so their FT work stays off the critical path. CLAIMS #155 shows FT's collectives within 2-6% of
the ULFM-only arm, which means the transport is not the cost.

## 2. Does commit_step serialize host and GPU? No (one fallback to guard)

- commit_step runs at pe:5994. That is after the **current** step's `_forward_step` (~pe:5986) is enqueued, and it
  waits for the **previous** step's sampler event (plr:703-704). With FT off, `_update_requests` -> `sampler.update_requests`
  waits for the same event a few lines later (pe:6005 -> sampler.py:1353-1354; MTP: spec_sampler_base.py:341). The
  wait just moves earlier and costs nothing extra. Overlap is preserved because the GPU already has step N+1 queued.
- Risk: the fallback `torch.cuda.current_stream().synchronize()` (plr:706) would also wait for the current forward,
  which removes the overlap and costs ~a full step. The `create()` requirement (plr:421-422) only checks that the
  sampler is a generation sampler, not that every `sample_state` carries an event. This cannot explain a uniform
  0.35 ms (it would cost several ms), but serving should count fallback hits: add a testhooks counter
  `WIDEEP_COMMIT_FALLBACK n=` and expect 0.
- Context steps only: `_wait_for_model_engine_input_copy()` + `_update_v2_context_resources` at commit (pe:5996-6003)
  add one wait per step whose previous batch had context requests. That step is not in the stall-free decode metric.

## 3. Peer-failure watch thread and GIL

- The watch is **native** (cpp/tensorrt_llm/nanobind/runtime/peerFailureWatch.cpp `watch()`): `MPI_Test` on a posted Irecv,
  then `condition_variable::wait_for(5 ms)` (comm:850). It takes the GIL only in `report()` on a failure. Python reads
  `failed` (an atomic). It never contends for the GIL when healthy.
- What remains: 200 `MPI_Test`/s under `MPI_THREAD_MULTIPLE` call `opal_progress` and take the PML/BTL locks
  (ob1/TCP), so a test can briefly delay the serving thread's progress inside a collective. The expected cost is
  ~us/step. The `watch` bisection piece and `TLLM_TEST_WATCH_POLL_S` measure it.
- HangEvidenceReporter (hang_evidence.py:291-300) is a Python thread that wakes every `heartbeat_s` (1 s). At most it
  forces one GIL switch-interval handoff per second, which is negligible. The native heartbeat publish is a store.
- Not FT-specific: the ULFM detector thread runs in the ulfm-only arm too (CLAIMS #98, #156: ulfm-only at 32 is -0.1%).

## 4. Lean fixes (all keep exact-token continuation, response history, owner bookkeeping)

**F0, measure (no product change).** Add cumulative `perf_counter_ns` per tap on rank 0 in testhooks: wrap each of
the 8 taps plus `_interruptible`'s `Get_failed` block, and log `WIDEEP_TAP name= calls= mean_us p99_us` every 1000
steps. Run the bisection at **32/GPU** with the pieces `resp`, `step`, `coll`, `watch` and `coll,resp,step,watch`,
on the same nodes, with mirrored order and n>=3 pairs, on the stall-free mean step. Decide on F1-F4 from these numbers. If
the all-pieces arm still shows >=0.25 ms, the cost is not in the taps. Then look at the device side (rank mask, ghost
slots: CLAIMS #148/#151 left "ft vs lb" unattributed) and at GC (FT keeps one iteration of responses in `_gathers`).

**F1, abort read without torch dispatch** (execution_state.py). Saves ~5-10 us/step.
```python
self.abort = device_visible_host_zeros(2, torch.int32)
self._abort_host = self.abort.numpy()          # same pinned memory, no copy
def aborted(self) -> bool:
    return bool(self._abort_host.any())
```
Guarantee: the read happens after `sampler_event.synchronize()`, so the kernels' writes to mapped pinned memory are
visible. numpy reads the same bytes. The watch thread's native store is seq_cst, as before.
Test: unit test where `request_abort()` and a direct `abort[1]=1` make `aborted()` True, and `rearm()` clears it.
The existing recovery unit tests (fake MoeExecutionState) must still pass.

**F2, rank 0 skips contributions without notes** (plr:833-838). Saves ~32 calls x 2 gathers.
```python
for contribution in contributions:
    if isinstance(contribution, tuple):
        if contribution and (contribution[1] or contribution[2] or contribution[3]):
            self._note_responses(*contribution[1:])
```
Guarantee: `_note_responses` with three empty lists is a no-op, so the result is identical. `_last_processed_gather` still
advances (plr:827) and `_gathers` is untouched. Test: a parametrized unit test that gives `responses_gathered` mixes of
empty-note, finished, first-streaming and progressed contributions and checks equal `_request_owners/_delivered/_interrupted`
against the current implementation.
This is the safe form of the monitor's proposal: skip on "no notes", not on "no final response" (section 5).

**F3, cheaper per-collective failure check** (comm:151). Saves ~2-5 us x 5-7.
```cpp
// nanobind, GIL released: number of failed procs ULFM knows of in comm
int failedCount(int64_t f) { MPI_Group g; MPIX_Comm_get_failed(MPI_Comm_f2c(f), &g);
                             int n; MPI_Group_size(g, &n); MPI_Group_free(&g); return n; }
```
```python
if _failed_count(self._world_comm_f) and (failed := mpi_failed_ranks(self._world_comm())):
    raise HostCollectiveInterrupted(func.__name__, failed)
```
Guarantee: the same ULFM query runs after **every** collective, so a collective that completed with a dead peer's undefined
data is still rejected before its result is used. Only the Python Group object churn goes away. Do not replace it with
the watch's `failed` flag: the watch can lag one poll (5 ms) behind ULFM's local knowledge. Ship F3 only if F0 shows `coll`
>= 20 us. Tests: the existing communicator ULFM tests (failed rank -> HostCollectiveInterrupted). Also an MPI test with 2
processes where one is killed between collectives, so that the next collective's post-check raises.

**F4, micro trims** (only if F0 shows `resp` >= 20 us after F2). In `gathering_responses`, skip `list()` when the input
is already a list. Hoist `LlmResponse`/`answering` lookups to locals (partly done). In `step_prepared`, skip
`context_capacities`/the list comprehension when `batch.context_requests` is empty (pass `{}`, `[]`). Guarantees are
unchanged because the data is identical. Tests: the existing plr unit tests plus a decode-only step_prepared/discard round trip.

**Not proposed:** moving the bookkeeping to a thread (it would race the serving thread's sets and need locks, which costs
more than it saves), and a longer watch poll (it adds detection latency, and the native thread costs little).

## 5. Red team: what breaks at recovery if bookkeeping is skipped or deferred

- **Skip on steps without a final response (monitor's literal proposal).** It loses the first-streaming notes, so
  `_delivered` stays unset for a request that already streamed tokens. After its rank is lost, recovery treats it as
  undelivered and re-runs it from the prompt (`_deliver_responses`, plr:1483, `streamed` at plr:1564; also `_departed_requests` plr:1446), and the client receives
  duplicate tokens. That breaks exact-token continuation. It also loses `progressed`, so `_interrupted` is not cleared
  and a request that made progress is **failed** on the next interruption (plr:332-339). Under non-streaming, notes appear
  only with finals, so the proposal adds nothing there.
- **Defer notes N steps (batching).** A loss inside the window has the same failure modes as above, for up to N steps of
  requests. Rank 0's `_gathers` and the other ranks' contributions are cleared at the next `fetching_requests`
  (plr:735), so the deferred notes would also need their own history: more state, not less.
- **Skip `requests_routed` on rank 0.** A new request has no owner, so losing its rank leaves its client waiting
  forever (no re-run, no error). Never skip it.
- **Skip `gathering_responses` history on non-final steps.** An interrupted gather cannot be redelivered, so the client
  loses tokens that were generated and committed: a gap in the stream.
- **Skip/defer the commit_step sync or abort read.** A step whose all-to-all read a dead peer's partial data is committed,
  and the tokens are silently wrong. The sync is free (section 2); keep it.
- **F2/F3 checks.** F2 is equivalent only while `_note_responses` stays a pure function of its three lists. A later
  edit that adds per-contribution side effects must revisit it, and the unit test from F2 pins this. F3 must keep the full
  `mpi_failed_ranks` on a nonzero count, so that the failed set reported to recovery is unchanged.
- **F1.** The numpy view must be rebuilt if `abort` is ever reallocated (today it is allocated once in `__init__`). Add an
  assert in `rearm()` that `self._abort_host.ctypes.data == self.abort.data_ptr()`.

## 6. Recommended order

1. F0 at 32/GPU (one job, no product change). 2. F1 + F2 (safe, ~20-50 us, unit-tested, a Python overlay suffices).
3. F3 only if `coll` is measurable (C++, needs a build). 4. If the taps total < 0.1 ms in F0, stop optimizing taps
and attribute the rest on the device side or through GC. Report it as "not host bookkeeping".

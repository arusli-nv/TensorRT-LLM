# Per-collective ULFM check in `_interruptible`: cost, safety, fix

Code read: worktree /tmp/claude-161238/wt-v5 (branch v5, e452975950). Paths are relative to it. No runs were made;
every µs figure below is an estimate until `bench_get_failed.py` (same directory) has been run.

## 1. Wrapped collectives in one healthy decode step (overlap loop, ADP, MTP, no disagg/EPLB)

`_executor_loop_overlap` (py_executor.py:5786). Each collective is entered by every rank, so the count is the same
on rank 0 and on the others. Rank 0 differs only as root (B, C) and as gather root (I, J).

| # | Collective (wrapped method) | Call site | Per step | Consumer of the result |
|---|---|---|---|---|
| A | `tp_allgather` rank states (pickled) | adp_router.py:220 via py_executor.py:6821/6829 | 1 | `route_requests`, active totals, `split_unservable_requests`, iter stats (py_executor.py:6833-6912) |
| B | `broadcast_int64` request count | request_utils.py:426 | 1 | whether C runs (request_utils.py:401) |
| C | `broadcast` request payload | request_utils.py:488 | 0/1 (only steps with arrivals) | waiting queue → routing; rank 0 `broadcast_requests` bookkeeping (peer_loss_recovery.py:737) |
| D | `tp_allgather_int64` ADP balance | py_executor.py:7069 (from 7402) | 0/1 (`attention_dp_config.enable_balance`) | context-request hold-back |
| E | `tp_allgather_int64` batch sizes | `_can_queue` py_executor.py:4069 (5865) | 1 | forward or skip this step |
| F | `tp_allgather_int64` graph info (pad) | cuda_graph_runner.py:299 via 1024 (`pad_batch`) | 1 | padded batch size |
| G | `tp_allgather_int64` graph info (key) | cuda_graph_runner.py:299 via 579 (`maybe_get_cuda_graph`) | 1 | graph key, sample tier |
| H | `tp_allgather_int64` token + spec counts | decoder/inputs.py:480 (MTP; else common.py:48) | 1 | `all_rank_num_tokens` for padding and MoE A2A |
| — | **`commit_step(prev)`** | py_executor.py:5995 | — | checks `peer_failed` (watch flag) and the MoE abort record |
| I | `tp_gather` pending transfer responses | py_executor.py:6032 → 1418 → 9320 | 1 | rank 0: client responses + `responses_gathered` (9359) |
| J | `tp_gather` step responses | `_process_previous_batch` 6090 → 9598 → 9320 (else `_enqueue_responses([])` 6094) | 1 | same as I |

Conditional extras: early first token +1 tp_gather (6022/6027); `TLLM_METRICS_ALL_RANKS` +1 (1455);
`enable_lm_head_tp_in_adp` with rejection sampling +1 (decoder/metadata.py:208); resource governor +1-2
(5261/5265); remote-tail ADP replaces E/H. **Typical: 8 per step, up to 10-11.**
Not wrapped: `cp_*`/`pp_*`, `send/recv`, and the recovery-only `*_among_survivors`.

## 2. Cost of `mpi_failed_ranks`

`_interruptible` (communicator.py:117-156) adds `in_host_collective` writes (135/150), a try/finally, and after
every **successful** collective `mpi_failed_ranks(self._world_comm())` (151). `mpi_failed_ranks` (_utils.py:635-649)
does `comm.Get_failed()` (MPIX_Comm_get_failed), then `Get_size()`, then `Free()`. In the healthy case it returns at
the size-0 branch, so `mpi_world_ranks` (Get_group/Translate_ranks, _utils.py:624) never runs. As I read Open MPI 5,
Get_failed takes the failed-procs mutex and intersects the 32-member group with an empty set, which returns a
retained `MPI_GROUP_EMPTY`. It makes no network calls and builds no new group. Estimate: ~1 µs native plus
~1-3 µs Python/mpi4py (one Group object, three calls), so **~2-5 µs per collective, ~20-50 µs per step**. That is
roughly 0.15-0.35% of a ~14.5 ms step, against the measured 0.35 ms (+2.41%), so it is unlikely to be the whole
regression. Measure it: `mpirun --with-ft ulfm -np 32 python3 -I bench_get_failed.py [--native-watch]`. The
script reports get_failed, mpi_failed_ranks, watch_flag, and a 1-int64 Allgather with and without the check, plus
a 10-collective "step". `--native-watch` runs the real 5 ms native watch, so the timings include MPI
progress-lock contention under THREAD_MULTIPLE.

## 3. Safety of deferring or dropping the post-success check

**Key fact.** The post-success check cannot be what makes the code correct. ULFM records a failure some time after
the death: microseconds to milliseconds after PRRTE notices it, longer with the heartbeat detector. A collective that
succeeds and then passes the check can still have had a peer die in it, if the failure is recorded one instruction
later. So every consumer below must already be correct after consuming a successful result from a collective a peer
died in. The check only narrows that window. Also, under `ERRORS_RETURN` a local MPI_SUCCESS means this rank's
buffers are complete. A transport does not complete a partial message. What ULFM leaves undefined are the buffers
of an operation that **raised**, and success is non-uniform across ranks. Garbled payloads surface as
Unpickling/EOF/Overflow errors, which the error path still classifies (communicator.py:137-148). Case by case:

- **A** (rank states). On success every live rank's state is real. Routing to a rank that has just died is what
  recovery already handles: rank 0 records owners in `requests_routed` (peer_loss_recovery.py:765), and recovery
  reruns or fails the departed rank's requests (`_departed_requests`, `_deliver_responses`). `fetching_requests`
  (:729) treats A's success as proof that every rank received the last broadcast, and success alone proves that.
- **B/C** (request broadcast). Root success on rank 0 says nothing about receipt even today. A survivor whose
  receive failed never holds the request. `_readmit_undelivered_requests` (:1602) re-admits it, using the
  survivors' `held` allgather (:1057). Its docstring describes exactly this case. A non-root that succeeded holds
  valid requests. No duplicates: the `held` sets come from served/answered ids.
- **D/E/F/G/H** (scheduling votes). Results feed only this step's device work: the forward is launched after them
  (5986) and accepted only by `commit_step`, which rejects on `peer_failed` or on an aborted A2A record
  (peer_loss_recovery.py:707). If ranks diverge (some raised, some did not), the MoE A2A misses a peer, and the
  native watch's abort word plus the device timeout end it, followed by a discard. Recovery rolls back uncommitted
  steps (`step_prepared` :643), so this is reversible. No crash: shapes come from fixed graphs and buckets.
- **I/J** (response gathers, the only irreversible client-visible effect). Rank 0 delivers only after its gather
  succeeded, which means it received every contribution intact, and it then advances `_last_processed_gather`
  (:827). Recovery broadcasts that counter from rank 0 (:1516), so only unprocessed gathers are redelivered. That
  gives neither loss nor duplication. The tokens are from steps already committed (I/J run after
  `commit_step`, 5995 < 6032), so they are not wrong.
- **Recovery-time wrapped collectives** (:447, :1057, :1077, :1516). These are not hot. Today a peer lost right
  after one of them raises at the check. Without the check it raises at the next collective, or at the new watch's
  `Dup` (:619). **Pre-existing gap:** `watch_peer_failures` (communicator.py:1059) is not wrapped, so a peer that
  fails between :1077 and :1083 makes `Dup` raise a raw `MPI.Exception` past `run()` (:544 catches only
  `HostCollectiveInterrupted`) and the job ends. Today's check narrows this window but does not close it. Fix it
  separately by classifying errors there too.

## 4. Options

(a) **One Get_failed per step at commit_step.** It is redundant: `commit_step` already checks the watch flag and the
abort record. ULFM records failures on every communicator, and the watch's ANY_SOURCE receive fails within 5 ms of
that. It adds ~5 µs and no correctness. Skip it.

(b) **Watch flag per collective instead of Get_failed.** A property read costs ~0.1 µs and keeps the fail-fast
behaviour. Two costs: MPIDist must hold the recovery's watch, which is a new coupling. And during recovery the old
watch has already fired while the new one is not armed until :1083, so recovery's own wrapped collectives (:1057,
:1077, :1516) would raise spuriously unless the flag is detached and re-attached. That is more state for ≤5 ms of
earlier detection.

(c) **Check only before irreversible consumers.** Section 3 shows none need it: rank 0's gather is already
authoritative. Caching Get_failed per step is (a) with extra state.

(d) **Recommended, and the leanest: delete the post-success check.** Keep the error-path classification and
`in_host_collective`. Detection stays with the watch: it sets the flag, the abort word, and `revoke()`
(peer_loss_recovery.py:626-640), so the next collective raises `ERR_REVOKED` and becomes
`HostCollectiveInterrupted` (communicator.py:141-143), and `commit_step` rejects the step.

```python
# communicator.py _interruptible wrapper, after the except block
        finally:
            self.in_host_collective = False
-        failed = mpi_failed_ranks(self._world_comm())
-        if failed:
-            raise HostCollectiveInterrupted(func.__name__, failed)
         return result
```
Update the docstring (122-128). It should say that a successful result is complete; a peer loss is raised by
the next collective (revoked by the watch) or rejected at `commit_step`. Remove the claim that the result counts
only once no failure is known.

**Unit tests** (tests/unittest/_torch/distributed/test_mpi_survivors.py):
1. Invert `test_result_completed_after_a_peer_failure_is_an_interruption` (:346) into
   `test_a_successful_collective_returns_its_result_even_if_a_peer_has_failed`: with `world.failed=(FAILED,)`,
   `tp_allgather_int64` returns rows and `Get_failed` is never called (FakeComm counter).
2. `test_healthy_collectives_make_no_failure_query`: a spy on `mpi_failed_ranks` sees 0 calls across all wrapped methods.
3. Keep :289, :307 and :327. The error path must still raise `HostCollectiveInterrupted`, including a garbled
   payload with a failure reported.
4. Recovery unit (fake executor): rank 0's broadcast succeeds and the survivor's receive fails, then
   `_readmit_undelivered_requests` re-admits once. Rank 0's gather succeeds and then the peer is lost:
   `processed` covers it and nothing is redelivered.
5. GPU/integration (FT kill test, standard recipe): kill a non-zero rank mid-decode. The test must see exact
   survivor tokens, no duplicate or lost responses, and `resumed` logged. Rerun the FT on/off cost with the fix.

## 5. Red team

| Failure mode | Covered by |
|---|---|
| Peer dies inside a collective; this rank succeeds | success means complete data; watch revokes in ≤5 ms; next collective raises; `commit_step` discards the step |
| Peer dies; this rank's collective errors | unchanged error path (137-148), including the 1 s evidence wait |
| Ranks diverge: some raise at k, others proceed to k+1 | k+1 lacks the raisers, then hangs until revoke; same as today (Get_failed was local and non-uniform too) |
| Requests routed to a dead or non-receiving rank | `requests_routed` owners plus `_departed_requests`, `_readmit_undelivered_requests` |
| Responses delivered, then a loss | rank-0 `processed` counter broadcast (:1516); redeliver only the unprocessed |
| Wrong tokens to clients | I/J carry only committed steps; uncommitted steps are discarded by the abort record |
| Watch misses a failure | any MPI error on the watch reports failure (peerFailureWatch.cpp `watch()`); a dead peer still fails the next collective; A2A device timeout |
| Hung, not dead, peer | Get_failed never saw it either; hang detection and the A2A timeout |
| Second loss during recovery | its own collectives raise at the next call; residual raw `MPI.Exception` at the watch `Dup` is pre-existing; fix separately |
| Fix gives <1% back | likely: est. 20-50 µs of 350 µs. Run the bench first. Next suspects: ResponseContribution pickling in I/J on rank 0, the watch thread's MPI_Test under THREAD_MULTIPLE (`--native-watch`), and whether the FT-off baseline also launches `--with-ft ulfm` |

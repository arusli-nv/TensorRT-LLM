# Event-driven peer-failure watch (replace the 5 ms MPI_Test thread)

Code read: worktree /tmp/claude-161238/wt-v5 (branch v5). Open MPI v5.0.x and OpenPMIx v5.0 sources were read from
GitHub (`ompi/errhandler/errhandler.c`, `ompi/instance/instance.c`, `opal/runtime/opal_progress.c`,
`opal/mca/threads/base/wait_sync.c`, `pmix/src/event/pmix_event_notification.c`). I ran nothing. Measured input:
BISECT32-8 job 8146122. Decode-only stall ms/step at 32/GPU: off 0.335, ft 0.568/0.605, ft-noT 0.366/0.461. These
numbers are not in CLAIMS yet; add a row before anyone cites them.

## 1. Why the watch perturbs the serving thread (code, not yet measured)
- Every `MPI_Test` runs `opal_progress()`. TCP (and the OMPI failure events) are progressed by `opal_progress_events()`. That function guards `opal_event_loop(opal_sync_event_base)` with a **try-lock** (`OPAL_THREAD_SWAP_32(&lock,1)`, opal_progress.c:173-199). While the watch thread is inside the event loop, the serving thread's own collectives skip TCP progress and spin. If the watch thread is preempted while it holds that lock (many threads per rank, CPU contention), the serving thread's allgather/gather stalls for a scheduler quantum. That fits "occasional long steps, trimmed means unchanged". It happens 200 times a second at 5 ms.
- How ULFM learns of a death, with the in-band detector off (DECISIONS 20:10, user decision d): PRRTE sends a PMIx event. `ompi_errhandler_callback` runs on the **PMIx progress thread**. It only queues an opal event on `opal_sync_event_base` ("transition this from the RTE thread to the MPI progress engine", errhandler.c:549). The rank is marked failed later, inside some thread's `opal_progress`. The watch thread exists only to supply that progress while the serving thread waits on the GPU.

## 2. Options
**(a) PMIx event handler: RECOMMENDED.** The PMIx client is already initialized in every worker. Open MPI's instance
init calls PMIx_Init, and `pmix_fence.py` already drives job control through the loaded libpmix (RTLD_NOLOAD). ULFM
itself registers for exactly `PMIX_ERR_PROC_TERM_WO_SYNC`, `PMIX_ERR_PROC_ABORTED_BY_SIG` and
`PMIX_ERR_PROC_ABORTED` as one multi-code handler (instance.c:461-472). PMIx runs handlers in this order: first,
single-code, multi-code, default (pmix_event_notification.c:726-811). A **single-code** handler per code therefore runs
before ULFM's handler. It must finish with `cbfunc(PMIX_SUCCESS, ...)`, never `PMIX_EVENT_ACTION_COMPLETE`, so that
the chain continues to ULFM. The handler runs on the existing PMIx thread, which sleeps in epoll and costs nothing on
the healthy path. It stores the abort word directly, a µs-scale atomic store, and never touches MPI or the GIL.
- ULFM's agreement does not need progress from us. revoke, shrink and the per-collective `mpi_failed_ranks` all run `opal_progress` on the thread that calls them, which drains ULFM's queued event then. Shrink is an agreement that tolerates late knowledge. With hang detection, `_shrink`'s `reported_failed_ranks` may miss a death that is only queued. It is used only to kill failed-but-alive ranks, so missing an already dead rank is harmless.
- A hung but live peer gets no event, the same as today. The authority sees silence, fences the rank, and rank 0's `kill_rank` goes through PMIx job control. The kill then produces `ABORTED_BY_SIG` like any death.

**(b) Blocking MPI_Wait in the native thread: REJECTED (worse).** Under THREAD_MULTIPLE, `ompi_sync_wait_mt`
(wait_sync.c:60-138) makes the head of the wait list the progress owner, which spins `opal_progress()` until its
request completes. For the watch request that is forever. Other waiters, including the serving thread's collectives
(`opal_max_thread_in_progress`=1), sleep on a condvar and are woken by the owner. The result is a core at 100%,
permanent event-lock contention, and a condvar wake on every host collective.

**(c) Longer poll interval: fallback knob only.** This keeps the mechanism and only lowers its rate. If stall
incidence scales with poll rate (unverified), the extra ~0.2-0.27 ms/step at 5 ms would be about 0.02 at 50 ms. The
cost is detection: worst case +interval, mean +interval/2, on top of today's. At 50 ms that is +25 ms mean against a
resume of 98-113 ms after the interruption (CLAIMS #1, RESTORE-E2E2-8034134, R1 DEP32, n=1/case), so +22-26%. At 1 s
it is +0.5 s mean.

**(d) Other routes: none usable.**
- ULFM error handlers (`MPI_Comm_set_errhandler`) run synchronously inside an MPI call on the calling thread, so they are blind while the rank waits on the GPU.
- Open MPI 5.0 ob1 has no async progress thread. `btl_tcp_progress_thread` progresses TCP only and does not drain `opal_sync_event_base`, where the failure event waits.
- The threaded in-band detector is deprecated and off (decision d).
- The runtime "errmgr callback" is this same PMIx event, i.e. option (a).
- Authority heartbeats take seconds (heartbeat_timeout_s 5).

## 3. Recommended design: a sticky, process-wide PMIx failure set plus subscriptions
- **One registration per process** at `MPIDist.enable_peer_loss_recovery()`, before its `Dup`. Three single-code handlers, filtered on own nspace (`PMIX_NAMESPACE`, as kill_rank does). Deregistered at atexit, which runs before mpi4py's MPI_Finalize because atexit is LIFO.
- A **sticky failed set** of PMIx ranks: monotonic and level-like. **Watches** subscribe with a member set: `pmix_ranks(comm)` minus self. If a member is already in the set at subscribe time, the watch fires at once. That matches today's "Dup/Irecv on a comm with a failed member fails at once", so no event is lost between watches.
- On a member failure, under the mutex: set `failed`, then store 1 to the abort word (today's order), then queue the watch. A **dispatcher thread** blocked on a condvar (no timeout, no MPI) calls `on_failure` with the GIL. `on_failure` still revokes, from that thread, once.
- What it keeps: the abort word is stored on the PMIx thread µs after the event reaches the client, with no MPI progress and no GIL. Detection latency: today it is PRRTE notify + ≤5 ms poll + up to one opal event tick (the default rate limit is 10 ms unless `num_event_users`>0) + MPI_Test. The new design is PRRTE notify + µs. For reference, kill detection today is 0.01-0.09 s (DECISIONS 19:55, from CLAIMS 57/119/121) and "0.5-9 ms" (CLAIMS #57, FAIL-1-7993388, dated).
- What still catches a death the event misses:
  1. The serving thread's next host collective. Its own progress drains ULFM, and it raises `HostCollectiveInterrupted`.
  2. If every rank is parked in an a2a kernel, hang detection: silent heartbeats, then fence, then the hard-kill path, ~35 s per decision d. The only loss compared with today is in-band TCP detection by the watch thread's progress. ULFM uses the same three PMIx events, so any death ULFM sees without in-band traffic is also seen by us.
- Support gate: require the PRRTE runtime (`mpirun --with-ft ulfm`, already demanded by the `mpi_ft_enable` check). Under srun's Slurm PMIx server, process-failure events are not delivered (to verify), so refuse there: check `PMIx_Get(PMIX_RM_NAME)` == PRRTE (to verify on the first run). A registration failure raises RuntimeError, so FT is refused, never silently blind.

### Code sketch (replaces most of peerFailureWatch.cpp, ~same size; Python ~30 lines)
```cpp
// Types from pmix_common.h (CMake: find_path(pmix.h) next to mpi.h; absent -> binding not built -> FT refused).
// Functions via dlopen(libpmixPath, RTLD_NOLOAD) + dlsym, so the bindings never get DT_NEEDED on libpmix.
class FailureEvents { // process-wide, static storage: a late PMIx callback never touches freed memory
    std::mutex mMutex; std::condition_variable mWake; std::set<uint32_t> mFailed; std::vector<Watch*> mWatches;
    std::deque<Watch*> mDue; std::thread mDispatcher; bool mStopping{false}; char mNspace[PMIX_MAX_NSLEN + 1];
public:
    void onFailed(uint32_t rank) { std::lock_guard l(mMutex);
        if (!mFailed.insert(rank).second) return;            // ABORTED_BY_SIG then TERM_WO_SYNC: once
        for (Watch* w : mWatches) if (w->members.count(rank)) fire(w); mWake.notify_one(); }
    void fire(Watch* w) { if (w->failed.exchange(true)) return;   // caller holds mMutex
        if (w->abort) __atomic_store_n(w->abort, 1, __ATOMIC_SEQ_CST); mDue.push_back(w); }
    void subscribe(Watch* w)  { std::lock_guard l(mMutex); mWatches.push_back(w);
        for (uint32_t r : mFailed) if (w->members.count(r)) { fire(w); mWake.notify_one(); break; } }
    void unsubscribe(Watch* w); // erase from mWatches/mDue; unless on the dispatcher thread, wait (GIL released)
                                // while w->inCallback, so on_failure never runs after stop() returns
    void dispatch();            // loop: wait mWake; pop w; inCallback=true; unlock; GIL; copy+call onFailure; ...
};
static void handler(size_t, pmix_status_t, const pmix_proc_t*, pmix_info_t info[], size_t n, pmix_info_t*, size_t,
                    pmix_event_notification_cbfunc_fn_t cbfunc, void* cbdata) {
    for (size_t i = 0; i < n; ++i)                            // same parse as ompi_errhandler_event_cb
        if (info[i].value.type == PMIX_PROC && !strncmp(info[i].value.data.proc->nspace, gEvents.nspace(), PMIX_MAX_NSLEN))
            gEvents.onFailed(info[i].value.data.proc->rank);
    if (cbfunc) cbfunc(PMIX_SUCCESS, nullptr, 0, nullptr, nullptr, cbdata);  // continue to ULFM's handler
}
// start(libpmixPath, nspace): register each code alone (single-code category), wait for each registration callback;
// start the dispatcher. stop(): deregister, wait, join. Bindings: start_peer_failure_events, stop_...,
// PeerFailureWatch(members: list[int], abort_word, on_failure) with .failed and .stop().
```
```python
# communicator.py: enable_peer_loss_recovery(): after the checks, before Dup:
#   start_peer_failure_events(_loaded_libpmix_path(), os.environ["PMIX_NAMESPACE"]); atexit.register(stop_...)
def watch_peer_failures(self, on_failure, abort_word=0):        # no longer collective, no Dup, no comm to free
    return PeerFailureWatch(self._members(self._world_comm()), on_failure, abort_word)
def watch_survivor_failures(self, on_failure):
    return PeerFailureWatch(self._members(self._shrunk[0]), on_failure)
def _members(self, comm): return [r for r in pmix_ranks(comm) if r != pmix_ranks(MPI.COMM_SELF)[0]]
# PeerFailureWatch.stop(): native stop only. Drop _PEER_FAILURE_POLL_INTERVAL_S, and in peer_loss_recovery
# _watch_survivors drop the is_mpi_peer_failure except (subscription cannot raise MPI errors; sticky set fires).
```
`_libpmix_path()` is factored out of `pmix_fence._loaded_libpmix` (the /proc/self/maps scan).

## 4. Tests
**CPU unit (fakes, no MPI):** a gtest (`add_tllm_gtest`, LABELS common) on `FailureEvents` with `onFailed` injected:
- a member failure fires once, `failed` is set before the abort word, and the callback runs on the dispatcher;
- a non-member or foreign-nspace failure is ignored (handler parse test with built `pmix_info_t`); a duplicate code fires once;
- **sticky**: a failure before subscribe fires at subscribe;
- after stop() returns there is no callback; stop() from inside the callback returns; destroying from inside the callback works; stop() races an in-flight callback (TSAN).

Python fakes: `watch_peer_failures` passes the current world's PMIx ranks minus self, and the survivor watch the shrunk
comm's. `enable_peer_loss_recovery` raises when events cannot start (no libpmix, RM not PRRTE). The existing
`test_peer_loss_recovery.py` mocks keep their signatures.

**CPU integration (decisive for the mechanism; `test_mpi_survivors_ulfm.py`, `mpirun --with-ft ulfm`, no GPU):**
survivors wait in `failure_seen.wait()` with **no MPI calls**; assert the event arrives (log the latency; bound
≤1 s). Cover these victims: `os._exit` (TERM_WO_SYNC), SIGKILL (ABORTED_BY_SIG), SIGSTOP plus `kill_rank`. Also: a
death before the survivor watch subscribes (sticky), the existing second victim, and a clean stop that never fires.
Assert that after the event, `shrink()` agrees on the victim, which proves ULFM's chain handler still ran.

**GPU (one job, frozen build, same nodes):**
- BISECT pattern: arms off | ft | ft-ev on the same 8 nodes, 32/GPU, 2 rounds with rotated order, plus 8/GPU 10 pairs. Pass: ft-ev decode-only rank-0 stall ms/step within noise of ft-noT (0.37-0.46), stall-free means unchanged.
- Kill matrix, logging per-rank event-receipt time against the harness kill timestamp:
  - rank SIGKILL in decode: detect and resume ≤ today (CLAIMS #1/#44);
  - whole-node kill (CLAIMS #8);
  - SIGSTOP with hang detection: ~35 s per decision d, recovered;
  - second loss during recovery (REPLAN, CLAIMS #72);
  - loss during startup: the job ends via the startup watch;
  - clean shutdown: no fire.

## 5. Red team
- **Startup race:** registration precedes enable's `Dup`. A death before that fails the Dup collective, and the job ends (no recovery before serving, as today). The startup and serving watches overlap (`_run` arms first), and the sticky set closes any gap anyway.
- **Stop race:** unsubscribe holds the mutex the PMIx thread holds while storing the abort word, so the abort memory can be freed after stop(). The dispatcher never holds the mutex while taking the GIL. stop() releases the GIL. Static storage plus a `mStopping` flag makes callbacks after deregistration no-ops.
- **Second loss during recovery:** the survivor watch's members exclude the agreed lost ranks. A late duplicate event for an already-lost rank is not a member, so there is no false `further_loss`. A survivor that died between shrink and subscribe is in the sticky set, so the watch fires at subscribe and triggers a replan.
- **Event before recovery is armed:** sticky, so it fires on subscribe. This includes a member that ULFM's shrink missed: today that would hang until the next collective; now it triggers `_shrink_again`.
- **PMIx thread under load:** our handler costs O(watches) µs and never blocks, so no GIL or deadlock risk with `kill_rank` (PMIx_Job_control waits on that thread). Delivery latency through busy PRRTE daemons (IOF of heavy logs) is shared with ULFM today. The GPU job's event-receipt timestamps measure it.
- **srun / Slurm PMIx:** failure events are not expected (to verify), so the gate refuses. Launch through the wrappers stays under exec (trtllm-llmapi-launch:302-315), so PRRTE sees the worker's own death. The SM-DIAG2 lesson (a shell wrapper hid deaths) becomes a hard requirement: assert in the integration test.
- **Node loss with its PRRTE daemon:** the HNP sees the daemon's OOB connection drop and declares that node's procs failed. Today's watch depends on the same thing. Power-off or partition (no RST) is slow either way, and the backstop is the hang path (unmeasured, as today). If the lost node hosts the HNP, rank 0 is lost too, which was never recoverable. If this rank's own daemon dies, `PMIX_ERR_LOST_CONNECTION` goes to Open MPI's default handler, which aborts this process; the peers see a node loss.
- **False alarm:** an event without a PMIx proc in our nspace is logged and ignored. Firing on it would run a recovery with an empty lost set, which `_check_recoverable` turns into a job end.
- **Residual risk:** a death that produces no PRRTE event, while every survivor is parked in an a2a kernel, is caught only by hang detection, ~35 s. Today in-band TCP progress might catch it sooner. If the integration test or kill matrix shows such a case, add a slow `MPI_Test` backstop (≥1 s, a config field) rather than restore the 5 ms poll.

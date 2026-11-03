# Red team: event-driven peer-failure watch (watch.md)

Reviewer: independent subagent, 2026-10-22 00:10 PT, read-only. Installed stack (RECIPE.txt, prte_info): Open MPI
v5.0.10, OpenPMIx 5.0.10, PRRTE 3.0.13. ompi-ft-ucx holds only the install, so I cloned exactly those tags from
GitHub and read them. Worktree /tmp/claude-161238/wt-v5 @ f96a8c1a53. I ran no MPI or GPU job: every verdict
below comes from reading code, none from a measurement.
## Verdict: SOUND WITH CHANGES. Two load-bearing claims are false (the kill_rank event, node loss).
The core works: a single-code PMIx handler that stores the abort word, then returns PMIX_SUCCESS. But it covers the
wrong code set. The stall mechanism (C1) is plausible, not shown. A smaller hybrid (C6) does the same job.
## C1. Try-lock as the stall mechanism: code CONFIRMED, mechanism PLAUSIBLE, NOT SHOWN
- opal_progress.c:170-200 guards `opal_event_loop(opal_sync_event_base)` with `OPAL_THREAD_SWAP_32`. A thread
  that loses the try-lock skips the loop.
- TCP runs on that base by default (btl_tcp_component.c:1159-1161). Each connected endpoint increments
  num_event_users (btl_tcp_endpoint.c:488/832/926). So across nodes every opal_progress call, on both threads,
  tries the lock, and the 10 ms tick does not apply. This corrects watch.md 3 ("up to one opal event tick"): that
  tick holds only for single-node sm jobs.
- Weak point: the lock holder is itself progressing TCP, including the serving thread's ob1 callbacks. Skipping
  costs the serving thread time only if the holder is descheduled inside the loop.
- Alternatives not ruled out: (a) the 200 Hz wakeups preempting the serving thread on a crowded core set;
  (b) THREAD_MULTIPLE ob1 and request locks taken in MPI_Test.
- ft vs ft-noT implicates the thread, not which part of it. Run T2 before citing C1.
## C2. Handler order and chain continuation: CONFIRMED
- pmix_event_notification.c:726-850 matches in order: first, single, multi, default, last. cycle_events
  (432-520) continues from the single list into the multi list. It stops only on ACTION_COMPLETE, LAST_OVERALL
  or endchain.
- ULFM registers one multi-code handler with PMIX_EVENT_HDLR_PREPEND (instance.c:461-472). PREPEND orders it only
  inside the multi list, so a single-code handler runs first. ULFM then returns ACTION_COMPLETE
  (errhandler.c:575). Calling cbfunc inline is safe: progress_local_event_hdlr THREADSHIFTs (line 668).
- Parse bug: PRRTE sends two PMIX_PROC infos, AFFECTED_PROC and CUSTOM_RANGE={nspace, WILDCARD}
  (errmgr_dvm.c:648-649). The sketch's "any PMIX_PROC in my nspace" check records rank 0xFFFFFFFE.
- Fix: match the PMIX_EVENT_AFFECTED_PROC key and require rank < PMIX_RANK_VALID.
## C3. ULFM still agrees on the failed set: CONFIRMED in principle (ERA agreement code not read)
ULFM's handler only queues `ompi_errhandler_event_cb` on opal_sync_event_base (errhandler.c:571-573).
- The failure is recorded (failed-procs group, wait_sync wakeup) when some thread runs that base.
- Revoke and shrink run opal_progress, so they drain it. Our earlier handler does not consume the event.
- "Shrink tolerates late local knowledge" is inferred, not read. Keep the shrink assertion in the tests.
## C4. Delivery per death type: PARTLY REFUTED
PRRTE notifies only for RECOVERABLE + NOTIFY_ERRORS jobs; `--with-ft ulfm` sets both (schizo_ompi.c:1067-1076).
The state comes from waitpid (odls_base_default_fns.c:1739-1860):
- SIGKILL or another fatal signal -> PMIX_ERR_PROC_ABORTED_BY_SIG. Covered.
- os._exit(0) -> TERM_WO_SYNC. Covered.
- os._exit(n!=0), or any non-zero exit -> TERM_NON_ZERO (error_non_zero_exit defaults to true,
  state_base_frame.c:82) -> PMIX_ERR_EXIT_NONZERO_TERM (errmgr_dvm.c:545-546). NOT covered.
  - It goes to MPI-Default. The default branch of ompi_errhandler_event_cb calls errors_are_fatal
    (errhandler.c:507-525): by the source, the SURVIVOR ABORTS.
- kill_rank (JOB_CTRL_KILL) -> terminate_procs (pmix_server_gen.c:1056) -> odls SIGKILL with the state forced to
  KILLED_BY_CMD (odls 2089, kept at 1739) -> PMIX_ERR_PROC_KILLED_BY_CMD (errmgr_dvm.c:394-407).
  - So watch.md's "the kill then produces ABORTED_BY_SIG" is FALSE. With 3 codes, a fenced hung rank never fires
    the watch.
  - By the source, MPI-Default would also abort survivors. CLAIMS #102 (survivors resumed after the runtime ended
    rank 2) contradicts that, or that run never exercised the path. UNVERIFIED: test first.
- Node loss including its prted: REFUTED. errmgr_dvm.c:236-343 handles a daemon COMM_FAILED that was not ordered
  as "Comm failure: daemon - aborting": prte_abnormal_term_ordered, then DAEMONS_TERMINATED. The whole DVM ends.
  - PRRTE 3.0.13 has no per-node proc-failure notification. Today's design is the same here (DECISIONS: node loss
    fails closed).
  - Strike watch.md 5 "HNP ... declares that node's procs failed". CLAIMS #8/#99 were process-level node kills,
    with prted alive.
- SIGSTOP: CONFIRMED no event. prte_wait.c:241 calls waitpid(-1, WNOHANG) without WUNTRACED, so only hang
  detection (~35 s, decision d) catches it.
- Promptness: an HNP xcast over the daemon tree, then each prted's PMIx server. That is the same path ULFM uses
  today. Unmeasured: log receipt times.
## C5. PMIx thread vs kill_rank: SAFE if three rules hold (design argument, untested)
PMIx_Job_control blocks its caller (ctypes releases the GIL) until a reply arrives on the PMIx thread, which also
runs our handler. It is deadlock-free if:
1. The handler never blocks, never takes the GIL, and never calls a blocking PMIx API. The sketch already does this.
2. No thread holds mMutex while it calls PMIx_Register/Deregister (both blocking) or MPI. stop() must unlock
   before it deregisters.
3. Nothing deregisters from inside a handler.
Put 2 and 3 in code comments and cover them with a TSAN test.
## C6. Simpler correct option: YES, a hybrid (recommended)
- "ULFM handler + opal_progress at known points" alone is insufficient. ULFM never stores our abort word, and a
  rank parked in an a2a kernel makes no MPI calls.
- Hybrid: keep PeerFailureWatch and its semantics (a private-comm Irecv; on_failure fires only once ULFM knows).
  - Add one process-wide single-code PMIx handler. It stores the abort word and signals the watch condvar, so
    MPI_Test runs at once.
  - Raise the poll period to >=1 s, as the backstop for deaths that produce no event.
  - Result: the abort word is stored µs after the event, ULFM agrees before we act, and there is no sticky set,
    subscription list or dispatcher. That is ~200x fewer MPI_Test calls on the healthy path.
  - Cost: the handler needs the job's already-lost ranks, so a late duplicate event does not abort a healthy step.
## Required changes (either design)
1. Codes. Add KILLED_BY_CMD, EXIT_NONZERO_TERM and REQUESTED_ABORT, or prove by T1 that they never arrive.
   - If they reach MPI-Default and abort survivors: end the chain for them with our own single-code
     ACTION_COMPLETE handler, then show that shrink still agrees with only in-band knowledge.
2. Parse AFFECTED_PROC only, and require rank < PMIX_RANK_VALID.
3. Correct watch.md: node loss with its prted ends the job; kill_rank yields KILLED_BY_CMD.
4. Follow the C5 lock rules in stop() and in deregistration.
5. Do not cite C1 as the cause until T2.
## Minimum falsifying tests (CPU only, minutes)
T1 (C2/C3/C4) setup: `mpirun --with-ft ulfm --mca pml ob1 --mca btl self,tcp -np 4`, on 2 nodes if available.
- Every rank uses ctypes on the loaded libpmix to register single-code handlers for TERM_WO_SYNC, ABORTED_BY_SIG,
  ABORTED, KILLED_BY_CMD, EXIT_NONZERO_TERM and REQUESTED_ABORT.
- Each handler logs (code, info keys, ranks, time) and returns PMIX_SUCCESS. Survivors sleep with no MPI calls.
- Victims, one per run: SIGKILL, os._exit(0), os._exit(1), and SIGSTOP + kill_rank from rank 0.
- After 3 s, survivors run MPIX_Comm_shrink and print the result.
- Falsified if a case logs no code, logs a code outside the design's 3, or a survivor aborts. C3 is falsified if
  shrink keeps the victim.
T2 (C1) setup: 2 ranks on 2 nodes over TCP, the main thread running 1e5 small Allgathers. A second thread runs
(a) nothing, (b) a 5 ms wake only, or (c) a 5 ms wake + MPI_Test.
- Run each pinned to 1 core and unpinned, and compare the max and p99.9 Allgather latency.
- If (b) ≈ (c), the try-lock is not the mechanism.

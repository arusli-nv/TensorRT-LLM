# ULFM MPI_Finalize stall, verbose run SHUTDOWN-R1-8139740 (OMPI_MCA_mpi_ft_verbose=40)

Data: mvp/runs/SHUTDOWN-R1-8139740/{ft-clean,ft-loss,noft-clean}, logs/bundle/8139740-8.log.
Source reference: open-mpi/ompi v5.0.x ompi/runtime/ompi_mpi_finalize.c and
ompi/communicator/ft/comm_ft_detector.c (fetched from GitHub; not local).

## Open MPI v5 finalize order with FT enabled
shrink_internal(COMM_WORLD) -> doublering barrier -> detector_finalize -> propagator/revoke/rbcast
finalize -> "Rank N: DONE WITH FINALIZE" (verbose 40) -> PMIx_Fence_nb (all ranks) -> instance finalize.

detector_finalize: send a quit to the observer with `from = hb_observer` (the source comment: "comm->c_my_rank;
except during finalize when it is equal to detector->hb_observer"), so the receiver sees its OWN rank.
fd_heartbeat_recv_cb treats `msg->from == my rank` as "a quit message from our observed process" and sets
hb_observing = PROC_NULL. So "Received heartbeat from N, which is my own rank" IS the neighbour's quit, by
design; not a ring wiring problem. After the wait it sets fd_thread_active = 0, calls
opal_event_base_loopbreak(fd_event_base) and opal_thread_join(fd_thread). The thread loop is
`while (fd_thread_active) { opal_event_loop(fd_event_base, ONCE); ompi_request_test(...) }`; on exit it
cancels its pending receive ("Recv_request_cancel ... has not matched"). hb_period is already INFINITY.

## ft-clean (32 ranks)
- GRACEFUL_STOP 1792633515.42 (18:45:15). Rank 0 teardown ends 18:45:29 (shutdown_abort: workers exited
  cleanly), i.e. ~14 s, same as FT off.
- MPI_Finalize entered ~529.5-530.6 (= RANK_EXIT - 30 s; the give-up timer starts right before it).
- 32/32 ranks log the quit ("own rank") line: every detector wait for the neighbour ended.
- 16/32 ranks then log the detector thread's receive cancel and DONE WITH FINALIZE (3,5,7,9,11,13,14,15,17,19,
  21,23,25,27,29,31). The other 16 log nothing after the quit: stuck between the quit and the thread's cancel,
  i.e. in opal_thread_join of the detector thread.
- 32/32 hit "MPI_Finalize (process end) did not end within 30 s". RANK_EXIT 559.5-560.6, MPIRUN_EXIT 561.1.
- Extra ~30 s = the give-up timer, exactly. The DONE ranks wait in the PMIx fence for the 16 that never
  leave detector_finalize; nobody completes.

## ft-loss (31 survivors after rank 5 SIGKILL)
- Stop 356.35; Finalize ("degraded world") entered ~360.7-368.0; 31/31 quit lines, 16/31 DONE, 31/31 give up
  at 30 s; exits 390.7-398.0 (rank 30's RANK_EXIT not recorded, its give-up line is). Same stall.
- mpirun rc 137 (here and PT-D 8138133): rank 5's SIGKILL status. mpirun reports the first abnormally
  ending process; it exited at 398.5 before rank 0's wrapper (399.2). Independent of the finalize stall.

## noft-clean
Stop 075.31 -> RANK_EXIT 090.9-091.9 -> mpirun 092.3 (rc 143), 17.0 s. No FT block, no stall.

## PT-D 8138133 (3 s grace, no verbose)
ft-clean 32/32 and ft-loss 30 give-ups at 3 s: the same stall, 2/2 runs.

## Verdict
Open MPI ULFM bug, threaded detector shutdown: lost wakeup. loopbreak lands while the thread is not inside
the event loop (libevent clears the break flag on loop entry) or the thread is blocked in
opal_event_loop(ONCE) with no events left (hb_period INFINITY, observer stopped), so the join never returns.
~50% of ranks per run. No misuse evidence: all ranks passed shrink + barrier + quit exchange.
Not proven by a stack yet: a native stack of a stuck rank (eu-stack/gdb at give-up) showing
opal_thread_join in ompi_comm_failure_detector_finalize would confirm it.

Fix candidates: (1) upstream: wake the thread reliably (e.g. activate an event or loop with a timeout).
(2) ours: keep the bounded give-up, derived from the detector timeout (8138133's 3 s gave ft-clean 16.6 s
vs noft 17.0 s). Safe because all state is flushed by the barrier before the stall.
Not mpi_ft_detector_thread=false: that drops detection while ranks sit in GPU work.
mpirun 137 after a loss needs separate launcher/harness handling.

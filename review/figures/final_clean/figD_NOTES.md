# figD_failure_detection_final: verified facts

Code: v5-final `27442bd91d` (/tmp/claude-161238/wt-split). Drawn by `failure_detection()` in fig_design_final.py
(backup before the change: diag/fig_design_final.pre-figD.py). Other figures regenerate byte-identical.

## Crash path
- The watch is a native C++ thread (`peer_fail_watch`). It posts an `MPI_Irecv` that no peer matches and polls it with
  `MPI_Test`. ULFM fails that receive once a peer fails. Sources: peerFailureWatch.cpp:47-52, 66, 144-163.
- The poll interval is `peer_failure_poll_s`, default 0.02 s (20 ms): config.py:67-72 and peerFailureWatch.cpp:162.
- On a failure the thread sets `failed`, then stores 1 to the abort word (`__atomic_store_n`), then calls
  `on_failure`, in that order: peerFailureWatch.cpp:166-181 and communicator.py:851-863.
- The abort word is `MoeExecutionState.abort`, an `int32[2]` record `(requested, status)`.
  - It is allocated by `device_visible_host_zeros`, which returns `torch.zeros(...).pin_memory()`, so it lives in
    pinned host memory and kernels read it at its mapped address: execution_state.py:33-36, 49 and _utils.py:1445-1447.
  - The watch gets its address at peer_loss_recovery.py:636-643.
  - `MoeA2AAbortState` is defined at moeAlltoAllKernels.h:64-76.
- The kernels poll the abort flag inside their spin loops, but only on long waits. `wait_round_flag` spins on the peer's
  round flag; once a wait has lasted `kAbortPollCycles` (100k cycles, about 50 us at an assumed 2 GHz), it reads
  `abort->requested` and repeats that read every period. On an abort it records `status = kCancelled` and returns
  false. Sources: moeAlltoAllKernels.h:59-62 and moeAlltoAllKernels.cu:270-320.
  - Dispatch: the call is at .cu:737-742 and sets `flag_val[kReleasedRoundIndex]`.
  - Combine: the call is at .cu:1822-1835. An aborted combine writes zeros (.cu:1857).
  - The abort record is used only on the rank-mask fence path, not CFT: .cu:1326-1327 and 2285-2286.
- `on_failure` is `PeerLossRecovery._on_peer_failure`, the recovery hook. It sets `peer_failed`, revokes the host
  communicators, which releases survivors blocked in host collectives, and wakes rank 0's request queue:
  peer_loss_recovery.py:645-663.

## Hang path (off by default)
- `hang_detection` defaults to **False**: config.py:73-77. The figure labels the band "opt-in".
- Defaults:
  - `hang_suspect_s` = 2.0 (config.py:78-80);
  - `hang_confirm_s` = 30.0 (config.py:81-85);
  - `heartbeat_s` = 1.0 (config.py:86-90);
  - `heartbeat_timeout_s` = 5.0 (config.py:91-94).
- Each worker's reporter thread polls every `heartbeat_s`. Each poll sends a heartbeat (iteration, waiting, host,
  pid) and reports a peer once that peer has lagged the device's all-to-all round for at least `hang_suspect_s`:
  hang_evidence.py:310-348.
  - A native heartbeat is also sent every `heartbeat_s` over authenticated UDP, not ZeroMQ:
    hang_evidence.py:160-178 and peer_loss_recovery.py:631.
- The authority's `HangMonitor` suspects rank r when every other rank with a fresh heartbeat waits on it and r makes
  no progress (its iteration does not rise). Source: hang_monitor.py:46-61 and 172-193.
  - A peer waits on r if its device waits on r, or if the peer waits on the host while r is not waiting or has no
    heartbeat for `heartbeat_timeout_s`: hang_monitor.py:203-213.
  - So "no heartbeat for 5 s" counts only together with peers blocked on the host. It is not a standalone trigger.
- The evidence must hold for `hang_confirm_s`. Any progress restarts the window: hang_monitor.py:183-185.
- Fencing (authority.py:338-359): the authority does not kill the process itself. It sends `Fence` over ZeroMQ to
  three places:
  - the hung rank's own evidence thread, which SIGKILLs its own process (hang_evidence.py:269-275);
  - two live helper ranks, which ask the MPI runtime (PMIx) to kill the rank after `RUNTIME_FENCE_GRACE_S` = 1 s
    (hang_evidence.py:276-290 and config.py:27-31);
  - the optional `--fence-command`, run as a last resort if no recovery has started after `COMMAND_FENCE_GRACE_S`
    = 5 s (authority.py:286-298, 785-826).
- The killed process then recovers like any other loss (authority.py:128-133). This is why the figure's "Fence" box is
  green (done by the ranks) and the arrow into it is a dashed ZeroMQ message.

## Survivors next
- Order inside `_recover` (peer_loss_recovery.py:929-1117):
  1. `request_abort`, stop the watch, revoke (952-958);
  2. **shrink** (962), the ULFM agreement on the failed set; with hang detection, the rank first has the MPI runtime
     end the ranks it sees failed (1199-1219);
  3. join device work and discard uncommitted steps;
  4. send `Stopped` (1008);
  5. wait for `Propose` (1023);
  6. remap workspaces;
  7. send `Ready`, receive `Install`;
  8. `install_recovery_placement`;
  9. send `Installed`, receive `Commit`.
- **Shrink happens before the proposal.** The authority proposes only after every survivor's `Stopped`. With
  `fence_departed` (always set by the deployment, authority.py:865), it first has rank 0 end every failed rank through
  the MPI runtime and waits for `Fenced`: authority.py:488-499 and 501-521. The figure leaves out this fence before
  Propose.
- The mask is written in place in two steps, so CUDA graphs keep their addresses:
  1. During the remap, after Propose, `release_workspaces` clears the departed ranks' bits in the device mask:
     execution_state.py:113-125 and peer_loss_recovery.py:1403.
  2. On Install, `install_recovery_placement` updates routing, marks the ranks failed, `rearm`s (clears the abort
     record), then `install_mask` copies the committed mask: peer_loss_recovery.py:1767-1802 and
     execution_state.py:143-158, 177-192.
- `active_rank_mask` is a uint64-word tensor in **GPU memory**, not host memory: execution_state.py:50.
- The kernels read the mask at every launch (`load_rank_mask_words`, .cu:549, 1772) and skip inactive ranks in sends
  and waits (.cu:731, 846, 861, 1800, 1819).
- "Descriptor" is the figure's name for the kernel parameters that point at both records: `active_rank_mask` and
  `abort` in DispatchKernelPointers and CombineKernelPointers (moeAlltoAllKernels.h:155-157, 185-189).

## Roles
- The authority is a separate process started by `main()` in authority.py. It uses a ZeroMQ ROUTER and the workers
  use DEALER sockets: client.py:14, 116-138 and authority.py:650. It is not an MPI rank.
- Rank 0's loss is never admitted, so a hang there ends the job: authority.py:345-347.

## Doc cross-check (DESIGN_DOC_final.md, Recovery Flow, lines 364-445)
- Order: Detect, then Agree (shrink, stop, tell the authority), then Plan (the authority fences the failed ranks and
  proposes). This matches the code.
- The table matches the code: 20 ms poll; device wait 2 s or heartbeat 5 s; held 30 s; then fenced.
- The doc does not say that hang detection is off by default.

## Differences from the brief
1. Hang detection is opt-in (`hang_detection=False` by default).
2. The authority orders the fence and the ranks carry it out:
   - the hung rank SIGKILLs itself;
   - peers ask the MPI runtime to kill it after 1 s;
   - the authority's fence command runs only as a last resort after 5 s.
3. A missing heartbeat counts as evidence only when peers are also blocked on the host.
4. The abort flag is in pinned host memory but the live-rank mask is in GPU memory. The descriptor is two separate
   records.
5. The mask is cleared once at the remap (after Propose) and the committed mask is installed at Install. Before
   proposing, the authority also has rank 0 fence the failed ranks.
6. Kernels poll the abort flag in their spin loops only after a wait has lasted about 50 us, then about every 50 us.
7. Canvas: the existing design figures render on #FFFFFF (`CANVAS` in fig_design_final.py, checked in the fig1 and
   fig4 SVGs), not #F5F5F5. figD uses the same `restyle` so that it matches them.

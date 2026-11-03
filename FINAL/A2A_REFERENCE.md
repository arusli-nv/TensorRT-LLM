# FT changes to the NVLinkOneSided all-to-all: verified reference

For appendix slides and Q&A with the all-to-all owners. Code: v5-final `27442bd91d` vs main `9cc37929e8`; all
`file:line` below are at v5-final. Kernel: `cpp/tensorrt_llm/kernels/moe/communication/moeAlltoAllKernels.{cu,h}`
(`.cu`/`.h` below). Kernel-side FT changes come from one commit, `34611ef619`; the expert-to-slot lookup from `b0a8905b44`.

## Summary table

What the measured "FT cost" compares: the same kernel with **rank-mask mode off (FT off, as deployed) vs on (FT on)**,
under CUDA-graph replay, EP32, 64 tokens/rank, R1 shapes. Main already has rank-mask mode (detection-only FT, off by
default, no CUDA graphs); FT turns it on and changes how it reads the mask. So the cost includes main's existing
per-peer checks, made more expensive by FT's device-memory mask (item M).

Cost = microseconds per dispatch + combine round trip, with online-EPLB statistics / without (ablation job 8293196,
median of 3, CLAIMS 210; each item removed alone, so savings overlap slightly).

| Item | Why FT needs it | When it runs | Cost (stats / no stats) |
|---|---|---|---|
| M: mask read from a device record | Survivors rewrite the mask in place; captured CUDA graphs must see it | Every call: 4 loads per CTA, before the grid dependency wait | in S and L below (spill alone ~0.16 µs, CLAIMS 205) |
| S: mask check when sending EPLB counts to each peer | Never write into a dead rank's buffer | Every dispatch, online EPLB only | **2.13 µs** / ~0 |
| F: "released" check before signalling | A rank whose dispatch was cut short must not tell peers "done" | Every dispatch and combine | 0.48 / 0.45 µs |
| L: mask checks per route, counter, signal and wait | Never send to or wait for a dead rank | Every call | 0.45 / 0.49 µs |
| A: abort-flag poll in waits | Release waits on a dead peer | Every spin adds a clock compare; the flag is read only after ~50 µs of waiting | 0.10 / 0.05 µs |
| SO: barrier with OR in combine | Whole block learns a wait ended by abort | Every combine | 0.03 / 0.03 µs |
| T: expert-to-slot lookup in dispatch | Route by a slot table survivors can rewrite (no EPLB only) | No EPLB with FT only | not in the kernel bench |
| **Total** (all of the above except T) | | | **+3.00 / +0.88 µs** (all removed: +0.09 / +0.02) |

## Each change in detail

**M. Mask in a device record (all calls).**
- Main passed the mask by value as a kernel parameter (`.h` diff), and rejected CUDA graphs in rank-mask mode.
- FT passes a device pointer (`.h` `DispatchKernelPointers::active_rank_mask`). Warp 0 of every CTA copies it into
  registers before `cudaGridDependencySynchronize` (`.cu:549`, `:1772`, `:984`), so the load hides behind PDL.
- Effect: in SASS the register copy spills to local memory, so each later check is a local-memory load where main's
  was a constant-cache load (`review/diag/a2a_ft_work_coderead.md`). That's what makes S and L cost more than on main.

**S. Mask check in the EPLB-statistics send loop (online EPLB only).**
- With online EPLB, each rank sends its per-expert token counts to every peer: a serial loop over peers in one warp,
  writing 1 KB per peer into the peer's gathered-stats buffer (`.cu:679-693`).
- FT's check (`.cu:683`) is the first statement of each peer iteration, so every peer's stores wait behind one
  load-store-unit round trip: ~70 ns × 32 peers.
- Measured 2.13 of the 3.00 µs (CLAIMS 210). Main runs the same check only in rank-mask mode.

**F. "Released" check before signalling (every call).**
- If a dispatch returns because of an abort without some live peer's data, it sets `flag_val[kReleasedRoundIndex]`
  (`.cu:740`). From then on this rank publishes no round flag until the workspace is reset at recovery (`rearm`), so
  no peer combines outputs computed from incomplete data (`.h` `MoeA2AAbortState` comment).
- Every dispatch and combine reads that marker before signalling (`.cu:706`, `:1794`). In dispatch the read sits after
  `fence.release.sys` (`.cu:700`), so it waits for the fence to drain. That's its 0.45 µs.
- The per-rank flag slot grew from 4 to 8 bytes to hold it (`moeAlltoAllOp.cpp:238`); completion flags stay 4 bytes.

**L. Per-lane mask checks (every call).**
- In routing (`.cu:375`), the send-counter loop (`:667`), the signal loop (`:712`), the dispatch wait (`:731`), and the
  combine signal and wait (`:1800`, `:1819`).
- All existed in main's rank-mask mode; FT reads them from the register copy (M). Lane-parallel; 0.45–0.49 µs.

**A. Abort-flag poll (every wait).**
- `wait_round_flag` (`.cu:295-315`): each spin also compares the elapsed clock. After 100,000 cycles (~50 µs at an
  assumed 2 GHz, `.h:62` `kAbortPollCycles`) it reads the abort record in pinned host memory, then again every
  ~50 µs. Healthy waits are shorter, so they never read it.
- On an abort it records `status = cancelled` (`.cu:265-277`) and returns: dispatch sets the released marker (F) and
  exits; combine joins SO.
- The 300 s device timeout trap is unchanged (`.h:57`).

**SO. Barrier with OR in combine (every combine).**
- `__syncthreads()` becomes `__syncthreads_or(aborted)` (`.cu:1845`): one barrier either way, now also reducing a flag.
- If any wait ended by abort, the block writes zeros (a finite output for an invalidated collective, `.cu:1863`) instead
  of combining.

**T. Expert-to-slot lookup in dispatch (no EPLB with FT only).**
- With one slot per expert, dispatch looks up each expert's slot in a device table (`.cu:364`) instead of a separate
  routing kernel, and sends slot ids as the expert-id payload. Survivors rewrite the table at recovery; graphs see it.
- FT-off and EPLB modes run main's routing (commit `b0a8905b44` guards). Not part of the kernel bench, so its cost is
  inside the no-EPLB serving number (+120 µs/step at 32/GPU, CLAIMS 206), not isolated.

## What does not change

- No extra atomics, no extra memory fences (the fences at `.cu:700` and `:1838` are main's), no extra `__syncwarp`.
- No extra global stores on the healthy path: the new stores (`.cu:740` released marker, `:265` cancelled status,
  `:1863` zeros) run only on an abort.
- Completion flags are still 4-byte values (`st/ld.relaxed.sys.u32`); main's packed fence path and row layout are kept.
- New healthy-path loads: the 4 mask words per CTA (M), one released-marker load per dispatch and combine (F), and with
  T one table load per routing lane.

## Outside the kernels

- **Workspace remap, only at recovery:** survivors release the workspace's physical pages and map fresh ones at the
  same addresses (`mnnvl_memory.py:943` `release_backing`, `:965` `remap`; `peer_loss_recovery.py:1379`). 0.50–0.96 s
  mean per mode (CLAIMS 241). No and offline EPLB are ~2× online because 28/31 survivors stall ~0.46 s on the first
  import of the dead GPU's node-mate's fresh handle (CLAIMS 214, 215; cause open).
- **Execution state** (`execution_state.py`): one shared mask record, an abort record in pinned host memory (`:49`),
  `request_abort` (`:160`), `rearm` (reset every workspace to its first round, `:143`), `install_mask` (`:177`).
- **Fence path required:** binding a layer to the execution state refuses CFT counted writes
  (`nvlink_one_sided.py bind_execution_state`). All runs used the fence path (driver 580); CFT is untested (doc Limits).
- **Host watchdog:** skipped for bound (FT) layers, since graph replay bypasses the host code; the abort record and
  the device timeout cover them.

## Per-step cost and how it shows in serving

- Calls per step: 59 (58 MoE layers + the MTP layer).
- Kernel bench × 59, if back to back: ~52 µs/step without stats, ~177 µs/step with stats (CLAIMS 200, 210). These
  are upper-bound extrapolations, not serving measurements.
- Serving, whole FT (not only the all-to-all), per decode step: no EPLB +120 / +150 µs, offline +70 / +103 µs, online
  +51 / +99 µs at 32 / 128 requests/GPU (pooled n=8, CLAIMS 206, 232). Online is not ~120 µs above offline, so most of
  the stats cost seems hidden by overlap (not measured).

## Known fixes (measured in the kernel bench only)

- **S1, ballot the mask once per 32 peers** in the stats loop: cuts 1.62 of S's 2.13 µs (CLAIMS 210).
- **F8, one 8-byte load of [round, released] before the fence**: cuts 0.18 / 0.14 µs (CLAIMS 210).
- Both exist only as experiment switches on branch `a2a-ablate` (`d47a3bcf88`, env `TRTLLM_A2A_FT_ABLATE`, not for a
  PR) and the draft `review/diag/a2a_ftcost_draft.diff`. Not in v5-final or the PR stack; no serving pair yet.
- Keeping the mask in shared memory (no spill) cut the no-stats cost 0.85 → 0.69 µs but not S (CLAIMS 205).

## Caveats

- All per-item numbers are a kernel-only microbenchmark (diagnostic, n=3 launches, same nodes, CLAIMS 210).
- In serving, nsys cannot isolate the cost: the all-to-all floor is ~3.0 ms/step with FT on and off alike (−1.7 µs
  [−116, +101]), because most of it is waiting for the other 30 ranks (CLAIMS 212).
- T (expert-to-slot lookup) has no isolated measurement.

## Answer for the NCCL and NIXL owners

"We started with NVLinkOneSided because it's the fastest backend for multi-node NVLink. Another backend would need:
(1) skip dead ranks via a mask the host rewrites in place, readable under CUDA graphs; (2) waits that a host-side abort
flag can release, plus a guard so a rank cut short doesn't signal completion; (3) a communication buffer whose memory
can be swapped at the same addresses, or another way to fence off a dead rank's late writes; (4) a reset to round one
for survivors. NCCL and NIXL are next steps."

Honest scope: today this is wired through NVLinkOneSided-specific pieces: MNNVL fabric handles for the remap
(`mnnvl_memory.py`), the fence path only (no CFT), and the workspace bound to `MoeExecutionState`.

## Answer for Ray vs MPI

"We use MPI with ULFM because TRT-LLM launches wide EP with MPI, and ULFM gives failure notification, revoke and an
agreement on survivors. Recovery uses about a dozen MPI-only methods, so a Ray backend would need equivalents of
them. Not built."

Honest scope: there is **no clean interface** today. The base `Distributed` class has only `live_ranks` and an
`enable_peer_loss_recovery` that raises `NotImplementedError`. The rest live on `MPIDist` only: `revoke`, `shrink`,
`reported_failed_ranks`, `install_survivors`, `watch_peer_failures`, `watch_survivor_failures`,
`barrier_among_survivors`, `broadcast_among_survivors`, `allgather_among_survivors` (`communicator.py:993-1143`).
Startup refuses anything but `MPIDist` (`peer_loss_recovery.py:417`). The authority (ZeroMQ) is independent of MPI.

## Corrections vs the earlier summary

1. **F** was wrong: it isn't about stale old signals. It stops a rank whose dispatch was cut short by an abort from
   signalling "done", so no peer combines incomplete data.
2. **S and L are main's existing rank-mask checks**, which FT turns on (FT off runs with them off). FT's own change is
   reading the mask from a device record so CUDA graphs see updates; that move (local-memory reads) is what makes S
   cost 2.13 µs.
3. **S** is when sending this rank's counts to every peer, not "gathering".
4. **Missing items:** M (the device mask), T (expert-to-slot lookup in dispatch, no EPLB), and the abort-only stores
   (released marker, cancelled status, zero outputs).
5. **"No extra … healthy-path global stores"** holds; new healthy-path work is loads only (M, F, T).
6. **Per-step 50–60 / 177 µs** are kernel-bench × 59 extrapolations, not serving numbers; serving FT cost (51–150 µs)
   covers all of FT.
7. **Fence path required:** FT can't use CFT counted writes (untested, driver ≥ 615).
8. **Backend requirements** are four things, not three, and today rely on MNNVL-specific remap.
9. **Ray vs MPI:** there is no thin interface; recovery depends on ~10 MPI-only methods.

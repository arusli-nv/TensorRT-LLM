<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Idle recovery acceptance contract

Target: DeepSeek EP32, fixed 288 slots, NVLinkOneSided fence, unfused finalization,
V1 KV, no overlap or prefix reuse. Kill rank 2 only after actual per-layer placement
and capacity admit it. Rank 0, required shards, a second loss and device/fabric loss
are terminal. Success means a correct fresh request, the original frontend and 31
worker processes, original mappings and graph executables, and no model reload,
recapture, compilation or retuning. **This acceptance test has not passed.**

S = source inspected; C = component tested; F = full-model tested; U = unproven.
An acknowledgment establishes only the evidence named below.

The current repository path implements controlled idle admission and STOP only.
The frontend closes admission; all 32 original schedulers must join their local
owners and seal GPU issuance before injection. After death, 31 new scheduler STOP
receipts are required. Neither quorum permits installation or replay. The preflight
package imports and 112 focused checks passed in `7875544`; its special CPU import
mode failed. Ordinary CPU imports and live no-GPU-context isolation passed in `7876691`.
The tiny graph inspector/replay passed in `7870806`. Full-model STOP and recovery
remain unproven. A second departure makes the authority terminal, even after STOP.

| Transition / owner | Required evidence before the next transition | Test catching premature progress | Current evidence |
|---|---|---|---|
| Detect / MPI adapter and frontend | Original process birth, task and generation; authoritative departure from a qualified lifecycle adapter. MPI error alone is suspicion. | Reused PID or suspected-but-live peer must not authorize removal. | Local pidfd adapter C; EP32 adapter U. |
| Release host calls / `MPIDist` | Actual blocked operation returns a typed failure; original logical mapping retained; independently progressing control channel. | Kill a peer during the actual host allgather; keep survivor tasks alive. | Earlier allgather and idle broadcast release C (`7875893`); composed serving path U. |
| STOP / original `PyExecutor` scheduler | Worker consumes its authenticated local STOP; admission paused; no prepared real request; pending request dispositions owned and retained. | Serialized, stale or conflicting STOP must not advance the scheduler. | S/C; physical handoff U. |
| Join / ModelEngine execution state | Actual graph and auxiliary stream events complete; native KV ownership reconciled; watchdog thread, callbacks and D2H copies retired within one deadline. | Empty watchdog queue with incomplete DMA must withhold readiness. | Individual native cancellation/KV C; full-model STOP U. |
| Quarantine / execution state and MNNVL allocation owner | Retained import/backing tokens; exact immutable source ranges below; all captured writers/readers covered; authentic local join receipt. | Old routing and a delayed old-source writer must not contaminate a resumed request. | Addressing S; cross-GPU late-writer C (`7861416`); current full capture U. |
| Prepare/install / EPLB and MPI adapter | Every layer retains expert coverage and fixed slot/weight addresses; compact host membership maps to fixed logical rows. Install only under a matching authority command while paused. | Missing expert, wrong logical row or partial routing copy stays terminal. | Preparation/translation S/C; actual installation U. |
| Commit / sole external authority | Matching worker READY, then matching INSTALLED from every survivor, for one proposal/base/next generation. Server HOLD is insufficient. | Missing, duplicate/conflicting or stale acknowledgments must not grant resume. | Required next integration; not implemented in the current STOP protocol. |
| Replay / original graph runner | Authorized installed generation; unchanged graph/address identities; fresh dispatch before combine; watchdog rearmed; no failed-epoch outputs. | Retained old dispatch metadata must not authorize combine after installation. | Healthy original-graph replay F (`7857880`); recovery U. |
| Return / proxy and result owner | Correct complete fresh result; prior requests disposed without duplicates; frontend opens admission last. | Interrupted or uncertain KV/output must never appear as a successful result. | Healthy parity F; post-loss request U. |

## Failed-source memory contract

For receiver R, let B be its original symmetric base, N=32, F=2, and T the
validated nonzero, 128-byte-aligned source stride. Native metainfo supplies offsets;
the owner must validate actual addresses and extents, not infer them from this table.
Constructor token capacity is fixed at 256 across the four captured batch variants;
source inspection and job `7857880` agree. Per-graph argument validation is pending.

| Writable range from F, on each receiver | Next replay's read/reset rule |
|---|---|
| Each dispatch payload: `B + payload_offset + F*T`, bounded by static token capacity | Gather excludes F before reading; its private row is zeroed. The local expert-output tail is outside every dispatch lane. |
| Receive counters: `B + counters_offset + 4*(parity*N + F)`, both parities | Counter reset, gather and prepare-combine skip F before accessing its cells. |
| Dispatch/combine flags: `B + flag_offset + 4*F` | Native waits skip F. Watchdog DMA copies the retained local table, but completion checks exclude F. Copy retirement remains mandatory. |
| EPLB statistics: `B + stats_offset + 4*(F*experts + expert)` | Gather excludes F and zeros its private statistics row; routing must not consume the quarantined row. |
| Expert outputs | Fence combine makes remote **reads**, not writes. FC2 and prepare-combine write their own local tail. Fresh dispatch routing must exclude F; combine does not independently mask each payload load. |

Captured attention, KV, sampling, copies and other output kernels still require
actual owner/alias bounds. Job `7849756` inventoried 128 graphs with no unknown,
NCCL or CFT nodes. Job `7857880` proved normal-owner graph identity and schedule,
but did not retain its complete DAG and kernel arguments. That gap blocks admission.

CUDA's [allocation lifetime contract](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html)
keeps backing alive until all mappings and handle references are released.
The workspace owner retains its imports and rejects unmap/release while execution
owns them. This protects backing lifetime, not completion of outstanding writes.

Disjoint source addresses do not prove failed-issuer drain. Process exit, a timeout,
rank mask, kernel return or authority reply cannot substitute for confinement and
retained physical backing. Idle admission also requires the last normal fence round
and all local GPU work to finish. Dispatch uses a release fence and relaxed flag
polling; its payload visibility needs qualification, not an assumed acquire.
Job `7861416` observed exact survivor outputs during live remote writes into the
quarantined lanes on EP4, with original graphs and imports. Its negative test also
confirmed masked combine can read stale routing. It proves neither failed-issuer
drain nor EP32 recovery. Current full-model capture and integrated SIGKILL tests
remain required. Until those gates pass, admission fails
closed. Standard store addressing and imported-allocation lifetime must be supported
for the declared process-loss platform; device/fabric faults remain excluded.

Repeated losses could retire additional lanes monotonically, with fresh coverage and
capacity admission each time; the MVP admits one only. Reusing a retired lane,
rejoining or reclaiming backing needs a qualified drain/lifetime guarantee. If
confinement fails, first reject the offending path. If it cannot be isolated, retain
disjoint replacement storage under the same ownership contract or remain terminal;
never alternate back to unqualified old storage.

# WideEP fault tolerance: PR stack

Code: branch `ft-rebase` (local clone `/tmp/claude-161238/work/repo`, worktree `/tmp/claude-161238/wt-stack`), 26
commits on upstream `main` 78145f88ea (2026-10-30). One commit is one PR. Fork branches (prepared, not opened):
`wide-ep-ft/mN-*` to `main`, `wide-ep-ft/NN-*` to `feat/wide-ep-ft-mvp`; `wide-ep-ft-mvp` = the whole stack.

## Topology

```
main ──┬─ M1 fake moe_a2a op args ..................... (independent)
       ├─ M2 EPGroupHealth deepcopy shares ............ (independent)
       ├─ M3 failed OpenAI streams end with error+[DONE] (independent; /health 503 builds on it)
       ├─ M4 attention-DP burst release ............... (independent)
       ├─ M5 a2a fence asm "memory" clobber ........... (independent)
       └─ M6 mpi4py 4.1.2 (needs an image rebuild) .... (ULFM needs it)

feat/wide-ep-ft-mvp (= main + M1-M6), stacked:
  01 recovery protocol, config, client
  └─ 02 recovery authority
     └─ 03 hang detection (heartbeats, evidence, PMIx fence)
  M6 ─ 04 ULFM survivor communicators
        └─ 05 MPI session / launcher survive a peer loss
  06 a2a rank mask by pointer + abort record ─┬─ 07 a2a FT cost cuts (stats ballot, 8-byte round state)
                                             ├─ 08 MNNVL workspace remap among survivors
                                             └─ 09 MoeExecutionState binding (CUDA graphs)
  10 EPLB ghost slots + C++ restore ── 11 extra EPLB slots + Python restore ─┬─ 12 ring host copies
                                                                           └─ 13 MoE tactics on loaded slots
  14 KV Cache Manager V2 step rollback
  04 ─ 15 attention-DP with absent ranks
  all of the above ─ 16 executor recovery (PeerLossRecovery) ─ 17 /health + 503 (M3)
  06 + 11 ─ 18 expert-to-slot dispatch lookup (perf)
  19 integration tests + CI lists ─ 20 docs
```

| # | Commit | PR | Size (+/-) |
|---|---|---|---|
| M1 | ee85d04775 | [None][fix] Accept the real arguments in the fake moe_a2a op registrations | +56 |
| M2 | ef9bee2078 | [None][fix] Share EPGroupHealth across ConfigurableMoE's model-config deepcopy | +32 |
| M3 | 9a506cbb97 | [None][fix] End every failed OpenAI stream with its error events and a single [DONE] | +211/-74 |
| M4 | dc88d7c4aa | [None][perf] Release an attention-DP context hold once a burst of arrivals has ended | +104/-1 |
| M5 | 2f9fec9b47 | [None][fix] Order the MoE all-to-all payload accesses against their fences | +4/-4 |
| M6 | 0d08759b16 | [None][infra] Upgrade mpi4py to 4.1.2 for ULFM | +45/-54 |
| 01 | 55e0771f48 | [None][feat] Add the WideEP recovery protocol, config and client | +996 |
| 02 | 02c5c56644 | [None][feat] Add the external WideEP recovery authority | +3042 |
| 03 | 8852da051c | [None][feat] Detect a hung WideEP rank with heartbeats, hang evidence and PMIx fencing | +1008 |
| 04 | faeea6f7e8 | [None][feat] Add ULFM survivor communicators for host collectives | +1654/-36 |
| 05 | 3d311ece85 | [None][feat] Keep the MPI session serving and exit cleanly after a WideEP peer loss | +981/-13 |
| 06 | 46b2988556 | [None][feat] Read the NVLinkOneSided rank mask through a pointer and add an abort record | +888/-124 |
| 07 | 2a4099dc5c | [None][perf] Test the all-to-all rank mask lane-parallel in the EPLB stats loop and load the round state at once | +56/-15 |
| 08 | f632a2aa92 | [None][feat] Remap MNNVL all-to-all workspaces in place among the survivors | +502/-11 |
| 09 | e2f154f6cc | [None][feat] Bind NVLinkOneSided layers to a shared MoeExecutionState for WideEP recovery | +1222/-51 |
| 10 | 0b5d759c81 | [None][feat] Back empty EPLB slots with ghost memory and restore lost experts into them | +2711/-102 |
| 11 | e78e8d0ecf | [None][feat] Add extra EPLB slots for WideEP recovery and restore lost experts into them | +1180/-41 |
| 12 | 9919215049 | [None][feat] Keep WideEP EPLB host copies in a ring of nodes and stage restores from them | +1333/-58 |
| 13 | dbe93aa8da | [None][perf] Tune and key the MoE tactics on the loaded slots under WideEP recovery | +410/-50 |
| 14 | 8e11ff3ef7 | [None][feat] Roll back a discarded step's KV cache allocations in KV Cache Manager V2 | +371/-1 |
| 15 | 4c427c228a | [None][feat] Keep attention-DP routing serving with absent ranks | +575/-98 |
| 16 | 8a5300d036 | [None][feat] Recover the PyTorch executor from a WideEP peer loss | +5868/-22 |
| 17 | 76de422dc3 | [None][feat] Report WideEP fault-tolerance state on /health and return 503 on peer loss | +489/-8 |
| 18 | 4f983edfc3 | [None][perf] Look up dispatch slots in a device expert-to-slot table under WideEP recovery | +433/-11 |
| 19 | a3b2f2b667 | [None][test] Add WideEP peer-loss recovery integration tests | +885 |
| 20 | 966d3788b3 | [None][doc] Document WideEP fault tolerance | +444 |

Most of the size is tests (about 60%). Hashes change with every fixup round; the table is regenerated at push time.

# v5-pr restack (batch 1: R1-R6 + D4), 2026-10-23 14:09 PT

Worktree /tmp/claude-161238/wt-pr, branch v5-pr, base 9cc37929e8. Proof: v5-pr^{tree} == 1117f7e0e7^{tree} == fa9d61bafc51c98f8f7fd6e7b7e71045a0d95ee6. 25 commits, each 1 Signed-off-by, author Andrew Rusli, no co-author. Messages byte-identical to their targets.

```
3274ada868 [None][fix] Accept the real arguments in the fake moe_a2a op registrations
6f40128f0f [None][fix] Share EPGroupHealth across ConfigurableMoE's model-config deepcopy
148ee4311b [None][fix] End every failed OpenAI stream with one error event and one [DONE]
de62f48333 [None][infra] Upgrade mpi4py to 4.1.2 for ULFM
dff466faf7 [None][feat] Add the WideEP recovery protocol, config and client
23d6209676 [None][feat] Add the external WideEP recovery authority
6ebe6fc8d6 [None][feat] Detect a hung WideEP rank with heartbeats, hang evidence and PMIx fencing
5287f02693 [None][feat] Add ULFM survivor communicators for host collectives
6d5028ad57 [None][feat] Keep the MPI session serving and exit cleanly after a WideEP peer loss
6724073b7c [None][feat] Read the NVLinkOneSided rank mask through a pointer and add an abort record
5dd297c7ae [None][feat] Remap the MNNVL all-to-all workspaces at the same addresses among the survivors
16f9b7c9da [None][feat] Bind NVLinkOneSided layers to a shared MoeExecutionState for WideEP recovery
e88623b0ad [None][feat] Restore lost experts into EPLB slots backed by ghost slot memory (C++ load balancer)
9797ef53d0 [None][feat] Append extra EPLB slots and restore lost experts into them (Python)
fb90301d4d [None][feat] Keep WideEP EPLB host copies in a ring of nodes and stage restores from them
94ad14d926 [None][perf] Tune and key the MoE tactics on the loaded slots under WideEP recovery
5f9783f4d5 [None][feat] Roll back a discarded step's KV cache allocations in KV Cache Manager V2
29f5a7babe [None][perf] Release an attention-DP context hold once a burst of arrivals has ended
db2b86766c [None][feat] Keep attention-DP routing serving with absent ranks
67d60e271f [None][feat] Recover the PyTorch executor from a WideEP peer loss
1cbf84a1cb [None][feat] Run a lost rank's undelivered requests again on the survivors
2ab9c57ca8 [None][feat] Report WideEP fault-tolerance state on /health and answer peer-loss failures with 503
121a0a3e11 [None][perf] Look up dispatch slots in a device expert-to-slot table under WideEP recovery
77748099a5 [None][test] Add WideEP peer-loss recovery integration tests
b3d36d0ebc [None][doc] Document WideEP fault tolerance
```

Squashes: R1 e74cadb97c+ddc2658518+65741c820a -> 67d60e271f; R3 17a047ea76 moved before 9cf51e9801 with self.dist.tp_size (29f5a7babe; its ADP test fake also gets dist.tp_size, replaced by 9cf51e9801) -> 29f5a7babe, db2b86766c; R4 db6f10dc0b+3d10673082+1117f7e0e7 -> 1cbf84a1cb; R5 6f5f1d080b+0e5a22ef51 -> 5dd297c7ae; R6 ffc8bc30c3+7e8dfb37df -> 77748099a5, c86566c184+e25e638054+6e311f2f04 -> b3d36d0ebc. D4: nvlink_one_sided.py device-mask publish (old 2b5005b9a1 :665-670 fields, :888-910 _publish_rank_mask, :1078-1079 publish before dispatch, :1096/:1245-1246 pass the record) moved into 6724073b7c (34611ef619), unconditional there; 16f9b7c9da keeps the bind/state, graph policy, docstrings.

Static smoke per commit (py_compile all touched .py: pass; ruff F821/F401/F811: only 2 hits that pre-exist on 9cc37929e8). OWED: per-commit import/collect smoke + GPU tests in the real container (tools/import_smoke.sh), C++ build at 6724073b7c.

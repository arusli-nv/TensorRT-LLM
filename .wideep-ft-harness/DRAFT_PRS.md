# v7 draft PRs vs our series (2026-10-11; none merged in main)

Local refs: pr/15785 pr/15788 pr/15789 pr/15895 pr/19610 (fetched read-only from origin).

| PR | What | Relation | MVP decision |
|---|---|---|---|
| 19610 (chulianz, open, updated 10-09) NVLinkOneSided overhaul | native fixed-capacity layout (`moe_a2a_get_workspace_layout`, `moeAlltoAllMeta.h`), combine rewrite (`wait_round_flag`), Python timeouts, auto-CFT, `_mnnvl_utils` -> `_torch/distributed/mnnvl_memory.py` | rewrites our kernel/op/comm lines; no abort/descriptor/fixed source slots | keep ours; rebase on it after it merges; put abort into `wait_round_flag`; force fence path under FT; keep ABI delta small |
| 15895 cooperative A2A abort (TRTLLM-12561) | `MoeA2AExecutionControl`, host epoch by value + mapped status | competing abort ABI; not graph-compatible | keep ours; borrow packed phase/peer/reason status + C++ kernel test; align with owner |
| 15785 MPI failure evidence (TRTLLM-13548) | `ep_failure_evidence.py`, gossip FT subcomm | conceptual overlap with our watcher/authority | keep ours; borrow evidence-vs-membership vocabulary |
| 15789 NCCL recovery | ncclCommAbort/reinit, rendezvous | unused on our path | ignore; ensure no NCCL group with dead rank is used |
| 15788 EP health telemetry (TRTLLM-13553) | `/metrics` export | observability; stale paths | ignore; later take `EPGroupHealth.__deepcopy__` identity fix |

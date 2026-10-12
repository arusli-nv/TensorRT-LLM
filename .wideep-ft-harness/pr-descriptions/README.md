# WideEP peer-loss recovery: PR descriptions

Drafts only; none of these PRs is open. Each file starts with the PR title. Merge in this order.

## Independent fixes, based on main

U1 to U5 do not depend on each other or on the series. Merge them first.

| File | PR | Title |
|------|----|-------|
| 01-rpc-fatal-errors.md | U1 | `[None][fix] propagate fatal RPC worker errors` |
| 02-rpc-socket-linger.md | U2 | `[None][fix] bound RPC socket teardown with unreachable peers` |
| 03-dlpack-metadata-lifetime.md | U3 | `[None][fix] retain DLPack metadata until the last storage alias` |
| 04-skip-absent-sample-state.md | U4 | `[None][fix] skip request updates without a sampling result` |
| 05-ep-group-health-deepcopy.md | U5 | `[None][fix] keep EPGroupHealth identity across ConfigurableMoE deepcopy` |

## Feature series

| File | PR | Title | Needs |
|------|----|-------|-------|
| 06-mpi4py-4.1.2.md | 1 | `[None][infra] upgrade mpi4py to 4.1.2` | none |
| 07-recovery-authority.md | 2 | `[None][feat] WideEP recovery authority` | none |
| 08-ulfm-survivor-communicators.md | 3 | `[None][feat] ULFM survivor communicators for host collectives` | PR 1 image for the ULFM test |
| 09-a2a-execution-descriptor.md | 4 | `[None][feat] fault-tolerant execution descriptor for the NVLinkOneSided all-to-all` | none |
| 10-moe-execution-state.md | 5 | `[None][feat] MoE execution state for WideEP recovery` | 2, 4, U5 |
| 11-resident-replica-eplb.md | 6 | `[None][feat] resident-replica EPLB placement after a rank loss` | 3, 5 |
| 12-adp-absent-ranks.md | 7 | `[None][refactor] let attention-DP routing skip absent ranks` | 3 |
| 13-mpi-session-peer-loss.md | 8 | `[None][feat] keep the MPI session serving after a peer loss` | 2, 3 |
| 14-executor-peer-loss-recovery.md | 9 | `[None][feat] recover the executor from a WideEP peer loss` | all |

## Open items before posting

- 06: fill in `jenkins/current_image_tags.properties` after the image rebuild. Until then CI keeps mpi4py 3.1.5 and every ULFM test of the series skips.
- 10: `test_moe_a2a_late_writes.py` moved here from PR 4, because it uses PR 5's `NVLinkOneSided.calculate_required_workspace_size(fixed_source_slots=True)`.
- 11 and 12: the split test file names `test_resident_placement.py` and `test_adp_absent_ranks.py` are proposals.

# WideEP peer-loss recovery: PR descriptions

Drafts only; none of these PRs is open. Each file starts with the PR title. The numbers match the
commits on branch `wideep-ft` and section 8 of `../DESIGN.md`. Merge in this order.

## Standalone fixes

PRs 1 to 4 do not depend on each other or on the feature series. Merge them first.

| PR | File | Title |
|----|------|-------|
| 1 | 01-rpc-fatal-errors.md | `[None][fix] propagate fatal RPC worker errors` |
| 2 | 02-rpc-socket-linger.md | `[None][fix] bound RPC socket teardown with unreachable peers` |
| 3 | 03-skip-absent-sample-state.md | `[None][fix] skip request updates without a sampling result` |
| 4 | 04-ep-group-health-deepcopy.md | `[None][fix] keep EPGroupHealth identity across ConfigurableMoE deepcopy` |

## Feature series

| PR | File | Title | Needs |
|----|------|-------|-------|
| 5 | 05-mpi4py-4.1.2.md | `[None][infra] upgrade mpi4py to 4.1.2` | none |
| 6 | 06-recovery-authority.md | `[None][feat] WideEP recovery authority` | none |
| 7 | 07-ulfm-survivor-communicators.md | `[None][feat] ULFM survivor communicators for host collectives` | 5 (image) for the ULFM test |
| 8 | 08-a2a-execution-descriptor.md | `[None][feat] fault-tolerant execution descriptor for the NVLinkOneSided all-to-all` | none |
| 9 | 09-moe-execution-state.md | `[None][feat] MoE execution state for WideEP recovery` | 4, 6, 8 |
| 10 | 10-resident-replica-eplb.md | `[None][feat] resident-replica EPLB placement after a rank loss` | 7, 9 |
| 11 | 11-adp-absent-ranks.md | `[None][feat] let attention-DP routing skip absent ranks` | 7 |
| 12 | 12-mpi-session-peer-loss.md | `[None][feat] keep the MPI session serving after a peer loss` | 6, 7 |
| 13 | 13-executor-peer-loss-recovery.md | `[None][feat] recover the executor from a WideEP peer loss` | all |

## Open items before posting

- PR 5: fill in `jenkins/current_image_tags.properties` after the image rebuild. Until then CI
  keeps mpi4py 3.1.5: PR 5's `test_mpi4py_exposes_ulfm_bindings` fails by design, because it
  guards the image bump, and the other ULFM tests of the series skip.
- PR 9: `test_moe_a2a_late_writes.py` lives here, not in PR 8, because it uses PR 9's
  `NVLinkOneSided.calculate_required_workspace_size(fixed_source_slots=True)`.

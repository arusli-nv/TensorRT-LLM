# WideEP peer-loss recovery: PR descriptions

Drafts only; none of these PRs is open. Each file starts with the PR title. The numbers match the
commits on branch `wideep-ft` and the deliverables table in `../DESIGN.md`. Merge in this order.

| PR | File | Title | Needs |
|----|------|-------|-------|
| 1 | 01-ep-group-health-deepcopy.md | `[None][fix] keep EPGroupHealth identity across ConfigurableMoE deepcopy` | none |
| 2 | 02-mpi4py-4.1.2.md | `[None][infra] upgrade mpi4py to 4.1.2 for ULFM fault tolerance` | none |
| 3 | 03-recovery-authority.md | `[None][feat] add an external recovery authority for WideEP peer loss` | none |
| 4 | 04-ulfm-survivor-communicators.md | `[None][feat] ULFM survivor communicators for host collectives` | 2 (image) for the ULFM test |
| 5 | 05-a2a-execution-descriptor.md | `[None][feat] add a run-time rank mask and abort to the NVLinkOneSided all-to-all` | none |
| 6 | 06-moe-execution-state.md | `[None][feat] bind MoE layers to the all-to-all execution descriptor for recovery` | 1, 3, 5 |
| 7 | 07-resident-replica-eplb.md | `[None][feat] route a lost rank's experts to their replicas after a peer loss` | 4, 6 |
| 8 | 08-adp-absent-ranks.md | `[None][feat] let attention-DP routing skip absent ranks` | 4 |
| 9 | 09-mpi-session-peer-loss.md | `[None][feat] keep the MPI session serving after a peer loss` | 3, 4 |
| 10 | 10-kv-cache-revert.md | `[None][feat] roll back KV Cache Manager V2 allocations of a discarded step` | none |
| 11 | 11-executor-peer-loss-recovery.md | `[None][feat] recover the executor from a WideEP peer loss` | all |

Three fixes found along the way are not needed by this series and are kept on the
backup branch `wideep-ft-exploration`: "propagate fatal RPC worker errors", "skip request updates without a sampling
result" and "bound RPC socket teardown with unreachable peers".

## Open items before posting

- PR 2: fill in `jenkins/current_image_tags.properties` after the image rebuild. Until then CI
  keeps mpi4py 3.1.5: PR 2's `test_mpi4py_exposes_ulfm_bindings` fails by design, because it
  guards the image bump, and the other ULFM tests of the series skip.
- PR 6: `test_moe_a2a_late_writes.py` lives here, not in PR 5, because it uses PR 6's
  `NVLinkOneSided.calculate_required_workspace_size(fixed_source_slots=True)`.

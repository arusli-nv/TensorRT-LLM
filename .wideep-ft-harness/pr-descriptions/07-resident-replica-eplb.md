[None][feat] resident-replica EPLB placement after a peer loss

## Description

After a rank loss, routing must stop selecting the dead rank's slots without moving weights and without changing anything a captured graph depends on. Static EPLB with spare slots can hold a replica of every expert of a rank on other ranks. This PR points routing at those resident replicas. It also decides which rank losses a placement can cover. It is PR 7 of the WideEP peer-loss recovery series and depends on PR 4 for `is_mpi_peer_failure` and on PR 6 for the execution state.

`MoeLoadBalancer` gains:

- `masked_placement_digest(dead_ranks)`, a SHA-256 over every layer's slot table with the dead ranks cleared. It raises when an expert would lose its last replica. Survivors compare digests through the authority, so ranks that agree route identically.
- `covers_loss_of(rank)`, which decides at startup which ranks are recoverable.

`ModelEngine.install_resident_placement` runs while the engine is idle, in this order: install the masked placement through main's mask-only EPLB API, mark the failed ranks in `EPGroupHealth` (a rank that is already failed is an error), rearm the workspace, then install the mask. Routing must exclude a rank before the mask does, because masked kernels drop routes to it silently.

The node-local shared-memory barrier in `MoeLoadBalancer` tolerates a dead peer only with recovery on.

Only static placement is supported, with `layer_updates_per_iter: 0` and `initial_global_assignments` set. Online EPLB needs a quiesced integration and is follow-up work.

FT-off impact: the new methods have no caller, and the shared barrier re-raises every error as before.

This PR does not stand alone: its methods get their first caller in PR 11.

## Test Coverage

New file `tests/unittest/_torch/executor/test_resident_placement.py`. The pre-merge stages that collect `unittest/_torch/executor` run it: `l0_cpu` runs them.

- `test_masked_placement_digest_is_deterministic_and_mask_sensitive` and `test_masked_placement_digest_rejects_lost_expert_and_bad_rank`.
- `test_install_resident_placement_commits_mask`, which also checks that routing is updated before the mask.

## PR Checklist

Please review the following before submitting your PR:
- PR description clearly explains what and why. If using CodeRabbit's summary, please make sure it makes sense.
- PR Follows [TRT-LLM CODING GUIDELINES](https://github.com/NVIDIA/TensorRT-LLM/blob/main/CODING_GUIDELINES.md) to the best of your knowledge.
- Test cases are provided for new code paths (see [test instructions](https://github.com/NVIDIA/TensorRT-LLM/tree/main/tests#1-how-does-the-ci-work))
- If PR introduces API changes, an appropriate PR label is added - either `api-compatible` or `api-breaking`. For `api-breaking`, include `BREAKING` in the PR title.
- Any new dependencies have been scanned for license and vulnerabilities
- [CODEOWNERS](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.github/CODEOWNERS) updated if ownership changes
- Documentation updated as needed
- Update [tava architecture diagram](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.github/tava_architecture_diagram.md) if there is a significant design change in PR.
- The reviewers assigned automatically/manually are appropriate for the PR.


- [ ] Please check this after reviewing the above items as appropriate for this PR.

## GitHub Bot Help

To see a list of available CI bot commands, please comment `/bot help`.

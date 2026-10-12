[None][feat] revert the KV cache allocations of a discarded step

## Description

WideEP peer-loss recovery (PR 14) discards a step that a peer loss interrupted and runs it again. The KV cache must then look as if the step was never scheduled, and nothing the discarded forward wrote may become a reusable prefix. This PR adds that rollback to both KV cache managers. It depends only on main.

- `KVCacheManager.revert_prepare_resources` (V1) undoes `prepare_resources`. First context chunks release the sequences added for them without storing their blocks for reuse, and rewind the request's reused-prefix state so the re-run can match a shorter prefix. Generation requests give back the tokens added for the step.
- `KVCacheManagerV2.revert_prepare_resources` undoes what V2 allocates during scheduling. First context chunks are freed before their blocks are committed for reuse, later chunks restore their previous capacity, and generation requests use the existing `revert_allocate_generation`.
- Cross-attention caches, helix context parallelism, SWA scratch reuse, draft caches and KV connectors raise `NotImplementedError`.
- The nanobind `remove_sequence` binding names its arguments and accepts `None` for `llm_request`, so a sequence can be released without storing its blocks.

Nothing calls the new methods yet, so behavior is unchanged until PR 14.

## Test Coverage

- `tests/unittest/_torch/executor/test_kv_cache_revert.py`, new, round-trips V1 and V2 rollback on real managers, for context and generation requests, and checks that a reverted context is not offered for reuse. It needs a GPU; the GPU pre-merge stages that collect `unittest/_torch/executor` (`l0_h100`, `l0_b300`) run it.

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

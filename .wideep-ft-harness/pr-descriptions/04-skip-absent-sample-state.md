[None][fix] skip request updates without a sampling result

## Description

`PyExecutor._sample_async` returns `None` when the forward produced no outputs and when sampling failed and already reported its error through `_handle_errors`. `_update_requests` then passed `None` to `sampler.update_requests`, which raised `AttributeError: 'NoneType' object has no attribute 'sampler_event'`. The executor caught it and ran `_handle_errors` a second time with that misleading message.

This PR returns early from `_update_requests` when there is no sample state. A valid sample state follows the same path as before, including the error handling.

## Test Coverage

`tests/unittest/_torch/executor/test_py_executor.py::test_update_requests_skips_absent_sample_state` is new; `l0_cpu` runs it through `unittest/_torch/executor`. The existing sampler tests cover the unchanged path with a sample state.

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

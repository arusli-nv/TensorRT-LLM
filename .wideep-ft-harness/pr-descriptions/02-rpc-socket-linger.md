[None][fix] bound RPC socket teardown with unreachable peers

## Description

`RPCClient.close()` can hang when its server is gone. ZMQ's default linger keeps undelivered messages, and terminating the context waits until they are sent. With a dead or never-started server that never happens.

This PR sets `ZMQ_LINGER` to 0 on the client socket. `close()` now discards undelivered sends and returns. Teardown must be bounded when worker processes die, for example on the terminal path of WideEP peer-loss recovery.

One behavior changes: a fire-and-forget call made right before `close()` can be dropped. `TestRpcBasics::test_rpc_without_wait_response` relied on a 100 ms sleep for such a call to arrive. It now waits for the server to acknowledge receipt, which is what a caller that needs delivery should do.

## Test Coverage

- `tests/unittest/executor/test_rpc.py::test_rpc_close_with_unreachable_peer` is new. It queues a send to an address with no server and closes the client in a subprocess with a 60 s timeout.
- `tests/unittest/executor/test_rpc.py::TestRpcBasics::test_rpc_without_wait_response` is updated.
- `l0_cpu` runs `test_rpc.py`.

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

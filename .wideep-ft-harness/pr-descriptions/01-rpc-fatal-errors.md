[None][fix] propagate fatal RPC worker errors

## Description

When the worker side of the RPC executor fails, the proxy does not notice. Its response fetch task logs the error and exits. Results that are already pending wait forever, `submit` keeps accepting requests, and `LLM.generate_async` reports "LLM is shutting down" without the cause.

This PR makes a response-stream failure terminal for the RPC proxy:

- `RpcWorkerMixin.fetch_responses` raises when the engine stored an event-loop error, so the failure crosses the RPC stream.
- The proxy keeps the first fatal error, wakes every pending result with `EngineDeadError`, and rejects later submissions with the same error. An unexpected cancellation or end of the response stream outside shutdown is fatal too.
- Submission registers the result before it sends the request and removes it if the send fails. An `RPCError` or `zmq.ZMQError` on send is fatal.
- `RayExecutor.submit` uses the same submission path and keeps its assigned client ID.
- `LLM.generate_async` raises `EngineDeadError` with the original cause.

The fix applies to every RPC and Ray deployment. A healthy executor behaves as before. WideEP peer-loss recovery, which follows in a separate series, relies on it: a recovery that fails takes the terminal path, and clients must receive that error instead of hanging.

## Test Coverage

- `tests/unittest/executor/test_rpc_executor_mixin.py` is new and added to `l0_cpu`. It covers sync and async waiters woken by a fatal error, failures during registration and during send (a transport error is fatal and keeps its cause, a local serialization error is not), a response stream that fails, closes or is cancelled, shutdown and cancellation that are not fatal, a worker-loop failure carried over a real RPC stream, and the Ray client ID.
- `tests/unittest/executor/test_rpc_worker_mixin.py::test_fetch_responses_surfaces_stored_executor_failure` is new; the file already runs in `l0_cpu`.

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

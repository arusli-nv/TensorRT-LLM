[None][feat] WideEP recovery authority

## Description

This PR adds the recovery authority: the single coordinator of a peer-loss recovery, and the place where the recoverable ranks are admitted. It is PR 3 of the WideEP peer-loss recovery series and has no dependency.

The authority runs outside the workers' launch and termination group, because agreement among survivors over MPI does not prove they can make progress. ULFM `Shrink` only rebuilds the communicator. The authority is one CPU process.

New package `tensorrt_llm/executor/recovery/`; it runs as a separate CPU process from a TensorRT-LLM install:

- `protocol.py` defines the messages `Register`, `Registered`, `Stopped`, `Propose`, `Ready`, `Install`, `Installed`, `Commit` and `Abort`, and signs every frame with HMAC.
- `authority.py` holds a state machine with no I/O and a ZeroMQ ROUTER loop. Run it with `python -m tensorrt_llm.executor.recovery.authority --endpoint tcp://0.0.0.0:PORT --world-size N`.
- `client.py` is the worker's DEALER client with deadlines. `recovery_requested()` is the environment check that later PRs import; it reads the environment once per process.

Each worker registers the ranks whose loss its expert placement covers. The authority admits the intersection of those sets and never rank 0. A recovery then runs `Stopped`, `Propose`, `Ready`, `Install`, `Installed`, `Commit`. `Ready` carries a placement digest that must match on every survivor. A protocol violation, a digest mismatch, an unadmitted loss, a second loss or the 60 s deadline sends `Abort`, which is terminal. Workers fail closed when they lose the authority.

The environment enables recovery with `TLLM_FAULT_TOLERANCE_MODE=1`, `TLLM_FAULT_TOLERANCE_AUTHORITY` and `TLLM_FAULT_TOLERANCE_AUTHORITY_KEY`, a hex key of at least 16 bytes. The key authenticates messages, not routing ids: on the trusted network it assumes, a peer without the key can at most take over a worker's route, which fails closed (timeout, then `Abort`). It does not encrypt messages, and the protocol has no replay protection. A public `LlmArgs` field is follow-up work once the feature leaves its experimental state.

FT-off impact: none. No existing code imports the package in this PR.

## Test Coverage

- `tests/unittest/executor/test_recovery_authority.py` is new and added to `l0_cpu`. It covers registration and admission, idle loss recovery, unadmitted and second losses, protocol violations, digest disagreement, stale generations, the deadline, the wire format and key handling, and end-to-end runs over ZeroMQ including rejected signatures.

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

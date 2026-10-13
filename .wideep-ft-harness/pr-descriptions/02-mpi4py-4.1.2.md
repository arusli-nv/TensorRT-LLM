[None][infra] upgrade mpi4py to 4.1.2 for ULFM fault tolerance

## Description

The container builds mpi4py 3.1.5 from source with a TRT-LLM patch. WideEP peer-loss recovery needs the ULFM bindings that mpi4py 4 adds: `Comm.Revoke`, `Comm.Shrink` and `Comm.Get_failed`. This PR moves the container to mpi4py 4.1.2, the version `security_scanning/poetry.lock` already pins.

This is PR 2 of the WideEP peer-loss recovery series. It has no code dependency.

- The `TRTLLM_USE_MPI_KVCACHE` patch selects the CUDA device for MPI KV-cache transfer when mpi4py.futures starts its manager. It moves from `futures/_lib.py` `_manager_split` to `futures/_core.py` `_comm_executor_helper`, where mpi4py 4 splits the communicator. A singleton executor still returns before the device selection, as in 3.1.5.
- The setuptools `< 69` pin and the aarch64 Cython 0.29.37 install go away. mpi4py 4 builds with current tooling.
- `jenkins/current_image_tags.properties` points at the rebuilt images. TODO before posting: fill in the tags after the image build.

FT-off impact: no TRT-LLM code changes. Every MPI path runs on the new mpi4py, so the existing MPI unit and multi-GPU tests are the regression check.

mpi4py.futures itself has no ULFM support. After a member dies, `MPICommExecutor` can neither collect results nor shut down. PR 9 bounds that with `MPI_Abort(0)`. A proper fix, upstream or as a TRT-LLM dispatcher, is follow-up work.

Until this image is in CI, the ULFM unit test of PR 4 and the WideEP integration test of PR 11 skip.

## Test Coverage

- `tests/unittest/llmapi/test_mpi_session.py::test_mpi4py_exposes_ulfm_bindings` is new. It checks that the installed mpi4py has `Comm.Revoke`, `Comm.Shrink` and `Comm.Get_failed`. `l0_cpu` runs this file with `ISOLATION`.
- The image build is the proof that the `TRTLLM_USE_MPI_KVCACHE` patch applies to 4.1.2's `futures/_core.py`: `install_mpi4py.sh` fails if it does not.
- All existing MPI tests in the pre-merge lists run on the rebuilt image.

Compatibility check before the image rebuild, on one GB200 node (ARM): main's MPI-dependent tests, run once with the image's mpi4py 3.1.5 and once with 4.1.2 built by this script, give the same results, apart from this PR's own ULFM guard test (see below).

| Test file | mpi4py 3.1.5 | mpi4py 4.1.2 |
|---|---|---|
| `_torch/moe/multi_gpu/test_moe_a2a.py` (`mpi4py.futures` workers) | 19 passed, 9 skipped | 19 passed, 9 skipped |
| `_torch/moe/multi_gpu/test_moe_a2a_workspace.py` | 8 passed, 8 skipped | 8 passed, 8 skipped |
| `_torch/executor/kv_cache/test_kvcm2_integration.py` | 165 passed | 165 passed |
| `_torch/disaggregation/test_disagg_inflight_cancel_gate.py` (`TRTLLM_USE_MPI_KVCACHE` path) | 51 passed | 51 passed |
| `llmapi/test_mpi_session.py` | 45 passed, 2 failed | 46 passed, 1 failed |

The 3.1.5-only failure is this PR's `test_mpi4py_exposes_ulfm_bindings`. `test_remote_mpi_session[flashinfer_temporary_cleanup]` fails identically on both versions because the test environment's temporary path makes a ZMQ IPC socket path exceed 107 characters. The build was also checked on a second ARM node; no x86 node was available, so the x86 build is first exercised by the image rebuild.

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

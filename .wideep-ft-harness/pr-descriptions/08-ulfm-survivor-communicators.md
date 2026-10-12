[None][feat] ULFM survivor communicators for host collectives

## Description

When one worker of a WideEP job dies, host collectives that include it hang or abort the job. This PR lets survivors interrupt those collectives, agree on the membership, and continue on a communicator without the dead rank. It implements v7 items 1c.3a and 1c.4a, survivor collectives, and the host side of 1b.4a, sparse membership mapped to logical slots. It is PR 3 of the WideEP peer-loss recovery series and has no code dependency.

`MPIDist` in `communicator.py` gains:

- `enable_peer_loss_recovery()`. It requires a pure attention-DP layout (`tp_size == moe_ep_size == world_size`, no PP or CP), mpi4py 4 or later, a ULFM launch and `MPI_THREAD_MULTIPLE`, so an unsupported job fails at startup. It checks `OMPI_MCA_mpi_ft_enable`, because the library exposes the ULFM calls even when the launch did not enable them. It moves host collectives to a private duplicate of `COMM_WORLD`, so revoking it leaves the MPI session usable.
- `revoke()`, `shrink()` and `install_survivors()`. `Shrink` gives every survivor the same failed set, and installation keeps logical ranks. `install_survivors()` rejects a committed failed set that differs from the shrink result.
- `watch_peer_failures()`, which returns a `PeerFailureWatch`. Its thread posts an `Irecv(ANY_SOURCE)` that no rank matches on a private communicator, and ULFM completes it with an error when a member dies. It calls back once and stops idempotently.
- With recovery on, peer-failure MPI errors from collectives become `HostCollectiveInterrupted`, which carries the known failed ranks. Other errors propagate unchanged.
- After installation, gathers keep each result at its logical rank. Object gathers return `None` for the departed rank, `tp_allgather_int64` takes a `fill_value`, and `live_ranks` lists the ranks still present.

The base `Distributed` gains `live_ranks` (every TP rank) and an `enable_peer_loss_recovery()` that raises `NotImplementedError`, so other backends reject recovery at startup. `_utils` gains `is_mpi_peer_failure()`, which classifies `ERR_PROC_FAILED`, `ERR_PROC_FAILED_PENDING` and `ERR_REVOKED` and ignores classes that alias a generic error class, plus `mpi_failed_ranks()` and `mpi_world_ranks()`, which PR 8 reuses.

ULFM needs `mpirun --with-ft ulfm --mca pml ob1 --mca btl self,tcp --mca coll ^hcoll,ucc`. MPI carries only host control traffic, and MoE data uses NVLink.

FT-off impact: nothing calls the new methods. The wrapper on each collective re-raises unchanged when recovery is off, and with every rank live the gathers return the same results as before.

## Test Coverage

- `tests/unittest/_torch/distributed/test_mpi_survivors.py` is new and uses a fake MPI. It covers revoke, shrink and install, logical results, interruption mapping, watch start and stop, the layout and launch checks, and the peer-failure error classes against a fake MPI that aliases one of them to `ERR_UNKNOWN`. `l0_cpu` runs it through `unittest/_torch/distributed`.
- `tests/unittest/_torch/distributed/test_mpi_survivors_ulfm.py::test_survivors_recover_from_a_departed_rank` is new and runs in `l0_cpu`. It launches `mpirun --with-ft ulfm -np 4`, and rank 2 dies. Survivors interrupt, shrink, install and gather. It needs the mpi4py 4.1.2 image from PR 1 and skips without it or without an Open MPI whose `ompi_info` reports `ftmpi`.

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

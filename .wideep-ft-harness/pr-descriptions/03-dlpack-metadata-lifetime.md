[None][fix] retain DLPack metadata until the last storage alias

## Description

`pack_strided_memory` turns borrowed device memory, such as an MNNVL workspace, into a torch tensor through a DLPack capsule. It kept the ctypes `DLManagedTensor` and its shape and stride arrays alive by attaching their wrapper to the returned tensor. Views, `detach()` and dtype views share the storage but not that attribute. When the root tensor died first, Python freed the metadata while the storage still pointed at it, and torch later called the deleter through freed memory. `create_dlpack_capsule` also leaked one reference to every capsule.

With this PR the storage owns the metadata:

- `pack_strided_memory` takes a reference to the wrapper and stores it in `manager_ctx`.
- The deleter, which torch calls when the last alias of the storage dies, clears `manager_ctx` and drops that reference. It still never frees the device memory, which stays owned by the caller.
- `create_dlpack_capsule` releases the extra capsule reference.

No caller changes. Metadata now lives exactly as long as the storage.

## Test Coverage

New tests in `tests/unittest/_torch/test_mnnvl_memory_lifecycle.py`, which runs in `l0_a10`:

- `test_dlpack_metadata_lives_until_last_storage_alias` for shape, detach, dtype and sliced dtype aliases, with and without inference mode.
- `test_dlpack_standalone_capsule_has_balanced_reference`
- `test_dlpack_refcount_helper_is_independent_of_shared_prototype`

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

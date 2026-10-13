[None][feat] bind MoE layers to the all-to-all execution descriptor for recovery

## Description

This PR binds the execution descriptor of PR 5 to the model, so survivors can abort in-flight MoE collectives and route around a dead rank on the CUDA graphs captured at startup, with no recapture. It is PR 6 of the WideEP peer-loss recovery series and depends on PR 3 for `recovery_requested()`, on PR 5 for the op arguments, and on PR 1, which keeps `EPGroupHealth` identity across the `ConfigurableMoE` deepcopy.

- `communication/execution_state.py` adds `MoeExecutionState`. It owns the descriptor at a fixed device address and provides `request_abort`, `rearm` and `install_mask`, which publishes the committed `EPGroupHealth` membership and refuses while the abort record is set. The abort record lives in pinned host memory that kernels read through its mapped address, so `request_abort` is a plain CPU store. It needs no CUDA work to make progress, so it reaches eagerly launched kernels that occupy the whole GPU; a side-stream copy did not complete while an eager dispatch spun. `tensorrt_llm/_utils.py` gains `device_visible_host_zeros` for this allocation. `for_model()` binds every NVLinkOneSided layer and returns `None` when recovery is off.
- `nvlink_one_sided.py` adds `bind_execution_state`, workspace sizing for fixed source slots and a private workspace reset that `rearm` calls, and passes the descriptor to dispatch and combine. Bound execution requires the fence path and relies on the device timeout instead of the host watchdog, which bound layers never start.
- `wide_ep_ft.py` adds `recovery_enabled()`, which delegates to the cached `recovery_requested()` of PR 3, and allows CUDA graphs in rank-mask mode only when recovery binds the layers.
- `_mnnvl_utils.py` adds `MnnvlMemory.pin` and `unpin`. Captured graphs address the workspace directly, so close and checkpoint leave a pinned allocation mapped.
- `DecoderRunner` creates the state, `ModelEngine.moe_execution_state` exposes it, and engine cleanup closes it after it releases the CUDA graphs.

FT-off impact: `for_model()` returns `None`, no layer is bound, the workspace keeps its size, the watchdog path is unchanged, and rank-mask mode still rejects CUDA graphs.

FT-on cost: one gather kernel per MoE layer. With the full series on DeepSeek-R1 NVFP4 at EP32 with all ranks active, throughput was 14,488 tok/s with FT on and 14,689 tok/s with FT off, 1.4% lower, three trials each. That figure also includes the per-step commit sync of PR 11.

## Test Coverage

- `tests/unittest/_torch/moe/test_moe_execution_state.py` is new and added to `l0_b200` next to PR 5's test; it needs one GPU and no MNNVL. It covers the descriptor layout and mask install, an abort of a dispatch stuck on a lost peer from another thread followed by rearm and resume on the same workspace, binding and workspace pinning, and the recovery flag. Graph replay across a mask change is covered by PR 5's `test_cancel_and_exclusion_on_replay`.
- `tests/unittest/_torch/moe/multi_gpu/test_moe_a2a_late_writes.py::test_killed_peer_writes_stay_in_its_storage` is new and added to `l0_gb200_multi_gpus` pre-merge with `TIMEOUT (10)`. Two processes share a real MNNVL workspace. The victim is SIGKILLed 0, 3 or 50 ms into capacity-sized dispatches. The survivor checks that no byte outside the victim's slot changed, then runs its captured graph with the victim masked and gets exact results.

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

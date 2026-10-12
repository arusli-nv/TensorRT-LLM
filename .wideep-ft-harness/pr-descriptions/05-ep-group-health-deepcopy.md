[None][fix] keep EPGroupHealth identity across ConfigurableMoE deepcopy

## Description

When a MoE layer has a per-layer quant override, `ConfigurableMoE` builds its backend from `copy.deepcopy(model_config)`. With the WideEP rank mask enabled through `TLLM_FAULT_TOLERANCE_MODE=1`, the first layer's communication strategy stores one `EPGroupHealth` in `model_config.extra_attrs`, and every MoE layer must share it. Later layers deep-copy that object. Copying its `threading.Lock` raises `TypeError`, and a successful copy would give the backend its own membership that recovery never updates.

This PR gives `EPGroupHealth` a `__deepcopy__` that returns the object itself: it is shared membership, not per-copy state. Every deepcopy of a config that holds it, including the backend config and the legacy `ep_group_health` key, now shares it. `ConfigurableMoE` is unchanged.

Without the rank mask nothing changes. The WideEP recovery series binds each NVLinkOneSided layer to an execution state and checks that both use the same health object.

## Test Coverage

- `tests/unittest/_torch/moe/fused_moe/test_configurable_moe.py::test_layer_quant_override_shares_ep_group_health` is new. It checks `copy.deepcopy(health) is health`, then builds a layer with `override_quant_config` while `extra_attrs` holds an `EPGroupHealth` and asserts the backend config holds the same object. `l0_cpu` runs this file through `unittest/_torch/moe`.
- `tests/unittest/_torch/moe/fused_moe/test_configurable_moe.py` and `tests/unittest/_torch/moe/test_ep_group_health.py` cover the unchanged paths.

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

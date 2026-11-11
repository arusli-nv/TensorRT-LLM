# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.custom_ops import torch_custom_ops
from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner
from tensorrt_llm._torch.utils import ActivationType

# Tactic selection and profiling only: the kernel runner is a mock, and the SM
# version is monkeypatched. The marker is also what makes the file reachable:
# the CPU stage collects only files that carry it.
pytestmark = pytest.mark.cpu_only


def _make_runner(x_dtype: torch.dtype, weight_dtype: torch.dtype) -> MoERunner:
    runner = MoERunner.__new__(MoERunner)
    runner.x_dtype = x_dtype
    runner.weight_dtype = weight_dtype
    runner.fused_moe_runner = Mock()
    runner.fused_moe_runner.get_tactic_num.side_effect = {1: 37, 2: 71}.get
    return runner


@pytest.mark.parametrize("gemm_idx, expected", [(1, 36), (2, 70)])
def test_sm107_bf16_moe_uses_last_tactic_for_fallback(
    monkeypatch: pytest.MonkeyPatch, gemm_idx: int, expected: int
) -> None:
    monkeypatch.setattr(torch_custom_ops, "get_sm_version", lambda: 107)
    runner = _make_runner(torch.bfloat16, torch.bfloat16)

    assert runner._resolve_fallback_tactic(-1, gemm_idx) == expected


@pytest.mark.parametrize(
    "sm_version,x_dtype,weight_dtype",
    [
        (100, torch.bfloat16, torch.bfloat16),
        (107, torch.float16, torch.float16),
        (107, torch.bfloat16, torch.float8_e4m3fn),
    ],
)
def test_moe_fallback_is_unchanged_outside_sm107_bf16(
    monkeypatch: pytest.MonkeyPatch,
    sm_version: int,
    x_dtype: torch.dtype,
    weight_dtype: torch.dtype,
) -> None:
    monkeypatch.setattr(torch_custom_ops, "get_sm_version", lambda: sm_version)
    runner = _make_runner(x_dtype, weight_dtype)

    assert runner._resolve_fallback_tactic(-1, 1) == -1
    runner.fused_moe_runner.get_tactic_num.assert_not_called()


def test_sm107_bf16_moe_preserves_tuned_tactic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch_custom_ops, "get_sm_version", lambda: 107)
    runner = _make_runner(torch.bfloat16, torch.bfloat16)

    assert runner._resolve_fallback_tactic(4, 1) == 4
    runner.fused_moe_runner.get_tactic_num.assert_not_called()


_NUM_LOCAL_EXPERTS, _HIDDEN_SIZE, _INTER_SIZE = 10, 16, 8


def _expert_weights(weight_dtype: torch.dtype, with_bias: bool) -> list[torch.Tensor | None]:
    """``[fc1 weights, fc1 biases, fc2 weights, fc2 biases]`` of ``_NUM_LOCAL_EXPERTS`` experts."""

    def weight(*shape: int) -> torch.Tensor:
        if weight_dtype == torch.quint4x2:
            # Packed int4 weights, viewed as the op reads them.
            return torch.zeros(*shape, dtype=torch.uint8).view(weight_dtype)
        return torch.zeros(*shape, dtype=weight_dtype)

    return [
        weight(_NUM_LOCAL_EXPERTS, 2 * _INTER_SIZE, _HIDDEN_SIZE),
        torch.zeros(_NUM_LOCAL_EXPERTS, 2 * _INTER_SIZE) if with_bias else None,
        weight(_NUM_LOCAL_EXPERTS, _HIDDEN_SIZE, _INTER_SIZE),
        torch.zeros(_NUM_LOCAL_EXPERTS, _HIDDEN_SIZE) if with_bias else None,
    ]


@pytest.mark.parametrize("tuner_num_local_experts", [None, 8])
def test_fused_moe_hands_the_tuner_layout_to_its_runner(
    monkeypatch: pytest.MonkeyPatch, tuner_num_local_experts: int | None
) -> None:
    """The op gives its runner the tuner layout, one int, and both GEMM choices and the kernel
    every expert's weights."""
    num_tokens = 3
    weights = _expert_weights(torch.bfloat16, with_bias=True)
    choices = []

    class _Tuner:
        def choose_one(self, custom_op, runners, tuning_config, inputs, **kwargs):
            choices.append((custom_op, inputs))
            return runners[0], -1

    runner = Mock()
    runner._resolve_fallback_tactic.side_effect = lambda tactic, gemm_idx: tactic
    runner.fused_moe_runner.run_moe.return_value = torch.zeros(num_tokens, _HIDDEN_SIZE)
    runner_cls = Mock(return_value=runner)
    monkeypatch.setattr(torch_custom_ops.AutoTuner, "get", lambda: _Tuner())
    monkeypatch.setattr(torch_custom_ops, "MoERunner", runner_cls)

    tuner_layout = {}
    if tuner_num_local_experts is not None:
        tuner_layout["tuner_num_local_experts"] = tuner_num_local_experts
    torch.ops.trtllm.fused_moe(
        torch.zeros(num_tokens, _HIDDEN_SIZE),
        torch.zeros(num_tokens, 8, dtype=torch.int32),
        torch.zeros(num_tokens, 8),
        *weights,
        torch.bfloat16,
        [],
        **tuner_layout,
    )

    assert runner_cls.call_args.kwargs["tuner_num_local_experts"] == tuner_num_local_experts
    assert [custom_op for custom_op, _ in choices] == [
        "trtllm::fused_moe::gemm1",
        "trtllm::fused_moe::gemm2",
    ]
    for _, inputs in choices:
        assert all(tuned is expected for tuned, expected in zip(inputs[1:], weights))
    run_args = runner.fused_moe_runner.run_moe.call_args.args
    assert run_args[3] is weights[0] and run_args[5] is weights[2]


def _moe_runner(monkeypatch: pytest.MonkeyPatch, **kwargs: object) -> MoERunner:
    """A BF16 MoERunner whose kernel runner is a mock."""
    monkeypatch.setattr(torch_custom_ops, "get_sm_version", lambda: 100)
    # BF16 activations, weights and output; no quantization mode.
    instance_key = (torch.bfloat16,) * 3 + (False,) * 5
    monkeypatch.setitem(MoERunner.runner_dict, instance_key, Mock())
    return MoERunner(
        x_dtype=torch.bfloat16,
        weight_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        top_k=2,
        tp_size=1,
        tp_rank=0,
        ep_size=4,
        ep_rank=1,
        cluster_size=1,
        cluster_rank=0,
        use_deepseek_fp8_block_scale=False,
        use_w4_group_scaling=False,
        use_int8_woq_per_channel=False,
        use_mxfp8_act_scaling=False,
        min_latency_mode=False,
        use_fused_finalize=True,
        activation_type=ActivationType.Swiglu,
        **kwargs,
    )


@pytest.mark.parametrize(
    "weight_dtype, with_bias", [(torch.bfloat16, True), (torch.quint4x2, False)], ids=["bf16", "w4"]
)
@pytest.mark.parametrize("tuner_num_local_experts", [None, 8])
def test_moe_runner_profiles_the_first_tuner_num_local_experts(
    monkeypatch: pytest.MonkeyPatch,
    tuner_num_local_experts: int | None,
    weight_dtype: torch.dtype,
    with_bias: bool,
) -> None:
    """The profiler routes tokens to every expert of the weights it gets: given a tuner layout, the
    runner profiles views of the first experts, and its tactics are cached apart."""
    runner = _moe_runner(monkeypatch, tuner_num_local_experts=tuner_num_local_experts)
    weights = _expert_weights(weight_dtype, with_bias)

    runner.forward([torch.zeros(4, _HIDDEN_SIZE), *weights], gemm_idx=1, tactic=3)

    profiled = runner.fused_moe_runner.run_gemm_profile.call_args.args[1:5]
    num_profiled = tuner_num_local_experts or _NUM_LOCAL_EXPERTS
    for tuned, weight in zip(profiled, weights):
        if weight is None:
            assert tuned is None
            continue
        assert tuned.data_ptr() == weight.data_ptr()
        assert (tuned.dtype, tuned.shape) == (weight.dtype, (num_profiled, *weight.shape[1:]))
    without_layout = _moe_runner(monkeypatch).unique_id()
    expected_id = without_layout
    if tuner_num_local_experts is not None:
        expected_id += (tuner_num_local_experts,)
    assert runner.unique_id() == expected_id

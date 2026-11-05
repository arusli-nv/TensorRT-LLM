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

from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.custom_ops import torch_custom_ops
from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner

# Tactic selection only: the runner is built without ``__init__``, its kernel
# runner is a mock, and the SM version is monkeypatched. The marker is also
# what makes the file reachable: the CPU stage collects only files that carry
# it.
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


@pytest.mark.parametrize("num_spare,num_loaded", [(2, 8), (1, 9)])
@pytest.mark.parametrize("num_tokens", [1, 64, 3072])
def test_moe_runner_keys_spare_slots_as_loaded_slots(
    num_spare: int, num_loaded: int, num_tokens: int
) -> None:
    """Weights of 10 local slots, ``num_spare`` of them empty, look tactics up under the key of
    the loaded slots alone: the key of the same deployment without fault tolerance."""
    runner = _make_runner(torch.bfloat16, torch.bfloat16)
    for name, value in dict(
        output_dtype=torch.bfloat16,
        top_k=8,
        tp_size=1,
        ep_size=32,
        cluster_size=1,
        enable_alltoall=False,
        use_deepseek_fp8_block_scale=False,
        use_w4_group_scaling=False,
        use_int8_woq_per_channel=False,
        use_mxfp8_act_scaling=False,
        min_latency_mode=False,
        use_fused_finalize=True,
        activation_type=0,
        unpadded_hidden_size=0,
        use_mxfp8_weight_scaling=False,
    ).items():
        setattr(runner, name, value)
    hidden_size, inter_size = 7168, 2048

    def key(num_local: int, num_spare_local: int, with_bias: bool):
        def tensor(*shape):
            return torch.empty(*shape, device="meta")

        inputs = [
            tensor(num_tokens, hidden_size),
            tensor(num_local, 2 * inter_size, hidden_size),
            tensor(num_local, 2 * inter_size) if with_bias else None,
            tensor(num_local, hidden_size, inter_size),
            tensor(num_local, hidden_size) if with_bias else None,
        ]
        tuner_inputs = torch_custom_ops.loaded_slot_moe_tuner_inputs(inputs, num_spare_local)
        tuner = AutoTuner.get()
        return tuner.profiling_cache.get_cache_key(
            "trtllm::fused_moe::gemm1",
            runner,
            tuple(tuner._get_input_sizes(tuner_inputs)),
            MoERunner.tuning_config,
        )

    for with_bias in (False, True):
        assert key(10, num_spare, with_bias) == key(num_loaded, 0, with_bias)
        assert key(10, 0, with_bias) != key(num_loaded, 0, with_bias)


@pytest.mark.parametrize("num_spare", [0, 2])
def test_fused_moe_profiles_the_weights_of_the_loaded_slots(
    monkeypatch: pytest.MonkeyPatch, num_spare: int
) -> None:
    """trtllm::fused_moe hands both GEMM choices the weights and biases of the loaded slots, and runs the
    kernel over every local slot."""
    num_local, num_tokens, hidden_size, inter_size = 10, 3, 16, 8
    choices = []

    class _Tuner:
        def choose_one(self, op, runners, config, inputs, **kwargs):
            choices.append((op, [None if t is None else tuple(t.shape) for t in inputs]))
            return runners[0], -1

    runner = Mock()
    runner._resolve_fallback_tactic.side_effect = lambda tactic, gemm_idx: tactic
    output = torch.zeros(num_tokens, hidden_size)
    runner.fused_moe_runner.run_moe.return_value = output
    monkeypatch.setattr(AutoTuner, "get", lambda: _Tuner())
    monkeypatch.setattr(torch_custom_ops, "MoERunner", Mock(return_value=runner))

    x = torch.zeros(num_tokens, hidden_size)
    fc1 = torch.zeros(num_local, 2 * inter_size, hidden_size)
    fc1_bias = torch.zeros(num_local, 2 * inter_size)
    fc2 = torch.zeros(num_local, hidden_size, inter_size)
    fc2_bias = torch.zeros(num_local, hidden_size)
    selected = torch.zeros(num_tokens, 8, dtype=torch.int32)
    scales = torch.zeros(num_tokens, 8)

    result = torch.ops.trtllm.fused_moe(
        x,
        selected,
        scales,
        fc1,
        fc1_bias,
        fc2,
        fc2_bias,
        torch.float32,
        [],
        ep_size=32,
        ep_rank=5,
        num_spare_local_experts=num_spare,
    )

    loaded = num_local - num_spare
    expected = [
        (num_tokens, hidden_size),
        (loaded, 2 * inter_size, hidden_size),
        (loaded, 2 * inter_size),
        (loaded, hidden_size, inter_size),
        (loaded, hidden_size),
    ]
    assert choices == [
        ("trtllm::fused_moe::gemm1", expected),
        ("trtllm::fused_moe::gemm2", expected),
    ]
    run_args = runner.fused_moe_runner.run_moe.call_args.args
    assert [tuple(t.shape) for t in (run_args[3], run_args[5])] == [
        (num_local, 2 * inter_size, hidden_size),
        (num_local, hidden_size, inter_size),
    ]
    torch.testing.assert_close(result[0], output)


def test_loaded_slot_inputs_slice_quint4x2_weights_by_expert() -> None:
    """W4AFP8 hands the kernel its weights viewed as quint4x2; the loaded slots are a slice of those views."""
    fc1 = torch.zeros(10, 32, 64, dtype=torch.uint8).view(torch.quint4x2)
    fc2 = torch.zeros(10, 128, 16, dtype=torch.uint8).view(torch.quint4x2)
    x = torch.zeros(3, 128)

    inputs = torch_custom_ops.loaded_slot_moe_tuner_inputs([x, fc1, None, fc2, None], 2)

    assert inputs[0] is x and inputs[2] is None and inputs[4] is None
    assert [(t.dtype, tuple(t.shape)) for t in (inputs[1], inputs[3])] == [
        (torch.quint4x2, (8, 32, 64)),
        (torch.quint4x2, (8, 128, 16)),
    ]

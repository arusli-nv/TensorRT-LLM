# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MoE backends read the dispatch receive slots in place exactly as they read a compact copy.

Under WideEP peer-loss recovery NVLinkOneSided hands the experts its fixed receive slots
(source s's token i at row s * capacity + i) instead of a compact copy (row s * runtime_rows + i);
the CuteDSL NVFP4 FC1 gather and the Cutlass row expansion map each token to its slot row.
"""

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.utils import ActivationType
from tensorrt_llm._utils import get_sm_version

_EP, _RUNTIME, _CAPACITY, _TOP_K = 4, 40, 128, 8
_HIDDEN, _INTERM, _EXPERTS, _LOCAL = 1024, 256, 32, 8
_TILE = 128


def _random_sf(*shape: int) -> torch.Tensor:
    # Finite positive E4M3 scale bytes.
    return torch.randint(0x30, 0x40, shape, dtype=torch.uint8, device="cuda")


def _fc1(a, a_sf, b, b_sf, routing, **slots):
    tile_to_group, tile_mn_limit, _, _, _, num_tiles = routing
    return torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell(
        a.view(torch.float4_e2m1fn_x2),
        b.view(torch.float4_e2m1fn_x2),
        a_sf,
        b_sf,
        torch.ones(_LOCAL, dtype=torch.float32, device="cuda"),
        tile_to_group,
        tile_mn_limit,
        routing[3],
        num_tiles,
        torch.ones(1, dtype=torch.float32, device="cuda"),
        num_experts=_EXPERTS,
        top_k=_TOP_K,
        num_local_experts=_LOCAL,
        local_expert_offset=0,
        tile_size=_TILE,
        scaling_vector_size=16,
        activation_type=ActivationType.Swiglu,
        swiglu_limit_scalar=float("inf"),
        **slots,
    )


@pytest.mark.skipif(get_sm_version() not in (100, 103), reason="Blackwell FC1 kernel")
@pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE, reason="cutlass-dsl is not available")
def test_fc1_reads_the_receive_slots_like_a_compact_copy() -> None:
    torch.manual_seed(0)
    tokens = _EP * _RUNTIME
    experts = torch.randint(0, _EXPERTS, (tokens, _TOP_K), dtype=torch.int32, device="cuda")
    experts.view(_EP, _RUNTIME, _TOP_K)[1, _RUNTIME // 2 :] = -1  # rows a source did not send
    weights = torch.rand(tokens, _TOP_K, dtype=torch.float32, device="cuda")
    a = torch.randint(0, 256, (tokens, _HIDDEN // 2), dtype=torch.uint8, device="cuda")
    a_sf = _random_sf(tokens, _HIDDEN // 16)
    b = torch.randint(0, 256, (_LOCAL, 2 * _INTERM, _HIDDEN // 2), dtype=torch.uint8, device="cuda")
    b_sf = _random_sf(_LOCAL, 2 * _INTERM, _HIDDEN // 16)

    # The slots: every source's rows at a fixed offset, the rest of each slot left over garbage.
    slot_a = torch.randint(
        0, 256, (_EP * _CAPACITY, _HIDDEN // 2), dtype=torch.uint8, device="cuda"
    )
    slot_a_sf = _random_sf(_EP * _CAPACITY, _HIDDEN // 16)
    slot_a.view(_EP, _CAPACITY, -1)[:, :_RUNTIME] = a.view(_EP, _RUNTIME, -1)
    slot_a_sf.view(_EP, _CAPACITY, -1)[:, :_RUNTIME] = a_sf.view(_EP, _RUNTIME, -1)

    routing = torch.ops.trtllm.moe_sort(
        token_selected_experts=experts,
        token_final_scales=weights,
        num_experts=_EXPERTS,
        top_k=_TOP_K,
        local_expert_offset=0,
        local_num_experts=_LOCAL,
        tile_tokens_dim=_TILE,
    )
    c, c_sf = _fc1(a, a_sf, b, b_sf, routing)
    slot_c, slot_c_sf = _fc1(
        slot_a, slot_a_sf, b, b_sf, routing, source_tokens=_RUNTIME, slot_tokens=_CAPACITY
    )
    # Rows of the sorted tiles that ran; scales are stored per 128-row block, blocks contiguous.
    valid = int(routing[4].item())
    assert valid > 0
    torch.testing.assert_close(
        slot_c[:valid].view(torch.uint8), c[:valid].view(torch.uint8), rtol=0, atol=0
    )
    valid_sf = valid * _INTERM // 16
    torch.testing.assert_close(
        slot_c_sf.view(torch.uint8)[:valid_sf], c_sf.view(torch.uint8)[:valid_sf], rtol=0, atol=0
    )


def _cutlass_bf16(inputs, experts, weights, w3_w1, w2, use_fused_finalize, **slots):
    (out,) = torch.ops.trtllm.fused_moe(
        inputs,
        experts,
        weights,
        w3_w1,
        None,
        w2,
        None,
        torch.bfloat16,
        quant_scales=[],
        use_fused_finalize=use_fused_finalize,
        **slots,
    )
    return out


@pytest.mark.parametrize("use_fused_finalize", [False, True])
@pytest.mark.parametrize("capacity", [_RUNTIME, _CAPACITY])
def test_cutlass_reads_bf16_hidden_states_in_the_receive_slots(
    use_fused_finalize: bool, capacity: int
) -> None:
    """The Cutlass MoE expands each token's input row from its source's slot."""
    torch.manual_seed(0)
    tokens, hidden, inter_size, num_experts, top_k = _EP * _RUNTIME, 256, 128, 8, 2
    experts = torch.randint(0, num_experts, (tokens, top_k), dtype=torch.int32, device="cuda")
    experts.view(_EP, _RUNTIME, top_k)[1, _RUNTIME // 2 :] = -1  # rows a source did not send
    experts.view(_EP, _RUNTIME, top_k)[3] = -1  # a source that sent nothing
    weights = torch.rand(tokens, top_k, dtype=torch.float32, device="cuda")
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda")
    # The slots hold every source's sent rows at a fixed offset; rows no token reads are NaN.
    slot_x = torch.full((_EP * capacity, hidden), float("nan"), dtype=torch.bfloat16, device="cuda")
    slot_view, x_view = slot_x.view(_EP, capacity, hidden), x.view(_EP, _RUNTIME, hidden)
    for source, sent in enumerate((_RUNTIME, _RUNTIME // 2, _RUNTIME, 0)):
        slot_view[source, :sent] = x_view[source, :sent]
    w3_w1 = (
        torch.randn(num_experts, 2 * inter_size, hidden, dtype=torch.bfloat16, device="cuda") / 16
    )
    w2 = torch.randn(num_experts, hidden, inter_size, dtype=torch.bfloat16, device="cuda") / 16

    expected = _cutlass_bf16(x, experts, weights, w3_w1, w2, use_fused_finalize)
    actual = _cutlass_bf16(
        slot_x,
        experts,
        weights,
        w3_w1,
        w2,
        use_fused_finalize,
        input_source_tokens=_RUNTIME,
        input_slot_tokens=capacity,
    )
    assert not actual.isnan().any()
    if use_fused_finalize:
        # The fused finalize may add a row's contributions in another order.
        torch.testing.assert_close(actual, expected, rtol=1.6e-2, atol=1e-2)
    else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_cutlass_rejects_inconsistent_input_slots() -> None:
    experts = torch.zeros((8, 1), dtype=torch.int32, device="cuda")
    weights = torch.ones((8, 1), dtype=torch.float32, device="cuda")
    x = torch.zeros((12, 128), dtype=torch.bfloat16, device="cuda")
    w3_w1 = torch.zeros((2, 128, 128), dtype=torch.bfloat16, device="cuda")
    w2 = torch.zeros((2, 128, 64), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="one slot of input_slot_tokens rows per source"):
        _cutlass_bf16(
            x, experts, weights, w3_w1, w2, False, input_source_tokens=4, input_slot_tokens=5
        )

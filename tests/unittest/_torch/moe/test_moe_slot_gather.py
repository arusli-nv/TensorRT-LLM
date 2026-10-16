# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FC1 reads the dispatch receive slots in place exactly as it reads a compact copy.

Under WideEP peer-loss recovery NVLinkOneSided hands the CuteDSL NVFP4 experts its fixed
receive slots (source s's token i at row s * capacity + i) instead of a compact copy
(row s * runtime_rows + i); the FC1 gather kernel maps each token to its slot row.
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

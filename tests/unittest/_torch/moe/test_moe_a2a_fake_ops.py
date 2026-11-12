# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check that the fake implementations of the MoE all-to-all ops accept every schema argument."""

import pytest
import torch
from torch._subclasses import FakeTensorMode

import tensorrt_llm  # noqa: F401

pytestmark = pytest.mark.cpu_only


def test_moe_a2a_fake_ops_accept_every_schema_argument():
    """Pass every schema argument positionally, as the NVLink one-sided backend does.

    The trailing arguments are not their defaults, so the dispatcher hands all of them to the fake
    implementations.
    """
    ep_rank, ep_size, max_tokens, top_k, num_experts, hidden = 0, 2, 4, 2, 8, 16
    eplb_local_stats = None
    use_cft_counted_writes = False
    expert_id_payload_index = 0
    invalid_token_expert_id = num_experts
    enable_rank_mask = True
    payload_in_workspace = False
    use_low_precision = False
    with FakeTensorMode():
        token_selected_experts = torch.empty(max_tokens, top_k, dtype=torch.int32)
        payload = torch.empty(max_tokens, hidden)
        workspace = torch.empty(ep_size, 1024, dtype=torch.uint8)
        metainfo = torch.empty(16, dtype=torch.int64)
        active_rank_mask = torch.empty(1, dtype=torch.uint64)
        recv, combine_payload_offset, _ = torch.ops.trtllm.moe_a2a_dispatch(
            token_selected_experts,
            [payload],
            workspace,
            metainfo,
            max_tokens,
            ep_rank,
            ep_size,
            top_k,
            num_experts,
            eplb_local_stats,
            use_cft_counted_writes,
            expert_id_payload_index,
            invalid_token_expert_id,
            enable_rank_mask,
            active_rank_mask,
        )
        output = torch.ops.trtllm.moe_a2a_combine(
            recv[0],
            max_tokens,
            workspace,
            metainfo,
            max_tokens,
            ep_rank,
            ep_size,
            top_k,
            combine_payload_offset,
            payload_in_workspace,
            use_low_precision,
            use_cft_counted_writes,
            enable_rank_mask,
            active_rank_mask,
        )
    assert recv[0].shape == (ep_size, max_tokens, hidden)
    assert output.shape == (max_tokens, hidden)

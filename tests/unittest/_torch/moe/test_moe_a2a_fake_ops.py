# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fake (meta) registrations of the MoE all-to-all ops accept the arguments of the real ops."""

import pytest
import torch
from torch._subclasses import FakeTensorMode

import tensorrt_llm  # noqa: F401 - loads the trtllm ops and registers their fakes

pytestmark = pytest.mark.cpu_only


def test_moe_a2a_fakes_take_every_argument_of_the_ops():
    """torch.compile traces the all-to-all through the fakes, with the arguments a real call passes."""
    ep_size, max_tokens, top_k, hidden = 2, 4, 2, 16
    with FakeTensorMode():
        experts = torch.empty(max_tokens, top_k, dtype=torch.int32)
        payload = torch.empty(max_tokens, hidden)
        workspace = torch.empty(ep_size, 1024, dtype=torch.uint8)
        metainfo = torch.empty(16, dtype=torch.int64)
        recv, _, _ = torch.ops.trtllm.moe_a2a_dispatch(
            experts,
            [payload],
            workspace,
            metainfo,
            max_tokens,
            0,
            ep_size,
            top_k,
            8,
            None,
            use_cft_counted_writes=False,
            expert_id_payload_index=0,
            invalid_token_expert_id=8,
        )
        output = torch.ops.trtllm.moe_a2a_combine(
            recv[0],
            max_tokens,
            workspace,
            metainfo,
            max_tokens,
            0,
            ep_size,
            top_k,
            0,
            False,
            use_low_precision=False,
            use_cft_counted_writes=False,
        )
    assert recv[0].shape == (ep_size, max_tokens, hidden)
    assert output.shape == (max_tokens, hidden)

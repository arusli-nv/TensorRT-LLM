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
"""The fake (meta) registrations of the MoE all-to-all ops accept the arguments of their schemas."""

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
    with FakeTensorMode():
        experts = torch.empty(max_tokens, top_k, dtype=torch.int32)
        payload = torch.empty(max_tokens, hidden)
        workspace = torch.empty(ep_size, 1024, dtype=torch.uint8)
        metainfo = torch.empty(16, dtype=torch.int64)
        # enable_rank_mask and active_rank_mask.
        rank_mask = (True, torch.empty(1, dtype=torch.uint64))
        layout = (workspace, metainfo, max_tokens, ep_rank, ep_size, top_k)
        # EPLB stats, CFT counted writes, expert-id payload index and invalid token expert id.
        dispatch_options = (None, False, 0, num_experts)
        recv, offset, _ = torch.ops.trtllm.moe_a2a_dispatch(
            experts, [payload], *layout, num_experts, *dispatch_options, *rank_mask
        )
        # payload_in_workspace, use_low_precision and use_cft_counted_writes.
        combine_options = (False, False, False)
        output = torch.ops.trtllm.moe_a2a_combine(
            recv[0], max_tokens, *layout, offset, *combine_options, *rank_mask
        )
    assert recv[0].shape == (ep_size, max_tokens, hidden)
    assert output.shape == (max_tokens, hidden)

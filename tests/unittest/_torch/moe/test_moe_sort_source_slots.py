# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The routing sort of the WideEP dispatch receive slots reads a compact copy of their routing.

Under WideEP peer-loss recovery the expert ids and router weights stay in their fixed receive slots
(source s's token i at row s * slot_tokens + i); only the first valid_counts[s] rows of a slot hold
tokens. moe_a2a_compact_source_slots must copy them to the compact list moe_sort reads, with expert
-1 and weight 0 for every other row.
"""

import pytest
import torch

_EXPERTS, _TOP_K = 256, 8


@pytest.mark.parametrize("sources,sent,slot", [(2, 2, 5), (8, 64, 100), (4, 4096, 4100)])
def test_compact_source_slots(sources: int, sent: int, slot: int) -> None:
    gen = torch.Generator(device="cuda").manual_seed(0)
    slot_experts = torch.randint(
        0, _EXPERTS, (sources * slot, _TOP_K), device="cuda", dtype=torch.int32, generator=gen
    )
    slot_weights = torch.rand((sources * slot, _TOP_K), device="cuda", generator=gen)
    counts = torch.randint(0, sent + 1, (sources,), device="cuda", dtype=torch.int32, generator=gen)
    counts[0] = sent  # a full slot
    counts[-1] = 0  # a lost or unfinished source
    # Rows past a source's count hold whatever an earlier round left; they must not be copied.
    received = (torch.arange(sent, device="cuda").view(1, -1) < counts.view(-1, 1)).unsqueeze(-1)
    expected_experts = torch.where(received, slot_experts.view(sources, slot, _TOP_K)[:, :sent], -1)
    expected_weights = torch.where(
        received, slot_weights.view(sources, slot, _TOP_K)[:, :sent], 0.0
    )

    experts, weights = torch.ops.trtllm.moe_a2a_compact_source_slots(
        slot_experts, slot_weights, counts, sent
    )

    torch.testing.assert_close(experts, expected_experts.view(-1, _TOP_K), rtol=0, atol=0)
    torch.testing.assert_close(weights, expected_weights.view(-1, _TOP_K), rtol=0, atol=0)


def test_compact_source_slots_rejects_a_slot_shorter_than_the_tokens_sent() -> None:
    experts = torch.zeros((8, _TOP_K), dtype=torch.int32, device="cuda")
    weights = torch.zeros((8, _TOP_K), dtype=torch.float32, device="cuda")
    counts = torch.zeros(2, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="source_tokens"):
        torch.ops.trtllm.moe_a2a_compact_source_slots(experts, weights, counts, 5)

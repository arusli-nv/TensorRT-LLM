# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EPLB host copies of a subset of the experts, as WideEP fault tolerance keeps them in a ring of nodes."""

import errno
import os
import uuid
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import HostMoeTensorSharer

pytestmark = pytest.mark.cpu_only


def _sharer(local_rank, local_size, expert_count=8, expert_ids=None):
    comm = Mock(Get_rank=Mock(return_value=local_rank), Get_size=Mock(return_value=local_size))
    return HostMoeTensorSharer(0, expert_count, comm, expert_ids)


@pytest.mark.parametrize("expert_count, local_size", [(8, 4), (10, 4), (3, 4)])
def test_sharer_splits_every_expert_by_default_as_before(expert_count, local_size):
    for local_rank in range(local_size):
        sharer = _sharer(local_rank, local_size, expert_count)
        start = local_rank * expert_count // local_size
        end = (local_rank + 1) * expert_count // local_size
        assert sharer.own_expert_ids == list(range(start, end))


def test_sharer_splits_the_experts_it_holds_over_the_node_s_ranks():
    held = [13, 5, 6, 7, 12]
    parts = [_sharer(rank, 2, 16, held).own_expert_ids for rank in range(2)]
    assert parts == [[5, 6], [7, 12, 13]]
    sharer = _sharer(1, 2, 16, held)
    assert sharer.rank_expert_ids(0) == [5, 6]
    # Another rank's experts and experts the node does not hold are only registered by shape.
    for expert in (5, 0):
        sharer.pre_register_host_tensor_with_shape(expert, "w", torch.float32, (4,))
    with pytest.raises(AssertionError):
        sharer.pre_register_host_tensor_with_shape(7, "w", torch.float32, (4,))
    with pytest.raises(AssertionError):
        sharer.share_host_tensor_with_shape(0, "w", torch.zeros(4))


def test_a_full_dev_shm_fails_with_an_error_and_leaves_no_segment(monkeypatch):
    sharer = _sharer(0, 1, 2)
    sharer.set_shared_memory_base_name(f"test_ring_{uuid.uuid4().hex[:8]}")
    for expert in range(2):
        sharer.share_host_tensor_with_shape(expert, "w", torch.zeros(64))

    def full(fd, offset, length):
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    monkeypatch.setattr(os, "posix_fallocate", full)
    with pytest.raises(RuntimeError, match="/dev/shm has no room for the 512 bytes"):
        sharer.finalize_layer_weights()
    assert sharer.own_shm is None
    assert not os.path.exists(f"/dev/shm/{sharer.get_shared_memory_name()}")

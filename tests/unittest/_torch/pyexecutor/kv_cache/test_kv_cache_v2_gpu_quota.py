# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resizing the GPU pool of KVCacheManagerV2 for memory the engine needs while serving."""

from functools import partial
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

MIB = 1 << 20


@pytest.fixture
def manager():
    # 32768 tokens of 8 KiB: a 256 MiB pool, which allocates in 2 MiB units.
    manager = KVCacheManagerV2(
        KvCacheConfig(max_tokens=32768, enable_block_reuse=False),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=4,
        num_kv_heads=4,
        head_dim=128,
        tokens_per_block=32,
        max_seq_len=1024,
        max_batch_size=4,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=tensorrt_llm.bindings.DataType.HALF,
    )
    yield manager
    manager.shutdown()


def test_a_resize_releases_device_memory_and_takes_it_back_with_the_kv_kept(manager):
    (request,) = manager.add_dummy_requests([0], [512])
    (pages,) = manager.get_batch_cache_indices([0], 0)
    manager.get_buffers(0)[pages[-1]].fill_(1.5)
    torch.cuda.synchronize()
    quota = manager.gpu_quota

    # The pool may move pages: every KV cache must be suspended first.
    with pytest.raises(RuntimeError, match="SUSPENDED"):
        manager.resize_gpu_quota(quota - 64 * MIB)
    # A quota that rounds down to the pool's own changes nothing, so it needs no suspension.
    assert manager.resize_gpu_quota(quota + MIB) == quota

    manager.suspend_request(request)
    free = torch.cuda.mem_get_info()[0]
    # Rounded down to the pool's 2 MiB allocation unit.
    assert manager.resize_gpu_quota(quota - 64 * MIB - 1) == quota - 66 * MIB
    assert manager.gpu_quota == quota - 66 * MIB
    assert torch.cuda.mem_get_info()[0] >= free + 66 * MIB

    assert manager.resize_gpu_quota(quota) == quota
    assert manager.resume_request(request)
    (pages,) = manager.get_batch_cache_indices([0], 0)
    assert manager.get_buffers(0)[pages[-1]].eq(1.5).all()


def test_a_pool_cannot_give_up_its_minimum(manager):
    quota = manager.gpu_quota
    with pytest.raises(RuntimeError, match="cannot be resized to 0 bytes"):
        manager.resize_gpu_quota(-1)
    assert manager.gpu_quota == quota


def test_the_executor_resumes_every_cache_when_a_resize_fails(manager):
    """A restore that shrinks the pool, then fails (a peer lost during the remap it includes), leaves every request
    active with its KV, so the executor keeps serving."""
    (request,) = manager.add_dummy_requests([0], [512])
    (pages,) = manager.get_batch_cache_indices([0], 0)
    manager.get_buffers(0)[pages[-1]].fill_(1.5)
    torch.cuda.synchronize()
    executor = SimpleNamespace(
        kv_cache_manager=manager,
        active_requests=[request],
        model_engine=SimpleNamespace(cuda_graph_runner=None),
        _read_kv_cache_capacity=lambda: None,
    )
    executor._suspend_padding_dummies_for_rebalance = partial(
        PyExecutor._suspend_padding_dummies_for_rebalance, executor
    )
    executor._resume_padding_dummies_after_rebalance = partial(
        PyExecutor._resume_padding_dummies_after_rebalance, executor
    )

    with pytest.raises(RuntimeError, match="a peer exited"):
        with PyExecutor._kv_pools_suspended(executor) as kv_cache_manager:
            kv_cache_manager.resize_gpu_quota(kv_cache_manager.gpu_quota - 64 * MIB)
            raise RuntimeError("a peer exited during the remap")
    assert manager.is_request_active(request.py_request_id)
    (pages,) = manager.get_batch_cache_indices([0], 0)
    assert manager.get_buffers(0)[pages[-1]].eq(1.5).all()
    manager.free_resources(request)

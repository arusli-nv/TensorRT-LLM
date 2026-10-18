# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rolling back a discarded step on real KV cache managers (WideEP peer-loss recovery)."""

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState, SamplingConfig
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _real_kv_cache_manager(enable_block_reuse=False):
    return KVCacheManagerV2(
        kv_cache_config=KvCacheConfig(max_tokens=1024, enable_block_reuse=enable_block_reuse),
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=2,
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=32,
        max_seq_len=256,
        max_batch_size=4,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.HALF,
        vocab_size=32000,
    )


def _prompt_request(request_id, chunk_size):
    request = LlmRequest(
        request_id=request_id,
        max_new_tokens=8,
        input_tokens=list(range(96)),
        sampling_config=SamplingConfig(1),
        is_streaming=False,
    )
    request.context_chunk_size = chunk_size
    return request


def _prepare_step(manager, context_requests, generation_requests):
    """Allocate one step as the executor does, up to the forward."""
    batch = ScheduledRequests()
    for request in context_requests:
        batch.append_context_request(request)
    batch.generation_requests = list(generation_requests)
    # The V2 scheduler allocates the step. It sizes each chunk after prepare_context, which spans a first
    # chunk to the end of the prompt when block reuse is on.
    for request in context_requests:
        chunk_size = request.context_chunk_size
        assert manager.prepare_context(request)
        request.context_chunk_size = min(chunk_size, request.context_remaining_length)
        assert manager.resize_context(request, request.context_chunk_size)
    for request in generation_requests:
        assert manager.try_allocate_generation(request)
    manager.prepare_resources(batch)
    return batch


def _revert(manager, batch):
    """Revert the step prepared last."""
    manager.revert_prepare_resources(batch, manager.context_capacities(batch))


def _commit_step(manager, batch):
    """Apply a step's results as the executor does after the forward."""
    for request in batch.context_requests:
        request.move_to_next_context_chunk()
    manager.update_context_resources(batch)
    for request in batch.context_requests:
        if request.context_remaining_length == 0:
            request.state = LlmRequestState.GENERATION_IN_PROGRESS
            request.add_new_token(0, 0)
    for request in batch.generation_requests:
        request.add_new_token(0, 0)
    manager.update_resources(batch)


def _kv_state(manager):
    per_request = {
        request_id: (kv_cache.capacity, kv_cache.history_length)
        for request_id, kv_cache in manager.kv_cache_map.items()
    }
    return manager.get_kv_cache_stats().free_num_blocks, per_request


def test_revert_prepare_resources_round_trip():
    """A reverted step leaves the KV cache as before, and the step runs again unchanged."""
    manager = _real_kv_cache_manager()
    try:
        chunked = _prompt_request(1, chunk_size=32)
        generating = _prompt_request(2, chunk_size=96)
        _commit_step(manager, _prepare_step(manager, [chunked, generating], []))
        assert generating.state == LlmRequestState.GENERATION_IN_PROGRESS
        # A committed chunk leaves the next chunk unsized; the scheduler sizes it.
        chunked.context_chunk_size = 32
        fresh = _prompt_request(3, chunk_size=32)
        before = _kv_state(manager)

        position = chunked.context_current_position
        batch = _prepare_step(manager, [fresh, chunked], [generating])
        prepared = _kv_state(manager)
        assert prepared != before
        _revert(manager, batch)
        assert _kv_state(manager) == before
        assert chunked.context_current_position == position

        # The scheduler sizes every context chunk again before the step runs.
        fresh.context_chunk_size = chunked.context_chunk_size = 32
        _prepare_step(manager, [fresh, chunked], [generating])
        assert _kv_state(manager) == prepared
    finally:
        manager.shutdown()


def test_reverted_context_is_not_offered_for_reuse():
    """KV a discarded step wrote never becomes a reusable prefix; committed KV does."""
    manager = _real_kv_cache_manager(enable_block_reuse=True)
    try:
        _revert(manager, _prepare_step(manager, [_prompt_request(1, 96)], []))
        probe = _prompt_request(2, 96)
        _prepare_step(manager, [probe], [])
        assert probe.context_current_position == 0

        committed = _prompt_request(3, 96)
        _commit_step(manager, _prepare_step(manager, [committed], []))
        manager.free_resources(committed)
        control = _prompt_request(4, 96)
        _prepare_step(manager, [control], [])
        assert control.context_current_position > 0
    finally:
        manager.shutdown()


def test_reverted_first_chunk_that_reused_a_prefix_runs_again_unchanged():
    """Reverting a first chunk that matched a reused prefix rewinds its cursor; the rerun matches again."""
    manager = _real_kv_cache_manager(enable_block_reuse=True)
    try:
        committed = _prompt_request(1, 96)
        _commit_step(manager, _prepare_step(manager, [committed], []))
        manager.free_resources(committed)
        reusing = _prompt_request(2, 96)
        before = _kv_state(manager)
        batch = _prepare_step(manager, [reusing], [])
        matched, prepared = reusing.context_current_position, _kv_state(manager)
        assert matched > 0
        _revert(manager, batch)
        assert reusing.context_current_position == 0 and reusing.prepopulated_prompt_len == 0
        assert _kv_state(manager) == before
        _prepare_step(manager, [reusing], [])
        assert reusing.context_current_position == matched and _kv_state(manager) == prepared
    finally:
        manager.shutdown()


def test_two_overlapped_steps_revert_newest_first():
    """The overlap scheduler prepares a step before the one before it commits; both revert exactly.

    The context request runs two chunks and the generation request two tokens, and the executor advances
    the context cursor between the steps, as its request-state update does before a step commits.
    """
    manager = _real_kv_cache_manager()
    try:
        chunked = _prompt_request(1, chunk_size=32)
        generating = _prompt_request(2, chunk_size=96)
        _commit_step(manager, _prepare_step(manager, [chunked, generating], []))
        chunked.context_chunk_size = 32
        before = _kv_state(manager)

        older = _prepare_step(manager, [chunked], [generating])
        older_capacities = manager.context_capacities(older)
        chunked.move_to_next_context_chunk()
        chunked.context_chunk_size = 32
        newer = _prepare_step(manager, [chunked], [generating])
        newer_capacities = manager.context_capacities(newer)

        chunked.context_current_position = 64
        manager.revert_prepare_resources(newer, newer_capacities)
        chunked.context_current_position = 32
        manager.revert_prepare_resources(older, older_capacities)

        assert _kv_state(manager) == before
    finally:
        manager.shutdown()


@pytest.mark.parametrize("overlap", [False, True], ids=["sequential", "overlap"])
def test_discarded_later_chunk_keeps_the_committed_chunks_reusable(overlap):
    """Discarding a later chunk keeps the blocks earlier chunks committed for reuse and frees its own.

    With the overlap scheduler the later chunk is prepared before the earlier chunk's step is accepted
    and commits its blocks.
    """
    manager = _real_kv_cache_manager(enable_block_reuse=True)
    try:
        chunked = _prompt_request(1, chunk_size=32)
        first = _prepare_step(manager, [chunked], [])
        chunked.move_to_next_context_chunk()
        chunked.context_chunk_size = 32
        if not overlap:
            manager.update_context_resources(first)
        free_before = manager.get_kv_cache_stats().free_num_blocks
        later = _prepare_step(manager, [chunked], [])
        later_capacities = manager.context_capacities(later)
        if overlap:
            manager.update_context_resources(first)
        kv_cache = manager.kv_cache_map[chunked.py_request_id]
        assert kv_cache.num_committed_tokens == 32

        manager.revert_prepare_resources(later, later_capacities)
        assert manager.get_kv_cache_stats().free_num_blocks == free_before
        assert kv_cache.capacity == 32 and kv_cache.num_committed_tokens == 32
        assert chunked.context_current_position == 32

        # Only the first chunk is offered for reuse.
        probe = _prompt_request(2, 96)
        probe_step = _prepare_step(manager, [probe], [])
        assert probe.context_current_position == 32
        _revert(manager, probe_step)
        assert manager.get_kv_cache_stats().free_num_blocks == free_before

        # The discarded chunk runs again and commits its blocks.
        chunked.context_chunk_size = 32
        _commit_step(manager, _prepare_step(manager, [chunked], []))
        assert kv_cache.num_committed_tokens == 64
        probe = _prompt_request(3, 96)
        _prepare_step(manager, [probe], [])
        assert probe.context_current_position == 64
    finally:
        manager.shutdown()

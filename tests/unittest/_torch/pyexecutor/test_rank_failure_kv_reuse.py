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
"""Tests that a context chunk's KV becomes reusable only after its step is accepted under rank failure recovery.

The tests use the real KV cache manager V2 with block reuse and chunked prefill, in the overlap executor's order.
"""

from functools import partial
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.py_executor import BatchState, PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.conversation_params import ConversationParams
from tensorrt_llm.llmapi.llm_args import BlockReuseConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

PROMPT_LEN = 96
CHUNK = 32  # one KV block


@pytest.fixture
def manager(request):
    policy = getattr(request, "param", "all_reusable")
    manager = KVCacheManagerV2(
        kv_cache_config=KvCacheConfig(
            max_tokens=1024,
            enable_block_reuse=True,
            block_reuse_config=BlockReuseConfig(policy=policy),
        ),
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=2,
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=CHUNK,
        max_seq_len=256,
        max_batch_size=4,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.HALF,
        vocab_size=32000,
    )
    yield manager
    manager.shutdown()


def _executor(manager, fault_tolerance):
    """Return the executor's context update and commit paths over ``manager``."""
    executor = SimpleNamespace(
        rank_failure_recovery=object() if fault_tolerance else None,
        kv_cache_manager=manager,
        enable_joint_kv_cache_reuse=False,
        resource_manager=SimpleNamespace(get_resource_manager=lambda _type: None),
        _wait_for_model_engine_input_copy=lambda: None,
    )
    executor.update = partial(PyExecutor._update_v2_context_resources, executor)
    executor.accept = partial(PyExecutor._commit_v2_context_blocks, executor)
    return executor


def _request(request_id, first_token):
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=8,
        input_tokens=list(range(first_token, first_token + PROMPT_LEN)),
        sampling_config=SamplingConfig(1),
        is_streaming=False,
    )


def _schedule_chunk(manager, request, chunk_size=CHUNK):
    """Allocate the request's next context chunk as the V2 scheduler does before the forward."""
    batch = ScheduledRequests()
    batch.append_context_request(request)
    if request.context_current_position == 0:
        assert manager.prepare_context(request)
    request.context_chunk_size = min(chunk_size, request.context_remaining_length)
    assert manager.resize_context(request, request.context_chunk_size)
    manager.prepare_resources(batch)
    return batch


def _launch(executor, batch):
    """Launch the forward, update the caches as the executor does, and return the step."""
    for request in batch.context_requests:
        request.move_to_next_context_chunk()
    context_commits = [] if executor.rank_failure_recovery is not None else None
    executor.update(batch, context_commits)
    return BatchState(scheduled_requests=batch, sample_state=None, context_commits=context_commits)


def _reused_prefix(manager, request_id, first_token):
    """Return how much of the prompt starting at ``first_token`` a new request can reuse."""
    probe = _request(request_id, first_token)
    _schedule_chunk(manager, probe, PROMPT_LEN)
    manager.free_resources(probe)
    return probe.context_current_position


@pytest.mark.parametrize("fault_tolerance", [False, True], ids=["off", "on"])
@pytest.mark.parametrize("accepted", [False, True], ids=["discarded", "accepted"])
def test_chunk_reusable_after_step_accepted(manager, fault_tolerance, accepted):
    executor = _executor(manager, fault_tolerance)
    request = _request(1, first_token=1000)
    step = _launch(executor, _schedule_chunk(manager, request))
    if accepted and fault_tolerance:  # the executor accepts a step's commits only in this case
        executor.accept(step)
    # A discarded step stands for a rank failure, which frees the request for recompute.
    manager.free_resources(request)
    # Without fault tolerance the chunk is committed at its launch.
    reusable = accepted or not fault_tolerance
    assert _reused_prefix(manager, 2, first_token=1000) == (CHUNK if reusable else 0)


def test_accept_skips_next_chunk_in_flight(manager):
    executor = _executor(manager, fault_tolerance=True)
    request = _request(1, first_token=1000)
    first = _launch(executor, _schedule_chunk(manager, request))
    # The next chunk is scheduled and launched before the first one's step is accepted.
    second = _launch(executor, _schedule_chunk(manager, request))
    assert request.context_current_position == 2 * CHUNK
    executor.accept(first)
    assert _reused_prefix(manager, 2, first_token=1000) == CHUNK
    executor.accept(second)
    assert _reused_prefix(manager, 3, first_token=1000) == 2 * CHUNK
    manager.free_resources(request)


@pytest.mark.parametrize("manager", ["per_conversation"], indirect=True)
@pytest.mark.parametrize("fault_tolerance", [False, True], ids=["off", "on"])
def test_drop_plan_covers_committed_prompt(manager, fault_tolerance):
    """Save a conversation's drop plan over every block its finished context committed.

    With fault tolerance the plan is saved when the step is accepted, because nothing is committed at the launch.
    """
    executor = _executor(manager, fault_tolerance)
    request = _request(1, first_token=1000)
    request.py_conversation_params = ConversationParams(conversation_id="conversation")
    committed_at_plan = []
    save_drop_plan = manager.conversation_manager.save_drop_plan

    def record(req, kv_cache):
        committed_at_plan.append(kv_cache.num_committed_tokens)
        save_drop_plan(req, kv_cache)

    manager.conversation_manager.save_drop_plan = record
    step = _launch(executor, _schedule_chunk(manager, request, PROMPT_LEN))
    if fault_tolerance:
        assert committed_at_plan == []
        executor.accept(step)
    assert committed_at_plan == [PROMPT_LEN]
    manager.free_resources(request)

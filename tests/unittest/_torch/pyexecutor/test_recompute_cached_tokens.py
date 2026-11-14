# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cache hits of a request that the executor pauses and computes again.

The recompute prefill reuses KV cache blocks that the request computed itself,
so its prefix hit says nothing about the client's prompt: ``cached_tokens``
keeps the value from the first prefill. The KV cache-aware attention-DP router
weighs the request by the prefill it runs now, so it uses the recompute's hit.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from _torch.pyexecutor.llm_request_factory import make_llm_request

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import KVCacheAwareADPRouter
from tensorrt_llm.bindings import executor as trtllm

pytestmark = pytest.mark.cpu_only

PROMPT = list(range(1, 17))
GENERATED = [100, 101, 102]
TOKENS_PER_BLOCK = 4


def _prefill(req, reuse: int) -> None:
    """What the KV cache manager and ``DecoderInputPreparer`` do at a prefill."""
    req.set_prepopulated_prompt_len(reuse, TOKENS_PER_BLOCK)
    req.cached_tokens = req.context_current_position


def _recomputed_request(first_reuse: int):
    req = make_llm_request(
        1, trtllm.RequestType.REQUEST_TYPE_CONTEXT_AND_GENERATION, input_token_ids=PROMPT
    )
    _prefill(req, first_reuse)
    for token in GENERATED:
        req.add_new_token(token, 0)

    PyExecutor._pause_recompute_request(SimpleNamespace(), req)
    assert req.prompt_len == len(PROMPT) + len(GENERATED)

    # The recompute prefill finds every full block of the prompt and the
    # generated tokens in the cache.
    _prefill(req, len(PROMPT))
    return req


@pytest.mark.parametrize("first_reuse", [0, 8], ids=["cold_cache", "prefix_hit"])
def test_recompute_keeps_the_first_prefill_cached_tokens(first_reuse: int) -> None:
    req = _recomputed_request(first_reuse)

    assert req.cached_tokens == first_reuse


@pytest.mark.parametrize("first_reuse", [0, 8], ids=["cold_cache", "prefix_hit"])
def test_kv_cache_aware_router_counts_the_recompute_prefill(first_reuse: int) -> None:
    req = _recomputed_request(first_reuse)
    dist = MagicMock(tp_rank=0, has_cp_helix=False)
    dist.mapping.dp_size = 1
    router = KVCacheAwareADPRouter(dist=dist, kv_cache_manager=MagicMock())

    state = router.create_rank_state([req], [])

    assert state.num_active_tokens == len(GENERATED)

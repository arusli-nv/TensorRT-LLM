# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The streams that CUDA graphs are captured on."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.deepseekv41_decoder_graph import PassGraphPool
from tensorrt_llm._torch.modules.multi_stream_utils import (
    maybe_execute_in_parallel,
    with_multi_stream,
)
from tensorrt_llm._torch.nccl_window_graph import nccl_window_graph_capture
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner, EncoderCUDAGraphRunner
from tensorrt_llm._torch.utils import decode_stream_priority

# PyTorch's stream pool holds 32 streams per priority and device; twice that covers it.
_POOL_COVER = 64
# The priority of a small engine's decode graphs (decode_stream_priority).
_SMALL_ENGINE_TOKENS = 128


def _capture_by_default(graph):
    """nccl_window_graph_capture without a stream: decode graphs of most engines, piecewise
    graphs, multimodal encoder graphs and the autotuner's graphs."""
    return nccl_window_graph_capture(graph, torch.cuda.graph_pool_handle())


def _capture_high_priority_decode(graph):
    runner = object.__new__(CUDAGraphRunner)
    runner._capture_stream = None
    runner.config = SimpleNamespace(
        prioritize_decode_graphs=True, max_num_tokens=_SMALL_ENGINE_TOKENS
    )
    stream = runner._priority_capture_stream()
    return nccl_window_graph_capture(graph, torch.cuda.graph_pool_handle(), stream=stream)


def _capture_encoder(graph):
    runner = object.__new__(EncoderCUDAGraphRunner)
    runner._capture_stream = None
    stream = runner._get_capture_stream()
    return nccl_window_graph_capture(graph, torch.cuda.graph_pool_handle(), stream=stream)


def _capture_pass_graph(graph):
    stream = PassGraphPool().stream(torch.device("cuda", torch.cuda.current_device()))
    return torch.cuda.graph(graph, stream=stream)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize(
    "capture, aux_priority",
    [
        (_capture_by_default, 0),
        (_capture_high_priority_decode, decode_stream_priority(_SMALL_ENGINE_TOKENS)),
        (_capture_encoder, 0),
        (_capture_pass_graph, 0),
    ],
    ids=["default", "high_priority_decode", "encoder", "deepseek_v41_pass_graph"],
)
def test_capture_forks_onto_every_pool_aux_stream(capture, aux_priority):
    """Every branch forked onto an aux stream from PyTorch's pool leaves the capture stream.

    Models take their aux streams from the pool, at the capture's priority too; none may be the
    capture stream.
    """
    aux_streams = [torch.cuda.Stream(priority=aux_priority) for _ in range(_POOL_COVER)]
    events = (torch.cuda.Event(), torch.cuda.Event())
    x = torch.zeros(16, device="cuda")
    fork_streams = []

    def fork():
        fork_streams.append(torch.cuda.current_stream().cuda_stream)
        return x + 2

    graph = torch.cuda.CUDAGraph()
    with with_multi_stream(True), capture(graph):
        capturing_stream = torch.cuda.current_stream().cuda_stream
        for aux_stream in aux_streams:
            maybe_execute_in_parallel(lambda: x + 1, fork, *events, aux_stream)

    assert len(fork_streams) == _POOL_COVER
    assert capturing_stream not in fork_streams


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_encoder_runners_capture_on_their_own_streams():
    """Each encoder runner keeps a capture stream of its own, shared with no other runner.

    A capture stream shared by two graph sets couples them through stream-keyed cuBLAS scratch
    (EncoderCUDAGraphRunner's comment on ``_capture_stream``).
    """
    first, second = object.__new__(EncoderCUDAGraphRunner), object.__new__(EncoderCUDAGraphRunner)
    first._capture_stream = second._capture_stream = None
    assert first._get_capture_stream().cuda_stream != second._get_capture_stream().cuda_stream

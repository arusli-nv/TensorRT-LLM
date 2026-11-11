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
"""The executor side of WideEP fault tolerance, against a fake membership, exchange and executor."""

import asyncio
import collections
import contextlib
import errno
import json
import os
import pickle
import queue
import re
import runpy
import signal
import socket
import subprocess
import sys
import threading
import time
import warnings
from concurrent.futures import Future
from datetime import timedelta
from functools import partial
from multiprocessing import shared_memory
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch

from tensorrt_llm import _utils
from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.distributed import membership, mnnvl_memory
from tensorrt_llm._torch.distributed.host_exchange import MnnvlDist, MnnvlRows, TPHostExchange
from tensorrt_llm._torch.distributed.membership import (
    DISCONNECTED_KEY,
    HOST_WEIGHTS_PREFIX_KEY,
    JOIN_SECRET_ENV,
    PROGRESS_TIMEOUT_S,
    RENDEZVOUS_ENV,
    TRANSACTION_KEY,
    MembershipAgent,
    MembershipRecord,
    MembershipState,
    Reply,
    ResumeState,
    SignedStore,
    StoreGroup,
    TransactionSummary,
)
from tensorrt_llm._torch.distributed.mnnvl_live_group import MnnvlMappingError, MnnvlPeerHandleError
from tensorrt_llm._torch.distributed.startup_record import StartupRecord, startup_record
from tensorrt_llm._torch.moe.fused_moe.eplb_expert_restore import (
    SHARED_MEMORY_DIR,
    DeploymentHostCopies,
    EplbExpertRestore,
)
from tensorrt_llm._torch.moe.fused_moe.impl_blocks import MoEEplbWeightLayoutMixin
from tensorrt_llm._torch.moe.fused_moe.moe_comm_timeout_guard import SERVING_TIMEOUT_ENV
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import MoeLoadBalancer
from tensorrt_llm._torch.pyexecutor import hang_detector
from tensorrt_llm._torch.pyexecutor import rank_failure_recovery as recovery_module
from tensorrt_llm._torch.pyexecutor.context_logits_plan import store_context_logits
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner
from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
    ExecutorRequestQueue,
    RequestQueueItem,
)
from tensorrt_llm._torch.pyexecutor.hang_detector import HangDetector
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmResponse, SamplingConfig
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.rank_failure_recovery import (
    AGREED_STARTUP_KEY,
    WORKER_ARGS_KEY,
    FaultToleranceHealth,
    RankFailureRecovery,
    StartupMemory,
    check_supported,
    prepare_model,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager, ResourceManagerType
from tensorrt_llm._torch.pyexecutor.sampler.sampler_common import SampleType
from tensorrt_llm._torch.pyexecutor.scheduler import FCFSWaitingQueue
from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import DefaultADPRouter, RankState
from tensorrt_llm._torch.pyexecutor.seq_slot_manager import SeqSlotManager
from tensorrt_llm._utils import FAULT_TOLERANCE_RANK_ENV, fault_tolerance_mode
from tensorrt_llm.executor.base_worker import BaseWorker
from tensorrt_llm.executor.executor import GenerationExecutor
from tensorrt_llm.executor.proxy import GenerationExecutorProxy
from tensorrt_llm.executor.utils import (
    RANK_FAILURE_CODE,
    ErrorResponse,
    RankFailureError,
    RequestError,
)
from tensorrt_llm.executor.worker_process_monitor import capture_worker_process_identity
from tensorrt_llm.llmapi import LLM, SamplingParams
from tensorrt_llm.llmapi.llm_args import AttentionDpConfig, KvCacheConfig, TorchLlmArgs
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

WORLD = 4
FAILED = 2
SURVIVORS = 0b1011
MIB = 1 << 20
# A rank's KV cache pool and free device memory once the deployment serves.
STARTUP_MEMORY = StartupMemory(kv_cache_quota=64 * MIB)
FREE_BYTES = 1 << 30


# Attention-DP routing over the ranks that serve.


def _item(req_id, target=None, relax=False, num_tokens=4):
    item = MagicMock()
    item.id = req_id
    item.child_req_ids = None
    item.request.num_input_tokens = num_tokens
    item.request.client_id = 100 + req_id
    item.request.py_scheduling_params = (
        None if target is None else Mock(attention_dp_rank=target, attention_dp_relax=relax)
    )
    return item


def _router():
    dist = MagicMock(tp_rank=0, tp_size=WORLD, has_cp_helix=False)
    return DefaultADPRouter(dist=dist)


def _survivor_states(active=0):
    return [RankState(rank=r, num_active_requests=active) for r in range(WORLD) if r != FAILED]


def test_gather_omits_absent_ranks():
    router = _router()
    states = [RankState(rank=r, num_active_requests=r).serialize() for r in range(WORLD)]
    states[FAILED] = None
    router.dist.tp_allgather.return_value = states
    assert [s.rank for s in router.gather_all_rank_states([])] == [0, 1, 3]


def test_route_requests_skips_absent_ranks():
    routed, expected = _router().route_requests(
        _survivor_states(), [_item(i) for i in range(6)], max_num_active_requests=4
    )
    assert set(routed) == {0, 1, 3}
    assert sorted(len(items) for items in routed.values()) == [2, 2, 2]
    assert expected == 2


def test_relaxed_request_pinned_to_absent_rank_is_balanced():
    routed, _ = _router().route_requests(
        _survivor_states(), [_item(0, target=FAILED, relax=True)], max_num_active_requests=4
    )
    assert sum(len(items) for items in routed.values()) == 1


def test_cuda_graph_agreement_skips_absent_ranks_and_keeps_the_captured_sizes():
    """The survivors keep replaying the CUDA graphs captured at startup: their graph vote and the batch size they pad
    to come from the serving ranks' rows, so the lost rank's empty row (no graph, batch size 0) neither sends them to
    eager mode nor changes the batch size they pad to."""
    sizes = [3, 4, None, 2]  # the batch each rank scheduled; rank 2 is lost
    exchange = TPHostExchange.__new__(TPHostExchange)
    exchange.nranks, exchange.live_mask, exchange._words = WORLD, SURVIVORS, 16
    exchange._masks = {}

    def one_round(chunk, length, readers, recipients=None, idle=False):
        """Every live rank r sent [can run a graph, sizes[r], FULL]; the absent rank's slot is stale."""
        readers = [r for r in range(WORLD) if readers >> r & 1]
        packets = np.full((WORLD, 7), 0xDEAD, dtype=np.uint64)
        for r in readers:
            packets[r, 0] = length
            row = np.array([1, sizes[r], SampleType.FULL.value], dtype=np.int64)
            packets[r, 1:] = row.view(np.uint32)
        return np.array([length if r in readers else 0 for r in range(WORLD)], np.uint32), packets

    exchange._round = one_round
    dist = MnnvlDist.__new__(MnnvlDist)
    dist.mapping = SimpleNamespace(tp_size=WORLD)
    dist._host_exchange = exchange
    dist._live_rows_index = (None, None)
    runner = CUDAGraphRunner.__new__(CUDAGraphRunner)
    runner.clear = lambda **_kwargs: None  # nothing was captured
    runner.enabled = True
    runner.config = SimpleNamespace(
        enable_attention_dp=True,
        mapping=SimpleNamespace(tp_size=WORLD),
        dist=dist,
        enable_in_graph_sampling=False,
    )
    batch = SimpleNamespace(batch_size=sizes[0])
    # Every serving rank pads to the widest serving rank's batch.
    assert runner._get_padding_batch_size(batch, True) == 4
    sizes[:] = [4, 4, None, 4]
    batch.batch_size = 4
    assert runner._agree_graph_shape(batch, True, frozenset()) == (True, SampleType.FULL)


def test_attention_dp_balance_waits_for_the_serving_ranks_only():
    """attention_dp_config.enable_balance holds context requests until every rank has some: a rank that left the
    group has none, and must not hold the survivors' context. Its counters move alike on every serving rank."""
    row = np.array([1, 4, 5, 4], dtype=np.int64).view(
        np.uint32
    )  # context, generation, tokens, real generation
    exchange = TPHostExchange.__new__(TPHostExchange)
    exchange.nranks, exchange.live_mask, exchange._words = WORLD, SURVIVORS, 16
    exchange._masks = {}

    def one_round(chunk, length, readers, recipients=None, idle=False):
        readers = [r for r in range(WORLD) if readers >> r & 1]
        packets = np.full((WORLD, 1 + row.size), 0xDEAD, dtype=np.uint64)
        for r in readers:
            packets[r, 0] = length
            packets[r, 1:] = row
        return np.array([length if r in readers else 0 for r in range(WORLD)], np.uint32), packets

    exchange._round = one_round
    dist = MnnvlDist.__new__(MnnvlDist)
    dist.mapping = SimpleNamespace(tp_size=WORLD)
    dist._host_exchange = exchange
    dist._live_rows_index = (None, None)
    executor = object.__new__(PyExecutor)
    executor.dist = dist
    executor.max_batch_size = 64
    executor.attention_dp_enable_balance = True
    executor.attention_dp_time_out_iters = 60
    executor.attention_dp_batching_wait_iters = 0
    executor.attention_dp_min_generation_requests = 0
    executor.attention_dp_low_occupancy_timeout_iters = 0
    executor.adp_ctx_waiting_iters_count = 0
    executor.adp_ctx_batching_wait_iters_count = 0
    context = [Mock(get_tokens=Mock(return_value=[0]))]
    generation = [Mock(is_attention_dp_dummy=False) for _ in range(4)]

    assert executor._balance_adp_requests(context, generation) == context
    assert executor.adp_ctx_waiting_iters_count == 0


# The startup check, before the model loads.


def _startup_config():
    """What the startup check reads beyond what the LLM arguments' validation checked."""
    llm_args = SimpleNamespace(
        attention_dp_config=None,
        kv_cache_config=SimpleNamespace(use_kv_cache_manager_v2=True),
    )
    mapping = SimpleNamespace(rank=0, moe_ep_size=WORLD, world_size=WORLD)
    return llm_args, mapping, MnnvlDist.__new__(MnnvlDist)


@pytest.fixture
def join_secret(monkeypatch):
    monkeypatch.setenv(JOIN_SECRET_ENV, "ab" * 32)
    monkeypatch.setenv(RENDEZVOUS_ENV, "head-node:29600")


@pytest.mark.parametrize("enable_balance", [False, True])
def test_a_supported_configuration_passes_the_startup_check(join_secret, enable_balance):
    llm_args, mapping, dist = _startup_config()
    llm_args.attention_dp_config = AttentionDpConfig(enable_balance=enable_balance)
    check_supported(llm_args, mapping, dist)


@pytest.mark.parametrize("rendezvous", [None, "head-node", "head-node:port"])
def test_a_rank_needs_the_rendezvous_address(join_secret, monkeypatch, rendezvous):
    """A replacement started by a launcher finds the deployment only there."""
    if rendezvous is None:
        monkeypatch.delenv(RENDEZVOUS_ENV)
    else:
        monkeypatch.setenv(RENDEZVOUS_ENV, rendezvous)
    with pytest.raises(
        ValueError, match=f"rendezvous address in {RENDEZVOUS_ENV}.*trtllm-llmapi-launch"
    ):
        recovery_module._launched_rank()


def test_a_rank_needs_its_identity_from_its_node_supervisor(join_secret, monkeypatch):
    """Only a node supervisor starts a rank, saying whether it is a replacement: a lost rank or node comes back by its
    node's launch, never by a rank started on its own."""
    monkeypatch.setattr(_utils, "_wide_ep_fault_tolerance", True)
    monkeypatch.delenv(FAULT_TOLERANCE_RANK_ENV, raising=False)
    with pytest.raises(ValueError, match="'<rank>,<world size>,<GPUs per node>,<replacement>'"):
        recovery_module._launched_rank()


@pytest.mark.parametrize("force_cft", ["0", "1"])
def test_the_startup_check_needs_the_fence_path(join_secret, monkeypatch, force_cft):
    """Bound execution runs on the fence path, so forcing CFT counted writes is refused before the model loads."""
    from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import FORCE_CFT_ENV

    monkeypatch.setenv(FORCE_CFT_ENV, force_cft)
    if force_cft == "0":
        check_supported(*_startup_config())
        return
    with pytest.raises(ValueError, match=f"fence path.*{FORCE_CFT_ENV}") as error:
        check_supported(*_startup_config())
    assert "enable_wide_ep_fault_tolerance=False" in str(error.value)


@pytest.mark.parametrize("margin_s, supported", [(-1, False), (0, True)])
def test_the_startup_check_needs_an_all_to_all_timeout_of_twice_the_hang_timeout(
    join_secret, monkeypatch, margin_s, supported
):
    """The kernels that wait on a hung rank are released when the authority excludes it: they must not time out
    first."""
    monkeypatch.setenv(SERVING_TIMEOUT_ENV, str(int(2 * PROGRESS_TIMEOUT_S) + margin_s))
    if supported:
        check_supported(*_startup_config())
    else:
        with pytest.raises(ValueError, match=f"{SERVING_TIMEOUT_ENV} of at least"):
            check_supported(*_startup_config())


@pytest.mark.parametrize("secret", [None, "not hex"])
def test_a_rank_needs_the_join_secret(join_secret, monkeypatch, secret):
    if secret is None:
        monkeypatch.delenv(JOIN_SECRET_ENV, raising=False)
    else:
        monkeypatch.setenv(JOIN_SECRET_ENV, secret)
    with pytest.raises(
        ValueError, match=f"hex join secret in {JOIN_SECRET_ENV}.*trtllm-llmapi-launch"
    ) as error:
        recovery_module._launched_rank()
    assert "openssl rand -hex 32" in str(error.value)
    assert "enable_wide_ep_fault_tolerance=False" in str(error.value)


@pytest.mark.parametrize("secret", ["", "ab" * 31])
def test_a_rank_rejects_a_join_secret_shorter_than_32_bytes(join_secret, monkeypatch, secret):
    """The secret keys the signatures of the store's values, which the ranks unpickle: an empty one, as `openssl rand
    -hex 32` yields without openssl, would let any writer to the store run code in every rank."""
    monkeypatch.setenv(JOIN_SECRET_ENV, secret)
    with pytest.raises(
        ValueError, match=f"join secret of at least 32 bytes .* in {JOIN_SECRET_ENV}"
    ) as error:
        recovery_module._launched_rank()
    assert "openssl rand -hex 32" in str(error.value)
    assert "enable_wide_ep_fault_tolerance=False" in str(error.value)


def test_an_empty_join_secret_counts_as_unset(monkeypatch):
    """The server then chooses one for the workers it starts, as when none is exported."""
    monkeypatch.setenv(JOIN_SECRET_ENV, "")
    monkeypatch.setenv(RENDEZVOUS_ENV, "head-node:29600")
    recovery_module.ensure_rendezvous()
    assert len(bytes.fromhex(os.environ[JOIN_SECRET_ENV])) == 32


def test_an_unsupported_configuration_fails_the_startup_check(join_secret):
    llm_args, mapping, dist = _startup_config()
    mapping.moe_ep_size = WORLD // 2
    with pytest.raises(
        ValueError, match="moe_expert_parallel_size equal to the world size"
    ) as error:
        check_supported(llm_args, mapping, dist)
    assert "enable_wide_ep_fault_tolerance=False" in str(error.value)


@pytest.mark.parametrize(
    "lacks, fix",
    [
        ("tp", "tensor_parallel_size equal to the world size"),
        ("cp", "context_parallel_size=1"),
        ("mnnvl", "multi-node NVLink (MNNVL) domain"),
        ("fabric", "fabric handles (GB200-class aarch64 nodes with IMEX"),
        ("mpi", "the MPI orchestrator (TLLM_DISABLE_MPI unset)"),
        # Distributed.get chose another implementation though every requirement is met.
        (None, "host collectives over MNNVL memory (MnnvlDist)"),
    ],
)
def test_the_startup_check_names_what_the_host_collectives_over_mnnvl_lack(
    join_secret, monkeypatch, lacks, fix
):
    from tensorrt_llm._torch.distributed import communicator

    llm_args, _, _ = _startup_config()
    mapping = SimpleNamespace(
        rank=0,
        moe_ep_size=WORLD,
        world_size=WORLD,
        enable_attention_dp=True,
        pp_size=1,
        cp_size=2 if lacks == "cp" else 1,
        tp_size=WORLD // 2 if lacks in ("tp", "cp") else WORLD,
    )
    monkeypatch.setattr(
        mnnvl_memory.MnnvlMemory, "supports_mnnvl", staticmethod(lambda: lacks != "mnnvl")
    )
    monkeypatch.setattr(
        MnnvlRows,
        "uses_fabric_handles",
        classmethod(lambda cls, dev_id: lacks != "fabric"),
    )
    monkeypatch.setattr(communicator.torch.cuda, "current_device", lambda: 0)
    if lacks == "mpi":
        monkeypatch.setenv("TLLM_DISABLE_MPI", "1")
    with pytest.raises(ValueError, match=re.escape(fix)) as error:
        check_supported(llm_args, mapping, Mock())
    assert "enable_wide_ep_fault_tolerance=False" in str(error.value)


@pytest.mark.parametrize("setting, preferred", [("auto", "V2"), ("auto", None), (False, "V2")])
def test_only_a_kv_cache_manager_resolved_to_v2_passes_the_startup_check(
    join_secret, setting, preferred
):
    """use_kv_cache_manager_v2="auto" resolves from the model's preference as the model loads, after the LLM
    arguments' validation and before the startup check: the startup check alone accepts V2 only, explicit or
    resolved."""
    from tensorrt_llm.llmapi.llm_utils import _resolve_kv_cache_manager_v2_auto

    llm_args = TorchLlmArgs(
        model="dummy",
        enable_wide_ep_fault_tolerance=True,
        enable_attention_dp=True,
        moe_config={"load_balancer": {"num_slots": 8}},
        kv_cache_config=KvCacheConfig(use_kv_cache_manager_v2=setting),
    )
    model_cls = type(
        "Model",
        (),
        {"get_preferred_kv_cache_manager_version": classmethod(lambda cls, config=None: preferred)},
    )
    _resolve_kv_cache_manager_v2_auto(llm_args, model_cls)
    _, mapping, dist = _startup_config()
    if setting == "auto" and preferred == "V2":  # as DeepSeek-V3 prefers
        check_supported(llm_args, mapping, dist)
        return
    with pytest.raises(ValueError, match="KV cache manager V2") as error:
        check_supported(llm_args, mapping, dist)
    assert "enable_wide_ep_fault_tolerance=False" in str(error.value)


# Host weights: every node keeps every expert's in /dev/shm.

_EXPERTS = 8
_LOCAL_RANKS = 2


def _host_tensor_sharer(local_rank=0, local_size=_LOCAL_RANKS):
    sharer = DeploymentHostCopies(
        0,
        _EXPERTS,
        Mock(Get_rank=Mock(return_value=local_rank), Get_size=Mock(return_value=local_size)),
    )
    sharer.set_shared_memory_base_name(f"wide_ep_ft_test_{os.getpid()}_{time.monotonic_ns()}")
    return sharer


class _MoeWeights(MoEEplbWeightLayoutMixin, torch.nn.Module):
    """A MoE layer's weights of 3 local slots, of 128 KiB and 4 bytes per slot (aligned to 256 bytes each), and a
    weight that is not per slot."""

    expert_size_per_partition = 3

    def __init__(self, sharer):
        super().__init__()
        self.layer_load_balancer = SimpleNamespace(host_tensor_sharer=sharer)
        self.w3_w1_weight = torch.nn.Parameter(torch.empty(3, 64, 1024, dtype=torch.float16))
        self.fc31_alpha = torch.nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.fc31_input_scale = torch.nn.Parameter(torch.empty((), dtype=torch.float32))


# Per expert its two aligned weights, and per local rank's segment its fingerprint's aligned block.
_NODE_BYTES = _EXPERTS * (128 * 1024 + 256) + _LOCAL_RANKS * 256


@pytest.fixture
def shm_free(monkeypatch):
    """Sets the free bytes of /dev/shm that the startup check reads."""
    monkeypatch.setattr(
        _utils, "_wide_ep_fault_tolerance", True
    )  # segments start with a fingerprint
    free = {}
    monkeypatch.setattr(
        os, "statvfs", lambda path: SimpleNamespace(f_bavail=free["bytes"], f_frsize=1)
    )
    return free


def test_the_host_weights_take_what_finalize_layer_weights_lays_out(monkeypatch):
    monkeypatch.setattr(_utils, "_wide_ep_fault_tolerance", True)
    sharer = _host_tensor_sharer(local_size=1)
    weights = _MoeWeights(sharer)
    for expert in range(_EXPERTS):
        for name, param in weights.slot_parameters():
            sharer.share_host_tensor_with_shape(expert, name, torch.zeros_like(param[0]))
    sharer.finalize_layer_weights()
    try:
        slot_bytes = [
            param[0].numel() * param.element_size() for _, param in weights.slot_parameters()
        ]
        path = os.path.join(SHARED_MEMORY_DIR, sharer.get_shared_memory_name())
        assert os.stat(path).st_size == sharer.node_bytes(slot_bytes)
    finally:
        sharer.own_shm.close()
        sharer.own_shm.unlink()


@pytest.mark.parametrize("shortfall", [0, 1])
def test_the_startup_check_needs_room_in_dev_shm_for_the_host_weights(shm_free, shortfall):
    """Docker and Kubernetes give a container 64 MB of /dev/shm by default."""
    model = torch.nn.ModuleList([_MoeWeights(_host_tensor_sharer())])
    shm_free["bytes"] = _NODE_BYTES - shortfall
    if not shortfall:
        recovery_module._check_host_weights_fit(model)
        return
    with pytest.raises(ValueError, match=f"needs .* \\({_NODE_BYTES} bytes\\) more") as error:
        recovery_module._check_host_weights_fit(model)
    message = str(error.value)
    assert "--shm-size" in message and "emptyDir" in message and "medium: Memory" in message
    assert "enable_wide_ep_fault_tolerance=False" in message


def test_the_host_weights_already_in_dev_shm_need_no_more_room(shm_free):
    """A replacement rank reattaches the segments of its node, which take their room already."""
    sharer = _host_tensor_sharer()
    model = torch.nn.ModuleList([_MoeWeights(sharer)])
    segment = shared_memory.SharedMemory(
        name=sharer.get_shared_memory_name(1), create=True, size=4096
    )
    try:
        segment.buf[:] = b"\x01" * 4096
        held = sharer.node_bytes_held()
        assert held >= 4096
        shm_free["bytes"] = _NODE_BYTES - held
        recovery_module._check_host_weights_fit(model)
        shm_free["bytes"] -= 1
        with pytest.raises(ValueError, match="/dev/shm"):
            recovery_module._check_host_weights_fit(model)
    finally:
        segment.close()
        segment.unlink()


def _supported_model(monkeypatch, state=None):
    """A model the model check accepts, whose MoE layers it binds to ``state``; returns the model and the binding."""
    from tensorrt_llm._torch.moe.fused_moe.communication.execution_state import MoeExecutionState

    for_model = Mock(return_value=state)
    monkeypatch.setattr(MoeExecutionState, "for_model", for_model)
    monkeypatch.setattr(recovery_module, "_moe_execution_state", None)
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.config_utils.is_hybrid_linear", lambda config: False
    )
    model = SimpleNamespace(model_config=SimpleNamespace(pretrained_config=SimpleNamespace()))
    return model, for_model


def test_the_model_check_checks_the_room_for_the_host_weights(monkeypatch):
    model, _ = _supported_model(monkeypatch)
    fit = Mock(side_effect=ValueError("/dev/shm"))
    monkeypatch.setattr(recovery_module, "_check_host_weights_fit", fit)
    with pytest.raises(ValueError, match="/dev/shm"):
        prepare_model(model)
    fit.assert_called_once_with(model)


def test_the_model_check_binds_the_moe_layers(monkeypatch):
    """Before warmup captures the CUDA graphs that read the state's records; recovery drives that state."""
    state = Mock()
    model, for_model = _supported_model(monkeypatch, state)
    monkeypatch.setattr(recovery_module, "_check_host_weights_fit", Mock())
    prepare_model(model)
    for_model.assert_called_once_with(model)
    assert recovery_module._moe_execution_state is state


# The flag reaches every process, and requests it does not support are rejected.


@pytest.mark.parametrize("enabled", [False, True])
def test_a_worker_takes_fault_tolerance_mode_from_its_args(monkeypatch, enabled):
    monkeypatch.setattr(_utils, "_wide_ep_fault_tolerance", not enabled)
    llm_args = TorchLlmArgs(
        model="dummy",
        enable_wide_ep_fault_tolerance=enabled,
        enable_attention_dp=True,
        moe_config={"load_balancer": {"num_slots": WORLD}},
    )
    worker = BaseWorker(llm_args=llm_args)
    worker.doing_shutdown = True  # nothing to shut down at exit
    assert fault_tolerance_mode() is enabled


def test_fault_tolerance_is_off_by_default():
    assert TorchLlmArgs(model="dummy").enable_wide_ep_fault_tolerance is False


@pytest.mark.parametrize(
    "params, rejected",
    [
        (SamplingParams(), False),
        (SamplingParams(n=2, temperature=0.8), True),
        (SamplingParams(best_of=2, temperature=0.8), True),
        (SamplingParams(use_beam_search=True), True),
    ],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_requests_with_several_sequences_are_rejected_under_fault_tolerance(
    params, rejected, enabled
):
    llm = SimpleNamespace(
        args=SimpleNamespace(
            backend="pytorch",
            enable_chunked_prefill=True,
            max_num_tokens=None,
            speculative_config=None,
            enable_wide_ep_fault_tolerance=enabled,
        ),
        _check_one_model_speculative_sampling=lambda params: None,
    )
    if rejected and enabled:
        with pytest.raises(
            RequestError,
            match="does not support requests with n > 1, best_of > 1 or beam search",
        ):
            LLM._check_arguments(llm, 8, params, is_gen_only=False)
    else:
        LLM._check_arguments(llm, 8, params, is_gen_only=False)


# Membership changes on the executor thread.


class FakeExchange:
    def __init__(self):
        self.interrupted = None
        self.on_interrupt = None
        self.waiting_for = 0
        self.stopped_epoch = 6
        self.epoch = 40
        self.rows = SimpleNamespace(stale_rows=lambda gather: frozenset(range(WORLD)))
        self.installs = []
        self.admits = []

    def install(self, gather, live_mask, epoch, fresh_rows):
        assert fresh_rows == frozenset(range(WORLD))
        self.installs.append((gather(b"row-mine"), live_mask, epoch))

    def admit(self, live_mask, epoch):
        self.admits.append((live_mask, epoch))


class FakeStore:
    """The SignedStore calls of a rank, in memory."""

    def __init__(self):
        self.values = {}
        self.store = self

    def set(self, key, value):
        self.values[key] = value

    def get(self, key):
        return self.values[key]

    def get_if_set(self, key):
        return self.values.get(key)

    def add(self, key, amount):
        self.values[key] = self.values.get(key, 0) + amount
        return self.values[key]

    def check(self, keys):
        return all(key in self.values for key in keys)

    def multi_get(self, keys):
        return [self.values[key] for key in keys]

    def delete_key(self, key):
        return self.values.pop(key, None) is not None


class FakeAgent:
    def __init__(self):
        self.reports = []

    connected = True

    def report(self, **changes):
        self.reports.append(changes)

    def stop(self):
        pass


def _executor(rank=0):
    return SimpleNamespace(
        dist=MagicMock(rank=rank, world_size=WORLD, tp_size=WORLD),
        iter_counter=5,
        hang_detector=Mock(paused=Mock(return_value=False), timeout=300),
        executor_request_queue=MagicMock(request_queue=queue.Queue()),
        waiting_queue=[],
        request_accumulated=[],
        active_requests=[],
        previous_batch=None,
        drafter=None,
        use_spec_decode=False,
        perf_manager=MagicMock(),
        model_engine=SimpleNamespace(moe_load_balancer=None),
        _pending_transfer_responses=[],
        _free_request_resources=MagicMock(),
        _pause_recompute_request=MagicMock(),
    )


def _recovery(rank=0):
    moe = MagicMock()
    moe.aborted.return_value = False
    moe.waited_rounds.return_value = {}
    moe.workspace_bytes.return_value = 0
    moe.stale_workspace_rows.return_value = frozenset(range(WORLD))
    recovery = RankFailureRecovery(
        _executor(rank), FakeExchange(), moe, FakeStore(), "host:1", STARTUP_MEMORY
    )
    recovery._agent = FakeAgent()
    executor = recovery._executor

    def retire():  # main's retire of the step in flight: it accepts (or drops) the step, then applies it
        if executor.previous_batch is not None:
            recovery.drop_invalid_step()
        executor.previous_batch = None

    executor._retire_inflight_batch_for_control = Mock(side_effect=retire)
    return recovery


def _record(state, generation=1, live_mask=SURVIVORS, resume=ResumeState()):
    return MembershipRecord(
        generation=generation,
        state=state,
        live_mask=live_mask,
        resume=resume,
        incarnations=(0,) * WORLD,
    )


def test_propose_aborts_the_all_to_all_and_wakes_rank_0():
    recovery = _recovery()
    assert not recovery._membership_changing()
    recovery._on_record(_record(MembershipState.PROPOSE))
    recovery._moe.request_abort.assert_called_once()
    recovery._executor.executor_request_queue.wake.assert_called_once()
    assert recovery._membership_changing()


@pytest.mark.parametrize(
    "live_mask, joining_mask, aborts",
    [
        (0b1111, 0b0100, False),  # a replacement for lost rank 2 joins: no process leaves
        (0b1011, 0b0010, True),  # a replacement takes live rank 1 over: its old process leaves
        (0b1101, 0b0100, True),  # rank 1 is lost while rank 2's replacement joins
    ],
)
def test_an_admission_does_not_abort_the_steps_in_flight_a_removal_does(
    live_mask, joining_mask, aborts
):
    """A change that only adds replacements keeps every process: the live ranks finish their steps in flight, and the
    host exchange stops them for the change. One that drops a process aborts the all-to-all, which may wait for it.
    Either way, rank 0 stops waiting for requests, to take part in the change."""
    recovery = _recovery()
    recovery._on_record(_record(MembershipState.COMMITTED))  # rank 2 was lost earlier
    proposal = _record(MembershipState.PROPOSE, generation=2, live_mask=live_mask).model_copy(
        update=dict(joining_mask=joining_mask)
    )
    recovery._on_record(proposal)
    assert recovery._moe.request_abort.called == aborts
    recovery._executor.executor_request_queue.wake.assert_called_once()
    # An admission stops the host exchange only after a forward vote (see the tests below).
    assert recovery._membership_changing() == aborts


def _vote(recovery, batch_size=4, peers=((3, 0, 0), (5, 0, 0), (2, 0, 0))):
    """``recovery``'s forward vote, with the other ranks' rows ``peers`` ([batch size, holds for a pending change,
    retries a give-back]); returns the row it sent and the batch sizes."""
    sent = []

    def gather(row):
        sent.append(row)
        return np.array([row, *peers])

    batch_sizes = recovery.forward_vote(batch_size, gather)
    return sent[0], batch_sizes.tolist()


def test_an_admission_does_not_stop_the_host_exchange_after_an_idle_vote():
    """Every rank idle votes an empty batch, so the vote launches no forward; yet a rank that then sees the PROPOSE
    that adds a replacement must not stop in the poll of the next round: rank 0, which has not seen it, may finish that
    round and wait in the next vote for this rank. It stops once a vote had a rank hold its steps for the change: that
    rank stops at the start of the next round, without sending, so no rank completes it."""
    recovery = _recovery(rank=1)
    recovery._on_record(_record(MembershipState.COMMITTED))  # rank 2 was lost earlier
    assert _vote(recovery, batch_size=0, peers=((0, 0, 0), (0, 0, 0), (0, 0, 0))) == (
        [0, 0, 0],
        [0, 0, 0, 0],
    )
    recovery._on_record(_joining_record(MembershipState.PROPOSE))
    assert recovery.holds_steps() and not recovery._membership_changing()
    assert _vote(recovery) == ([0, 1, 0], [0, 3, 5, 2])  # it holds its steps for the change
    assert recovery._membership_changing()


def test_an_admission_stops_the_host_exchange_once_a_rank_held_for_it_in_a_vote():
    """A rank that voted before it saw the PROPOSE learns from the vote that another rank held for it; it stops once it
    has seen it. While it holds its steps, rank 0 does not wait for requests. A change that drops a process stops the
    host exchange at once."""
    recovery = _recovery()
    recovery._on_record(_record(MembershipState.COMMITTED))  # rank 2 was lost earlier
    wake = recovery._executor.executor_request_queue.wake
    assert _vote(recovery, peers=((0, 1, 0), (5, 0, 0), (2, 0, 0))) == (
        [4, 0, 0],
        [4, 0, 5, 2],
    )
    wake.assert_not_called()
    assert not recovery._membership_changing()  # it has not seen the change yet
    recovery._on_record(_joining_record(MembershipState.PROPOSE))
    assert recovery._membership_changing()
    wake.reset_mock()
    _vote(recovery)
    wake.assert_called_once()
    recovery._on_record(_record(MembershipState.COMMITTED, generation=3))
    _vote(recovery)
    recovery._on_record(_record(MembershipState.PROPOSE, generation=4, live_mask=0b0011))
    assert recovery._membership_changing()


def test_after_a_commit_an_admission_waits_for_a_vote_in_which_a_rank_holds_for_it(monkeypatch):
    """The vote that stopped the live ranks for the last change had a rank hold for it; once that change is committed,
    the next admission still waits for a vote of its own, as on a joining rank, which took part in none."""
    monkeypatch.setattr(recovery_module.torch.cuda, "synchronize", lambda: None)
    recovery = _resuming_recovery()
    recovery._steps_quiet = True  # the vote before the live ranks stopped for rank 2's loss
    recovery.before_iteration()
    assert _replies(recovery) == [Reply.INSTALLED] and not recovery.holds_steps()
    recovery._on_record(_joining_record(MembershipState.PROPOSE))
    assert not recovery._membership_changing()
    _vote(recovery)
    assert recovery._membership_changing()


def test_membership_change_replies_and_returns_once_the_host_exchange_is_installed():
    recovery = _recovery(rank=1)
    for r in (0, 3):
        recovery._store.set(f"host_exchange_rows/1/0/{r}", pickle.dumps(f"rows-{r}".encode()))
    recovery._on_record(_record(MembershipState.PROPOSE))
    done = threading.Thread(target=recovery._change_membership, daemon=True)
    done.start()
    _wait_until(lambda: recovery._agent.reports)
    assert recovery._agent.reports[-1] == {
        "reply": Reply.READY,
        "reply_generation": 1,
        "stopped_at": ResumeState(iteration=5, epoch=6),
    }
    install = _record(MembershipState.INSTALL, resume=ResumeState(iteration=9, epoch=8))
    recovery._on_record(install)
    # INSTALLED follows at the resume point (before_iteration), so the interrupted round can finish first.
    done.join(5)
    assert not done.is_alive()
    assert all(report.get("reply") != Reply.INSTALLED for report in recovery._agent.reports)
    # The INSTALL record carries the READY payloads: the reports the agent keeps sending drop this rank's, and a
    # joining rank's readiness, which stood for its READY. Then the rank reports the install step it completed.
    assert recovery._agent.reports[-2:] == [
        {"ready_payload": None, "joining_ready": False},
        {"install_steps": 1},
    ]
    # The record's epoch, which the authority took from the READY replies.
    assert recovery._exchange.installs == [
        ([b"rows-0", b"row-mine", None, b"rows-3"], SURVIVORS, 8)
    ]
    assert recovery._resume_after == 9 and recovery.holds_steps()
    # The installed change no longer stops the host exchange.
    assert not recovery._membership_changing()
    recovery._on_record(_record(MembershipState.PROPOSE, generation=2, live_mask=0b0011))
    assert recovery._membership_changing()


def test_a_peer_lost_during_the_install_waits_for_the_next_record(monkeypatch):
    recovery = _recovery(rank=1)
    recovery._exchange.install = Mock(side_effect=MnnvlPeerHandleError([3]))
    recovery._next_record = Mock(
        side_effect=[_record(MembershipState.INSTALL), _record(MembershipState.ABORT)]
    )
    monkeypatch.setattr(recovery_module.os, "_exit", Mock(side_effect=SystemExit))
    with pytest.raises(SystemExit):  # the ABORT that follows, not the error
        recovery._change_membership()
    recovery._exchange.install.assert_called_once()
    recovery_module.os._exit.assert_called_once_with(1)


@pytest.mark.parametrize(
    "error, reply",
    [
        (MnnvlMappingError([1], "MnnvlRows failed on live ranks:\nrank 1: ..."), [Reply.FAILED]),
        # Only the failed rank replies FAILED: the authority aborts, or goes on without a joining rank.
        (MnnvlMappingError([3], "MnnvlRows failed on live ranks:\nrank 3: ..."), []),
        (RuntimeError("host exchange stopped more than one round before"), [Reply.FAILED]),
    ],
)
def test_an_install_of_the_host_exchange_that_fails_reaches_the_authority(
    monkeypatch, error, reply
):
    recovery = _recovery(rank=1)
    recovery._exchange.install = Mock(side_effect=error)
    recovery._next_record = Mock(
        side_effect=[_record(MembershipState.INSTALL), _record(MembershipState.ABORT)]
    )
    monkeypatch.setattr(recovery_module.os, "_exit", Mock(side_effect=SystemExit))
    with pytest.raises(SystemExit):  # the ABORT that follows, not the error
        recovery._change_membership()
    assert _replies(recovery) == reply


@pytest.mark.parametrize(
    "record, status",
    [
        (_record(MembershipState.ABORT), 1),
        (_record(MembershipState.PROPOSE, live_mask=0b0101), 1),  # rank 1 is excluded
        (_record(MembershipState.SHUTDOWN), 0),
    ],
)
def test_change_ends_the_process(monkeypatch, record, status):
    def exit_(code):
        raise SystemExit(code)

    monkeypatch.setattr(recovery_module.os, "_exit", exit_)
    recovery = _recovery(rank=1)
    recovery._on_record(record)
    with pytest.raises(SystemExit) as exited:
        recovery._change_membership()
    assert exited.value.code == status


@pytest.mark.parametrize("completed, status", [(True, 0), (False, 1)])
def test_stop_exits_with_the_loop_status_without_the_mpi_teardown(monkeypatch, completed, status):
    """Also without a membership change: a rank may have failed since the last record, and the MPI teardown at exit
    would wait for it forever."""

    def exit_(code):
        raise SystemExit(code)

    monkeypatch.setattr(recovery_module.os, "_exit", exit_)
    recovery = _recovery(rank=1)
    recovery._record = _record(MembershipState.SHUTDOWN)
    with pytest.raises(SystemExit) as exited:
        recovery.stop(completed)
    assert exited.value.code == status


def test_idle_follows_the_hang_detector():
    detector = HangDetector(timeout=600)
    detector.start()
    try:
        assert detector.paused()  # before the first checkpoint
        detector.checkpoint()
        assert not detector.paused()
        with detector.pause():
            assert detector.paused()
        assert not detector.paused()
    finally:
        detector.stop()

    recovery = _recovery()
    paused = recovery._executor.hang_detector.paused
    for is_paused in (True, True, False):
        paused.return_value = is_paused
        recovery._on_tick()
        assert recovery._agent.reports[-1]["idle"] == is_paused


def test_a_rank_reports_whom_it_waits_for_and_the_rounds_it_waits_in():
    recovery = _recovery()
    recovery._exchange.waiting_for = 1 << 1
    recovery._moe.waited_rounds.return_value = {3: 9}
    recovery._on_tick()
    report = recovery._agent.reports[-1]
    assert report["waiting_for"] == 0b1010 and report["waited_rounds"] == ((3, 9),)
    # The host exchange waits for rank 1 in its current round.
    assert report["waited_epochs"] == ((1, recovery._exchange.epoch),)


@pytest.mark.parametrize("paused_before", [False, True])
def test_main_s_hang_detector_is_paused_while_a_rank_takes_part_in_a_membership_change(
    paused_before,
):
    """The authority bounds each phase of the change; the detector's kill would end a survivor of a long one. A change
    that interrupts a wait the detector is paused for (the request broadcast) leaves it paused for that wait."""
    detector = HangDetector(timeout=600)
    detector.start()
    try:
        if not paused_before:
            detector.checkpoint()
        recovery = _recovery(rank=1)
        recovery._executor.hang_detector = detector
        seen = []
        # The committed record: no change to take part in.
        recovery._next_record = lambda record: seen.append(detector.paused()) or recovery._record
        recovery._finish_install = lambda record: seen.append(detector.paused())
        recovery._change_membership()
        assert detector.paused() == paused_before
        recovery._installed = _record(MembershipState.INSTALL)
        recovery._resume_after = recovery._executor.iter_counter - 1
        recovery.before_iteration()
        assert detector.paused() == paused_before
        assert seen == [True, True]
    finally:
        detector.stop()


def _mark_arrived(store, prefix, rank):
    """Mark ``rank``'s value of the call under ``prefix`` set, as its StoreGroup does."""
    store.store.add(StoreGroup._counter(prefix, rank), 1 << rank % 63)


@pytest.mark.timeout(60)
def test_a_gather_through_the_store_reports_the_ranks_whose_values_it_waits_for():
    """The authority then takes only those ranks for the end of the wait-for chain, not the ranks waiting for them."""
    recovery = _recovery(rank=1)
    record = _record(MembershipState.INSTALL)
    recovery._record = record
    gather = recovery._live_group(record, "rows").allgather
    recovery._store.set("rows/1/0/0", pickle.dumps("rows-0"))
    _mark_arrived(recovery._store, "rows/1/0", 0)
    gathered = []
    gathering = threading.Thread(target=lambda: gathered.append(gather("rows-1")), daemon=True)
    gathering.start()
    _wait_until(lambda: recovery._waiting_in_store)
    recovery._on_tick()
    assert recovery._agent.reports[-1]["waiting_for"] == 1 << 3
    recovery._store.set("rows/1/0/3", pickle.dumps("rows-3"))
    gathering.join(10)
    assert gathered == [["rows-0", "rows-1", None, "rows-3"]]
    assert recovery._waiting_in_store == 0


@pytest.mark.timeout(60)
def test_a_value_not_signed_with_the_join_secret_counts_as_not_set():
    """As everywhere in the membership store: a forged value neither ends the gather nor the change; the signed value
    that overwrites it does."""
    raw = FakeStore()
    store = SignedStore(raw, b"join-secret")
    recovery = _recovery(rank=1)
    recovery._store = store
    record = recovery._record = _record(MembershipState.INSTALL)
    gather = recovery._live_group(record, "rows").allgather
    store.set("rows/1/0/0", pickle.dumps("rows-0"))
    _mark_arrived(store, "rows/1/0", 0)
    raw.set("rows/1/0/3", bytes(32) + pickle.dumps("forged"))
    gathered = []
    gathering = threading.Thread(target=lambda: gathered.append(gather("rows-1")), daemon=True)
    gathering.start()
    _wait_until(lambda: recovery._waiting_in_store == 1 << 3)
    assert gathering.is_alive()
    store.set("rows/1/0/3", pickle.dumps("rows-3"))
    gathering.join(10)
    assert gathered == [["rows-0", "rows-1", None, "rows-3"]]


@pytest.mark.timeout(60)
def test_a_gather_reads_the_ranks_it_waits_for_from_arrival_counters_once_per_report_interval(
    monkeypatch,
):
    """At 256 ranks, one store call per missing rank and wait would load the authority's store with every rank
    waiting for a slow one: the waiting ranks read one counter per 63 ranks, once per report interval."""
    world = 256
    recovery = _recovery(rank=1)
    recovery.world_size = world
    record = recovery._record = MembershipRecord(
        generation=1, state=MembershipState.INSTALL, live_mask=(1 << world) - 1
    )
    adds = []
    checked = []  # per store check, the number of keys it checks
    store = recovery._store
    add, check = store.add, store.check
    monkeypatch.setattr(store, "add", lambda *args: (adds.append(args), add(*args))[1])
    monkeypatch.setattr(store, "check", lambda keys: (checked.append(len(keys)), check(keys))[1])
    gather = recovery._live_group(record, "rows").allgather
    for r in range(world):
        if r not in (1, 200):
            store.set(f"rows/1/0/{r}", pickle.dumps(r))
            _mark_arrived(store, "rows/1/0", r)
    adds.clear()
    checked.clear()
    gathered = []
    gathering = threading.Thread(target=lambda: gathered.append(gather(1)), daemon=True)
    started_s = time.monotonic()
    gathering.start()
    try:
        _wait_until(lambda: recovery._waiting_in_store == 1 << 200)
        time.sleep(1.0)
        waited_s = time.monotonic() - started_s
        # A poll for the values checks all of them in one store call, never a rank's alone.
        assert checked and set(checked) == {world}
        assert len(checked) <= 10 + waited_s / membership.STORE_POLL_S
        # This rank's own arrival, then 5 counters per read, at most once per report interval.
        assert len(adds) <= 1 + 5 * (waited_s / membership.REPORT_INTERVAL_S + 1)
    finally:
        store.set("rows/1/0/200", pickle.dumps(200))
        _mark_arrived(store, "rows/1/0", 200)
        gathering.join(10)
    assert gathered == [list(range(world))]


# Dropping steps and continuing requests.


def _request(request_id, dummy=False):
    return SimpleNamespace(
        py_request_id=request_id, is_dummy=dummy, is_finished=False, py_seq_slot=request_id
    )


def _batch(requests, context_rows=()):
    return SimpleNamespace(
        scheduled_requests=SimpleNamespace(all_requests=lambda: list(requests)),
        sample_state=SimpleNamespace(sampler_event=MagicMock()),
        gpu_forward_events_from_perf_pool=False,
        context_rows=list(context_rows),
    )


@pytest.mark.parametrize("aborted", [False, True])
def test_drop_invalid_step_follows_the_abort(aborted):
    recovery = _recovery()
    recovery._moe.aborted.return_value = aborted
    request = _request(1)
    recovery._executor.previous_batch = batch = _batch([request])
    recovery.drop_invalid_step()
    batch.sample_state.sampler_event.synchronize.assert_called_once()
    assert (recovery._executor.previous_batch is None) == aborted
    assert recovery._interrupted == ([request] if aborted else [])


_PROMPT_LEN = 10
_VOCAB = 31


def _context_rows_request():
    """A request that returns its context logits and prompt logprobs, kept in host memory."""
    return LlmRequest(
        request_id=1,
        max_new_tokens=4,
        input_tokens=[(7 * i) % _VOCAB for i in range(_PROMPT_LEN)],
        sampling_config=SamplingConfig(1),
        is_streaming=False,
        return_context_logits=True,
        prompt_logprobs=2,
        return_logits_device_memory=False,
    )


def _store_chunk(request, size, logits):
    """The forward of the request's next context chunk of ``size`` positions; ``logits`` holds a row per position."""
    request.context_chunk_size = size
    begin = request.context_current_position
    store_context_logits(request, logits[begin : begin + size])
    request.move_to_next_context_chunk()


@pytest.mark.parametrize("reused", [0, 4], ids=["no_reuse", "reused_accepted_chunk"])
def test_a_discarded_step_drops_its_context_rows_and_the_recompute_stores_them_again(
    monkeypatch, reused
):
    """Context logits and prompt logprobs are stored during the forward, before the step is accepted. The steps an
    abort discards drop their rows (invalid MoE outputs), and the recompute stores them again, once, whether its
    prefill starts at the beginning or after the accepted chunk it reuses: the result is that of a clean run."""
    monkeypatch.setattr(_utils, "_wide_ep_fault_tolerance", True)
    generator = torch.Generator().manual_seed(5)
    logits = torch.randn(_PROMPT_LEN, _VOCAB, generator=generator)
    clean = _context_rows_request()
    for size in (4, 4, 2):
        _store_chunk(clean, size, logits)

    recovery = _recovery()
    recovery._moe.aborted.return_value = True
    request = _context_rows_request()
    _store_chunk(request, 4, logits)  # its step is accepted
    steps = []
    for size in (4, 2):  # their steps run on invalid MoE outputs and are discarded
        mark = request.py_result.mark_context_rows()
        steps.append(_batch([request], context_rows=[(request, mark)]))
        _store_chunk(request, size, torch.full_like(logits, 99.0))
    for step in steps:
        recovery._executor.previous_batch = step
        recovery.drop_invalid_step()
    request.reset_for_recompute(1 << 20)
    request.context_current_position = reused
    while request.context_remaining_length > 0:
        _store_chunk(request, min(4, request.context_remaining_length), logits)

    result, expected = request.py_result, clean.py_result
    torch.testing.assert_close(result.context_logits, expected.context_logits, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.cat(result.context_prompt_logprob_rows),
        torch.cat(expected.context_prompt_logprob_rows),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        result.context_prompt_last_logprobs, expected.context_prompt_last_logprobs, rtol=0, atol=0
    )


@pytest.mark.parametrize("resume_after, held", [(None, False), (5, True), (4, True)])
def test_no_forward_until_the_change_commits(resume_after, held):
    recovery = _recovery()
    recovery._resume_after = resume_after
    assert recovery.holds_steps() == held  # the executor is at iteration 5


# One lost expert (3): rank 0 restores it into its spare slot. The agreed view still lists lost rank 2's slots.
_PLACEMENT = ((4, ((0, -1), (1, -1), (3, -1), (2, -1))),)
_INSTALL = _record(MembershipState.INSTALL, resume=ResumeState(iteration=9, epoch=8))


def _resuming_recovery(
    gathered_from_peers=(None, None, set()), record=_INSTALL, placement=None, rank=0
):
    """``rank`` past the iteration the survivors resume after; the authority commits once it replies INSTALLED.
    ``placement``: the agreed placement, from the READY payloads the INSTALL record carries."""
    recovery = _recovery(rank)
    if placement is not None:
        payloads = ((r, recovery_module.ready_payload(record, r, placement)) for r in range(WORLD))
        record = record.model_copy(
            update={"ready_payloads": tuple((r, p) for r, p in payloads if p is not None)}
        )
    executor = recovery._executor
    executor.dist.allgather.side_effect = lambda local: [local, *gathered_from_peers]
    recovery._record = recovery._installed = record
    recovery._resume_after = 9
    recovery._moe.aborted.return_value = True
    executor.iter_counter = 10
    reports = recovery._agent.reports

    def report(**changes):
        reports.append(changes)
        if changes.get("reply") == Reply.INSTALLED:
            recovery._on_record(
                record.model_copy(update=dict(state=MembershipState.COMMITTED, joining_mask=0))
            )

    recovery._agent.report = report
    return recovery


def _replies(recovery):
    return [report["reply"] for report in recovery._agent.reports if "reply" in report]


def test_resume_recomputes_the_interrupted_requests_and_fails_the_lost_ones(monkeypatch):
    monkeypatch.setattr(recovery_module.torch.cuda, "synchronize", lambda: None)
    recovery = _resuming_recovery([{20}, None, set()])
    executor = recovery._executor
    in_dropped_step, in_pending_step, untouched = _request(10), _request(14), _request(15)
    executor.active_requests = [
        in_dropped_step,
        in_pending_step,
        untouched,
        _request(-1, dummy=True),
    ]
    recovery._interrupted = [in_dropped_step]
    # Every request holds a sequence slot of the executor's resource manager, which the executor frees.
    slots = SeqSlotManager(len(executor.active_requests))
    for request in executor.active_requests:
        request.request_id = request.py_request_id
        request.seq_slot = slots.slot_manager.add_slot(request.request_id)
    executor.resource_manager = ResourceManager({ResourceManagerType.SEQ_SLOT_MANAGER: slots})
    executor.enable_attention_dp = True
    executor.sampler = SimpleNamespace(release_request=lambda request: None)
    executor.disagg = SimpleNamespace(forget_request=lambda request_id: None)
    executor._prefetched_request_ids = set()
    executor.draft_model_engine = None
    executor._free_request_resources = partial(PyExecutor._free_request_resources, executor)
    freed_when_paused = []
    executor._pause_recompute_request = Mock(
        side_effect=lambda request: freed_when_paused.append(
            slots.slot_manager.get_slot(request.request_id) is None
        )
    )
    executor.previous_batch = _batch(
        [in_pending_step]
    )  # its forward ran before the change committed
    executor.waiting_queue = [SimpleNamespace(id=12)]
    executor.executor_request_queue.request_queue.put(RequestQueueItem(13))
    # A cancel of lost request 11 does not hold it.
    executor.executor_request_queue.request_queue.put(
        RequestQueueItem(11, is_canceled_request=True)
    )
    recovery.set_outstanding_requests(lambda: {r: 100 + r for r in (10, 11, 12, 13, 20)})

    recovery._resume_after, executor.iter_counter = 10, 10
    recovery.before_iteration()  # not yet past the last iteration a survivor reached
    recovery._moe.rearm.assert_not_called()
    assert recovery.holds_steps()

    recovery._resume_after = 9
    executor.adp_ctx_waiting_iters_count = executor.adp_ctx_batching_wait_iters_count = 3
    info = Mock()
    monkeypatch.setattr(recovery_module.logger, "info", info)
    recovery.before_iteration()
    (resumed,) = [c.args[0] for c in info.call_args_list if "resumed serving" in c.args[0]]
    assert resumed.endswith(
        "2 interrupted requests continue by recompute, 1 requests no live rank holds ended by the rank failure"
    )
    assert executor.previous_batch is None
    # A replacement starts its attention-DP balance counters at 0: every live rank restarts them alike.
    assert executor.adp_ctx_waiting_iters_count == executor.adp_ctx_batching_wait_iters_count == 0
    # Only the requests of the discarded steps run again; the rest of the prefix cache stays.
    assert executor._pause_recompute_request.call_args_list == [
        ((in_dropped_step,),),
        ((in_pending_step,),),
    ]
    # Each gave its resources back before it was paused; the others keep theirs.
    assert freed_when_paused == [True, True]
    assert sorted(slots.slot_manager.slot_mapping) == [-1, untouched.request_id]
    recovery._moe.health.mark_failed.assert_called_once_with(FAILED)
    # The workspaces move to fresh pages among the survivors before the all-to-all is reset and masked.
    steps = ("remap_workspaces", "rearm", "install_mask")
    assert [name for name, _, _ in recovery._moe.method_calls if name in steps] == list(steps)
    assert _replies(recovery) == [Reply.INSTALLED]
    assert not recovery.holds_steps()
    ((request_id, response),) = executor._pending_transfer_responses  # request 11: on no survivor
    assert request_id == 11 and response.client_id == 111
    assert isinstance(response.error_msg, RankFailureError)


def test_the_workspace_remap_gathers_among_the_survivors_through_the_store(monkeypatch):
    monkeypatch.setattr(recovery_module.torch.cuda, "synchronize", lambda: None)
    recovery = _resuming_recovery()
    for call in range(2):
        for r in (1, 3):
            recovery._store.set(
                f"moe_a2a_workspaces/1/{call}/{r}", pickle.dumps(("handle", r, call))
            )
    gathered = []
    recovery._moe.remap_workspaces.side_effect = lambda gather, fresh_rows: gathered.extend(
        [gather(("handle", 0, 0)), gather(None)]
    )
    recovery.before_iteration()
    assert gathered == [
        [("handle", 0, 0), ("handle", 1, 0), None, ("handle", 3, 0)],
        [None, ("handle", 1, 1), None, ("handle", 3, 1)],
    ]


@pytest.mark.parametrize("change_after_remap", [False, True])
def test_a_newer_change_stops_the_install_without_resetting_the_all_to_all(
    monkeypatch, change_after_remap
):
    """The record is checked first, in every wait for a peer, and just before the all-to-all is reset."""
    monkeypatch.setattr(recovery_module.torch.cuda, "synchronize", lambda: None)
    recovery = _resuming_recovery()
    recovery.set_outstanding_requests(lambda: {11: 111})
    newer = _record(MembershipState.PROPOSE, generation=2, live_mask=0b0011)
    if change_after_remap:
        recovery._moe.remap_workspaces.side_effect = lambda gather, fresh_rows: recovery._on_record(
            newer
        )
    else:
        recovery._on_record(newer)
    recovery.before_iteration()
    assert recovery._moe.remap_workspaces.call_count == int(change_after_remap)
    recovery._moe.rearm.assert_not_called()
    assert _replies(recovery) == [] and recovery.holds_steps()
    assert recovery._executor._pending_transfer_responses == []
    recovery.before_iteration()  # once per install: the newer change finishes after its own
    assert recovery._executor.dist.allgather.call_count == 1


@pytest.mark.parametrize(
    "error, reply",
    [
        (RuntimeError("cuMemCreate failed"), [Reply.FAILED]),  # the authority aborts
        (MnnvlPeerHandleError([3]), []),  # a peer exited: the authority proposes again
    ],
)
def test_an_install_that_fails_holds_the_steps(monkeypatch, error, reply):
    monkeypatch.setattr(recovery_module.torch.cuda, "synchronize", lambda: None)
    recovery = _resuming_recovery()
    report = recovery._agent.report

    def answer_failed(**changes):  # the authority answers FAILED with the next change
        report(**changes)
        if changes.get("reply") == Reply.FAILED:
            recovery._on_record(_record(MembershipState.PROPOSE, generation=2, live_mask=0b0011))

    recovery._agent.report = answer_failed
    recovery._moe.remap_workspaces.side_effect = error
    recovery.before_iteration()
    assert _replies(recovery) == reply
    recovery._moe.rearm.assert_not_called()
    assert recovery.holds_steps()


def test_a_rank_whose_install_failed_exits_once_the_authority_s_connection_closes(monkeypatch):
    """After FAILED it waits for the authority's answer, as after INSTALLED, instead of holding its steps for good
    when the authority is gone."""
    monkeypatch.setattr(recovery_module.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(recovery_module.os, "_exit", Mock(side_effect=SystemExit))
    recovery = _resuming_recovery()
    recovery._agent.connected = False
    recovery._moe.remap_workspaces.side_effect = RuntimeError("cuMemCreate failed")
    with pytest.raises(SystemExit):
        recovery.before_iteration()
    assert _replies(recovery) == [Reply.FAILED]
    recovery_module.os._exit.assert_called_once_with(1)


def test_a_wait_for_the_next_record_ends_as_soon_as_the_authority_s_connection_closes(monkeypatch):
    """The agent's error ends the wait at once: no wait polls the connection."""
    monkeypatch.setattr(recovery_module.os, "_exit", Mock(side_effect=SystemExit))
    recovery = _recovery()

    def lose_the_authority():
        recovery._agent.connected = False
        recovery._on_agent_error(ConnectionError("the authority closed the liveness connection"))

    threading.Timer(0.05, lose_the_authority).start()
    start_s = time.monotonic()
    with pytest.raises(SystemExit):
        recovery._next_record(recovery._record)
    assert time.monotonic() - start_s < 0.5
    recovery_module.os._exit.assert_called_once_with(1)


def test_an_original_rank_waits_for_every_rank_s_startup_without_polling(monkeypatch):
    """The record that ends the startup wakes the wait."""
    agent = Mock(
        record=MembershipRecord(generation=0, state=MembershipState.STARTING, live_mask=0b1111)
    )
    monkeypatch.setattr(recovery_module, "_agent", agent)
    monkeypatch.setattr(recovery_module.time, "sleep", Mock(side_effect=AssertionError("polls")))

    def every_rank_started():
        agent.record = MembershipRecord(
            generation=0, state=MembershipState.COMMITTED, live_mask=0b1111
        )
        recovery_module._on_record_at_startup(agent.record)

    threading.Timer(0.05, every_rank_started).start()
    recovery_module.wait_for_every_rank()
    agent.report.assert_called_once_with(started=True)


@pytest.mark.parametrize("aborted", [False, True])
def test_a_forward_that_fails_while_the_all_to_all_is_aborted_is_discarded(aborted):
    """It failed for the membership change: its requests, and the previous step's, continue by recompute instead of
    failing; a forward that fails by itself fails its requests as without fault tolerance."""
    recovery = _recovery()
    recovery._moe.aborted.return_value = aborted
    executor = recovery._executor
    executor._accepted_previous_batch = Mock(side_effect=recovery.drop_invalid_step)
    executor.previous_batch = _batch([_request(10)])
    failed = _request(11)
    scheduled = SimpleNamespace(all_requests=lambda: [failed])
    assert recovery.discard_failed_forward(scheduled) is aborted
    if aborted:
        assert [request.py_request_id for request in recovery._interrupted] == [11, 10]
        assert executor.previous_batch is None
    else:
        assert recovery._interrupted == [] and executor.previous_batch is not None


def test_a_request_the_scheduler_paused_for_recompute_is_not_freed_again():
    """Its sequence slot is gone: freeing it again would release the slot None (MTP's relaxed-acceptance pool then
    zeroes every request's delta)."""
    recovery = _recovery()
    executor = recovery._executor
    paused, interrupted = _request(10), _request(11)
    paused.py_seq_slot = None
    executor.active_requests = [paused, interrupted]
    recovery._interrupted = [paused, interrupted]
    assert recovery._recompute_interrupted_requests() == 1
    executor._free_request_resources.assert_called_once_with(interrupted)
    executor._pause_recompute_request.assert_called_once_with(interrupted)


class _Device:
    """A rank's device memory since the deployment started serving: its free memory, the spare slots its EPLB backs
    (``backed``) and its KV cache pool (this is also the KV cache manager), whose quota changes in whole ``unit``s
    above an 8 MiB minimum and within the free memory. ``calls`` records the resizes."""

    def __init__(self, extra_free=0, gpu_quota=STARTUP_MEMORY.kv_cache_quota, backed=0, unit=MIB):
        self.calls = []
        self.free = FREE_BYTES + extra_free
        self.gpu_quota = gpu_quota
        self.backed = backed
        self.unit = unit

    def take(self, num_bytes):
        self.free -= num_bytes

    def resize_gpu_quota(self, quota):
        self.calls.append(("resize", quota))
        quota = max(0, quota) // self.unit * self.unit
        if quota < 8 * MIB or quota - self.gpu_quota > self.free:
            raise RuntimeError(f"The GPU KV cache pool cannot be resized to {quota} bytes")
        self.take(quota - self.gpu_quota)
        self.gpu_quota = quota
        return quota


def _eplb(monkeypatch, recovery, restored=6 * MIB, workspace=4 * MIB, device=None, released=0):
    """Rank 0's EPLB, whose plan backs one spare slot of 6 MiB (``restored``: the device memory the restore maps),
    and an all-to-all workspace of ``workspace`` bytes, on ``device`` (by default, as the deployment started
    serving). ``released``: the device memory that releasing the empty spare slots frees."""
    device = _Device() if device is None else device
    monkeypatch.setattr(recovery_module.torch.cuda, "mem_get_info", lambda: (device.free, 1 << 40))
    balancer = MoeLoadBalancer.__new__(MoeLoadBalancer)
    balancer.ep_rank = 0
    # Rank 0 takes lost expert 3 into its spare slot.
    planned = (np.array([[0, 3], [1, -1], [-1, -1], [2, -1]]),)
    plan = SimpleNamespace(device_bytes=restored, digest="restored", placements=planned)
    calls = device.calls

    def back(num_bytes):
        device.take(num_bytes)
        device.backed += num_bytes
        balancer.expert_restore.backed_spare_slot_bytes = device.backed

    def restore_lost_experts(restore, times):
        calls.append(("restore", restore))
        back(restored)
        times.update({"back": 1.0, "load": 2.0, "placement install": 3.0})
        return restored

    def release_empty_spare_slots():
        calls.append("release")
        back(-released)
        return released

    balancer.expert_restore = SimpleNamespace(
        plan_expert_restore=lambda live, current, rejoining: (
            calls.append(("plan", live, current, rejoining)) or plan
        ),
        restore_lost_experts=restore_lost_experts,
        release_empty_spare_slots=release_empty_spare_slots,
        backed_spare_slot_bytes=device.backed,
    )
    balancer.abandon_iterations = lambda: calls.append("abandon")

    @contextlib.contextmanager
    def kv_pools_suspended():  # every request suspended around it
        calls.append("suspend")
        try:
            yield device
        finally:
            calls.append("resume")

    recovery._executor.model_engine.moe_load_balancer = balancer
    recovery._executor._kv_pools_suspended = kv_pools_suspended
    recovery._moe.workspace_bytes.return_value = workspace
    recovery._moe.remap_workspaces.side_effect = lambda gather, fresh_rows: calls.append("remap")
    return calls, plan, device


def _resizes(device):
    return [call[1] for call in device.calls if call[0] == "resize"]


def test_the_install_restores_the_lost_experts_from_the_agreed_placement(monkeypatch):
    monkeypatch.setattr(recovery_module.torch.cuda, "synchronize", lambda: None)
    recovery = _resuming_recovery(placement=_PLACEMENT)
    calls, plan, _ = _eplb(monkeypatch, recovery, workspace=4 * MIB - 4096)
    info = Mock()
    monkeypatch.setattr(recovery_module.logger, "info", info)
    recovery.before_iteration()
    # One line per rank tells operators what the restore did and how long each phase of the install took, and how
    # the KV cache pool's quota followed the memory needed and the free device memory.
    (restored,) = [c.args[0] for c in info.call_args_list if "restored membership" in c.args[0]]
    assert "Loaded 1 experts (6291456 bytes)" in restored
    assert (
        f"KV cache quota {64 * MIB} -> {54 * MIB} bytes ({10 * MIB - 4096} bytes needed, "
        f"{FREE_BYTES} bytes free). Phases: "
    ) in restored
    phases = restored.split(". Phases: ")[1].split(", ")
    assert [phase.rsplit(" ", 2)[0] for phase in phases] == [
        "prepare",
        "request gather",
        "KV cache shrink",
        "back",
        "load",
        "placement install",
        "remap",
        "rearm",
    ]
    assert "back 1 ms, load 2 ms, placement install 3 ms" in restored
    assert calls[0] == "abandon"  # at the resume point, on every live rank
    (_, live, (current,), rejoining) = calls[1]
    assert live == [True, True, False, True] and rejoining == [False] * WORLD
    # The agreed placement from the INSTALL record's READY payloads: the plan ignores the lost rank's slots.
    np.testing.assert_array_equal(current, [[0, -1], [1, -1], [3, -1], [2, -1]])
    assert calls == [
        "abandon",
        calls[1],
        "suspend",
        # The spare slot, and the workspace's fresh pages, mapped before the pages they replace are freed.
        ("resize", 54 * MIB + 4096),
        ("restore", plan),
        "remap",
        "resume",
        # Once committed, the pool takes back the workspace's part (no spare slot to release), once per change.
        "release",
        "suspend",
        ("resize", 58 * MIB),
        "resume",
    ]
    assert _replies(recovery) == [Reply.INSTALLED]
    assert recovery._agent.reports[-1]["placement_digest"] == "restored"


def test_each_completed_install_step_is_reported_as_progress(monkeypatch):
    """The gather of the requests held, the plan, the KV cache memory given up, the experts restored and the remap:
    the authority counts each step as progress, so an install that takes long but keeps completing steps is not
    taken for a hang."""
    monkeypatch.setattr(recovery_module.torch.cuda, "synchronize", lambda: None)
    recovery = _resuming_recovery(placement=_PLACEMENT)
    _eplb(monkeypatch, recovery)
    recovery.before_iteration()
    steps = [report.get("install_steps", report.get("reply")) for report in recovery._agent.reports]
    assert steps == [1, 2, 3, 4, 5, Reply.INSTALLED]


@pytest.mark.parametrize("extra_free", [0, 7 * MIB, 20 * MIB, -3 * MIB])
def test_the_kv_cache_pool_gives_up_exactly_what_the_change_takes(monkeypatch, extra_free):
    """The 6 MiB spare slot and the 4 MiB workspace come out of the KV cache pool, whatever serving allocated or
    freed since the startup (``extra_free``); once committed, the pool takes back the workspace's part."""
    recovery = _resuming_recovery(placement=_PLACEMENT)
    _, _, device = _eplb(monkeypatch, recovery, device=_Device(extra_free=extra_free))
    recovery._restore(recovery._installed)
    recovery._give_back_kv_cache()  # once committed
    assert _resizes(device) == [54 * MIB, 58 * MIB] and device.gpu_quota == 58 * MIB
    # Serving keeps the memory it had free.
    assert device.free == FREE_BYTES + extra_free


def test_the_kv_cache_pool_takes_back_its_startup_quota_after_serving_allocated_memory(monkeypatch):
    """Serving allocates 160 MiB after the startup (the caching allocator grows): a failover and the rejoin after
    it charge none of it to the KV cache pool, which resizes in units of 32 MiB, and it is back at its startup
    quota."""
    unit = 32 * MIB
    startup = StartupMemory(kv_cache_quota=64 * unit)
    device = _Device(gpu_quota=startup.kv_cache_quota, unit=unit)
    device.take(5 * unit)  # serving
    loss = _resuming_recovery(placement=_PLACEMENT)
    loss._startup_memory = startup
    _eplb(monkeypatch, loss, restored=3 * unit, workspace=2 * unit, device=device)
    loss._restore(loss._installed)
    loss._give_back_kv_cache()  # once committed
    assert device.gpu_quota == startup.kv_cache_quota - 3 * unit
    rejoin = _resuming_recovery(
        record=_INSTALL.model_copy(update=dict(live_mask=0b1111, joining_mask=1 << FAILED)),
        placement=_PLACEMENT,
    )
    rejoin._startup_memory = startup
    _eplb(monkeypatch, rejoin, restored=0, workspace=2 * unit, released=3 * unit, device=device)
    rejoin._restore(rejoin._installed)
    rejoin._give_back_kv_cache()  # once committed
    assert device.gpu_quota == startup.kv_cache_quota
    assert device.free == FREE_BYTES - 5 * unit


def test_the_kv_cache_pool_leaves_out_the_pages_a_process_that_is_ending_maps(monkeypatch):
    """At the commit, an excluded process that has not ended yet still maps the pages the remap replaced: the pool
    takes back only memory freed since the change started, so serving keeps the memory it had free, and takes the
    rest back at the next change."""
    device = _Device()
    loss = _resuming_recovery(placement=_PLACEMENT)
    _eplb(monkeypatch, loss, device=device)
    loss._moe.remap_workspaces.side_effect = lambda gather, fresh_rows: device.take(4 * MIB)
    loss._restore(loss._installed)
    loss._give_back_kv_cache()  # once committed
    assert device.gpu_quota == 54 * MIB and device.free == FREE_BYTES
    device.take(-4 * MIB)  # the excluded process ended
    rejoin = _resuming_recovery(
        record=_INSTALL.model_copy(update=dict(live_mask=0b1111, joining_mask=1 << FAILED)),
        placement=_PLACEMENT,
    )
    _eplb(monkeypatch, rejoin, restored=0, released=6 * MIB, device=device)
    rejoin._restore(rejoin._installed)
    rejoin._give_back_kv_cache()  # once committed
    assert device.gpu_quota == STARTUP_MEMORY.kv_cache_quota and device.free == FREE_BYTES


def _short_give_back(monkeypatch):
    """Rank 0 after a failover whose commit found the pages the remap replaced still mapped by the process of the
    excluded rank 2: its give-back fell 4 MiB short. A second passed since."""
    monkeypatch.setattr(recovery_module.torch.cuda, "synchronize", lambda: None)
    recovery = _resuming_recovery(placement=_PLACEMENT)
    _, _, device = _eplb(monkeypatch, recovery)
    recovery._moe.remap_workspaces.side_effect = lambda gather, fresh_rows: device.take(4 * MIB)
    recovery.before_iteration()  # the install, then the commit
    assert device.gpu_quota == 54 * MIB and device.free == FREE_BYTES
    recovery._give_back_retry.check_after_s = 0.0
    recovery._executor._retire_inflight_batch_for_control.reset_mock()
    return recovery, device


def test_a_rank_whose_give_back_fell_short_asks_for_a_retry_once_memory_is_freed(monkeypatch):
    """No retry while no device memory was freed since the commit, and at most one request a second; every rank takes
    part in a retry another rank asked for, unless a rank held its steps for a pending change."""
    recovery, device = _short_give_back(monkeypatch)
    assert _vote(recovery)[0] == [4, 0, 0] and not recovery._give_back_agreed
    recovery._give_back_retry.check_after_s = 0.0
    device.take(-4 * MIB)  # the excluded process ended
    assert _vote(recovery)[0] == [4, 0, 1] and recovery._give_back_agreed
    assert _vote(recovery)[0] == [4, 0, 0]  # within the second
    other = _recovery(rank=1)
    _vote(other, peers=((4, 0, 1), (5, 0, 0), (2, 0, 0)))
    assert other._give_back_agreed and not other._give_back_due
    _vote(other, peers=((4, 0, 1), (5, 1, 0), (2, 0, 0)))
    assert not other._give_back_agreed


def test_a_retried_give_back_takes_back_the_memory_freed_since_the_commit(monkeypatch):
    """At the iteration after the vote, the step in flight is retired, then the pool takes back what the ended process
    freed: it holds its quota for the backed spare slot again, and the retries end."""
    recovery, device = _short_give_back(monkeypatch)
    retire = recovery._executor._retire_inflight_batch_for_control
    device.take(-4 * MIB)  # the excluded process ended
    _vote(recovery)
    recovery.before_iteration()
    retire.assert_called_once()
    assert device.gpu_quota == 58 * MIB and device.free == FREE_BYTES
    assert recovery._give_back_retry is None
    _vote(recovery)
    recovery.before_iteration()
    retire.assert_called_once()


def test_the_give_back_retries_end_once_the_excluded_process_ended(monkeypatch):
    """Once the authority saw the liveness connection of rank 2's excluded process close, the process has ended: with
    no memory freed since the last give-back, the retries end without another one."""
    recovery, device = _short_give_back(monkeypatch)
    recovery._store.set(
        DISCONNECTED_KEY, json.dumps([0 if r == FAILED else -1 for r in range(WORLD)]).encode()
    )
    assert _vote(recovery)[0] == [4, 0, 0]
    assert recovery._give_back_retry is None and device.gpu_quota == 54 * MIB


def test_a_healthy_step_checks_no_device_memory(monkeypatch):
    """Without a give-back that fell short, the vote asks for no retry, reads no free device memory, and the next
    iteration retires no step."""
    recovery = _recovery()
    mem_get_info = Mock()
    monkeypatch.setattr(recovery_module.torch.cuda, "mem_get_info", mem_get_info)
    assert _vote(recovery)[0] == [4, 0, 0] and not recovery._give_back_agreed
    recovery.before_iteration()
    mem_get_info.assert_not_called()
    recovery._executor._retire_inflight_batch_for_control.assert_not_called()


def test_a_rejoin_releases_the_emptied_spare_slots_once_committed(monkeypatch):
    """The spare slots the joining rank's experts leave keep their memory until the change is committed, so no live
    rank waits for their release in the transaction; then their memory goes back to the KV cache pool."""
    monkeypatch.setattr(recovery_module.torch.cuda, "synchronize", lambda: None)
    recovery = _resuming_recovery(
        record=_INSTALL.model_copy(update=dict(live_mask=0b1111, joining_mask=1 << FAILED)),
        placement=((4, ((0, -1), (1, -1), (3, -1), (2, -1))),),
    )
    calls, plan, device = _eplb(
        monkeypatch,
        recovery,
        restored=0,
        released=6 * MIB,
        device=_Device(gpu_quota=58 * MIB, backed=6 * MIB),
    )
    report = recovery._agent.report

    def report_reply(**changes):
        if "reply" in changes:
            calls.append(changes["reply"])
        report(**changes)

    recovery._agent.report = report_reply
    info = Mock()
    monkeypatch.setattr(recovery_module.logger, "info", info)
    recovery.before_iteration()
    assert calls[0] == "abandon"
    (_, live, _, rejoining) = calls[1]
    assert live == [True] * WORLD and rejoining == [False, False, True, False]
    assert calls[2:] == [
        "suspend",
        ("resize", 54 * MIB),
        ("restore", plan),
        "remap",
        "resume",
        Reply.INSTALLED,
        "release",
        "suspend",
        ("resize", STARTUP_MEMORY.kv_cache_quota),
        "resume",
    ]
    assert device.gpu_quota == STARTUP_MEMORY.kv_cache_quota
    recovery._moe.health.mark_active.assert_any_call(FAILED)
    (resumed,) = [c.args[0] for c in info.call_args_list if "resumed serving" in c.args[0]]
    assert (
        f". Released {6 * MIB} bytes of empty spare slots, KV cache quota "
        f"{STARTUP_MEMORY.kv_cache_quota} bytes, in "
    ) in resumed


@pytest.mark.parametrize("released", [6 * MIB, 0], ids=["spare slots backed", "none backed"])
def test_an_admission_that_needs_no_memory_leaves_the_kv_cache_pool_alone(monkeypatch, released):
    """When every process that maps the workspaces' pages stays live, the joining rank's rows are mapped into them (no
    fresh pages), and the restore backs no spare slot: no request is suspended and the KV cache pool is not resized.
    Once committed, the spare slots backed at the failover, which the joining rank's experts left, are released and
    the pool takes their memory back; with none released, the pool stays as it is."""
    recovery = _resuming_recovery(
        record=_INSTALL.model_copy(update=dict(live_mask=0b1111, joining_mask=1 << FAILED)),
        placement=((4, ((0, -1), (1, -1), (3, -1), (2, -1))),),
    )
    calls, plan, device = _eplb(
        monkeypatch,
        recovery,
        restored=0,
        released=released,
        device=_Device(gpu_quota=58 * MIB, backed=released),
    )
    recovery._moe.stale_workspace_rows.return_value = frozenset()
    recovery._moe.remap_workspaces.side_effect = lambda gather, fresh_rows: calls.append(fresh_rows)
    recovery._restore(recovery._installed)
    assert calls[1:] == [("restore", plan), frozenset()]
    given_back = recovery._give_back_kv_cache()  # once committed
    if released:
        assert calls[3:] == [
            "release",
            "suspend",
            ("resize", STARTUP_MEMORY.kv_cache_quota),
            "resume",
        ]
        assert device.gpu_quota == STARTUP_MEMORY.kv_cache_quota
    else:
        assert calls[3:] == ["release"] and given_back == ""
        assert device.gpu_quota == 58 * MIB


def test_repeated_failures_do_not_ratchet_the_kv_cache_pool_down(monkeypatch):
    """Every change derives the pool's quota afresh: memory a change cannot take back yet (the replaced pages that an
    excluded process, not ended yet, still maps) comes back at the next one."""
    device = _Device()
    loss = _resuming_recovery(placement=_PLACEMENT)
    _eplb(monkeypatch, loss, device=device)
    loss._moe.remap_workspaces.side_effect = lambda gather, fresh_rows: device.take(4 * MIB)
    loss._restore(loss._installed)
    assert device.gpu_quota == 54 * MIB
    device.take(-4 * MIB)  # the excluded process ended
    for _ in range(2):
        rejoin = _resuming_recovery(
            record=_INSTALL.model_copy(update=dict(live_mask=0b1111, joining_mask=1 << FAILED)),
            placement=_PLACEMENT,
        )
        _eplb(monkeypatch, rejoin, restored=0, released=6 * MIB, device=device)
        rejoin._restore(rejoin._installed)
        rejoin._give_back_kv_cache()  # once committed
        assert device.gpu_quota == STARTUP_MEMORY.kv_cache_quota
        loss = _resuming_recovery(placement=_PLACEMENT)
        _eplb(monkeypatch, loss, device=device)
        loss._restore(loss._installed)
        assert device.gpu_quota == 54 * MIB
        loss._give_back_kv_cache()
        assert device.gpu_quota == 58 * MIB


def test_kv_memory_given_up_for_an_install_that_fails_is_taken_back_at_the_next_change(monkeypatch):
    """The pool gave up 10 MiB, 6 of them for the restored slot; the remap then fails: the next change takes the
    other 4 back once committed."""
    recovery = _resuming_recovery(placement=_PLACEMENT)
    _, _, device = _eplb(monkeypatch, recovery)
    recovery._moe.remap_workspaces.side_effect = RuntimeError("cuMemCreate failed")
    with pytest.raises(RuntimeError):
        recovery._restore(recovery._installed)
    assert device.gpu_quota == 54 * MIB
    _eplb(monkeypatch, recovery, restored=0, device=device)
    recovery._restore(recovery._installed)
    assert device.gpu_quota == 54 * MIB
    recovery._give_back_kv_cache()
    assert device.gpu_quota == 58 * MIB


def test_without_eplb_or_workspace_nothing_is_suspended():
    recovery = _resuming_recovery()
    recovery._executor._kv_pools_suspended = Mock()
    recovery._restore(recovery._installed)
    assert recovery._give_back_kv_cache() == ""  # the change took no KV cache memory
    recovery._executor._kv_pools_suspended.assert_not_called()
    recovery._moe.remap_workspaces.assert_called_once()


def _bare_balancer():
    """A MoeLoadBalancer with its expert restore, both without their startup: the test sets the fields they read."""
    balancer = MoeLoadBalancer.__new__(MoeLoadBalancer)
    balancer.expert_restore = EplbExpertRestore.__new__(EplbExpertRestore)
    balancer.expert_restore.balancer = balancer
    return balancer


def _online_eplb(recovery, on_abandon=lambda: None):
    """``recovery``'s online EPLB; ``on_abandon()`` runs while its blocking end of the interrupted iterations waits.
    Returns the calls: each end and each read of the placement."""
    balancer = _bare_balancer()
    balancer.ep_rank = recovery.rank
    calls = []

    def abandon_iterations():
        calls.append(("abandon",))
        on_abandon()

    balancer.abandon_iterations = abandon_iterations
    balancer.single_layer_load_balancers = [
        SimpleNamespace(
            expert_count=3,
            get_old_rank_expert_ids=lambda: calls.append(("read",)) or [[0], [1], [2], [0]],
        )
    ]
    recovery._executor.model_engine.moe_load_balancer = balancer
    return calls


def test_ready_reads_the_placement_once_the_online_eplb_updates_already_started_ended(monkeypatch):
    """One blocking call ends them; it has no deadline of its own: the authority waits on a rank that has not
    replied, so its hang rule bounds the wait."""
    _exit_raises(monkeypatch)
    recovery = _recovery(rank=1)
    calls = _online_eplb(recovery)
    proposal = _record(MembershipState.PROPOSE)
    recovery._on_record(proposal)
    recovery._next_record = Mock(side_effect=[proposal, _record(MembershipState.ABORT)])
    with pytest.raises(SystemExit):
        recovery._change_membership()
    assert calls == [("abandon",), ("read",)]
    assert _replies(recovery) == [Reply.READY]


def test_a_change_that_replaces_the_proposal_while_the_updates_end_gets_no_ready(monkeypatch):
    _exit_raises(monkeypatch)
    recovery = _recovery(rank=1)
    newer = _record(MembershipState.PROPOSE, generation=2, live_mask=0b1011)
    _online_eplb(recovery, on_abandon=lambda: recovery._on_record(newer))
    proposal = _record(MembershipState.PROPOSE)
    recovery._on_record(proposal)
    recovery._next_record = Mock(side_effect=[proposal, _record(MembershipState.ABORT)])
    with pytest.raises(SystemExit):
        recovery._change_membership()
    assert _replies(recovery) == []


def test_every_live_rank_restarts_the_online_eplb_update_plan_at_the_resume_point(monkeypatch):
    """After a failure, rank 0 ends its online EPLB work before READY, then finishes the host-exchange round it
    stopped in after INSTALL and runs that step, one more EPLB iteration, which rank 1 ran before its own end. Their
    update plans would then update different layers from there; both restart them at the resume point."""
    monkeypatch.setattr(recovery_module.torch.cuda, "synchronize", lambda: None)
    plan_steps = {}
    for rank, iterations_after_ready in ((0, 1), (1, 0)):
        recovery = _resuming_recovery(placement=_PLACEMENT, rank=rank)
        _eplb(monkeypatch, recovery)
        balancer = recovery._executor.model_engine.moe_load_balancer
        balancer.abandon_iterations = lambda balancer=balancer: setattr(balancer, "plan_step", 0)
        balancer.abandon_iterations()  # before READY
        balancer.plan_step += iterations_after_ready
        recovery.before_iteration()
        assert _replies(recovery) == [Reply.INSTALLED]
        plan_steps[rank] = balancer.plan_step
    assert plan_steps == {0: 0, 1: 0}


@pytest.mark.parametrize("rank", [0, 1])
def test_waiting_requests_pinned_to_an_absent_rank_leave_the_queue_and_fail_on_rank_0(rank):
    recovery = _recovery(rank)
    waiting = FCFSWaitingQueue()
    waiting.add_requests([_item(1), _item(2, target=FAILED), _item(3, target=FAILED, relax=True)])
    all_states = [RankState(rank=r) for r in range(WORLD)]
    recovery.fail_unservable_requests(waiting, all_states)  # every rank serves
    assert [item.id for item in waiting] == [1, 2, 3]

    recovery.fail_unservable_requests(waiting, _survivor_states())
    assert [item.id for item in waiting] == [1, 3]
    responses = recovery._executor._pending_transfer_responses
    if rank == 0:
        ((request_id, response),) = responses
        assert request_id == 2 and response.client_id == 102
        # Not one to send again: it would fail the same way.
        assert not isinstance(response.error_msg, RankFailureError)
    else:
        assert responses == []


# One MoE layer of four experts as every rank started: rank r holds expert r and an empty spare slot.
_STARTUP = ((4, ((0, -1), (1, -1), (2, -1), (3, -1))),)


def _installed(record, views):
    """The INSTALL record that follows the PROPOSE ``record``, with the READY payloads of the ranks' ``views``."""
    payloads = ((r, recovery_module.ready_payload(record, r, view)) for r, view in views.items())
    return record.model_copy(
        update={
            "state": MembershipState.INSTALL,
            "ready_payloads": tuple((r, p) for r, p in payloads if p is not None),
        }
    )


def test_the_agreed_placement_is_the_lowest_survivor_s_view_with_the_joining_ranks_own_slots():
    """An interrupted online EPLB update left the survivors with different views; a joining rank knows only its
    startup placement."""
    views = {
        0: ((4, ((1, -1), (0, -1), (3, -1), (2, -1))),),
        1: _STARTUP,  # rank 1 missed the update
        3: _STARTUP,  # rank 3 joins
    }
    record = MembershipRecord(
        generation=2, state=MembershipState.PROPOSE, live_mask=0b1011, joining_mask=1 << 3
    )
    installed = _installed(record, views)
    assert [r for r, _ in installed.ready_payloads] == [0, 3]  # rank 1's carries nothing
    agreed = recovery_module.agreed_placement(installed)
    assert agreed == [(4, [[1, -1], [0, -1], [3, -1], [3, -1]])]
    # With rank 1 the lowest survivor.
    record = record.model_copy(update=dict(live_mask=0b1010))
    installed = _installed(record, {r: view for r, view in views.items() if r})
    startup = [(4, [[0, -1], [1, -1], [2, -1], [3, -1]])]
    assert recovery_module.agreed_placement(installed) == startup
    # The lowest survivor, or a joining rank, sent no placement (no EPLB).
    for missing in (1, 3):
        payloads = tuple((r, p) for r, p in installed.ready_payloads if r != missing)
        without = installed.model_copy(update=dict(ready_payloads=payloads))
        assert recovery_module.agreed_placement(without) is None


def test_at_deepseek_r1_size_one_ready_payload_carries_the_placement_every_rank_plans_from():
    """EP 32, 58 MoE layers of 10 slots per rank: the lowest survivor's payload carries its view, a joining rank's its
    own 580 slots and every other rank's none, so the INSTALL record stays far below a liveness line's limit."""
    world, layers, experts, spare_slots = 32, 58, 256, 2
    share = experts // world
    ranks = tuple(
        tuple(range(r * share, (r + 1) * share)) + (-1,) * spare_slots for r in range(world)
    )
    view = ((experts, ranks),) * layers
    record = MembershipRecord(
        generation=3,
        state=MembershipState.PROPOSE,
        live_mask=(1 << world) - 1,
        joining_mask=1 << 5,
    )
    payloads = {r: recovery_module.ready_payload(record, r, view) for r in range(world)}
    assert json.loads(payloads[0]) == json.loads(json.dumps(view))
    assert json.loads(payloads[5]) == [list(ranks[5])] * layers and len(payloads[5]) < 4000
    assert all(payloads[r] is None for r in range(world) if r not in (0, 5))
    installed = _installed(record, {r: view for r in range(world)})
    assert (
        recovery_module.agreed_placement(installed)
        == [(experts, [list(slots) for slots in ranks])] * layers
    )
    assert len(installed.model_dump_json()) < membership._MAX_LINE_BYTES // 4


@pytest.mark.parametrize(
    "rank, joining, payload",
    [
        # The lowest survivor: every rank's slots as it sees them, the view every live rank plans from.
        (0, False, [[3, [[0, 1], [1, 2], [2, -1], [0, -1]]]]),
        # Another survivor: none of the placement.
        (1, False, None),
        # A joining rank: its own slots only.
        (FAILED, True, [[2, -1]]),
    ],
)
def test_ready_carries_the_rank_s_part_of_the_placement(rank, joining, payload):
    balancer = _bare_balancer()
    balancer.ep_rank = rank
    balancer.abandon_iterations = lambda: None
    # Per rank, the expert of each slot; -1: an empty spare slot.
    view = [[0, 1], [1, 2], [2, -1], [0, -1]]
    balancer.single_layer_load_balancers = [
        SimpleNamespace(expert_count=3, get_old_rank_expert_ids=lambda: view)
    ]
    if joining:  # rank 2's replacement
        recovery = RankFailureRecovery(
            _executor(rank),
            FakeExchange(),
            MagicMock(),
            FakeStore(),
            "host:1",
            STARTUP_MEMORY,
            incarnation=1,
        )
        recovery._agent = FakeAgent()
        proposal = MembershipRecord(
            generation=2,
            state=MembershipState.PROPOSE,
            live_mask=0b1111,
            incarnations=(0, 0, 1, 0),
            joining_mask=1 << FAILED,
        )
    else:
        recovery = _recovery(rank)
        proposal = _record(MembershipState.PROPOSE)
    recovery._executor.model_engine.moe_load_balancer = balancer
    recovery._on_record(proposal)
    recovery._next_record = Mock(side_effect=[proposal, _record(MembershipState.ABORT)])
    with (
        patch.object(recovery_module.os, "_exit", side_effect=SystemExit),
        pytest.raises(SystemExit),
    ):
        recovery._change_membership()
    ready = recovery._agent.reports[-1]
    assert ready["reply"] == Reply.READY
    sent = ready["ready_payload"]
    assert (None if sent is None else json.loads(sent)) == payload


def test_a_replacement_reports_that_it_is_ready_to_join_with_its_ready():
    """Before any PROPOSE adds it, a started replacement reports that it is ready to join, with what its READY carries
    (its own slots): the authority stops the live ranks for it only then, and takes its READY from this report."""
    balancer = _bare_balancer()
    balancer.ep_rank = FAILED
    balancer.single_layer_load_balancers = [
        SimpleNamespace(
            expert_count=3, get_old_rank_expert_ids=lambda: [[0, 1], [1, 2], [2, -1], [0, -1]]
        )
    ]
    recovery = RankFailureRecovery(
        _executor(FAILED),
        FakeExchange(),
        MagicMock(),
        FakeStore(),
        "host:1",
        STARTUP_MEMORY,
        incarnation=1,
    )
    recovery._agent = FakeAgent()
    # Its own EPLB statistics and update plan restart before its READY, as the live ranks' do before theirs: it may
    # never see the PROPOSE that adds it.
    balancer.abandon_iterations = lambda: recovery._agent.reports.append("abandoned")
    recovery._executor.model_engine.moe_load_balancer = balancer
    recovery._change_membership = Mock()  # the change that adds it
    recovery._resume_after = 20
    recovery._join()
    assert recovery._agent.reports == [
        "abandoned",
        {"joining_ready": True, "ready_payload": json.dumps([[2, -1]])},
    ]
    recovery._change_membership.assert_called_once()


# Hooks outside the executor.


def test_wake_returns_a_waiting_request_fetch():
    request_queue = ExecutorRequestQueue(MagicMock(), 8, False, 0)
    with request_queue.enqueue_lock:  # wake enqueues like the other enqueue_* calls
        waker = threading.Thread(target=request_queue.wake, daemon=True)
        waker.start()
        waker.join(0.2)
        assert request_queue.request_queue.empty()
    waker.join(5)
    fetched = request_queue.get_from_request_queue(timedelta(seconds=30))
    assert fetched == []


def test_hard_kill_stays_local_under_fault_tolerance(monkeypatch):
    from mpi4py import MPI

    monkeypatch.setattr(_utils, "_wide_ep_fault_tolerance", True)
    comm = MagicMock()
    killed = []
    # Without fault tolerance, these make the kill go through MPI_Abort.
    monkeypatch.setattr(hang_detector, "ENABLE_MULTI_DEVICE", True)
    monkeypatch.setattr(hang_detector, "mpi_comm", lambda: comm)
    monkeypatch.setattr(MPI, "Is_initialized", lambda: True)
    monkeypatch.setattr(MPI, "Is_finalized", lambda: False)
    monkeypatch.setattr(MPI, "Query_thread", lambda: MPI.THREAD_MULTIPLE)
    monkeypatch.setattr(hang_detector.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    hang_detector.propagate_hard_kill(diagnostics=False)
    comm.Abort.assert_not_called()
    assert killed == [(os.getpid(), signal.SIGKILL)]


def _worker(rank):
    return capture_worker_process_identity(rank)._replace(pid=1000 + rank)


def _proxy(dead_worker=None, remote_error=None, membership=True):
    """A proxy past worker startup, with or without the fault-tolerance membership established."""
    proxy = GenerationExecutorProxy.__new__(GenerationExecutorProxy)
    proxy.fault_tolerance_health = Mock() if membership else None
    proxy._worker_process_monitor = Mock(find_dead_worker=Mock(return_value=dead_worker))
    # As under trtllm-llmapi-launch: the remote session forwards worker deaths.
    proxy.mpi_session = Mock(check_worker_error=Mock(return_value=remote_error))
    exited = Future()
    exited.set_result(None)
    proxy.mpi_futures = [exited]  # rank 0's, the session's only worker
    proxy._set_fatal_error = Mock()
    proxy._handle_worker_death = Mock()
    proxy.doing_shutdown = True
    return proxy


def test_joining_the_membership_watches_the_session_s_worker_rank_0(monkeypatch):
    monkeypatch.setattr(_utils, "_wide_ep_fault_tolerance", True)
    monkeypatch.setenv(JOIN_SECRET_ENV, "ab" * 32)
    monkeypatch.setenv(membership.RENDEZVOUS_ENV, "")
    monkeypatch.setattr(recovery_module, "open_store", Mock(side_effect=OSError("no store")))
    proxy = _proxy(membership=False)
    # Every rank's identity, gathered over the startup group; the session holds rank 0's process only.
    workers = [_worker(rank) for rank in range(WORLD)]
    proxy._join_membership(workers, "host:1")
    assert isinstance(proxy.fault_tolerance_health, FaultToleranceHealth)
    # The store is read when the server's probes are served, not before.
    recovery_module.open_store.assert_not_called()
    proxy._worker_process_monitor.register.assert_called_once_with(workers[:1])


@pytest.mark.parametrize("membership", [False, True])
def test_the_death_of_the_session_s_worker_rank_0_is_fatal_with_the_authority_alive(membership):
    """The session holds rank 0's worker only: the authority handles the other ranks, which the proxy never sees."""
    proxy = _proxy(membership=membership)
    assert proxy._check_mpi_workers()
    assert proxy._set_fatal_error.called


@pytest.mark.parametrize("membership", [False, True])
def test_a_remote_worker_death_is_fatal_with_the_authority_alive(membership):
    """Under trtllm-llmapi-launch the server's session holds rank 0's worker only."""
    proxy = _proxy(remote_error=RuntimeError("Remote MPI worker died"), membership=membership)
    assert proxy._check_remote_worker_death()
    assert proxy._set_fatal_error.called


def test_deepseek_v4_cache_manager_keeps_the_reuse_commits_for_step_acceptance(monkeypatch):
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.cache_manager import (
        DeepseekV4CacheManager,
    )
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

    update = Mock()
    monkeypatch.setattr(KVCacheManagerV2, "update_context_resources", update)
    manager = DeepseekV4CacheManager.__new__(DeepseekV4CacheManager)
    manager._order_sparse_history_update = Mock()
    batch, commit_ends = SimpleNamespace(context_requests=[]), {}
    manager.update_context_resources(batch, commit_ends)
    update.assert_called_once_with(batch, commit_ends)


@pytest.mark.parametrize("fault_tolerance", [False, True])
def test_userbuffers_off_under_fault_tolerance(monkeypatch, fault_tolerance):
    """The userbuffers bootstrap runs MPI_COMM_WORLD collectives, and under fault tolerance every rank is its own
    MPI world: torch.compile runs without userbuffers there."""
    from tensorrt_llm._torch.pyexecutor.engine import model_setup

    initialize = Mock()
    monkeypatch.setattr(model_setup, "mpi_disabled", lambda: False)
    monkeypatch.setattr(model_setup.ub, "ub_supported", lambda: True)
    monkeypatch.setattr(model_setup.ub, "initialize_userbuffers_manager", initialize)
    llm_args = SimpleNamespace(
        enable_wide_ep_fault_tolerance=fault_tolerance, allreduce_strategy="AUTO"
    )
    mapping = SimpleNamespace(tp_size=WORLD, pp_size=1, cp_size=1, rank=0, gpus_per_node=WORLD)

    enabled = model_setup._init_userbuffers(mapping, llm_args, hidden_size=8, max_num_tokens=8)

    assert enabled is not fault_tolerance
    assert initialize.called is not fault_tolerance


class _PastModelBuild(Exception):
    """Raised by the step of the model load that follows the checks of the built model."""


@pytest.mark.parametrize(
    "fault_tolerance, checkpoint_architecture, built_architecture, moe_layer, refused",
    [
        (True, "DeepseekV3ForCausalLM", "DeepseekV3ForCausalLM", True, None),
        (True, "Qwen3_5MoeForCausalLM", "Qwen3_5MoeForCausalLM", True, "hybrid linear-attention"),
        # The built multimodal model keeps its language model's configuration.
        (
            True,
            "Qwen3_5MoeForConditionalGeneration",
            "Qwen3_5MoeForCausalLM",
            True,
            "hybrid linear-attention",
        ),
        (True, "DeepseekV3ForCausalLM", "DeepseekV3ForCausalLM", False, "a model with MoE layers"),
        (False, "Qwen3_5MoeForCausalLM", "Qwen3_5MoeForCausalLM", False, None),
    ],
    ids=["supported", "hybrid", "hybrid_multimodal", "no_moe_layers", "without_fault_tolerance"],
)
def test_unsupported_models_are_refused_once_built_before_any_weight_loads(
    tmp_path,
    monkeypatch,
    fault_tolerance,
    checkpoint_architecture,
    built_architecture,
    moe_layer,
    refused,
):
    from contextlib import nullcontext

    from tensorrt_llm._torch.moe.fused_moe.configurable_moe import ConfigurableMoE
    from tensorrt_llm._torch.pyexecutor import model_loader as model_loader_module
    from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader

    model = torch.nn.Module()
    model.model_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(architectures=[built_architecture])
    )
    if moe_layer:
        layer = ConfigurableMoE.__new__(ConfigurableMoE)
        torch.nn.Module.__init__(layer)
        layer.comm = SimpleNamespace(bind_execution_state=Mock(), ep_group_health=object())
        model.experts = layer
    checkpoint_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(architectures=[checkpoint_architecture])
    )
    loader = ModelLoader(
        llm_args=TorchLlmArgs(model=str(tmp_path)),
        mapping=Mapping(),
        spec_config=None,
        sparse_attention_config=None,
        max_num_tokens=8,
        max_seq_len=8,
    )
    monkeypatch.setattr(loader, "_load_and_validate_config", lambda *_args: checkpoint_config)
    monkeypatch.setattr(model_loader_module, "MetaInitMode", nullcontext)
    monkeypatch.setattr(
        model_loader_module, "maybe_create_moe_load_balancer", lambda *_args: nullcontext()
    )
    monkeypatch.setattr(model_loader_module, "_checkpoint_startup_metadata", lambda *_args: {})
    monkeypatch.setattr(
        model_loader_module.AutoModelForCausalLM, "from_config", Mock(return_value=model)
    )
    monkeypatch.setattr(
        loader, "_qualify_post_transform_profile", Mock(side_effect=_PastModelBuild)
    )
    monkeypatch.setattr(_utils, "_wide_ep_fault_tolerance", fault_tolerance)
    checkpoint_loader = MagicMock()

    with pytest.raises(ValueError if refused else _PastModelBuild) as error:
        loader.load(str(tmp_path), checkpoint_loader)
    if refused:
        assert refused in str(error.value)
        assert "enable_wide_ep_fault_tolerance=False" in str(error.value)


def _ipc_worker(monkeypatch, fault_tolerance=True):
    """Rank 0's executor worker past its startup, under the proxy, with two postprocessing workers: the responses of a
    request go to its postprocessing worker, its errors to the result queue."""
    from tensorrt_llm.executor import base_worker

    monkeypatch.setattr(
        base_worker,
        "GenerationResult",
        lambda request, **kwargs: SimpleNamespace(
            _generation_request=request,
            _params_transmitted=False,
            _streaming=True,
            sampling_params="sampling params",
            postproc_params="postproc params",
            disaggregated_params=None,
        ),
    )
    monkeypatch.setattr(base_worker, "_maybe_wrap_response", lambda worker, response: response)
    worker = BaseWorker.__new__(BaseWorker)
    worker.rank = 0
    worker._results = {}
    worker._client_id_to_request_id = {}
    worker._refused_client_ids = set()
    worker._unanswered_requests = {} if fault_tolerance else None
    worker.start = Mock()
    worker._get_logprob_params = Mock(return_value=None)
    worker._handle_background_error = Mock()
    worker._has_background_error = Mock(return_value=False)
    worker._enqueue_request = Mock(side_effect=range(100, 200))
    worker.postproc_config = SimpleNamespace(num_postprocess_workers=2)
    worker.postproc_queues = [Mock(), Mock()]
    worker.frontend_result_queues = None
    worker.result_queue = Mock()
    helper = base_worker.AwaitResponseHelper.__new__(base_worker.AwaitResponseHelper)
    helper.worker = worker
    helper.enable_postprocprocess_parallel = True
    return worker, helper


def _submit(worker, client_id, multimodal_params=None):
    request = SimpleNamespace(
        id=client_id, multimodal_params=multimodal_params, disaggregated_params=None
    )
    worker.submit(request)
    return request


def _sent(worker):
    """client id -> what reached the client: the responses sent to the postprocessing workers, and the errors'
    types."""
    sent = collections.defaultdict(list)
    for postproc_queue in worker.postproc_queues:
        for (batch,), _ in postproc_queue.put.call_args_list:
            for item in batch:
                sent[item.rsp.client_id].append(("output", item.sampling_params))
    for (batch,), _ in worker.result_queue.put.call_args_list:
        for error in batch:
            assert isinstance(error, ErrorResponse)
            sent[error.client_id].append(("error", type(error.error_msg)))
    return dict(sent)


def _output(client_id, final=False):
    return LlmResponse(request_id=1, result=SimpleNamespace(is_final=final), client_id=client_id)


def _ended_by_rank_failure(client_id):
    return LlmResponse(request_id=1, error_msg=RankFailureError(), client_id=client_id)


def test_the_worker_enqueues_a_request_a_rank_failure_ended_again_once_if_it_sent_none_of_it(
    monkeypatch,
):
    """Rank 0's worker enqueues such a request again from its own copy, under the same client id: the client, and the
    postprocessing worker that receives the request's first response with its parameters, see none of the first
    attempt. Once the request sent a response, or was enqueued again once already, the client gets the error."""
    worker, helper = _ipc_worker(monkeypatch)
    requests = {client_id: _submit(worker, client_id) for client_id in (7, 8)}
    helper.handle_for_ipc_batched([_output(8), _ended_by_rank_failure(7)])
    assert worker._enqueue_request.call_args_list[-1].args == (requests[7],)
    assert worker._client_id_to_request_id == {7: 102, 8: 101}
    assert _sent(worker) == {8: [("output", "sampling params")]}

    helper.handle_for_ipc_batched([_output(7), _ended_by_rank_failure(8)])
    assert _sent(worker) == {
        7: [("output", "sampling params")],  # the first response of the request enqueued again
        8: [("output", "sampling params"), ("error", RankFailureError)],
    }
    assert 8 not in worker._results
    helper.handle_for_ipc_batched([_ended_by_rank_failure(7)])
    assert _sent(worker)[7][-1] == ("error", RankFailureError)
    assert worker._enqueue_request.call_count == 3

    _submit(worker, 9)
    helper.handle_for_ipc_batched([_ended_by_rank_failure(9)])
    helper.handle_for_ipc_batched([_ended_by_rank_failure(9)])  # enqueued again once already
    assert _sent(worker)[9] == [("error", RankFailureError)]
    assert worker._enqueue_request.call_count == 5
    assert worker._unanswered_requests == {} and worker._results == {}


def test_the_worker_enqueues_a_multimodal_request_again_with_its_multimodal_input(monkeypatch):
    """The first enqueue consumes the multimodal input, and an in-process executor then drops the request's
    multimodal parameters: the request is enqueued again with both, as first sent."""
    worker, helper = _ipc_worker(monkeypatch)
    multimodal_input = object()
    params = SimpleNamespace(multimodal_input=multimodal_input)
    enqueued = []

    def enqueue(request):
        enqueued.append((request.multimodal_params, request.multimodal_params.multimodal_input))
        request.multimodal_params.multimodal_input = None
        return 100 + len(enqueued)

    worker._enqueue_request = Mock(side_effect=enqueue)
    request = _submit(worker, 7, multimodal_params=params)
    del request.multimodal_params
    helper.handle_for_ipc_batched([_ended_by_rank_failure(7)])
    assert enqueued == [(params, multimodal_input)] * 2
    assert _sent(worker) == {}
    assert worker._client_id_to_request_id == {7: 102}


def test_without_fault_tolerance_the_worker_enqueues_no_request_again(monkeypatch):
    worker, helper = _ipc_worker(monkeypatch, fault_tolerance=False)
    _submit(worker, 7)
    helper.handle_for_ipc_batched([_ended_by_rank_failure(7)])
    assert _sent(worker) == {7: [("error", RankFailureError)]}
    assert worker._enqueue_request.call_count == 1


def test_a_request_whose_enqueue_fails_is_not_enqueued_again(monkeypatch):
    worker, helper = _ipc_worker(monkeypatch)
    worker._enqueue_request = Mock(side_effect=RequestError("refused"))
    with pytest.raises(RequestError):
        _submit(worker, 7)
    assert worker._unanswered_requests == {}


def test_the_rank_failure_error_reaches_the_client_as_its_type():
    """The worker's error response crosses the IPC boundary pickled; the client's result raises the error itself."""
    received = pickle.loads(pickle.dumps(ErrorResponse(7, RankFailureError(), 1)))
    assert isinstance(received.error_msg, RankFailureError)
    with pytest.raises(RankFailureError, match=str(RankFailureError())):
        GenerationExecutor._handle_background_error(Mock(), received.error_msg)


@pytest.mark.parametrize(
    "error, status, code",
    [
        (RankFailureError(), 503, RANK_FAILURE_CODE),
        (RequestError(str(RankFailureError())), 400, 400),  # the type decides, not the text
        (RequestError("bad request"), 400, 400),
        ("bad request", 400, 400),
    ],
)
def test_a_request_a_rank_failure_ended_answers_503_with_its_code(error, status, code):
    """Clients send a request again after a 503, never after a 400."""
    from tensorrt_llm.serve.openai_server import OpenAIServer

    response = OpenAIServer.create_error_response(error)
    body = json.loads(response.body)
    assert response.status_code == status and body["code"] == code
    assert "retry-after" not in response.headers


@pytest.mark.parametrize("rank_failure", [False, True])
def test_a_streamed_request_a_rank_failure_ended_gets_its_code(rank_failure):
    """A stream that already answered 200 ends with an error event that carries the code."""
    from tensorrt_llm.serve.openai_server import _stream_error_events

    error = RankFailureError() if rank_failure else RequestError(str(RankFailureError()))
    events = _stream_error_events(error)
    assert events[-1] == "data: [DONE]\n\n"
    error = json.loads(events[0][len("data: ") :])["error"]
    assert error["code"] == (RANK_FAILURE_CODE if rank_failure else None)


@pytest.mark.parametrize("fault_tolerance", [False, True])
def test_the_ranks_the_llm_spawns_receive_the_join_secret_and_the_rendezvous(
    fault_tolerance, monkeypatch
):
    """Plain trtllm-serve and the LLM API spawn their ranks in the LLM's MPI pool, before the executor proxy exists.

    With fault tolerance and neither variable exported, the pool chooses both before it captures the environment
    its workers receive.
    """
    from tensorrt_llm.llmapi import llm as llm_module
    from tensorrt_llm.llmapi import mpi_session, startup_checks

    for name in (JOIN_SECRET_ENV, RENDEZVOUS_ENV):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)  # unset now, and unset again afterwards
    monkeypatch.setattr(recovery_module, "_rendezvous_ports", [])
    # The LLM sets it; restored afterwards.
    monkeypatch.setattr(_utils, "_wide_ep_fault_tolerance", False)
    monkeypatch.setattr(llm_module, "MpiPoolSession", mpi_session.MpiPoolSession)
    monkeypatch.setattr(llm_module, "get_spawn_proxy_process_env", lambda: False)
    monkeypatch.setattr(startup_checks, "get_device_count", lambda: 4)
    pool = Mock(side_effect=RuntimeError("stop"))
    monkeypatch.setattr(mpi_session, "MPIPoolExecutor", pool, raising=False)
    with pytest.raises(RuntimeError, match="stop"):
        LLM(
            model="dummy",
            tensor_parallel_size=4,
            enable_attention_dp=True,
            moe_config={"load_balancer": {"num_slots": 8}},
            enable_wide_ep_fault_tolerance=fault_tolerance,
        )
    env = pool.call_args.kwargs["env"]
    for name in (JOIN_SECRET_ENV, RENDEZVOUS_ENV):
        assert (name in env) == fault_tolerance
        assert env.get(name) == os.environ.get(name)


def test_rank_0_ends_the_membership_with_its_report(monkeypatch):
    """After a normal end, rank 0's report asks the authority for the SHUTDOWN record."""
    monkeypatch.setattr(recovery_module.os, "_exit", Mock(side_effect=SystemExit))
    recovery = _recovery(rank=0)
    recovery._record = _record(MembershipState.SHUTDOWN)
    with pytest.raises(SystemExit):
        recovery.stop(True)
    assert {"shutdown": True} in recovery._agent.reports


def test_the_rendezvous_ports_the_server_chooses_stay_reserved_for_the_authority(monkeypatch):
    """The authority's store and liveness listener bind them only once the model has loaded; until then no other
    process takes them. The address is an IP address, which every node reaches without resolving a host name."""
    from torch.distributed import TCPStore

    for name in (JOIN_SECRET_ENV, RENDEZVOUS_ENV):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)  # unset now, and unset again afterwards
    monkeypatch.setattr(recovery_module, "_rendezvous_ports", [])
    recovery_module.ensure_rendezvous()
    bytes.fromhex(os.environ[JOIN_SECRET_ENV])
    host, store_port, liveness_port = membership.authority_ports(os.environ[RENDEZVOUS_ENV])
    socket.inet_aton(host)
    for port in (store_port, liveness_port):
        with socket.socket() as other, pytest.raises(OSError):
            other.bind(("", port))
    # As the authority binds them.
    store = TCPStore(host, store_port, is_master=True, wait_for_workers=False)
    assert store.port == store_port
    with socket.create_server((host, liveness_port)) as listener:
        assert listener.getsockname()[1] == liveness_port


class FakeSignedStore:
    def __init__(self, record, transaction=None):
        self.values = {
            key: value.model_dump_json().encode()
            for key, value in ((membership.RECORD_KEY, record), (TRANSACTION_KEY, transaction))
            if value is not None
        }

    def get_if_set(self, key):
        return self.values.get(key)


@pytest.mark.parametrize(
    "record, state",
    [
        (_record(MembershipState.COMMITTED, live_mask=0b1111), "healthy"),
        (_record(MembershipState.COMMITTED), "degraded"),
        (_record(MembershipState.INSTALL), "recovering"),
        (_record(MembershipState.ABORT), "stopped"),
        (None, "unprotected"),
    ],
)
def test_health_state(monkeypatch, record, state):
    monkeypatch.setattr(
        recovery_module, "open_store", lambda *args, **kwargs: FakeSignedStore(record)
    )
    assert FaultToleranceHealth("host:1", b"secret").status()["state"] == state


# Not JSON; JSON without a record's fields; JSON that is not an object.
@pytest.mark.parametrize("value", [b"not a membership record", b"{}", b"[1, 2]"])
def test_health_keeps_the_record_past_a_value_it_cannot_parse(monkeypatch, value):
    """A store value that does not parse is logged and counts as absent; the next one that parses is read."""
    store = FakeSignedStore(_record(MembershipState.COMMITTED))
    monkeypatch.setattr(recovery_module, "open_store", lambda *args, **kwargs: store)
    health = FaultToleranceHealth("host:1", b"secret")
    assert health.status()["state"] == "degraded"
    store.values[membership.RECORD_KEY] = value
    store.values[TRANSACTION_KEY] = value
    status = health.status()
    assert status["state"] == "degraded" and status["last_transaction"] is None
    store.values[membership.RECORD_KEY] = (
        _record(MembershipState.COMMITTED, live_mask=0b1111).model_dump_json().encode()
    )
    assert health.status()["state"] == "healthy"


def test_health_connects_to_a_store_that_does_not_answer_at_most_once_per_read_timeout(monkeypatch):
    """Probes served while the authority's store does not answer do not each wait for it."""
    monkeypatch.setattr(recovery_module, "_STATUS_READ_TIMEOUT_S", 0.5)
    open_store = Mock(side_effect=OSError("the store does not answer"))
    monkeypatch.setattr(recovery_module, "open_store", open_store)
    health = FaultToleranceHealth("host:1", b"secret")
    for _ in range(3):
        status = health.status()
        assert status["state"] == "unprotected" and not status["authority_alive"]
    assert open_store.call_count == 1
    time.sleep(0.5)
    open_store.side_effect = None
    open_store.return_value = FakeSignedStore(_record(MembershipState.COMMITTED, live_mask=0b1111))
    assert health.status()["state"] == "healthy"
    assert open_store.call_count == 2


def test_health_counts_the_authority_alive_while_its_store_answers(monkeypatch):
    """The authority's process hosts the store: the store answers exactly while that process runs."""
    store = FakeSignedStore(_record(MembershipState.COMMITTED, live_mask=0b1111))
    monkeypatch.setattr(recovery_module, "open_store", lambda *args, **kwargs: store)
    health = FaultToleranceHealth("host:1", b"secret")
    assert health.status()["authority_alive"] and health.status()["state"] == "healthy"
    store.get_if_set = Mock(side_effect=RuntimeError("the store does not answer"))
    status = health.status()
    assert not status["authority_alive"] and status["state"] == "unprotected"


def test_health_reads_no_store_once_the_authority_ended(monkeypatch):
    """The authority hosts the store and is not restarted: once a read of the store that answered fails, later reads
    answer unprotected at once, with the last membership read, without the store."""
    monkeypatch.setattr(recovery_module, "_STATUS_READ_TIMEOUT_S", 0.0)
    store = FakeSignedStore(_record(MembershipState.COMMITTED, live_mask=0b1111))
    open_store = Mock(return_value=store)
    monkeypatch.setattr(recovery_module, "open_store", open_store)
    health = FaultToleranceHealth("host:1", b"secret")
    assert health.status()["state"] == "healthy"
    store.get_if_set = Mock(side_effect=RuntimeError("the store does not answer"))
    assert health.status()["state"] == "unprotected"
    store.get_if_set.reset_mock()
    for _ in range(3):
        status = health.status()
        assert status["state"] == "unprotected" and not status["authority_alive"]
        assert status["live_ranks"] == list(range(WORLD))
    store.get_if_set.assert_not_called()
    assert open_store.call_count == 1


def _server(monkeypatch, store):
    """A server whose executor reads the membership from ``store`` when a probe is served (None: the store does not
    answer)."""
    from tensorrt_llm.serve.openai_server import OpenAIServer

    monkeypatch.setattr(_utils, "_wide_ep_fault_tolerance", True)
    monkeypatch.setattr(
        recovery_module,
        "open_store",
        Mock(side_effect=OSError("no store")) if store is None else Mock(return_value=store),
    )
    server = OpenAIServer.__new__(OpenAIServer)
    server.generator = MagicMock(_check_health=Mock(return_value=True))
    server.embedding_batcher = None
    server.generator._executor.fault_tolerance_health = FaultToleranceHealth("host:1", b"secret")
    return server


def test_status_endpoint_reports_the_membership_and_the_last_transaction(monkeypatch):
    began_unix_ms = 1_762_000_000_000
    transaction = TransactionSummary(
        generation=1,
        outcome="COMMITTED",
        removed_ranks=(FAILED,),
        added_ranks=(),
        detected_unix_ms=began_unix_ms - 12,
        began_unix_ms=began_unix_ms,
        ready_unix_ms=began_unix_ms + 400,
        ended_unix_ms=began_unix_ms + 1500,
    )
    server = _server(monkeypatch, FakeSignedStore(_record(MembershipState.COMMITTED), transaction))
    response = asyncio.run(server.fault_tolerance_status())
    assert response.status_code == 200
    assert json.loads(response.body) == {
        "state": "degraded",
        "authority_alive": True,
        "generation": 1,
        "live_ranks": [0, 1, 3],
        "lost_ranks": [FAILED],
        "incarnations": [0] * WORLD,
        "last_transaction": {
            "generation": 1,
            "outcome": "COMMITTED",
            "removed_ranks": [FAILED],
            "added_ranks": [],
            "detected_unix_ms": began_unix_ms - 12,
            "began_unix_ms": began_unix_ms,
            "ready_unix_ms": began_unix_ms + 400,
            "ended_unix_ms": began_unix_ms + 1500,
        },
    }


def test_status_endpoint_while_the_store_does_not_answer(monkeypatch):
    server = _server(monkeypatch, None)
    status = json.loads(asyncio.run(server.fault_tolerance_status()).body)
    assert status["state"] == "unprotected" and not status["authority_alive"]
    assert status["live_ranks"] is None and status["last_transaction"] is None


@pytest.mark.parametrize(
    "record, code",
    [
        (_record(MembershipState.COMMITTED), 200),
        # A liveness probe must not restart the server while the membership changes.
        (_record(MembershipState.PROPOSE), 200),
        (_record(MembershipState.INSTALL), 200),
        (_record(MembershipState.ABORT), 503),
        (None, 200),
    ],
)
def test_health_answers_503_only_once_the_deployment_stopped(monkeypatch, record, code):
    """The status code carries the health; the body stays main's (the state is the status endpoint's)."""
    server = _server(monkeypatch, None if record is None else FakeSignedStore(record))
    response = asyncio.run(server.health())
    assert response.status_code == code
    if code == 200:
        assert response.body == b""
    else:
        assert response.body.startswith(b"LLM is unavailable")


@pytest.mark.parametrize("fault_tolerance", [False, True])
def test_health_after_a_fatal_engine_error_answers_503(monkeypatch, fault_tolerance):
    server = _server(monkeypatch, None)
    monkeypatch.setattr(_utils, "_wide_ep_fault_tolerance", fault_tolerance)
    server.generator._check_health.return_value = False
    server.generator._executor._fatal_error = RuntimeError("engine died")
    server.generator._executor.doing_shutdown = True  # the shutdown already started: no signal
    response = asyncio.run(server.health())
    assert response.status_code == 503
    assert response.body.startswith(b"LLM is unavailable")


@pytest.mark.timeout(60)
def test_health_stays_stopped_after_abort_while_a_rank_lingers(monkeypatch):
    """ABORT takes main's fatal path (/health answers 503) while the store answers: the authority keeps its store up
    until every live rank has left."""
    from torch.distributed import TCPStore

    secret = b"join-secret"
    master = TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
    store = SignedStore(master, secret)
    authority = membership.AuthorityProcess(2, store, membership.Authority(2))
    thread = threading.Thread(target=authority.run, kwargs={"host": "127.0.0.1"}, daemon=True)
    thread.start()
    _wait_until(lambda: store.get_if_set(membership.LIVENESS_KEY) is not None)
    connections = [membership.open_liveness_connection(store, rank, 0) for rank in range(2)]
    for connection in connections:  # every rank's startup is done
        connection.sock.sendall(membership._line(membership.RankReport(started=True, sequence=1)))
    health = FaultToleranceHealth(f"127.0.0.1:{master.port}", secret)
    try:
        _wait_until(lambda: health.status()["state"] == "healthy", timeout_s=10)
        connections[0].sock.close()  # rank 0's process ends: the deployment aborts
        _wait_until(lambda: health.status()["state"] == "stopped", timeout_s=10)
        # Rank 1 has not left yet.
        assert thread.is_alive()
        status = health.status()
        assert status["authority_alive"]
    finally:
        for connection in connections:
            connection.sock.close()
        thread.join(10)
    assert status["state"] == "stopped"
    server = _server(monkeypatch, None)
    server.generator._executor.fault_tolerance_health = health
    response = asyncio.run(server.health())
    assert response.status_code == 503
    assert response.body.startswith(b"LLM is unavailable")


def _start_authority_at_a_free_port(monkeypatch):
    """Rank 0's authority process at a free rendezvous port. The authority binds the port itself, so another process
    may take it first: another port is then tried."""
    for attempt in range(3):
        port = _utils.get_free_port()
        monkeypatch.setenv(RENDEZVOUS_ENV, f"localhost:{port}")
        try:
            return recovery_module._start_authority(world_size=1, timeout_s=120.0)
        except RuntimeError:
            if attempt == 2 or not _port_in_use(port):
                raise


def _port_in_use(port):
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # as a listening store binds
        try:
            probe.bind(("localhost", port))
        except OSError as error:
            if error.errno != errno.EADDRINUSE:
                raise
            return True
    return False


@pytest.mark.timeout(600)
def test_authority_process(monkeypatch):
    """Rank 0's authority process starts, hosts its store at the rendezvous address and reports it, serves a rank,
    and stops once the rank has left after the SHUTDOWN record."""
    from torch.distributed import TCPStore

    secret = bytes.fromhex("ab" * 32)
    monkeypatch.setenv(JOIN_SECRET_ENV, secret.hex())
    process, address = _start_authority_at_a_free_port(monkeypatch)
    try:
        # Where a restarted process finds the deployment.
        assert address == os.environ[RENDEZVOUS_ENV]
        host, port = address.rsplit(":", 1)
        store = SignedStore(TCPStore(host, int(port), is_master=False), secret)
        agent = MembershipAgent(0, store, on_record=lambda record: None)
        agent.start()
        _wait_until(lambda: agent.record is not None and agent.connected, timeout_s=30)
        agent.report(shutdown=True)
        _wait_until(lambda: agent.record.state == MembershipState.SHUTDOWN, timeout_s=30)
        with pytest.raises(subprocess.TimeoutExpired):
            process.wait(3)  # the store stays up while the rank is connected
        agent.stop()
        assert process.wait(30) == 0
    finally:
        process.kill()


@pytest.mark.timeout(600)
def test_the_authority_process_ends_once_the_process_that_started_it_ends(monkeypatch):
    """That process holds the only writing end of the authority's standard input: once it ends, the authority ends
    too, before any rank connected or after."""
    monkeypatch.setenv(JOIN_SECRET_ENV, ("ab" * 32))
    process, _ = _start_authority_at_a_free_port(monkeypatch)
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            process.wait(3)
        process.stdin.close()  # as when this process ends
        assert process.wait(30) == 1
    finally:
        process.kill()


def test_the_authority_starts_outside_the_launcher_job(monkeypatch):
    """With the launcher's rank identity, MPI would initialize the authority as a member of the job."""
    for name in ("OMPI_COMM_WORLD_RANK", "PMIX_RANK", "PMI_RANK", "PRTE_LAUNCHED", "SLURM_PROCID"):
        monkeypatch.setenv(name, "0")
    launched = {}

    def popen(args, env, pass_fds, stdin):
        launched.update(env)
        os.write(pass_fds[0], b"host:1\n")
        return Mock()

    monkeypatch.setattr(recovery_module.subprocess, "Popen", popen)
    _, address = recovery_module._start_authority(world_size=4, timeout_s=120.0)
    assert address == "host:1"
    assert not [
        name for name in launched if name.startswith(("OMPI_", "PMIX_", "PMI_", "PRTE_", "SLURM_"))
    ]
    assert launched["CUDA_VISIBLE_DEVICES"] == ""
    assert launched["MPI4PY_RC_INITIALIZE"] == "false"
    assert launched["TRT_LLM_NO_LIB_INIT"] == "1"  # importing tensorrt_llm would run MPI_Init


@pytest.mark.timeout(60)
def test_an_authority_that_does_not_report_its_address_is_killed(monkeypatch):
    authority = Mock(wait=Mock(return_value=-9))
    held = []

    def popen(args, env, pass_fds, stdin):
        held.append(os.dup(pass_fds[0]))  # a hung authority keeps the pipe open and writes nothing
        return authority

    monkeypatch.setattr(recovery_module.subprocess, "Popen", popen)
    try:
        with pytest.raises(RuntimeError, match="did not report where its store listens"):
            recovery_module._start_authority(world_size=4, timeout_s=0.1)
    finally:
        os.close(held[0])
    authority.kill.assert_called_once()


@pytest.mark.parametrize("rank", [0, 1])
def test_a_rank_keeps_serving_unprotected_when_its_membership_agent_fails(monkeypatch, rank):
    """The authority is gone: the deployment serves on; were rank 0 gone too, main's hang detector ends the others."""
    monkeypatch.setattr(recovery_module.os, "_exit", Mock(side_effect=SystemExit))
    recovery = _recovery(rank=rank)
    recovery._on_agent_error(OSError("the authority is gone"))
    recovery_module.os._exit.assert_not_called()
    recovery._moe.request_abort.assert_not_called()


@pytest.mark.parametrize(
    "state, status",
    [(MembershipState.ABORT, 1), (MembershipState.SHUTDOWN, 0), (MembershipState.STARTING, None)],
)
def test_before_its_executor_serves_a_rank_exits_once_the_deployment_ends(
    monkeypatch, state, status
):
    """A loss during the startup aborts the deployment: every rank ends at once rather than at a timeout."""
    _exit_raises(monkeypatch)
    monkeypatch.setattr(recovery_module, "_agent", Mock(rank=1))
    record = MembershipRecord(generation=0, state=state, live_mask=0b1111)
    if status is None:
        recovery_module._on_record_at_startup(record)
        return
    with pytest.raises(SystemExit) as exited:
        recovery_module._on_record_at_startup(record)
    assert exited.value.code == status
    with pytest.raises(SystemExit) as exited:
        recovery_module._on_error_at_startup(ConnectionError("the authority is gone"))
    assert exited.value.code == 1


# Rejoin: a replacement rank starts alone, then the live ranks admit it.

JOINED = (0, 0, 1, 0)  # rank 2's first replacement stands for it


def _joining_record(state, joining=1 << FAILED, generation=2):
    resume = ResumeState(iteration=20, epoch=40)
    return MembershipRecord(
        generation=generation,
        state=state,
        live_mask=0b1111,
        resume=resume,
        incarnations=JOINED,
        joining_mask=joining,
    )


def _exit_raises(monkeypatch):
    def exit_(code):
        raise SystemExit(code)

    monkeypatch.setattr(recovery_module.os, "_exit", exit_)


@pytest.fixture
def agreed_startup(monkeypatch):
    """Fault tolerance on, an original rank, nothing recorded yet."""
    monkeypatch.setattr(_utils, "_wide_ep_fault_tolerance", True)
    monkeypatch.delenv(FAULT_TOLERANCE_RANK_ENV, raising=False)
    for name, value in (
        ("_startup_fingerprint", None),
        ("_worker_main_args", None),
        ("_published_record", None),
    ):
        monkeypatch.setattr(recovery_module, name, value)
    for name, value in vars(StartupRecord()).items():
        monkeypatch.setattr(startup_record, name, value)


def _published(monkeypatch, worker_main_args, identity="2,4,4,1"):
    """A rank other than 0 whose launcher gave it ``identity``, and the rendezvous store it reads, with
    ``worker_main_args`` as rank 0 published them; its membership agent is not started. Returns the store."""
    monkeypatch.setenv(JOIN_SECRET_ENV, "ab" * 32)
    monkeypatch.setenv(RENDEZVOUS_ENV, "head-node:29600")
    monkeypatch.setenv(FAULT_TOLERANCE_RANK_ENV, identity)
    store = FakeStore()
    store.set(WORKER_ARGS_KEY, pickle.dumps(worker_main_args))
    monkeypatch.setattr(recovery_module, "_startup_store", lambda: store)
    monkeypatch.setattr(recovery_module, "_join_at_entry", lambda store, identity: None)
    return store


class FakeProfilingCache:
    """Tactics by key; serialized with string keys."""

    def __init__(self, cache):
        self.cache = dict(cache)

    def _partition_cache_by_strategy(self):
        return dict(self.cache), {}

    def _serialize_cache_data(self, cache):
        return {json.dumps(key): value for key, value in cache.items()}

    def _deserialize_cache_data(self, data):
        return {tuple(json.loads(key)): tuple(value) for key, value in data.items()}


def _replacement_of(monkeypatch, published: bytes) -> None:
    """This process becomes a replacement rank that replays ``published``, the agreed-startup record rank 0
    published, as ``worker_node_main`` makes it."""
    for name, value in vars(StartupRecord()).items():
        setattr(startup_record, name, value)
    monkeypatch.setenv(FAULT_TOLERANCE_RANK_ENV, "2,4,4,1")
    monkeypatch.setattr(recovery_module, "_published_record", pickle.loads(published))
    startup_record.replay(recovery_module._agreed_startup())


def test_a_replacement_takes_the_startup_values_rank_0_recorded(agreed_startup, monkeypatch):
    gathers = []

    def gather(value):
        return lambda: (gathers.append(value), value)[1]

    startup_record.record()
    assert startup_record.agree("rank_hosts", gather(["a", "a", "b", "b"])) == ["a", "a", "b", "b"]
    # A second split, in the same order on every rank.
    assert startup_record.agree("startup_group_split", gather([[0, 0]] * WORLD)) == [[0, 0]] * WORLD
    assert startup_record.agree("startup_group_split", gather([[1, 0]] * WORLD)) == [[1, 0]] * WORLD
    # Rank 0 runs each collective once.
    assert gathers == [["a", "a", "b", "b"], [[0, 0]] * WORLD, [[1, 0]] * WORLD]
    tuned = FakeProfilingCache({("gemm", 8): (0, 3, 0.1)})
    monkeypatch.setattr(AutoTuner, "get", lambda: SimpleNamespace(profiling_cache=tuned))
    startup_record.adopt_tuned_tactics()  # rank 0 records its tactics and keeps them
    assert tuned.cache == {("gemm", 8): (0, 3, 0.1)}
    startup_record.close()
    _replacement_of(monkeypatch, recovery_module._agreed_startup_record())

    # Its peers are absent: it runs none of the collectives, and takes rank 0's results.
    gathers.clear()
    assert startup_record.agree("rank_hosts", gather(["c"] * WORLD)) == ["a", "a", "b", "b"]
    assert startup_record.agree("startup_group_split", gather(None)) == [[0, 0]] * WORLD
    with pytest.raises(SystemExit, match="takes the startup values") as exited:
        startup_record.close()  # a call of the original ranks not made yet
    assert exited.value.code == membership.REPLACEMENT_CANNOT_START
    assert startup_record.agree("startup_group_split", gather(None)) == [[1, 0]] * WORLD
    own = FakeProfilingCache({("gemm", 8): (0, 5, 0.2), ("gemm", 16): (0, 1, 0.3)})
    monkeypatch.setattr(AutoTuner, "get", lambda: SimpleNamespace(profiling_cache=own))
    startup_record.adopt_tuned_tactics()
    assert own.cache == {("gemm", 8): (0, 3, 0.1), ("gemm", 16): (0, 1, 0.3)}
    assert gathers == []
    startup_record.close()
    with pytest.raises(SystemExit, match="startup_group_split 3 times") as exited:
        startup_record.agree("startup_group_split", gather(None))  # one call more than rank 0 made
    assert exited.value.code == membership.REPLACEMENT_CANNOT_START


def test_without_fault_tolerance_startup_values_pass_through(agreed_startup, monkeypatch):
    monkeypatch.setattr(_utils, "_wide_ep_fault_tolerance", False)
    assert startup_record.agree("rank_hosts", lambda: ["a"] * WORLD) == ["a"] * WORLD
    assert startup_record.published() == {"values": {}, "exchanges": []}


class _LlmArgs:
    def __init__(self, **fields):
        self.fields = fields

    def model_dump(self):
        return dict(self.fields)


def test_a_replacement_that_does_not_match_the_deployment_refuses_to_start(
    agreed_startup, monkeypatch, tmp_path
):
    (tmp_path / "config.json").write_text('{"num_hidden_layers": 2}')
    # The configuration comes from rank 0: only the version and the checkpoint can differ.
    args = {
        "llm_args": _LlmArgs(max_batch_size=8, tokenizer=object()),
        "hf_model_dir": str(tmp_path),
    }
    fingerprint = recovery_module.startup_fingerprint(args)
    assert recovery_module.startup_fingerprint({**args, "llm_args": None}) == fingerprint
    (tmp_path / "config.json").write_text('{"num_hidden_layers": 3}')
    assert recovery_module.startup_fingerprint(args) != fingerprint

    monkeypatch.setattr(recovery_module, "_published_record", {"fingerprint": fingerprint})
    _published(monkeypatch, args)
    with pytest.raises(SystemExit, match="does not match") as exited:
        recovery_module.worker_node_main()
    # Another replacement would not match either: its supervisor starts none.
    assert exited.value.code == membership.REPLACEMENT_CANNOT_START


def test_the_deployment_nonce_is_rank_0_s_on_every_rank(agreed_startup, monkeypatch):
    """Rank 0 draws it in a startup collective; a replacement takes it from the startup record."""
    monkeypatch.setattr(recovery_module, "_deployment_nonce", None)
    broadcasts = []
    monkeypatch.setattr(
        recovery_module, "mpi_broadcast", lambda value: (broadcasts.append(value), value)[1]
    )
    monkeypatch.setattr(recovery_module, "mpi_rank", lambda: 0)
    startup_record.record()
    nonce = recovery_module.deployment_nonce()
    assert recovery_module.deployment_nonce() == nonce  # drawn once
    assert broadcasts == [nonce]
    startup_record.close()
    published = pickle.dumps(startup_record.published())

    _replacement_of(monkeypatch, published)
    monkeypatch.setattr(recovery_module, "_deployment_nonce", None)
    monkeypatch.setattr(recovery_module, "mpi_broadcast", Mock(side_effect=AssertionError("MPI")))
    assert recovery_module.deployment_nonce() == nonce


def test_a_rank_turns_fault_tolerance_on_before_it_imports_the_executor(
    agreed_startup, monkeypatch
):
    monkeypatch.setattr(_utils, "_wide_ep_fault_tolerance", False)
    seen = []
    worker_module = ModuleType("tensorrt_llm.executor.worker")

    def lookup(name):
        seen.append((name, fault_tolerance_mode()))
        return lambda **kwargs: None

    worker_module.__getattr__ = lookup
    monkeypatch.setitem(sys.modules, "tensorrt_llm.executor.worker", worker_module)
    args = {"hf_model_dir": None}
    monkeypatch.setattr(
        recovery_module,
        "_published_record",
        {**StartupRecord().published(), "fingerprint": recovery_module.startup_fingerprint(args)},
    )
    _published(monkeypatch, args)
    recovery_module.worker_node_main()
    # The import machinery also looks up module attributes such as __path__; worker_main is what matters.
    assert ("worker_main", True) in seen
    assert ("worker_main", False) not in seen


def test_module_entry_runs_the_imported_module(monkeypatch):
    """``python -m`` runs the module as ``__main__``: the rank joins in the module the executor imports."""
    worker_node_main = Mock()
    monkeypatch.setattr(recovery_module, "worker_node_main", worker_node_main)
    with warnings.catch_warnings():
        # runpy warns that the module it runs is imported already.
        warnings.simplefilter("ignore", RuntimeWarning)
        runpy.run_module(recovery_module.__name__, run_name="__main__")
    worker_node_main.assert_called_once_with()


@pytest.mark.parametrize("identity", ["2,4,4,0", "2,4,4,1"])
def test_every_rank_but_0_joins_then_starts_with_the_arguments_rank_0_published(
    agreed_startup, monkeypatch, identity
):
    """The original process of a rank and its replacements alike: no process hands it anything over MPI."""
    events = []
    worker_module = ModuleType("tensorrt_llm.executor.worker")
    worker_module.worker_main = lambda **kwargs: events.append(("worker_main", kwargs))
    monkeypatch.setitem(sys.modules, "tensorrt_llm.executor.worker", worker_module)
    args = {"hf_model_dir": None, "log_level": "info"}
    published = {
        "values": {"rank_hosts": [["a"] * WORLD]},
        "exchanges": [],
        "fingerprint": recovery_module.startup_fingerprint(args),
    }
    monkeypatch.setattr(recovery_module, "_published_record", published)
    _published(monkeypatch, args, identity)
    monkeypatch.setattr(
        recovery_module, "_join_at_entry", lambda store, rank: events.append(("join", rank))
    )
    recovery_module.worker_node_main()
    # A replacement replays rank 0's startup record; an original rank records nothing.
    replaying = identity.endswith(",1")
    assert startup_record.replaying == replaying
    assert startup_record.values == (published["values"] if replaying else {})
    rank = _utils.RankIdentity(*(int(field) for field in identity.split(",")))
    assert events == [("join", rank), ("worker_main", args)]


@pytest.mark.parametrize(
    "identity, entry",
    [("0,4,4,0", "worker_node_main"), ("1,4,4,0", "start_worker"), (None, "start_worker")],
)
def test_a_rank_started_by_the_wrong_entry_or_without_its_rank_fails_with_the_reason(
    agreed_startup, join_secret, monkeypatch, identity, entry
):
    if identity is None:
        monkeypatch.delenv(FAULT_TOLERANCE_RANK_ENV, raising=False)
    else:
        monkeypatch.setenv(FAULT_TOLERANCE_RANK_ENV, identity)
    monkeypatch.setattr(recovery_module, "_agent", None)
    with pytest.raises(ValueError, match="own process"):
        if entry == "start_worker":
            recovery_module.start_worker({"hf_model_dir": None})
        else:
            recovery_module.worker_node_main()


def test_rank_0_starts_the_authority_then_publishes_its_arguments_without_its_ipc_keys_and_joins(
    agreed_startup, join_secret, monkeypatch
):
    """The store signs its values but does not encrypt them; no rank but 0 uses the IPC keys."""
    from tensorrt_llm.executor.utils import WorkerCommIpcAddrs

    events = []
    store = FakeStore()
    monkeypatch.setenv(FAULT_TOLERANCE_RANK_ENV, "0,4,4,0")
    monkeypatch.setattr(recovery_module, "_agent", None)
    monkeypatch.setattr(
        recovery_module,
        "_start_authority",
        lambda world_size, timeout_s: (
            events.append(("authority", world_size)),
            (Mock(), "head-node:29600"),
        )[1],
    )
    monkeypatch.setattr(recovery_module, "_startup_store", lambda: store)
    monkeypatch.setattr(
        recovery_module,
        "_join_at_entry",
        lambda joined, identity: events.append(("join", sorted(store.values), identity.rank)),
    )
    queues = WorkerCommIpcAddrs(
        ("ipc://request", b"request-key"),
        ("ipc://status", b"status-key"),
        ("ipc://result", b"result-key"),
    )
    args = {
        "worker_queues": queues,
        "rpc_addr": "ipc://rpc",
        "hmac_key": b"rpc-key",
        "log_level": "info",
        "hf_model_dir": None,
        "llm_args": None,
    }
    recovery_module.start_worker(args)
    assert events == [
        ("authority", 4),
        ("join", [WORKER_ARGS_KEY], 0),
    ]
    # Rank 0 records its startup for the replacements.
    assert startup_record.agree("rank_hosts", lambda: ["a"] * WORLD) == ["a"] * WORLD
    assert startup_record.values == {"rank_hosts": [["a"] * WORLD]}
    published = store.values[WORKER_ARGS_KEY]
    assert not [
        key
        for key in (b"request-key", b"status-key", b"result-key", b"rpc-key")
        if key in published
    ]
    assert pickle.loads(published) == {
        **args,
        "worker_queues": WorkerCommIpcAddrs(None, None, None),
        "rpc_addr": None,
        "hmac_key": b"",
    }
    assert recovery_module._startup_fingerprint == recovery_module.startup_fingerprint(args)


@pytest.mark.parametrize("rank", [0, 1])
def test_an_original_rank_takes_over_its_membership_once_every_rank_has_started(
    agreed_startup, join_secret, monkeypatch, rank
):
    """Rank 0 publishes what a replacement starts with; nothing goes over MPI. First, every rank checks that the
    largest failure's restore fits its KV cache pool and keeps its device memory as the deployment serves; last, it
    reports that its startup is done, waits for the record that says every rank's is, and takes over the agent it
    joined with."""
    events = []
    store = FakeStore()
    agent = Mock(
        record=MembershipRecord(generation=0, state=MembershipState.STARTING, live_mask=0b1111),
        bind=Mock(side_effect=lambda *args, **kwargs: events.append("bind")),
        incarnation=0,
    )

    def report(**changes):
        events.append(("report", changes))
        agent.record = MembershipRecord(
            generation=0, state=MembershipState.COMMITTED, live_mask=0b1111
        )
        recovery_module._on_record_at_startup(agent.record)  # as the agent's thread delivers it

    agent.report = report
    monkeypatch.setattr(recovery_module, "_agent", agent)
    monkeypatch.setattr(recovery_module, "_every_rank_started", threading.Event())
    monkeypatch.setattr(recovery_module, "_authority", "rank 0's authority" if rank == 0 else None)
    monkeypatch.setattr(
        recovery_module, "rendezvous_store", lambda: (events.append("store"), store)[1]
    )
    monkeypatch.setattr(recovery_module, "_agreed_startup_record", lambda: b"agreed")
    executor = _executor(rank)
    executor.kv_cache_manager = device = _Device()
    monkeypatch.setattr(recovery_module.torch.cuda, "mem_get_info", lambda: (device.free, 1 << 40))

    @contextlib.contextmanager
    def kv_pools_suspended():
        yield device

    executor._kv_pools_suspended = kv_pools_suspended
    executor.dist.allgather.side_effect = lambda value: [value] * WORLD
    monkeypatch.setattr(
        recovery_module, "_moe_execution_state", Mock(workspace_bytes=Mock(return_value=4 * MIB))
    )
    executor.model_engine.moe_load_balancer = MagicMock(spec=MoeLoadBalancer)
    expert_restore = MagicMock(spec=EplbExpertRestore)
    executor.model_engine.moe_load_balancer.expert_restore = expert_restore
    expert_restore.largest_failure_restore_bytes.side_effect = lambda: (
        events.append("restore fits"),
        6 * MIB,
    )[1]
    expert_restore.keep_host_copies_for_the_deployment.side_effect = lambda: (
        events.append("host copies"),
        "moe_shared_deployment_",
    )[1]
    recovery = RankFailureRecovery.create(executor)
    # Until the deployment serves, a rank's failure ends it, and the rank's host copies with it; from then on the
    # node supervisor keeps them until the deployment ended, by the prefix rank 0 publishes.
    assert events == [
        "restore fits",
        "host copies",
        "store",
        ("report", {"started": True}),
        "bind",
    ]
    assert recovery.address == "head-node:29600"
    assert recovery._startup_memory == STARTUP_MEMORY
    # The check shrank the KV cache pool for the largest failure's restore, then took its quota back.
    assert _resizes(device) == [54 * MIB, STARTUP_MEMORY.kv_cache_quota]
    assert recovery._authority == ("rank 0's authority" if rank == 0 else None)
    expected = {HOST_WEIGHTS_PREFIX_KEY: b"moe_shared_deployment_", AGREED_STARTUP_KEY: b"agreed"}
    assert store.values == (expected if rank == 0 else {})


@pytest.mark.parametrize("margin_s, supported", [(-1, False), (0, True)])
def test_the_executor_s_hang_detection_must_outlast_twice_the_hang_timeout(
    agreed_startup, monkeypatch, margin_s, supported
):
    """A rank whose peer hangs waits until the authority excludes the peer, up to 1.5 hang timeouts: main's hang
    detector must not end the rank first."""
    executor = _executor()
    executor.hang_detector.timeout = 2 * PROGRESS_TIMEOUT_S + margin_s
    if not supported:
        with pytest.raises(ValueError, match="hang detection timeout of at least"):
            RankFailureRecovery.create(executor)
        return
    # Past the check, create() goes on to the startup memory: stop it there.
    executor.kv_cache_manager = None
    with pytest.raises(AttributeError, match="gpu_quota"):
        RankFailureRecovery.create(executor)


def test_a_rank_the_launcher_started_for_fault_tolerance_needs_it_on(monkeypatch):
    from tensorrt_llm.executor.worker import worker_main

    monkeypatch.setattr(_utils, "_wide_ep_fault_tolerance", False)
    monkeypatch.setenv(FAULT_TOLERANCE_RANK_ENV, "1,4,4,0")
    with pytest.raises(ValueError, match="enable_wide_ep_fault_tolerance is off"):
        worker_main(worker_queues=None, log_level="info", llm_args=None)


@pytest.mark.parametrize(
    "mpi_size, identity, refused",
    [(1, None, False), (4, None, True), (4, "0,4,4,0", False)],
    ids=["one-process", "every-rank-of-an-mpi-job", "under-trtllm-llmapi-launch"],
)
def test_fault_tolerance_starts_rank_0_s_worker_in_the_session_whatever_the_mpi_size(
    monkeypatch, mpi_size, identity, refused
):
    """Every rank is its own process: the MPI world of the frontend, which main's launch selection reads, plays no
    part, and a frontend started on every rank of an MPI job is refused. Under trtllm-llmapi-launch the frontend has
    rank 0's identity."""
    from tensorrt_llm.executor import executor as executor_module

    monkeypatch.setattr(_utils, "_wide_ep_fault_tolerance", True)
    if identity is None:
        monkeypatch.delenv(FAULT_TOLERANCE_RANK_ENV, raising=False)
    else:
        monkeypatch.setenv(FAULT_TOLERANCE_RANK_ENV, identity)
    monkeypatch.setattr(executor_module, "global_mpi_size", lambda: mpi_size)
    for name in ("external_mpi_comm_available", "need_spawn_mpi_workers"):
        monkeypatch.setattr(executor_module, name, Mock(side_effect=AssertionError(name)))
    created = Mock(return_value="proxy")
    monkeypatch.setattr(GenerationExecutor, "_create_ipc_executor", created)
    session = Mock()
    if refused:
        with pytest.raises(ValueError, match="not on every rank of an MPI job"):
            GenerationExecutor.create(model_world_size=4, mpi_session=session, return_logits=True)
        return
    assert (
        GenerationExecutor.create(model_world_size=4, mpi_session=session, return_logits=True)
        == "proxy"
    )
    assert created.call_args.kwargs["use_worker"] is False
    assert created.call_args.kwargs["mpi_session"] is session


@pytest.mark.parametrize(
    "restore_bytes, peer_fits, short",
    [(52 * MIB, True, None), (53 * MIB, True, [0]), (52 * MIB, False, [1])],
)
def test_the_startup_rejects_a_largest_failure_no_kv_cache_pool_can_restore(
    monkeypatch, restore_bytes, peer_fits, short
):
    """The startup applies the rule of every change (StartupMemory.fit_kv_cache) to the largest failure: the restore
    takes the backed spare slots and the remapped all-to-all workspaces (4 MiB) from the KV cache pool (64 MiB), down
    to the pool's minimum (8 MiB) at most, and the pool takes them back. All ranks take the same decision."""
    device = _Device()
    monkeypatch.setattr(recovery_module.torch.cuda, "mem_get_info", lambda: (device.free, 1 << 40))
    suspended = []

    @contextlib.contextmanager
    def kv_pools_suspended():
        suspended.append(True)
        yield device

    monkeypatch.setattr(
        recovery_module, "_moe_execution_state", Mock(workspace_bytes=Mock(return_value=4 * MIB))
    )
    executor = SimpleNamespace(
        dist=Mock(rank=0, allgather=lambda fits: [fits, peer_fits, True, True]),
        _kv_pools_suspended=kv_pools_suspended,
    )
    balancer = Mock(
        expert_restore=Mock(largest_failure_restore_bytes=Mock(return_value=restore_bytes))
    )
    if short is None:
        recovery_module._check_largest_failure_fits(executor, balancer, STARTUP_MEMORY)
    else:
        with pytest.raises(ValueError, match=re.escape(f"on ranks {short}")) as error:
            recovery_module._check_largest_failure_fits(executor, balancer, STARTUP_MEMORY)
        assert "more ranks, fewer EPLB slots per rank" in str(error.value)
    assert suspended == [True]
    # The pool tried the largest failure's quota, then took its startup quota back.
    assert _resizes(device)[0] == 64 * MIB - restore_bytes - 4 * MIB
    assert device.gpu_quota == STARTUP_MEMORY.kv_cache_quota


@pytest.mark.parametrize("identity", ["2,4,4,0", "2,4,4,1"], ids=["original", "replacement"])
def test_no_rank_creates_a_process_group(monkeypatch, identity):
    """Every rank is its own process: a rank's replacement would never join a torch.distributed (NCCL) group."""
    monkeypatch.setattr(_utils, "_wide_ep_fault_tolerance", True)
    monkeypatch.setenv(FAULT_TOLERANCE_RANK_ENV, identity)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="process group"):
        RankFailureRecovery.create(_executor(rank=FAILED))


def test_a_replacement_waits_for_its_admission_and_starts_where_the_live_ranks_resume(
    monkeypatch,
):
    _exit_raises(monkeypatch)
    moe = MagicMock()
    recovery = RankFailureRecovery(
        _executor(rank=FAILED),
        FakeExchange(),
        moe,
        FakeStore(),
        "host:1",
        STARTUP_MEMORY,
        incarnation=1,
    )
    recovery._agent = FakeAgent()
    recovery._executor.iter_counter = 0
    for r in (0, 1, 3):
        recovery._store.set(f"host_exchange_rows/2/0/{r}", pickle.dumps(f"rows-{r}".encode()))
    errors = []

    def join():
        try:
            recovery._join()
        except BaseException as error:  # noqa: BLE001 - reported by the test
            errors.append(error)

    joined = threading.Thread(target=join, daemon=True)
    joined.start()
    recovery._on_record(_record(MembershipState.COMMITTED))  # without rank 2: not admitted yet
    time.sleep(0.2)
    # Started, it reports that it is ready to join (no EPLB here: no placement), and only that.
    assert joined.is_alive() and recovery._agent.reports == [{"joining_ready": True}] and not errors
    moe.request_abort.assert_not_called()  # not a member yet: nothing to stop
    recovery._on_record(_joining_record(MembershipState.PROPOSE))
    _wait_until(lambda: len(recovery._agent.reports) == 2)
    assert recovery._agent.reports[-1]["reply"] == Reply.READY
    recovery._on_record(_joining_record(MembershipState.INSTALL))
    joined.join(5)
    assert not joined.is_alive() and not errors
    # Its fresh row is mapped by every live rank at INSTALL; it enters the host exchange at the resume point,
    # where it also restores and replies INSTALLED.
    assert recovery._exchange.installs == [
        ([b"rows-0", b"rows-1", b"row-mine", b"rows-3"], SURVIVORS, 40)
    ]
    assert all(report.get("reply") != Reply.INSTALLED for report in recovery._agent.reports)
    assert recovery._executor.iter_counter == 21 and recovery.holds_steps()


def _replacement():
    recovery = RankFailureRecovery(
        _executor(rank=FAILED),
        FakeExchange(),
        MagicMock(),
        FakeStore(),
        "host:1",
        STARTUP_MEMORY,
        incarnation=1,
    )
    recovery._agent = FakeAgent()
    return recovery


def test_a_replacement_is_admitted_by_the_committed_record_that_lists_it():
    recovery = _replacement()
    recovery._on_record(_joining_record(MembershipState.PROPOSE))
    assert not recovery._admitted
    recovery._on_record(_joining_record(MembershipState.COMMITTED, joining=0))
    assert recovery._admitted


def test_a_replacement_the_change_leaves_out_exits(monkeypatch):
    """Its supervisor starts another replacement; the live ranks commit without it."""
    _exit_raises(monkeypatch)
    recovery = _replacement()
    recovery._on_record(_joining_record(MembershipState.PROPOSE))
    recovery._next_record = Mock(
        side_effect=[
            _joining_record(MembershipState.PROPOSE),
            _record(MembershipState.PROPOSE, generation=3),  # without rank 2
        ]
    )
    with pytest.raises(SystemExit) as exited:
        recovery._change_membership()
    assert exited.value.code == 1
    assert _replies(recovery) == [Reply.READY]


def test_the_live_ranks_keep_the_host_exchange_to_themselves_while_a_rank_joins():
    """They map the joining rank's fresh row, but keep it out of the exchange until they resume."""
    recovery = _recovery(rank=1)
    for r in (0, 2, 3):
        recovery._store.set(f"host_exchange_rows/2/0/{r}", pickle.dumps(f"rows-{r}".encode()))
    recovery._on_record(_joining_record(MembershipState.PROPOSE))
    changed = threading.Thread(target=recovery._change_membership, daemon=True)
    changed.start()
    recovery._on_record(_joining_record(MembershipState.INSTALL))
    changed.join(5)
    assert not changed.is_alive()
    assert recovery._exchange.installs == [
        ([b"rows-0", b"row-mine", b"rows-2", b"rows-3"], SURVIVORS, 40)
    ]


def _admitting(monkeypatch, rank):
    """``rank`` at the first iteration after the change that adds rank 2."""
    monkeypatch.setattr(recovery_module.torch.cuda, "synchronize", lambda: None)
    recovery = _resuming_recovery(
        [set()] * (WORLD - 1), record=_joining_record(MembershipState.INSTALL), rank=rank
    )
    recovery.incarnation = JOINED[rank]
    recovery._committed_mask = SURVIVORS  # rank 2 was lost earlier
    recovery._moe.aborted.return_value = False  # an admission does not abort its steps
    recovery._resume_after = 20
    executor = recovery._executor
    executor.iter_counter = 21
    executor.waiting_queue = FCFSWaitingQueue()
    return recovery


def test_rank_0_admits_the_joining_rank_into_the_host_exchange(monkeypatch):
    recovery = _admitting(monkeypatch, rank=0)
    executor = recovery._executor
    executor.waiting_queue.add_requests([RequestQueueItem(7)])
    report = recovery._agent.report
    handed = {}

    def report_installed(**changes):
        if changes.get("reply") == Reply.INSTALLED:  # the joining rank has read them by then
            handed.update(recovery._store.values)
        report(**changes)

    recovery._agent.report = report_installed
    recovery.before_iteration()
    epoch = membership.next_epoch(40)
    assert handed["host_exchange_epoch/2"] == str(epoch).encode()
    assert recovery._exchange.admits == [(0b1111, epoch)] and recovery._exchange.installs == []
    # The joining ranks take the waiting queue from the store: no collective with the live ranks.
    queued = pickle.loads(handed["waiting_queue/2"])
    assert [item.id for item in queued] == [7]
    # Once committed, the store keeps neither.
    assert "host_exchange_epoch/2" not in recovery._store.values
    assert "waiting_queue/2" not in recovery._store.values
    executor.dist.broadcast.assert_not_called()
    recovery._moe.health.mark_active.assert_any_call(FAILED)
    assert _replies(recovery) == [Reply.INSTALLED] and not recovery.holds_steps()


@pytest.mark.timeout(60)
def test_a_live_rank_admits_the_joining_rank_from_its_own_epoch(monkeypatch):
    """The live ranks stand between the same two rounds: none waits for rank 0's epoch, which only joiners read."""
    recovery = _admitting(monkeypatch, rank=1)
    recovery.before_iteration()
    assert recovery._exchange.admits == [(0b1111, membership.next_epoch(40))]
    assert recovery._store.get_if_set("host_exchange_epoch/2") is None
    assert _replies(recovery) == [Reply.INSTALLED]


def test_a_joining_rank_enters_the_host_exchange_and_takes_rank_0_s_waiting_queue(monkeypatch):
    recovery = _admitting(monkeypatch, rank=FAILED)
    recovery._store.set("host_exchange_epoch/2", b"41")
    recovery._store.set("waiting_queue/2", pickle.dumps([RequestQueueItem(7)]))
    executor = recovery._executor
    recovery.before_iteration()
    assert recovery._exchange.admits == [(0b1111, 41)] and recovery._exchange.installs == []
    assert [item.id for item in executor.waiting_queue] == [7]
    executor.dist.broadcast.assert_not_called()
    # It ran no step, and enters the host exchange only at its admission.
    executor._retire_inflight_batch_for_control.assert_not_called()
    recovery._moe.remap_workspaces.assert_called_once()
    recovery._moe.rearm.assert_called_once()
    recovery._moe.install_mask.assert_called_once()
    assert _replies(recovery) == [Reply.INSTALLED] and not recovery.holds_steps()


def test_the_active_ranks_retire_the_step_in_flight_before_the_kv_cache_pool_is_resized(
    monkeypatch,
):
    """As main does before it acts on the KV cache, the active ranks retire the step in flight at the resume point
    (an admission does not abort it, so its results are applied), among themselves, before the joining rank enters
    the host exchange and before the restore resizes the KV cache pool."""
    recovery = _admitting(monkeypatch, rank=0)
    executor = recovery._executor
    executor.previous_batch = _batch([_request(14)])
    steps = []
    executor._retire_inflight_batch_for_control.side_effect = lambda: steps.append(
        ("retire", executor.previous_batch is not None)
    )
    recovery._exchange.admit = lambda live_mask, epoch: steps.append("admit")
    recovery._moe.workspace_bytes.return_value = 4 << 20
    device = _Device()
    monkeypatch.setattr(recovery_module.torch.cuda, "mem_get_info", lambda: (device.free, 1 << 40))

    @contextlib.contextmanager
    def kv_pools_suspended():
        steps.append("suspend")
        yield device

    executor._kv_pools_suspended = kv_pools_suspended
    recovery.before_iteration()
    # The second suspension gives the KV cache memory back, once committed.
    assert steps == [("retire", True), "admit", "suspend", "suspend"]
    assert _replies(recovery) == [Reply.INSTALLED]


@pytest.mark.parametrize(
    "installed", [False, True], ids=["proposed after the retire", "installed inside the retire"]
)
def test_a_newer_change_during_the_resume_keeps_the_live_ranks_in_the_same_collectives(
    monkeypatch, installed
):
    """Rank 3 is lost while the live ranks retire the step in flight. Its PROPOSE reaches each live rank at its own
    time, and a rank descheduled in the retire's last round installs the newer change inside it, while its peers
    install it at their next round. Every live rank runs the gather of the requests held next all the same; only a
    rank that installed the newer change keeps the replaced change's joining rank out of its host exchange (which
    it would otherwise wait for in that gather). No rank replies to the replaced change."""
    recovery = _admitting(monkeypatch, rank=0)

    def retire():
        if installed:
            recovery._installed, recovery._resume_after = _NEWER_INSTALL, 30
        recovery._on_record(_NEWER_INSTALL if installed else _NEWER)

    recovery._executor._retire_inflight_batch_for_control.side_effect = retire
    recovery.before_iteration()
    recovery._executor.dist.allgather.assert_called_once()
    epoch = membership.next_epoch(40)
    assert recovery._exchange.admits == ([] if installed else [(0b1111, epoch)])
    assert (recovery._store.get_if_set("host_exchange_epoch/2") is None) == installed
    assert _replies(recovery) == []


def test_rank_0_deletes_the_admission_keys_of_a_replaced_change_once_a_later_one_commits(
    monkeypatch,
):
    """Rank 0 handed rank 2 its admission into generation 2 before the newer change reached it; generation 3, which
    adds rank 2 again, commits: the store keeps the keys of neither."""
    recovery = _admitting(monkeypatch, rank=0)
    retire = recovery._executor._retire_inflight_batch_for_control
    retire.side_effect = lambda: recovery._on_record(_NEWER)
    recovery.before_iteration()
    assert recovery._store.get_if_set("host_exchange_epoch/2") is not None
    # The newer change, installed at the next round.
    retire.side_effect = None
    recovery._record = recovery._installed = _NEWER_INSTALL
    recovery._resume_after, recovery._executor.iter_counter = 30, 31
    committed = _NEWER_INSTALL.model_copy(
        update=dict(state=MembershipState.COMMITTED, joining_mask=0)
    )

    def report(**changes):
        if changes.get("reply") == Reply.INSTALLED:
            recovery._on_record(committed)

    recovery._agent.report = report
    recovery.before_iteration()
    assert not recovery.holds_steps()
    assert not [
        key
        for key in recovery._store.values
        if key.startswith(("host_exchange_epoch/", "waiting_queue/"))
    ]


@pytest.mark.timeout(60)
def test_an_admission_lets_the_step_in_flight_finish_and_recomputes_nothing(monkeypatch):
    """The PROPOSE that adds rank 2's replacement keeps every process: nothing aborts the all-to-all, the step in
    flight finishes and its results are applied, and no request runs again once the live ranks resume. Every
    outstanding request is held by a live rank, so none fails."""
    monkeypatch.setattr(recovery_module.torch.cuda, "synchronize", lambda: None)
    recovery = _recovery()
    executor, moe = recovery._executor, recovery._moe
    moe.aborted.side_effect = lambda: moe.request_abort.called and not moe.rearm.called
    recovery._on_record(_record(MembershipState.COMMITTED))  # rank 2 was lost earlier
    in_flight, other = _request(10), _request(11)
    executor.active_requests = [in_flight, other]
    in_flight_step = _batch([in_flight])
    executor.previous_batch = in_flight_step

    recovery._on_record(_joining_record(MembershipState.PROPOSE))
    moe.request_abort.assert_not_called()
    assert recovery.holds_steps()  # from its next forward vote
    recovery.drop_invalid_step()  # before its results are applied: the step is kept
    assert executor.previous_batch is in_flight_step and recovery._interrupted == []

    install = _joining_record(MembershipState.INSTALL)
    recovery._on_record(install)
    for r in (1, FAILED, 3):  # the other live ranks' fresh host-exchange rows
        recovery._store.set(f"host_exchange_rows/2/0/{r}", pickle.dumps(f"rows-{r}".encode()))
    recovery._install(install)
    reports = recovery._agent.reports

    def report(**changes):  # the authority commits once this rank installed
        reports.append(changes)
        if changes.get("reply") == Reply.INSTALLED:
            recovery._on_record(
                install.model_copy(update=dict(state=MembershipState.COMMITTED, joining_mask=0))
            )

    recovery._agent.report = report
    executor.dist.allgather.side_effect = lambda local: [local, set(), set(), set()]
    recovery.set_outstanding_requests(lambda: {10: 110, 11: 111})
    executor.iter_counter = 21
    info = Mock()
    monkeypatch.setattr(recovery_module.logger, "info", info)
    recovery.before_iteration()
    executor._pause_recompute_request.assert_not_called()
    assert executor._pending_transfer_responses == []
    (resumed,) = [c.args[0] for c in info.call_args_list if "resumed serving" in c.args[0]]
    assert resumed.endswith(
        "0 interrupted requests continue by recompute, 0 requests no live rank holds ended by the rank failure"
    )
    assert _replies(recovery) == [Reply.INSTALLED] and not recovery.holds_steps()
    moe.request_abort.assert_not_called()


@pytest.mark.parametrize("rank", [0, FAILED])
def test_only_the_rank_whose_mapping_failed_replies_failed(monkeypatch, rank):
    """The joining rank could not map the remapped all-to-all workspace, which every live rank learns from the
    outcome exchange (MnnvlMappingError names it): it replies FAILED, so the authority goes on without it, while rank
    0 only waits for the authority's next record."""
    recovery = _admitting(monkeypatch, rank)
    recovery._store.set("host_exchange_epoch/2", b"41")
    recovery._store.set("waiting_queue/2", pickle.dumps([]))
    recovery._moe.remap_workspaces.side_effect = MnnvlMappingError(
        [FAILED], "MNNVL remap failed on live ranks:\nrank 2: RuntimeError: 2"
    )
    # The authority's next record, a change without rank 2.
    recovery._next_record = Mock(return_value=_record(MembershipState.PROPOSE, generation=3))
    if rank == FAILED:
        _exit_raises(monkeypatch)
        # Left out: its supervisor starts another replacement.
        with pytest.raises(SystemExit) as exited:
            recovery.before_iteration()
        assert exited.value.code == 1
    else:
        installed = recovery._installed
        recovery.before_iteration()
        recovery._next_record.assert_called_once_with(installed)
    assert _replies(recovery) == ([Reply.FAILED] if rank == FAILED else [])
    recovery._moe.rearm.assert_not_called()
    assert recovery.holds_steps()


def _newer_change(recovery, answers):
    """Rank 2's admission in generation 2 meets a newer change, generation 3, in which rank 3 failed and rank 2 still
    joins: the store holds what rank 0 and the other live ranks publish for both, and the authority answers this
    rank's replies, ``answers`` (reply, generation) -> the record it publishes next."""
    store = recovery._store
    for generation in (2, 3):
        store.set(f"host_exchange_epoch/{generation}", b"41")
        store.set(f"waiting_queue/{generation}", pickle.dumps([RequestQueueItem(7)]))
    for r in (0, 1):
        store.set(f"host_exchange_rows/3/0/{r}", pickle.dumps(f"rows-{r}".encode()))
    reports = recovery._agent.reports

    def report(**changes):
        reports.append(changes)
        answer = answers.get((changes.get("reply"), changes.get("reply_generation")))
        if answer is not None:
            recovery._on_record(answer)

    recovery._agent.report = report


_NEWER = _joining_record(MembershipState.PROPOSE, generation=3).model_copy(
    update=dict(live_mask=0b0111)
)
_NEWER_INSTALL = _NEWER.model_copy(
    update=dict(state=MembershipState.INSTALL, resume=ResumeState(iteration=30, epoch=50))
)


@pytest.mark.timeout(60)
@pytest.mark.parametrize("reached", ["waiting for rank 0's epoch", "waiting for the commit"])
def test_a_joining_rank_follows_a_newer_change_that_still_adds_it(monkeypatch, reached):
    """Rank 3 fails during rank 2's admission. Rank 2, which has run no step with the live ranks, follows the newer
    change as it joined (READY, then the newer INSTALL, whose rows it maps) and finishes its admission from it,
    instead of exiting, which would lose the experts it holds once the survivors gave them back."""
    _exit_raises(monkeypatch)
    recovery = _admitting(monkeypatch, rank=FAILED)
    committed = _NEWER_INSTALL.model_copy(
        update=dict(state=MembershipState.COMMITTED, joining_mask=0)
    )
    answers = {(Reply.READY, 3): _NEWER_INSTALL, (Reply.INSTALLED, 3): committed}
    _newer_change(recovery, answers)
    if reached == "waiting for the commit":
        answers[(Reply.INSTALLED, 2)] = _NEWER
        admits = [(0b1111, 41), (0b0111, 41)]
    else:
        recovery._store.values.pop("host_exchange_epoch/2")  # rank 0 has not resumed yet
        threading.Timer(0.2, recovery._on_record, (_NEWER,)).start()
        admits = [(0b0111, 41)]
    recovery.before_iteration()
    replies = [(r["reply"], r["reply_generation"]) for r in recovery._agent.reports if "reply" in r]
    assert replies[-2:] == [(Reply.READY, 3), (Reply.INSTALLED, 3)]
    # Mapped as it joined: the rows of the newer change's live ranks, rank 3 left out.
    assert recovery._exchange.installs[-1] == (
        [b"rows-0", b"rows-1", b"row-mine", None],
        0b0011,
        50,
    )
    assert recovery._exchange.admits == admits
    assert [item.id for item in recovery._executor.waiting_queue] == [7]  # rank 0's queue, once
    assert recovery._executor.iter_counter == 31 and not recovery.holds_steps()


def test_a_joining_rank_that_a_newer_change_leaves_out_exits(monkeypatch):
    """Its supervisor starts another replacement, which a later change adds."""
    _exit_raises(monkeypatch)
    recovery = _admitting(monkeypatch, rank=FAILED)
    without_rank_2 = _NEWER.model_copy(update=dict(live_mask=0b1011, joining_mask=0))
    _newer_change(recovery, {(Reply.INSTALLED, 2): without_rank_2})
    with pytest.raises(SystemExit) as exited:
        recovery.before_iteration()
    assert exited.value.code == 1


@pytest.mark.parametrize("rank", [0, FAILED])
def test_a_joining_rank_leaves_the_collective_a_newer_change_interrupts(monkeypatch, rank):
    """Its admission's collective (the gather of the requests held), which a newer change has the live ranks finish
    without it: once it installed the newer change, it leaves the collective, and before_iteration admits it from
    there. A live rank goes on with the collective."""
    recovery = _admitting(monkeypatch, rank)

    def follow():  # the newer change, as installed
        recovery._installed, recovery._resume_after = _NEWER_INSTALL, 30

    recovery._change_membership = follow
    if rank == FAILED:
        with pytest.raises(recovery_module._MembershipRestarted):
            recovery._on_exchange_interrupt()
    else:
        recovery._on_exchange_interrupt()
    assert recovery._installed is _NEWER_INSTALL


@pytest.mark.parametrize("replacement", [False, True], ids=["original", "replacement"])
def test_every_rank_takes_its_identity_from_its_node_supervisor(monkeypatch, replacement):
    """Its MPI world is only itself; the rank and the world size come from the node supervisor."""
    monkeypatch.setattr(_utils, "_wide_ep_fault_tolerance", True)
    monkeypatch.setenv(FAULT_TOLERANCE_RANK_ENV, f"2,4,4,{int(replacement)}")
    assert _utils.wide_ep_fault_tolerance_rank() == (2, 4, 4, replacement)
    assert _utils.fault_tolerance_replacement() == ((2, 4, 4, True) if replacement else None)
    assert (_utils.mpi_rank(), _utils.global_mpi_rank()) == (2, 2)
    assert (_utils.mpi_world_size(), _utils.global_mpi_size()) == (4, 4)
    monkeypatch.setattr(_utils, "_wide_ep_fault_tolerance", False)
    assert _utils.wide_ep_fault_tolerance_rank() is None


def _wait_until(condition, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)

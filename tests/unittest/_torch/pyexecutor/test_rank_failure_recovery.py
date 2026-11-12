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
"""Tests for the executor side of Wide-EP fault tolerance with a fake membership, exchange and executor."""

import asyncio
import collections
import contextlib
import json
import os
import pickle
import queue
import re
import signal
import socket
import threading
import time
from datetime import timedelta
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import torch

from tensorrt_llm import _wide_ep_fault_tolerance
from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.distributed import membership, mnnvl_memory
from tensorrt_llm._torch.distributed.host_exchange import (
    HostExchangeMnnvlMemory,
    MnnvlDist,
    TPHostExchange,
)
from tensorrt_llm._torch.distributed.membership import (
    DISCONNECTED_KEY,
    HANG_TIMEOUT_S,
    HOST_WEIGHTS_PREFIX_KEY,
    RENDEZVOUS_ENV,
    SHARED_SECRET_ENV,
    TRANSACTION_KEY,
    MembershipRecord,
    MembershipState,
    Reply,
    ResumeState,
    StoreGroup,
    TransactionSummary,
)
from tensorrt_llm._torch.distributed.mnnvl_active_group import MnnvlPeerHandleError
from tensorrt_llm._torch.distributed.startup_record import StartupRecord, startup_record
from tensorrt_llm._torch.moe.fused_moe.communication.execution_state import MoeExecutionState
from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import FORCE_CFT_ENV
from tensorrt_llm._torch.moe.fused_moe.eplb_expert_restore import (
    DeploymentHostWeights,
    EplbExpertRestore,
)
from tensorrt_llm._torch.moe.fused_moe.impl_blocks import MoEEplbWeightLayoutMixin
from tensorrt_llm._torch.moe.fused_moe.moe_comm_timeout_guard import SERVING_TIMEOUT_ENV
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import MoeLoadBalancer
from tensorrt_llm._torch.pyexecutor import hang_detector
from tensorrt_llm._torch.pyexecutor import rank_failure_recovery as recovery_module
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner
from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
    ExecutorRequestQueue,
    RequestQueueItem,
)
from tensorrt_llm._torch.pyexecutor.hang_detector import HangDetector
from tensorrt_llm._torch.pyexecutor.llm_request import LlmResponse
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
from tensorrt_llm._wide_ep_fault_tolerance import (
    FAULT_TOLERANCE_RANK_ENV,
    wide_ep_fault_tolerance_enabled,
)
from tensorrt_llm.executor.base_worker import BaseWorker
from tensorrt_llm.executor.utils import (
    RANK_FAILURE_CODE,
    ErrorResponse,
    RankFailureError,
    RequestError,
)
from tensorrt_llm.llmapi import LLM, SamplingParams
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

pytestmark = pytest.mark.cpu_only

WORLD = 4
FAILED = 2
SURVIVORS = 0b1011
JOINED = (0, 0, 1, 0)  # The first replacement of rank 2 stands for it.
MIB = 1 << 20
# A rank's KV cache pool and free device memory once the deployment serves.
STARTUP_MEMORY = StartupMemory(kv_cache_quota=64 * MIB)
FREE_BYTES = 1 << 30


@pytest.fixture(autouse=True)
def _no_cuda_synchronize(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)


@pytest.fixture
def shared_secret(monkeypatch):
    monkeypatch.setenv(SHARED_SECRET_ENV, "ab" * 32)
    monkeypatch.setenv(RENDEZVOUS_ENV, "head-node:29600")


# One fake per collaborator of RankFailureRecovery.


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

    def install(self, gather, active_mask, epoch, fresh_rows):
        assert fresh_rows == frozenset(range(WORLD))
        self.installs.append((gather(b"row-mine"), active_mask, epoch))

    def admit(self, active_mask, epoch):
        self.admits.append((active_mask, epoch))


class FakeStore:
    """In-memory RendezvousStore for one rank."""

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


class FakeClient:
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
    recovery._client = FakeClient()
    executor = recovery._executor

    def retire():  # Accept or drop the step in flight, as the executor does.
        if executor.previous_batch is not None:
            recovery.drop_invalid_step()
        executor.previous_batch = None

    executor._retire_inflight_batch_for_control = Mock(side_effect=retire)
    return recovery


def _replacement():
    """Return the recovery of the first replacement of rank 2 before its admission."""
    recovery = RankFailureRecovery(
        _executor(FAILED),
        FakeExchange(),
        MagicMock(),
        FakeStore(),
        "host:1",
        STARTUP_MEMORY,
        incarnation=1,
    )
    recovery._client = FakeClient()
    return recovery


def _record(state, generation=1, active_mask=SURVIVORS, resume=ResumeState()):
    return MembershipRecord(
        generation=generation,
        state=state,
        active_mask=active_mask,
        resume=resume,
        incarnations=(0,) * WORLD,
    )


def _joining_record(state, joining=1 << FAILED, generation=2):
    return MembershipRecord(
        generation=generation,
        state=state,
        active_mask=0b1111,
        resume=ResumeState(iteration=20, epoch=40),
        incarnations=JOINED,
        joining_mask=joining,
    )


def _replies(recovery):
    return [report["reply"] for report in recovery._client.reports if "reply" in report]


def _exit_raises(monkeypatch):
    def exit_(code):
        raise SystemExit(code)

    monkeypatch.setattr(recovery_module.os, "_exit", exit_)


def _wait_until(condition, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


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
    return DefaultADPRouter(dist=MagicMock(tp_rank=0, tp_size=WORLD, has_cp_helix=False))


def _survivor_states(active=0):
    return [RankState(rank=r, num_active_requests=active) for r in range(WORLD) if r != FAILED]


def _host_exchange(row_of):
    """Return a host exchange whose rounds return ``row_of(rank)`` of the active ranks and a stale absent row."""

    def one_round(chunk, length, readers, recipients=None, idle=False):
        readers = [r for r in range(WORLD) if readers & (1 << r)]
        rows = {r: np.array(row_of(r), dtype=np.int64).view(np.uint32) for r in readers}
        packets = np.full((WORLD, 1 + rows[readers[0]].size), 0xDEAD, dtype=np.uint64)
        for r, row in rows.items():
            packets[r, 0] = length
            packets[r, 1:] = row
        return np.array([length if r in readers else 0 for r in range(WORLD)], np.uint32), packets

    exchange = TPHostExchange.__new__(TPHostExchange)
    exchange.nranks, exchange.active_mask, exchange._words = WORLD, SURVIVORS, 16
    exchange._masks = {}
    exchange._round = one_round
    return exchange


def _mnnvl_dist(exchange=None):
    """Return an MnnvlDist without MNNVL memory, over ``exchange``."""
    dist = MnnvlDist.__new__(MnnvlDist)
    dist.mapping = SimpleNamespace(tp_size=WORLD)
    dist._host_exchange = exchange
    dist._active_rows_index = (None, None)
    return dist


def test_router_skips_absent_ranks():
    router = _router()
    states = [RankState(rank=r, num_active_requests=r).serialize() for r in range(WORLD)]
    states[FAILED] = None
    router.dist.tp_allgather.return_value = states
    assert [s.rank for s in router.gather_all_rank_states([])] == [0, 1, 3]

    routed, expected = _router().route_requests(
        _survivor_states(), [_item(i) for i in range(6)], max_num_active_requests=4
    )
    assert set(routed) == {0, 1, 3}
    assert sorted(len(items) for items in routed.values()) == [2, 2, 2]
    assert expected == 2

    # A relaxed pin to the absent rank is balanced over the serving ranks.
    routed, _ = _router().route_requests(
        _survivor_states(), [_item(0, target=FAILED, relax=True)], max_num_active_requests=4
    )
    assert sum(len(items) for items in routed.values()) == 1


def test_cuda_graph_agreement_skips_absent_ranks():
    """Check that the failed rank's stale row changes neither the graph vote nor the padded batch size."""
    sizes = [3, 4, None, 2]  # The batch each rank scheduled. Rank 2 failed.
    dist = _mnnvl_dist(_host_exchange(lambda r: [1, sizes[r], SampleType.FULL.value]))
    runner = CUDAGraphRunner.__new__(CUDAGraphRunner)
    runner.clear = lambda **_kwargs: None  # Nothing was captured.
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


def test_attention_dp_balance_ignores_absent_ranks():
    """Check that a rank that left the group does not hold the survivors' context requests."""
    executor = object.__new__(PyExecutor)
    # Context, generation, tokens and real generation requests of every serving rank.
    executor.dist = _mnnvl_dist(_host_exchange(lambda r: [1, 4, 5, 4]))
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


# The startup checks, before the model loads.


def _startup_config(**mapping_fields):
    """Return the LLM arguments and the mapping that the startup check reads."""
    llm_args = SimpleNamespace(
        attention_dp_config=None,
        disable_overlap_scheduler=False,
        kv_cache_config=SimpleNamespace(use_kv_cache_manager_v2=True),
    )
    mapping = SimpleNamespace(
        rank=0,
        moe_ep_size=WORLD,
        world_size=WORLD,
        enable_attention_dp=True,
        pp_size=1,
        cp_size=1,
        tp_size=WORLD,
    )
    vars(mapping).update(mapping_fields)
    return llm_args, mapping


_SECRET_OR_RENDEZVOUS = f"hex shared secret in {SHARED_SECRET_ENV} and the rendezvous address"
_SHORT_SECRET = f"shared secret of at least 32 bytes (64 hex digits) in {SHARED_SECRET_ENV}"
_SERVING_TIMEOUT_S = int(2 * HANG_TIMEOUT_S)


@pytest.mark.parametrize(
    "env, lacks, fix",
    [
        ({}, None, None),
        ({FORCE_CFT_ENV: "0"}, None, None),
        ({SERVING_TIMEOUT_ENV: str(_SERVING_TIMEOUT_S)}, None, None),
        (
            {FORCE_CFT_ENV: "1"},
            None,
            f"fence path of the NVLink one-sided all-to-all ({FORCE_CFT_ENV}",
        ),
        (
            {SERVING_TIMEOUT_ENV: str(_SERVING_TIMEOUT_S - 1)},
            None,
            f"{SERVING_TIMEOUT_ENV} of at least",
        ),
        ({}, "ep", "moe_expert_parallel_size equal to the world size"),
        ({}, "kv_v2", "KV cache manager V2"),
        ({}, "kv_auto", "KV cache manager V2"),  # "auto" resolves only as the model loads.
        # The executor creator turns it off after the arguments' validation in some modes.
        ({}, "overlap", "the overlap scheduler, which speculative decoding modes"),
        ({}, "tp", "tensor_parallel_size equal to the world size"),
        ({}, "cp", "context_parallel_size=1"),
        ({}, "mnnvl", "multi-node NVLink (MNNVL) domain"),
        ({}, "fabric", "fabric handles (GB200-class aarch64 nodes with IMEX"),
        ({"TLLM_DISABLE_MPI": "1"}, "mpi", "the MPI orchestrator (TLLM_DISABLE_MPI unset)"),
        # Distributed.get chose another implementation although every requirement is met.
        ({}, "dist", "host collectives over MNNVL memory (MnnvlDist)"),
        ({}, "hybrid", "does not support hybrid linear-attention models"),
        ({}, "no_moe", "requires a model with MoE layers"),
        # The authority excludes a hung rank before the rank's own hang detector would end it.
        ({}, "hang", "hang detection timeout of at least"),
        ({RENDEZVOUS_ENV: None}, "launch", _SECRET_OR_RENDEZVOUS),
        ({RENDEZVOUS_ENV: "head-node"}, "launch", _SECRET_OR_RENDEZVOUS),
        ({RENDEZVOUS_ENV: "head-node:port"}, "launch", _SECRET_OR_RENDEZVOUS),
        ({SHARED_SECRET_ENV: None}, "launch", _SECRET_OR_RENDEZVOUS),
        ({SHARED_SECRET_ENV: "not hex"}, "launch", _SECRET_OR_RENDEZVOUS),
        # The secret signs the store's values, which the ranks unpickle.
        ({SHARED_SECRET_ENV: ""}, "launch", _SHORT_SECRET),
        ({SHARED_SECRET_ENV: "ab" * 31}, "launch", _SHORT_SECRET),
    ],
)
def test_startup_rejects_unsupported(shared_secret, monkeypatch, env, lacks, fix):
    """Check that each unsupported setup is refused with its reason and how to turn fault tolerance off."""
    from tensorrt_llm._torch.distributed import communicator

    for name, value in env.items():
        if value is None:
            monkeypatch.delenv(name)
        else:
            monkeypatch.setenv(name, value)
    monkeypatch.setattr(_wide_ep_fault_tolerance, "_enabled", True)
    monkeypatch.setattr(
        mnnvl_memory.MnnvlMemory, "supports_mnnvl", staticmethod(lambda: lacks != "mnnvl")
    )
    monkeypatch.setattr(
        HostExchangeMnnvlMemory,
        "uses_fabric_handles",
        classmethod(lambda cls, dev_id: lacks != "fabric"),
    )
    monkeypatch.setattr(communicator.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.config_utils.is_hybrid_linear",
        lambda config: lacks == "hybrid",
    )
    llm_args, mapping = _startup_config(
        moe_ep_size=WORLD // 2 if lacks == "ep" else WORLD,
        cp_size=2 if lacks == "cp" else 1,
        tp_size=WORLD // 2 if lacks in ("tp", "cp") else WORLD,
    )
    llm_args.kv_cache_config.use_kv_cache_manager_v2 = {"kv_v2": False, "kv_auto": "auto"}.get(
        lacks, True
    )
    llm_args.disable_overlap_scheduler = lacks == "overlap"
    host_collectives_unmet = lacks in ("tp", "cp", "mnnvl", "fabric", "mpi", "dist")
    check = partial(
        check_supported, llm_args, mapping, Mock() if host_collectives_unmet else _mnnvl_dist()
    )
    if fix is None:
        check()
        return
    model = torch.nn.Module()  # A model without MoE layers.
    model.model_config = SimpleNamespace(pretrained_config=SimpleNamespace())
    executor = _executor()
    executor.hang_detector.timeout = _SERVING_TIMEOUT_S - 1
    calls = {
        "launch": recovery_module._launched_rank,
        "hybrid": partial(prepare_model, model),
        "no_moe": partial(prepare_model, model),
        "hang": partial(RankFailureRecovery.create, executor),
    }
    with pytest.raises(ValueError, match=re.escape(fix)) as error:
        calls.get(lacks, check)()
    assert "enable_wide_ep_fault_tolerance=False" in str(error.value)


@pytest.mark.parametrize("replacement", [False, True], ids=["original", "replacement"])
def test_rank_identity_from_node_supervisor(shared_secret, monkeypatch, replacement):
    """Check that the rank, the world size and whether it is a replacement come from the node supervisor."""
    monkeypatch.setattr(_wide_ep_fault_tolerance, "_enabled", True)
    monkeypatch.setenv(FAULT_TOLERANCE_RANK_ENV, f"2,4,4,{int(replacement)}")
    identity = _wide_ep_fault_tolerance.wide_ep_fault_tolerance_rank()
    assert identity == (2, 4, 4, replacement)
    assert identity.env_value() == f"2,4,4,{int(replacement)}"
    expected = (2, 4, 4, True) if replacement else None
    assert _wide_ep_fault_tolerance.fault_tolerance_replacement() == expected
    monkeypatch.delenv(FAULT_TOLERANCE_RANK_ENV)
    with pytest.raises(ValueError, match="'<rank>,<world size>,<GPUs per node>,<replacement>'"):
        recovery_module._launched_rank()
    monkeypatch.setattr(_wide_ep_fault_tolerance, "_enabled", False)
    assert _wide_ep_fault_tolerance.wide_ep_fault_tolerance_rank() is None
    assert _wide_ep_fault_tolerance.fault_tolerance_replacement() is None


# Host weights. Every node keeps the weights of every expert in /dev/shm.

_EXPERTS = 8
_LOCAL_RANKS = 2


def _host_tensor_sharer(local_rank=0, local_size=_LOCAL_RANKS):
    sharer = DeploymentHostWeights(
        0,
        _EXPERTS,
        Mock(Get_rank=Mock(return_value=local_rank), Get_size=Mock(return_value=local_size)),
    )
    sharer.set_shared_memory_base_name(f"wide_ep_ft_test_{os.getpid()}_{time.monotonic_ns()}")
    return sharer


class _MoeWeights(MoEEplbWeightLayoutMixin, torch.nn.Module):
    """MoE layer weights of 3 local slots of 128 KiB and 4 bytes, and one weight that is not per slot."""

    expert_size_per_partition = 3

    def __init__(self, sharer):
        super().__init__()
        self.layer_load_balancer = SimpleNamespace(host_tensor_sharer=sharer)
        self.w3_w1_weight = torch.nn.Parameter(torch.empty(3, 64, 1024, dtype=torch.float16))
        self.fc31_alpha = torch.nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.fc31_input_scale = torch.nn.Parameter(torch.empty((), dtype=torch.float32))


# Two weights per expert aligned to 256 bytes, plus one aligned fingerprint block per local rank's segment.
_NODE_BYTES = _EXPERTS * (128 * 1024 + 256) + _LOCAL_RANKS * 256


@pytest.fixture
def shm_free(monkeypatch):
    """Set the free bytes of /dev/shm that the startup check reads."""
    # Segments start with a fingerprint only under fault tolerance.
    monkeypatch.setattr(_wide_ep_fault_tolerance, "_enabled", True)
    free = {}
    monkeypatch.setattr(
        os, "statvfs", lambda path: SimpleNamespace(f_bavail=free["bytes"], f_frsize=1)
    )
    return free


@pytest.mark.parametrize("shortfall", [0, 1])
def test_startup_check_needs_dev_shm_room(shm_free, shortfall):
    """Check the room for the host weights in /dev/shm, which is 64 MB in Docker and Kubernetes by default."""
    model = torch.nn.ModuleList([_MoeWeights(_host_tensor_sharer())])
    shm_free["bytes"] = _NODE_BYTES - shortfall
    if not shortfall:
        recovery_module._check_host_weights_fit(model)
        return
    with pytest.raises(ValueError, match=f"needs .* \\({_NODE_BYTES} bytes\\) more") as error:
        recovery_module._check_host_weights_fit(model)
    message = str(error.value)
    assert "--shm-size" in message
    assert "emptyDir" in message
    assert "medium: Memory" in message
    assert "enable_wide_ep_fault_tolerance=False" in message


# The flag reaches every process, and requests it does not support are rejected.


@pytest.mark.parametrize("enabled", [False, True])
def test_worker_takes_flag_from_args(monkeypatch, enabled):
    monkeypatch.setattr(_wide_ep_fault_tolerance, "_enabled", not enabled)
    llm_args = TorchLlmArgs(
        model="dummy",
        enable_wide_ep_fault_tolerance=enabled,
        enable_attention_dp=True,
        moe_config={"load_balancer": {"num_slots": WORLD}},
    )
    worker = BaseWorker(llm_args=llm_args)
    worker.doing_shutdown = True  # Nothing to shut down at exit.
    assert wide_ep_fault_tolerance_enabled() is enabled


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
def test_multi_sequence_requests_rejected(params, rejected, enabled):
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


@pytest.mark.parametrize(
    "active_mask, joining_mask, aborts",
    [
        (0b1111, 0b0100, False),  # A replacement for failed rank 2 joins and no process leaves.
        (0b1011, 0b0010, True),  # A replacement takes over active rank 1, whose old process leaves.
        (0b1101, 0b0100, True),  # Rank 1 fails while the replacement of rank 2 joins.
    ],
)
def test_only_removal_aborts_steps_in_flight(active_mask, joining_mask, aborts):
    """Check that only a change that drops a process aborts the all-to-all."""
    recovery = _recovery()
    recovery._on_record(_record(MembershipState.COMMITTED))  # Rank 2 failed earlier.
    proposal = _record(MembershipState.PROPOSE, generation=2, active_mask=active_mask).model_copy(
        update=dict(joining_mask=joining_mask)
    )
    recovery._on_record(proposal)
    assert recovery._moe.request_abort.called == aborts
    recovery._executor.executor_request_queue.wake.assert_called_once()
    # An admission stops the host exchange only after a forward vote.
    assert recovery._membership_changing() == aborts


def _vote(recovery, batch_size=4, peers=((3, 0, 0), (5, 0, 0), (2, 0, 0))):
    """Run the forward vote of ``recovery`` and return the row it sent and the batch sizes.

    Each row of ``peers`` holds a batch size, whether the rank holds for a pending change, and whether it retries
    a give-back.
    """
    sent = []

    def gather(row):
        sent.append(row)
        return np.array([row, *peers])

    batch_sizes = recovery.forward_vote(batch_size, gather)
    return sent[0], batch_sizes.tolist()


def test_admission_stops_exchange_after_held_vote():
    """Check that an admission stops the host exchange once a vote had a rank hold for it."""
    recovery = _recovery()
    recovery._on_record(_record(MembershipState.COMMITTED))  # Rank 2 failed earlier.
    wake = recovery._executor.executor_request_queue.wake
    assert _vote(recovery, peers=((0, 1, 0), (5, 0, 0), (2, 0, 0))) == (
        [4, 0, 0],
        [4, 0, 5, 2],
    )
    wake.assert_not_called()
    assert not recovery._membership_changing()  # It has not seen the change yet.
    recovery._on_record(_joining_record(MembershipState.PROPOSE))
    assert recovery._membership_changing()
    wake.reset_mock()
    _vote(recovery)
    wake.assert_called_once()
    recovery._on_record(_record(MembershipState.COMMITTED, generation=3))
    _vote(recovery)
    recovery._on_record(_record(MembershipState.PROPOSE, generation=4, active_mask=0b0011))
    assert recovery._membership_changing()


def test_change_returns_after_exchange_install():
    recovery = _recovery(rank=1)
    for r in (0, 3):
        recovery._store.set(f"host_exchange_rows/1/0/{r}", pickle.dumps(f"rows-{r}".encode()))
        _mark_arrived(recovery._store, "host_exchange_rows/1/0", r)
    recovery._on_record(_record(MembershipState.PROPOSE))
    done = threading.Thread(target=recovery._change_membership, daemon=True)
    done.start()
    _wait_until(lambda: recovery._client.reports)
    assert recovery._client.reports[-1] == {
        "reply": Reply.READY,
        "reply_generation": 1,
        "paused_at": ResumeState(iteration=5, epoch=6),
    }
    install = _record(MembershipState.INSTALL, resume=ResumeState(iteration=9, epoch=8))
    recovery._on_record(install)
    # INSTALLED follows in before_iteration, so the interrupted round can finish first.
    done.join(5)
    assert not done.is_alive()
    assert all(report.get("reply") != Reply.INSTALLED for report in recovery._client.reports)
    # The INSTALL record carries the READY payloads, so the client stops sending this rank's payload and
    # readiness. The rank then reports the install step it completed.
    assert recovery._client.reports[-2:] == [
        {"ready_payload": None, "joining_ready": False},
        {"install_steps": 1},
    ]
    # The authority took the record's epoch from the READY replies.
    assert recovery._exchange.installs == [
        ([b"rows-0", b"row-mine", None, b"rows-3"], SURVIVORS, 8)
    ]
    assert recovery._resume_after == 9
    assert recovery.pauses_forward()
    # The installed change no longer stops the host exchange.
    assert not recovery._membership_changing()
    recovery._on_record(_record(MembershipState.PROPOSE, generation=2, active_mask=0b0011))
    assert recovery._membership_changing()


@pytest.mark.parametrize(
    "record, status",
    [
        (_record(MembershipState.ABORTED), 1),
        (_record(MembershipState.PROPOSE, active_mask=0b0101), 1),  # Rank 1 is excluded.
        (_record(MembershipState.SHUTDOWN), 0),
    ],
)
def test_change_ends_the_process(monkeypatch, record, status):
    _exit_raises(monkeypatch)
    recovery = _recovery(rank=1)
    recovery._on_record(record)
    with pytest.raises(SystemExit) as exited:
        recovery._change_membership()
    assert exited.value.code == status


@pytest.mark.parametrize("rank, completed, status", [(1, True, 0), (1, False, 1), (0, True, 0)])
def test_stop_exits_without_mpi_teardown(monkeypatch, rank, completed, status):
    """Check that stop exits with the loop status, and that rank 0 asks the authority for SHUTDOWN."""
    _exit_raises(monkeypatch)
    recovery = _recovery(rank=rank)
    recovery._record = _record(MembershipState.SHUTDOWN)
    with pytest.raises(SystemExit) as exited:
        recovery.stop(completed)
    assert exited.value.code == status
    assert ({"shutdown": True} in recovery._client.reports) == (rank == 0)


@pytest.mark.parametrize("paused_before", [False, True])
def test_hang_detector_paused_during_change(paused_before):
    """Check that the hang detector is paused while the rank takes part in a membership change."""
    detector = HangDetector(timeout=600)
    detector.start()
    try:
        if not paused_before:
            detector.checkpoint()
        recovery = _recovery(rank=1)
        recovery._executor.hang_detector = detector
        seen = []
        # The record is committed, so there is no change to take part in.
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
    store.store.add(StoreGroup._count_key(prefix), 1)


@pytest.mark.parametrize("world", [4, 256])
def test_gather_poll_cost_independent_of_ranks(monkeypatch, world):
    """Check that each poll of a waiting gather reads one store key, whatever the number of ranks.

    The gather reads its values once all ranks counted their arrival. To report the missing ranks, it reads one
    arrival counter per 63 ranks once per report interval.
    """
    now_s = [0.0]
    polls = []
    recovery = _recovery(rank=1)
    recovery.world_size = world
    record = recovery._record = MembershipRecord(
        generation=1, state=MembershipState.INSTALL, active_mask=(1 << world) - 1
    )
    gather = recovery._active_group(record, "rows").allgather
    store = recovery._store
    last = world - 1
    for r in range(world):
        if r not in (1, last):
            store.set(f"rows/1/0/{r}", pickle.dumps(r))
            _mark_arrived(store, "rows/1/0", r)
    calls = []  # Each store call of the gathering rank, with its key or its number of keys.
    add, check, multi_get = store.add, store.check, store.multi_get
    monkeypatch.setattr(
        store, "add", lambda key, amount: (calls.append(("add", key, amount)), add(key, amount))[1]
    )
    monkeypatch.setattr(
        store, "check", lambda keys: (calls.append(("check", len(keys))), check(keys))[1]
    )
    monkeypatch.setattr(
        store,
        "multi_get",
        lambda keys: (calls.append(("multi_get", len(keys))), multi_get(keys))[1],
    )

    def sleep(seconds):
        polls.append(seconds)
        now_s[0] += seconds
        if now_s[0] >= 1.0 and not check([f"rows/1/0/{last}"]):
            store.set(f"rows/1/0/{last}", pickle.dumps(last))
            add(StoreGroup._counter("rows/1/0", last), 1 << last % 63)
            add(StoreGroup._count_key("rows/1/0"), 1)

    monkeypatch.setattr(
        membership, "time", SimpleNamespace(sleep=sleep, monotonic=lambda: now_s[0])
    )
    assert gather(1) == list(range(world))
    count_reads = calls.count(("add", "rows/1/0/count", 0))
    assert count_reads == len(polls) + 1
    counter_reads = [c for c in calls if c[0] == "add" and "/arrived/" in c[1] and c[2] == 0]
    counters = -(-world // 63)
    assert len(counter_reads) <= counters * (1 + now_s[0] / membership.REPORT_INTERVAL_S)
    assert [c for c in calls if c[0] != "add"] == [("check", world), ("multi_get", world)]
    # Besides these, the rank only marks its own arrival in two adds and reads the values once.
    assert len(calls) == count_reads + len(counter_reads) + 4


def test_gather_returns_soon_after_last_value(monkeypatch):
    """Check that a gather returns within a few milliseconds of its last value, however long it waited."""
    now_s = [0.0]
    last_value_s = 1.0
    recovery = _recovery(rank=1)
    store = recovery._store
    record = recovery._record = _record(MembershipState.INSTALL)
    gather = recovery._active_group(record, "rows").allgather
    store.set("rows/1/0/0", pickle.dumps("rows-0"))
    _mark_arrived(store, "rows/1/0", 0)

    def sleep(seconds):
        now_s[0] += seconds
        if now_s[0] >= last_value_s and not store.check(["rows/1/0/3"]):
            store.set("rows/1/0/3", pickle.dumps("rows-3"))
            _mark_arrived(store, "rows/1/0", 3)

    monkeypatch.setattr(
        membership, "time", SimpleNamespace(sleep=sleep, monotonic=lambda: now_s[0])
    )
    assert gather("rows-1") == ["rows-0", "rows-1", None, "rows-3"]
    assert now_s[0] - last_value_s <= 0.005


# Dropping steps and continuing requests.


def _request(request_id, dummy=False):
    return SimpleNamespace(
        py_request_id=request_id, is_dummy=dummy, is_finished=False, py_seq_slot=request_id
    )


def _batch(requests):
    return SimpleNamespace(
        scheduled_requests=SimpleNamespace(all_requests=lambda: list(requests)),
        sample_state=SimpleNamespace(sampler_event=MagicMock()),
        gpu_forward_events_from_perf_pool=False,
        context_rows=[],
    )


@pytest.mark.parametrize("aborted", [False, True])
def test_forward_failed_by_abort_is_discarded(aborted):
    """Check that the requests of a forward that failed for the abort continue by recompute."""
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
        assert recovery._interrupted == []
        assert executor.previous_batch is not None


# Rank 0 restores lost expert 3 into its spare slot. The agreed view still lists the slots of failed rank 2.
_PLACEMENT = ((4, ((0, -1), (1, -1), (3, -1), (2, -1))),)
_INSTALL = _record(MembershipState.INSTALL, resume=ResumeState(iteration=9, epoch=8))
_REJOIN_INSTALL = _INSTALL.model_copy(update=dict(active_mask=0b1111, joining_mask=1 << FAILED))


def _resuming_recovery(
    gathered_from_peers=(None, None, set()), record=_INSTALL, placement=None, rank=0
):
    """Return the recovery of ``rank`` past the iteration that the survivors resume after.

    The authority commits once the rank replies INSTALLED. ``placement`` is the agreed placement that the INSTALL
    record carries in its READY payloads.
    """
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
    reports = recovery._client.reports

    def report(**changes):
        reports.append(changes)
        if changes.get("reply") == Reply.INSTALLED:
            recovery._on_record(
                record.model_copy(update=dict(state=MembershipState.COMMITTED, joining_mask=0))
            )

    recovery._client.report = report
    return recovery


def test_resume_recomputes_interrupted_fails_lost(monkeypatch):
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
    # Its forward ran before the change committed.
    executor.previous_batch = _batch([in_pending_step])
    executor.waiting_queue = [SimpleNamespace(id=12)]
    executor.executor_request_queue.request_queue.put(RequestQueueItem(13))
    # A cancel of lost request 11 does not hold it.
    executor.executor_request_queue.request_queue.put(
        RequestQueueItem(11, is_canceled_request=True)
    )
    recovery.set_outstanding_requests(lambda: {r: 100 + r for r in (10, 11, 12, 13, 20)})

    recovery._resume_after, executor.iter_counter = 10, 10
    recovery.before_iteration()  # Not yet past the last iteration a survivor reached.
    recovery._moe.rearm.assert_not_called()
    assert recovery.pauses_forward()

    recovery._resume_after = 9
    executor.adp_ctx_waiting_iters_count = executor.adp_ctx_batching_wait_iters_count = 3
    info = Mock()
    monkeypatch.setattr(recovery_module.logger, "info", info)
    recovery.before_iteration()
    (resumed,) = [c.args[0] for c in info.call_args_list if "resumed serving" in c.args[0]]
    assert resumed.endswith(
        "2 interrupted requests continue by recompute, 1 requests no active rank holds ended by the rank failure"
    )
    assert executor.previous_batch is None
    # A replacement starts its attention-DP balance counters at 0, so every active rank restarts them.
    assert executor.adp_ctx_waiting_iters_count == 0
    assert executor.adp_ctx_batching_wait_iters_count == 0
    # Only the requests of the discarded steps run again. The rest of the prefix cache stays.
    assert executor._pause_recompute_request.call_args_list == [
        ((in_dropped_step,),),
        ((in_pending_step,),),
    ]
    # Each gave its resources back before it was paused. The others keep theirs.
    assert freed_when_paused == [True, True]
    assert sorted(slots.slot_manager.slot_mapping) == [-1, untouched.request_id]
    recovery._moe.health.mark_failed.assert_called_once_with(FAILED)
    # The workspaces move to fresh pages among the survivors before the all-to-all is reset and masked.
    steps = ("remap_workspaces", "rearm", "install_mask")
    assert [name for name, _, _ in recovery._moe.method_calls if name in steps] == list(steps)
    assert _replies(recovery) == [Reply.INSTALLED]
    assert not recovery.pauses_forward()
    # No survivor holds request 11.
    ((request_id, response),) = executor._pending_transfer_responses
    assert request_id == 11
    assert response.client_id == 111
    assert isinstance(response.error_msg, RankFailureError)


@pytest.mark.parametrize(
    "error, reply",
    [
        (RuntimeError("cuMemCreate failed"), [Reply.FAILED]),  # The authority aborts.
        (MnnvlPeerHandleError([3]), []),  # A peer exited, so the authority proposes again.
    ],
)
def test_failed_install_holds_the_steps(error, reply):
    recovery = _resuming_recovery()
    report = recovery._client.report

    def answer_failed(**changes):  # The authority answers FAILED with the next change.
        report(**changes)
        if changes.get("reply") == Reply.FAILED:
            recovery._on_record(_record(MembershipState.PROPOSE, generation=2, active_mask=0b0011))

    recovery._client.report = answer_failed
    recovery._moe.remap_workspaces.side_effect = error
    recovery.before_iteration()
    assert _replies(recovery) == reply
    recovery._moe.rearm.assert_not_called()
    assert recovery.pauses_forward()


# Device memory: the restore of lost experts, the KV cache pool and its give-back.


class _Device:
    """Device memory of a rank since the deployment started serving.

    It tracks the free memory, the spare slots EPLB backs (``backed``) and the KV cache pool. It also acts as the
    KV cache manager, whose quota changes in whole ``unit``s above 8 MiB and within the free memory. ``calls``
    records the resizes and the suspensions of every request.
    """

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

    @contextlib.contextmanager
    def suspended(self):
        """Stand for the executor's KV cache pools while every request is suspended."""
        self.calls.append("suspend")
        try:
            yield self
        finally:
            self.calls.append("resume")


def _resizes(device):
    return [call[1] for call in device.calls if call[0] == "resize"]


def _eplb(monkeypatch, recovery, restored=6 * MIB, workspace=4 * MIB, device=None, released=0):
    """Set up the EPLB of rank 0 and return its calls, its restore plan and the device.

    The plan backs one spare slot with ``restored`` bytes of device memory. The all-to-all workspace takes
    ``workspace`` bytes. Releasing the empty spare slots frees ``released`` bytes.
    """
    device = _Device() if device is None else device
    monkeypatch.setattr(recovery_module.torch.cuda, "mem_get_info", lambda: (device.free, 1 << 40))
    calls = device.calls
    balancer = MoeLoadBalancer.__new__(MoeLoadBalancer)
    balancer.ep_rank = 0
    balancer.abandon_iterations = lambda: calls.append("abandon")
    # Rank 0 takes lost expert 3 into its spare slot.
    planned = (np.array([[0, 3], [1, -1], [-1, -1], [2, -1]]),)
    plan = SimpleNamespace(device_bytes=restored, digest="restored", placements=planned)

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
        plan_expert_restore=lambda active, current, rejoining: (
            calls.append(("plan", active, current, rejoining)) or plan
        ),
        restore_lost_experts=restore_lost_experts,
        release_empty_spare_slots=release_empty_spare_slots,
        backed_spare_slot_bytes=device.backed,
    )
    recovery._executor.model_engine.moe_load_balancer = balancer
    recovery._executor.kv_pools_suspended = device.suspended
    recovery._moe.workspace_bytes.return_value = workspace
    recovery._moe.remap_workspaces.side_effect = lambda gather, fresh_rows: calls.append("remap")
    return calls, plan, device


def test_install_restores_lost_experts(monkeypatch):
    recovery = _resuming_recovery(placement=_PLACEMENT)
    calls, plan, _ = _eplb(monkeypatch, recovery, workspace=4 * MIB - 4096)
    recovery.before_iteration()
    assert calls[0] == "abandon"  # At the resume point, on every active rank.
    (_, active, (current,), rejoining) = calls[1]
    assert active == [True, True, False, True]
    assert rejoining == [False] * WORLD
    # The plan takes the agreed placement from the READY payloads and ignores the failed rank's slots.
    np.testing.assert_array_equal(current, [[0, -1], [1, -1], [3, -1], [2, -1]])
    assert calls == [
        "abandon",
        calls[1],
        "suspend",
        # The spare slot and the fresh workspace pages are mapped before the pages they replace are freed.
        ("resize", 54 * MIB + 4096),
        ("restore", plan),
        "remap",
        "resume",
    ]
    # Each completed install step is reported, so a long install is not taken for a hang.
    steps = [
        report.get("install_steps", report.get("reply")) for report in recovery._client.reports
    ]
    assert steps == [1, 2, 3, 4, 5, Reply.INSTALLED]
    assert recovery._client.reports[-1]["placement_digest"] == "restored"
    _vote(recovery)
    recovery.before_iteration()
    # Once serving resumed, the pool takes back the workspace memory once per change. No spare slot is released.
    assert calls[7:] == ["release", "suspend", ("resize", 58 * MIB), "resume"]


@pytest.mark.parametrize("extra_free", [0, 7 * MIB, 20 * MIB, -3 * MIB])
def test_kv_pool_gives_up_change_memory(monkeypatch, extra_free):
    """Check that the KV cache pool gives up exactly the spare slot and the workspace."""
    recovery = _resuming_recovery(placement=_PLACEMENT)
    _, _, device = _eplb(monkeypatch, recovery, device=_Device(extra_free=extra_free))
    recovery._restore(recovery._installed)
    recovery._give_back_kv_cache()  # Once committed.
    assert _resizes(device) == [54 * MIB, 58 * MIB]
    assert device.gpu_quota == 58 * MIB
    # Serving keeps the memory it had free.
    assert device.free == FREE_BYTES + extra_free


def _short_give_back(monkeypatch, record=_INSTALL):
    """Return rank 0 after a failover whose give-back fell 4 MiB short, and its device.

    The excluded rank 2 still mapped the pages that the remap replaced at the commit.
    """
    recovery = _resuming_recovery(record=record, placement=_PLACEMENT)
    _, _, device = _eplb(monkeypatch, recovery)
    recovery._moe.remap_workspaces.side_effect = lambda gather, fresh_rows: device.take(4 * MIB)
    recovery.before_iteration()  # The install and the commit.
    _vote(recovery)
    recovery.before_iteration()  # The give-back.
    assert device.gpu_quota == 54 * MIB
    assert device.free == FREE_BYTES
    recovery._give_back_retry.check_after_s = 0.0
    recovery._executor._retire_inflight_batch_for_control.reset_mock()
    return recovery, device


def test_retried_give_back_regains_freed_memory(monkeypatch):
    """Check that the retry retires the step in flight and takes back what the ended process freed."""
    recovery, device = _short_give_back(monkeypatch)
    retire = recovery._executor._retire_inflight_batch_for_control
    device.take(-4 * MIB)  # The excluded process ended.
    _vote(recovery)
    recovery.before_iteration()
    retire.assert_called_once()
    assert device.gpu_quota == 58 * MIB
    assert device.free == FREE_BYTES
    assert recovery._give_back_retry is None
    _vote(recovery)
    recovery.before_iteration()
    retire.assert_called_once()


def test_give_back_ignores_replacement_hello(monkeypatch):
    """Check that the retries end once the excluded process ended, though its replacement said hello.

    The replacement of rank 2 said hello while the survivors waited on the hung process of rank 2, so the INSTALL
    record names the replacement. The retries wait only for the process that the last commit listed.
    """
    install = _INSTALL.model_copy(update=dict(incarnations=JOINED))
    recovery, device = _short_give_back(monkeypatch, install)
    recovery._store.set(
        DISCONNECTED_KEY, json.dumps([0 if r == FAILED else -1 for r in range(WORLD)]).encode()
    )
    assert _vote(recovery)[0] == [4, 0, 0]
    assert recovery._give_back_retry is None


def test_healthy_step_checks_no_device_memory(monkeypatch):
    """Check that a healthy vote asks for no retry and reads no free device memory."""
    recovery = _recovery()
    mem_get_info = Mock()
    monkeypatch.setattr(recovery_module.torch.cuda, "mem_get_info", mem_get_info)
    assert _vote(recovery)[0] == [4, 0, 0]
    assert not recovery._give_back_agreed
    recovery.before_iteration()
    mem_get_info.assert_not_called()
    recovery._executor._retire_inflight_batch_for_control.assert_not_called()


def test_rejoin_gives_back_memory_after_resume(monkeypatch):
    """Check that the emptied spare slots and the KV cache pool wait for a step boundary after serving resumes.

    Every active rank gives back at the step boundary after a forward vote in which no rank held for a change.
    """
    recovery = _resuming_recovery(record=_REJOIN_INSTALL, placement=_PLACEMENT)
    calls, _, device = _eplb(
        monkeypatch,
        recovery,
        restored=0,
        released=6 * MIB,
        device=_Device(gpu_quota=58 * MIB, backed=6 * MIB),
    )
    info = Mock()
    monkeypatch.setattr(recovery_module.logger, "info", info)
    recovery.before_iteration()
    (_, active, _, rejoining) = calls[1]
    assert active == [True] * WORLD
    assert rejoining == [False, False, True, False]
    assert _replies(recovery) == [Reply.INSTALLED]
    assert not recovery.pauses_forward()
    recovery._moe.health.mark_active.assert_any_call(FAILED)
    # Serving resumed with the spare slots still backed and the pool short of the workspace.
    assert device.backed == 6 * MIB
    assert device.gpu_quota == 54 * MIB
    retire = recovery._executor._retire_inflight_batch_for_control
    retire.reset_mock()
    # Rank 1 holds its steps for the next change, so no rank gives back at this step boundary.
    assert _vote(recovery, peers=((3, 1, 0), (5, 0, 0), (2, 0, 0)))[0] == [4, 0, 1]
    recovery.before_iteration()
    retire.assert_not_called()
    assert device.backed == 6 * MIB
    assert _vote(recovery)[0] == [4, 0, 1]
    recovery.before_iteration()
    retire.assert_called_once()
    assert device.backed == 0
    assert device.gpu_quota == STARTUP_MEMORY.kv_cache_quota
    # A rank with nothing to give back still retires its step in flight at the same step boundary.
    other = _recovery(rank=1)
    _vote(other, peers=((4, 0, 1), (5, 0, 0), (2, 0, 0)))
    other.before_iteration()
    other._executor._retire_inflight_batch_for_control.assert_called_once()
    (given_back,) = [c.args[0] for c in info.call_args_list if "gave back" in c.args[0]]
    assert given_back.startswith(
        f"Rank 0 gave back the memory of membership generation 1. Released {6 * MIB} bytes of empty spare "
        f"slots, KV cache quota {STARTUP_MEMORY.kv_cache_quota} bytes, in "
    )


# EPLB and waiting requests across a membership change.


def test_eplb_plan_restarts_at_resume(monkeypatch):
    """Check that every active rank restarts its online EPLB update plan at the resume point."""
    plan_steps = {}
    for rank, iterations_after_ready in ((0, 1), (1, 0)):
        recovery = _resuming_recovery(placement=_PLACEMENT, rank=rank)
        _eplb(monkeypatch, recovery)
        balancer = recovery._executor.model_engine.moe_load_balancer
        balancer.abandon_iterations = lambda balancer=balancer: setattr(balancer, "plan_step", 0)
        balancer.abandon_iterations()  # Before READY.
        balancer.plan_step += iterations_after_ready
        recovery.before_iteration()
        assert _replies(recovery) == [Reply.INSTALLED]
        plan_steps[rank] = balancer.plan_step
    assert plan_steps == {0: 0, 1: 0}


@pytest.mark.parametrize("rank", [0, 1])
def test_requests_pinned_to_absent_rank_fail(rank):
    """Check that waiting requests pinned to an absent rank leave the queue and fail on rank 0."""
    recovery = _recovery(rank)
    waiting = FCFSWaitingQueue()
    waiting.add_requests([_item(1), _item(2, target=FAILED), _item(3, target=FAILED, relax=True)])
    all_states = [RankState(rank=r) for r in range(WORLD)]
    recovery.fail_unservable_requests(waiting, all_states)  # Every rank serves.
    assert [item.id for item in waiting] == [1, 2, 3]

    recovery.fail_unservable_requests(waiting, _survivor_states())
    assert [item.id for item in waiting] == [1, 3]
    responses = recovery._executor._pending_transfer_responses
    if rank == 0:
        ((request_id, response),) = responses
        assert request_id == 2
        assert response.client_id == 102
        # The request is not sent again because it would fail the same way.
        assert not isinstance(response.error_msg, RankFailureError)
    else:
        assert responses == []


# One MoE layer of four experts at startup. Rank r holds expert r and an empty spare slot.
_STARTUP = ((4, ((0, -1), (1, -1), (2, -1), (3, -1))),)


def _installed(record, views):
    """Return the INSTALL record that follows ``record`` with the READY payloads of ``views``."""
    payloads = ((r, recovery_module.ready_payload(record, r, view)) for r, view in views.items())
    return record.model_copy(
        update={
            "state": MembershipState.INSTALL,
            "ready_payloads": tuple((r, p) for r, p in payloads if p is not None),
        }
    )


def test_agreed_placement_uses_lowest_survivor_view():
    """Check that the agreed placement is the lowest survivor's view plus the joining ranks' own slots."""
    views = {
        0: ((4, ((1, -1), (0, -1), (3, -1), (2, -1))),),
        1: _STARTUP,  # Rank 1 missed the update.
        3: _STARTUP,  # Rank 3 joins.
    }
    record = MembershipRecord(
        generation=2, state=MembershipState.PROPOSE, active_mask=0b1011, joining_mask=1 << 3
    )
    installed = _installed(record, views)
    assert [r for r, _ in installed.ready_payloads] == [0, 3]  # The payload of rank 1 is empty.
    agreed = recovery_module.agreed_placement(installed)
    assert agreed == [(4, [[1, -1], [0, -1], [3, -1], [3, -1]])]
    # Rank 1 is the lowest survivor.
    record = record.model_copy(update=dict(active_mask=0b1010))
    installed = _installed(record, {r: view for r, view in views.items() if r})
    startup = [(4, [[0, -1], [1, -1], [2, -1], [3, -1]])]
    assert recovery_module.agreed_placement(installed) == startup
    # Without EPLB, the lowest survivor or a joining rank sends no placement.
    for missing in (1, 3):
        payloads = tuple((r, p) for r, p in installed.ready_payloads if r != missing)
        without = installed.model_copy(update=dict(ready_payloads=payloads))
        assert recovery_module.agreed_placement(without) is None


# Hooks outside the executor.


def test_wake_returns_a_waiting_request_fetch():
    request_queue = ExecutorRequestQueue(MagicMock(), 8, False, 0)
    with request_queue.enqueue_lock:  # Wake enqueues like the other enqueue_* calls.
        waker = threading.Thread(target=request_queue.wake, daemon=True)
        waker.start()
        waker.join(0.2)
        assert request_queue.request_queue.empty()
    waker.join(5)
    fetched = request_queue.get_from_request_queue(timedelta(seconds=30))
    assert fetched == []


def test_hard_kill_stays_local(monkeypatch):
    from mpi4py import MPI

    monkeypatch.setattr(_wide_ep_fault_tolerance, "_enabled", True)
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


# Requests that a rank failure ended, in the worker and the server.


def _ipc_worker(monkeypatch, fault_tolerance=True):
    """Return the executor worker of rank 0 past its startup and its response helper.

    The worker has two postprocessing workers. A request's responses go to its postprocessing worker and its errors
    go to the result queue.
    """
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
    worker.postproc_config = SimpleNamespace(num_postprocess_workers=2, enabled=True)
    worker.postproc_queues = [Mock(), Mock()]
    worker.frontend_result_queues = None
    worker.result_queue = Mock()
    return worker, base_worker.AwaitResponseHelper(worker)


def _submit(worker, client_id, multimodal_params=None):
    request = SimpleNamespace(
        id=client_id, multimodal_params=multimodal_params, disaggregated_params=None
    )
    worker.submit(request)
    return request


def _sent(worker):
    """Map each client id to the outputs and error types that reached the client."""
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


def test_worker_reenqueues_unanswered_failed_request(monkeypatch):
    """Check that the worker enqueues a request ended by a rank failure again once if it sent no response."""
    worker, helper = _ipc_worker(monkeypatch)
    requests = {client_id: _submit(worker, client_id) for client_id in (7, 8)}
    helper.handle_for_ipc_batched([_output(8), _ended_by_rank_failure(7)])
    assert worker._enqueue_request.call_args_list[-1].args == (requests[7],)
    assert worker._client_id_to_request_id == {7: 102, 8: 101}
    assert _sent(worker) == {8: [("output", "sampling params")]}

    helper.handle_for_ipc_batched([_output(7), _ended_by_rank_failure(8)])
    assert _sent(worker) == {
        7: [("output", "sampling params")],  # The first response of the request enqueued again.
        8: [("output", "sampling params"), ("error", RankFailureError)],
    }
    assert 8 not in worker._results
    helper.handle_for_ipc_batched([_ended_by_rank_failure(7)])
    assert _sent(worker)[7][-1] == ("error", RankFailureError)
    assert worker._enqueue_request.call_count == 3

    _submit(worker, 9)
    helper.handle_for_ipc_batched([_ended_by_rank_failure(9)])
    helper.handle_for_ipc_batched([_ended_by_rank_failure(9)])  # It was enqueued again already.
    assert _sent(worker)[9] == [("error", RankFailureError)]
    assert worker._enqueue_request.call_count == 5
    assert worker._unanswered_requests == {}
    assert worker._results == {}


def test_worker_reenqueues_multimodal_input(monkeypatch):
    """Check that a multimodal request is enqueued again with its multimodal input and parameters."""
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


def test_worker_without_flag_never_reenqueues(monkeypatch):
    worker, helper = _ipc_worker(monkeypatch, fault_tolerance=False)
    _submit(worker, 7)
    helper.handle_for_ipc_batched([_ended_by_rank_failure(7)])
    assert _sent(worker) == {7: [("error", RankFailureError)]}
    assert worker._enqueue_request.call_count == 1


@pytest.mark.parametrize(
    "error, status, code",
    [
        (RankFailureError(), 503, RANK_FAILURE_CODE),
        (RequestError(str(RankFailureError())), 400, 400),  # The type decides, not the text.
        (RequestError("bad request"), 400, 400),
        ("bad request", 400, 400),
    ],
)
def test_rank_failure_answers_503_with_code(error, status, code):
    """Check that only a rank failure answers 503, after which clients send the request again."""
    from tensorrt_llm.serve.openai_server import OpenAIServer

    response = OpenAIServer.create_error_response(error)
    body = json.loads(response.body)
    assert response.status_code == status
    assert body["code"] == code
    assert "retry-after" not in response.headers


# The rendezvous, the authority process and the server's health.


@pytest.mark.parametrize("secret", [None, ""], ids=["unset", "empty"])
def test_rendezvous_ports_stay_reserved(monkeypatch, secret):
    """Check that no other process takes the rendezvous ports before the authority binds them."""
    from torch.distributed import TCPStore

    for name in (SHARED_SECRET_ENV, RENDEZVOUS_ENV):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)  # Unset now and again afterwards.
    if secret is not None:
        monkeypatch.setenv(SHARED_SECRET_ENV, secret)
    monkeypatch.setattr(recovery_module, "_rendezvous_ports", [])
    recovery_module.ensure_rendezvous()
    assert len(bytes.fromhex(os.environ[SHARED_SECRET_ENV])) == 32
    host, store_port, liveness_port = membership.authority_ports(os.environ[RENDEZVOUS_ENV])
    socket.inet_aton(host)
    for port in (store_port, liveness_port):
        with socket.socket() as other, pytest.raises(OSError):
            other.bind(("", port))
    # Bind them as the authority does.
    store = TCPStore(host, store_port, is_master=True, wait_for_workers=False)
    assert store.port == store_port
    with socket.create_server((host, liveness_port)) as listener:
        assert listener.getsockname()[1] == liveness_port


class FakeRendezvousStore:
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
        (_record(MembershipState.COMMITTED, active_mask=0b1111), "healthy"),
        (_record(MembershipState.COMMITTED), "degraded"),
        (_record(MembershipState.INSTALL), "recovering"),
        (_record(MembershipState.ABORTED), "failed"),
        (None, "unprotected"),
    ],
)
def test_health_state(monkeypatch, record, state):
    monkeypatch.setattr(
        recovery_module, "open_store", lambda *args, **kwargs: FakeRendezvousStore(record)
    )
    assert FaultToleranceHealth("host:1", b"secret").status()["state"] == state


def test_health_skips_store_after_authority_ends(monkeypatch):
    """Check that reads after a failed store read answer unprotected at once without the store.

    The authority hosts the store and is not restarted. The status keeps the last membership read.
    """
    monkeypatch.setattr(recovery_module, "_STATUS_READ_TIMEOUT_S", 0.0)
    store = FakeRendezvousStore(_record(MembershipState.COMMITTED, active_mask=0b1111))
    open_store = Mock(return_value=store)
    monkeypatch.setattr(recovery_module, "open_store", open_store)
    health = FaultToleranceHealth("host:1", b"secret")
    status = health.status()
    assert status["state"] == "healthy"
    assert status["authority_alive"]
    store.get_if_set = Mock(side_effect=RuntimeError("the store does not answer"))
    assert health.status()["state"] == "unprotected"
    store.get_if_set.reset_mock()
    for _ in range(3):
        status = health.status()
        assert status["state"] == "unprotected"
        assert not status["authority_alive"]
        assert status["active_ranks"] == list(range(WORLD))
    store.get_if_set.assert_not_called()
    assert open_store.call_count == 1


def _server(monkeypatch, store):
    """Return a server whose executor reads the membership from ``store`` when a probe is served.

    A ``store`` of None does not answer.
    """
    from tensorrt_llm.serve.openai_server import OpenAIServer

    monkeypatch.setattr(_wide_ep_fault_tolerance, "_enabled", True)
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


def test_status_endpoint_reports_membership(monkeypatch):
    transaction = TransactionSummary(
        generation=1,
        outcome="COMMITTED",
        removed_ranks=(FAILED,),
        added_ranks=(),
        detected_unix_ms=988,
        began_unix_ms=1000,
        ready_unix_ms=1400,
        ended_unix_ms=2500,
    )
    server = _server(
        monkeypatch, FakeRendezvousStore(_record(MembershipState.COMMITTED), transaction)
    )
    response = asyncio.run(server.fault_tolerance_status())
    assert response.status_code == 200
    assert json.loads(response.body) == {
        "state": "degraded",
        "authority_alive": True,
        "generation": 1,
        "active_ranks": [0, 1, 3],
        "failed_ranks": [FAILED],
        "incarnations": [0] * WORLD,
        "last_transaction": {
            "generation": 1,
            "outcome": "COMMITTED",
            "removed_ranks": [FAILED],
            "added_ranks": [],
            "detected_unix_ms": 988,
            "began_unix_ms": 1000,
            "ready_unix_ms": 1400,
            "ended_unix_ms": 2500,
        },
    }


@pytest.mark.parametrize(
    "record, fatal, code",
    [
        (_record(MembershipState.COMMITTED), False, 200),
        # A liveness probe must not restart the server while the membership changes.
        (_record(MembershipState.PROPOSE), False, 200),
        (_record(MembershipState.INSTALL), False, 200),
        (_record(MembershipState.ABORTED), False, 503),
        (None, False, 200),
        (_record(MembershipState.COMMITTED), True, 503),
    ],
)
def test_health_503_only_after_deployment_stops(monkeypatch, record, fatal, code):
    """Check that the status code carries the health and the body stays unchanged."""
    server = _server(monkeypatch, None if record is None else FakeRendezvousStore(record))
    if fatal:
        server.generator._check_health.return_value = False
        server.generator._executor._fatal_error = RuntimeError("engine died")
        server.generator._executor.doing_shutdown = True  # No signal, since the shutdown started.
    response = asyncio.run(server.health())
    assert response.status_code == code
    if code == 200:
        assert response.body == b""
    else:
        assert response.body.startswith(b"LLM is unavailable")


def test_authority_starts_outside_launcher_job(monkeypatch):
    """Check that the authority does not inherit the launcher's rank identity, which would make it a job member."""
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
    assert launched["TRT_LLM_NO_LIB_INIT"] == "1"  # Importing tensorrt_llm would run MPI_Init.


# The startup of each rank as its own process, and the startup record a replacement replays.


@pytest.mark.parametrize("last", ["commit", "predecessor end"])
def test_replacement_waits_for_predecessor_memory(monkeypatch, last):
    """Check that a replacement starts only once its predecessor ended and the active ranks committed without it.

    The active ranks release the memory that they imported from the predecessor before that commit, so the
    replacement measures the free device memory for its KV cache only afterwards.
    """
    store = FakeStore()
    client = Mock(rank=FAILED, incarnation=1)
    monkeypatch.setattr(recovery_module, "_client", None)
    monkeypatch.setattr(recovery_module, "MembershipClient", lambda *args, **kwargs: client)
    monkeypatch.setattr(recovery_module, "_predecessor_memory_released", threading.Event())
    commit = [
        partial(recovery_module._on_record_at_startup, _record(MembershipState.INSTALL)),
        partial(recovery_module._on_record_at_startup, _record(MembershipState.COMMITTED)),
    ]
    predecessor_end = [partial(store.set, DISCONNECTED_KEY, json.dumps([-1, -1, 0, -1]).encode())]
    steps = predecessor_end + commit if last == "commit" else commit + predecessor_end
    monkeypatch.setattr(recovery_module.time, "sleep", lambda _s: steps.pop(0)())
    identity = _wide_ep_fault_tolerance.RankIdentity(FAILED, WORLD, WORLD, True)
    recovery_module._join_at_entry(store, identity)
    assert steps == []


@pytest.mark.parametrize(
    "ended, awaited",
    [(False, r"predecessor \(incarnation 0\) to end"), (True, "release the memory")],
)
def test_replacement_wait_times_out(monkeypatch, ended, awaited):
    store = FakeStore()
    if ended:
        store.set(DISCONNECTED_KEY, json.dumps([-1, -1, 0, -1]).encode())
    monkeypatch.setattr(recovery_module, "_predecessor_memory_released", threading.Event())
    clock = SimpleNamespace(now_s=0.0)

    def sleep(seconds):
        clock.now_s += seconds

    monkeypatch.setattr(recovery_module.time, "monotonic", lambda: clock.now_s)
    monkeypatch.setattr(recovery_module.time, "sleep", sleep)
    with pytest.raises(TimeoutError, match=awaited):
        recovery_module._wait_for_predecessor_memory(store, FAILED, 1)


@pytest.fixture
def agreed_startup(monkeypatch):
    """Enable fault tolerance for an original rank with nothing recorded yet."""
    monkeypatch.setattr(_wide_ep_fault_tolerance, "_enabled", True)
    monkeypatch.delenv(FAULT_TOLERANCE_RANK_ENV, raising=False)
    monkeypatch.setattr(recovery_module, "_startup_fingerprint", None)
    for name, value in vars(StartupRecord()).items():
        monkeypatch.setattr(startup_record, name, value)


def _published(monkeypatch, worker_main_args, identity="2,4,4,1"):
    """Set up a rank other than 0 with the launcher ``identity`` and return the store it reads.

    The store holds ``worker_main_args`` as rank 0 published them. The membership client is not started.
    """
    monkeypatch.setenv(SHARED_SECRET_ENV, "ab" * 32)
    monkeypatch.setenv(RENDEZVOUS_ENV, "head-node:29600")
    monkeypatch.setenv(FAULT_TOLERANCE_RANK_ENV, identity)
    store = FakeStore()
    store.set(WORKER_ARGS_KEY, pickle.dumps(worker_main_args))
    monkeypatch.setattr(recovery_module, "_startup_store", lambda: store)
    monkeypatch.setattr(recovery_module, "_join_at_entry", lambda store, identity: None)
    return store


class FakeProfilingCache:
    """Tactics by profiling cache key."""

    def __init__(self, cache):
        self.cache = dict(cache)

    def merge_cache_data(self, cache_data):
        self.cache.update(cache_data)


def _replacement_of(monkeypatch, published: bytes) -> None:
    """Make this process a replacement rank that replays the startup record ``published`` by rank 0."""
    for name, value in vars(StartupRecord()).items():
        setattr(startup_record, name, value)
    monkeypatch.setenv(FAULT_TOLERANCE_RANK_ENV, "2,4,4,1")
    startup_record.replay(pickle.loads(published))


def test_replacement_takes_recorded_startup_values(agreed_startup, monkeypatch):
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
    startup_record.adopt_tuned_tactics()  # Rank 0 records its tactics and keeps them.
    assert tuned.cache == {("gemm", 8): (0, 3, 0.1)}
    startup_record.close()
    # What rank 0 publishes under AGREED_STARTUP_KEY once its startup is done.
    _replacement_of(monkeypatch, pickle.dumps({**startup_record.published(), "fingerprint": None}))

    # Its peers are absent, so it runs none of the collectives and takes the results of rank 0.
    gathers.clear()
    assert startup_record.agree("rank_hosts", gather(["c"] * WORLD)) == ["a", "a", "b", "b"]
    assert startup_record.agree("startup_group_split", gather(None)) == [[0, 0]] * WORLD
    with pytest.raises(SystemExit, match="takes the startup values") as exited:
        startup_record.close()  # The original ranks made a call it has not made yet.
    assert exited.value.code == membership.REPLACEMENT_CANNOT_START
    assert startup_record.agree("startup_group_split", gather(None)) == [[1, 0]] * WORLD
    own = FakeProfilingCache({("gemm", 8): (0, 5, 0.2), ("gemm", 16): (0, 1, 0.3)})
    monkeypatch.setattr(AutoTuner, "get", lambda: SimpleNamespace(profiling_cache=own))
    startup_record.adopt_tuned_tactics()
    assert own.cache == {("gemm", 8): (0, 3, 0.1), ("gemm", 16): (0, 1, 0.3)}
    assert gathers == []
    startup_record.close()
    with pytest.raises(SystemExit, match="startup_group_split 3 times") as exited:
        startup_record.agree("startup_group_split", gather(None))  # One call more than rank 0 made.
    assert exited.value.code == membership.REPLACEMENT_CANNOT_START


def test_startup_values_pass_through_without_flag(agreed_startup, monkeypatch):
    monkeypatch.setattr(_wide_ep_fault_tolerance, "_enabled", False)
    assert startup_record.agree("rank_hosts", lambda: ["a"] * WORLD) == ["a"] * WORLD
    assert startup_record.published() == {"values": {}, "exchanges": []}


class _LlmArgs:
    def __init__(self, **fields):
        self.fields = fields

    def model_dump(self):
        return dict(self.fields)


def test_mismatched_replacement_refuses_to_start(agreed_startup, monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text('{"num_hidden_layers": 2}')
    # The configuration comes from rank 0, so only the version and the checkpoint can differ.
    args = {
        "llm_args": _LlmArgs(max_batch_size=8, tokenizer=object()),
        "hf_model_dir": str(tmp_path),
    }
    fingerprint = recovery_module.startup_fingerprint(args)
    assert recovery_module.startup_fingerprint({**args, "llm_args": None}) == fingerprint
    (tmp_path / "config.json").write_text('{"num_hidden_layers": 3}')
    assert recovery_module.startup_fingerprint(args) != fingerprint

    store = _published(monkeypatch, args)
    store.set(AGREED_STARTUP_KEY, pickle.dumps({"fingerprint": fingerprint}))
    with pytest.raises(SystemExit, match="does not match") as exited:
        recovery_module.worker_node_main()
    # Another replacement would not match either, so its supervisor starts none.
    assert exited.value.code == membership.REPLACEMENT_CANNOT_START


def test_rank0_publishes_args_without_ipc_keys(agreed_startup, shared_secret, monkeypatch):
    """Check that rank 0 starts the authority, publishes its arguments without the IPC keys, and joins."""
    from tensorrt_llm.executor.utils import WorkerCommIpcAddrs

    events = []
    store = FakeStore()
    monkeypatch.setenv(FAULT_TOLERANCE_RANK_ENV, "0,4,4,0")
    monkeypatch.setattr(recovery_module, "_client", None)
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
    assert events == [("authority", 4), ("join", [WORKER_ARGS_KEY], 0)]
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
def test_original_rank_binds_client_after_startup(agreed_startup, shared_secret, monkeypatch, rank):
    """Check the steps of an original rank at the end of its startup."""
    events = []
    store = FakeStore()
    client = Mock(
        record=MembershipRecord(generation=0, state=MembershipState.STARTING, active_mask=0b1111),
        bind=Mock(side_effect=lambda *args, **kwargs: events.append("bind")),
        rank=rank,
        incarnation=0,
    )

    def report(**changes):
        events.append(("report", changes))
        client.record = MembershipRecord(
            generation=0, state=MembershipState.COMMITTED, active_mask=0b1111
        )
        recovery_module._on_record_at_startup(client.record)  # As the client's thread delivers it.

    client.report = report
    monkeypatch.setattr(recovery_module, "_client", client)
    monkeypatch.setattr(recovery_module, "_every_rank_started", threading.Event())
    monkeypatch.setattr(recovery_module, "_authority", "rank 0's authority" if rank == 0 else None)
    monkeypatch.setattr(
        recovery_module, "rendezvous_store", lambda: (events.append("store"), store)[1]
    )
    monkeypatch.setattr(recovery_module, "_startup_fingerprint", "rank 0's fingerprint")
    executor = _executor(rank)
    executor.kv_cache_manager = device = _Device()
    monkeypatch.setattr(recovery_module.torch.cuda, "mem_get_info", lambda: (device.free, 1 << 40))
    executor.kv_pools_suspended = device.suspended
    executor.dist.allgather.side_effect = lambda value: [value] * WORLD
    # The state that the model check bound the MoE layers to.
    moe_state = Mock(workspace_bytes=Mock(return_value=4 * MIB))
    executor.model_engine.model = model = object()
    monkeypatch.setattr(
        MoeExecutionState,
        "model_comms",
        lambda bound: [SimpleNamespace(execution_state=moe_state)] if bound is model else [],
    )
    executor.model_engine.moe_load_balancer = MagicMock(spec=MoeLoadBalancer)
    expert_restore = MagicMock(spec=EplbExpertRestore)
    executor.model_engine.moe_load_balancer.expert_restore = expert_restore
    expert_restore.largest_failure_restore_bytes.side_effect = lambda: (
        events.append("restore fits"),
        6 * MIB,
    )[1]
    expert_restore.keep_host_weights_for_the_deployment.side_effect = lambda: (
        events.append("host weights"),
        "moe_shared_deployment_",
    )[1]
    recovery = RankFailureRecovery.create(executor)
    # Until the deployment serves, a rank's failure ends it with the rank's host weights. After that, the node
    # supervisor keeps the host weights under the prefix rank 0 publishes until the deployment ends.
    assert events == [
        "restore fits",
        "host weights",
        "store",
        ("report", {"started": True}),
        "bind",
    ]
    assert recovery.address == "head-node:29600"
    assert recovery._moe is moe_state
    assert recovery._startup_memory == STARTUP_MEMORY
    # The check shrank the KV cache pool for the largest failure's restore and took its quota back.
    assert _resizes(device) == [54 * MIB, STARTUP_MEMORY.kv_cache_quota]
    assert recovery._authority == ("rank 0's authority" if rank == 0 else None)
    agreed = {**startup_record.published(), "fingerprint": "rank 0's fingerprint"}
    expected = {
        HOST_WEIGHTS_PREFIX_KEY: b"moe_shared_deployment_",
        AGREED_STARTUP_KEY: pickle.dumps(agreed),
    }
    assert store.values == (expected if rank == 0 else {})


@pytest.mark.parametrize(
    "restore_bytes, peer_fits, short",
    [(52 * MIB, True, None), (53 * MIB, True, [0]), (52 * MIB, False, [1])],
)
def test_startup_rejects_unrestorable_largest_failure(monkeypatch, restore_bytes, peer_fits, short):
    """Check that the startup applies StartupMemory.fit_kv_cache to the largest failure on every rank."""
    device = _Device()
    monkeypatch.setattr(recovery_module.torch.cuda, "mem_get_info", lambda: (device.free, 1 << 40))
    moe_state = Mock(workspace_bytes=Mock(return_value=4 * MIB))
    executor = SimpleNamespace(
        dist=Mock(rank=0, allgather=lambda fits: [fits, peer_fits, True, True]),
        kv_pools_suspended=device.suspended,
    )
    balancer = Mock(
        expert_restore=Mock(largest_failure_restore_bytes=Mock(return_value=restore_bytes))
    )
    if short is None:
        recovery_module._check_largest_failure_fits(executor, balancer, STARTUP_MEMORY, moe_state)
    else:
        with pytest.raises(ValueError, match=re.escape(f"on ranks {short}")) as error:
            recovery_module._check_largest_failure_fits(
                executor, balancer, STARTUP_MEMORY, moe_state
            )
        assert "more ranks, fewer EPLB slots per rank" in str(error.value)
    assert device.calls.count("suspend") == 1
    # The pool tried the largest failure's quota and took its startup quota back.
    assert _resizes(device)[0] == 64 * MIB - restore_bytes - 4 * MIB
    assert device.gpu_quota == STARTUP_MEMORY.kv_cache_quota


@pytest.mark.parametrize("identity", ["2,4,4,0", "2,4,4,1"], ids=["original", "replacement"])
def test_no_rank_creates_a_process_group(monkeypatch, identity):
    """Check that no rank creates a torch.distributed group, which a replacement could never join."""
    monkeypatch.setattr(_wide_ep_fault_tolerance, "_enabled", True)
    monkeypatch.setenv(FAULT_TOLERANCE_RANK_ENV, identity)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="process group"):
        RankFailureRecovery.create(_executor(rank=FAILED))


# Rejoin. A replacement rank starts alone and the active ranks admit it.


def test_replacement_waits_for_admission(monkeypatch):
    _exit_raises(monkeypatch)
    recovery = _replacement()
    recovery._executor.iter_counter = 0
    for r in (0, 1, 3):
        recovery._store.set(f"host_exchange_rows/2/0/{r}", pickle.dumps(f"rows-{r}".encode()))
        _mark_arrived(recovery._store, "host_exchange_rows/2/0", r)
    errors = []

    def join():
        try:
            recovery._join()
        except SystemExit as error:  # The patched os._exit.
            errors.append(error)

    joined = threading.Thread(target=join, daemon=True)
    joined.start()
    recovery._on_record(_record(MembershipState.COMMITTED))  # Without rank 2, so not admitted yet.
    time.sleep(0.2)
    # It reports only that it is ready to join. There is no EPLB here, so it sends no placement.
    assert joined.is_alive()
    assert recovery._client.reports == [{"joining_ready": True}]
    assert not errors
    recovery._moe.request_abort.assert_not_called()  # It is not a member yet, so there is nothing to stop.
    recovery._on_record(_joining_record(MembershipState.PROPOSE))
    _wait_until(lambda: len(recovery._client.reports) == 2)
    assert recovery._client.reports[-1]["reply"] == Reply.READY
    recovery._on_record(_joining_record(MembershipState.INSTALL))
    joined.join(5)
    assert not joined.is_alive()
    assert not errors
    # Every active rank maps its fresh row at INSTALL. It enters the host exchange at the resume point, where it
    # also restores and replies INSTALLED.
    assert recovery._exchange.installs == [
        ([b"rows-0", b"rows-1", b"row-mine", b"rows-3"], SURVIVORS, 40)
    ]
    assert all(report.get("reply") != Reply.INSTALLED for report in recovery._client.reports)
    assert recovery._executor.iter_counter == 21
    assert recovery.pauses_forward()


def _admitting(monkeypatch, rank):
    """Return the recovery of ``rank`` at the first iteration after the change that adds rank 2."""
    monkeypatch.setattr(recovery_module.torch.cuda, "synchronize", lambda: None)
    recovery = _resuming_recovery(
        [set()] * (WORLD - 1), record=_joining_record(MembershipState.INSTALL), rank=rank
    )
    recovery.incarnation = JOINED[rank]
    recovery._admitted = JOINED[rank] == 0  # A replacement is admitted only at its commit.
    recovery._committed_mask = SURVIVORS  # Rank 2 failed earlier.
    recovery._moe.aborted.return_value = False  # An admission does not abort its steps.
    recovery._resume_after = 20
    executor = recovery._executor
    executor.iter_counter = 21
    executor.waiting_queue = FCFSWaitingQueue()
    return recovery


def test_rank0_admits_joining_rank(monkeypatch):
    recovery = _admitting(monkeypatch, rank=0)
    executor = recovery._executor
    executor.waiting_queue.add_requests([RequestQueueItem(7)])
    report = recovery._client.report
    handed = {}

    def report_installed(**changes):
        if changes.get("reply") == Reply.INSTALLED:  # The joining rank has read them by then.
            handed.update(recovery._store.values)
        report(**changes)

    recovery._client.report = report_installed
    recovery.before_iteration()
    epoch = membership.next_epoch(40)
    assert handed["host_exchange_epoch/2"] == str(epoch).encode()
    assert recovery._exchange.admits == [(0b1111, epoch)]
    assert recovery._exchange.installs == []
    # The joining ranks take the waiting queue from the store, without a collective with the active ranks.
    queued = pickle.loads(handed["waiting_queue/2"])
    assert [item.id for item in queued] == [7]
    # Once committed, the store keeps neither.
    assert "host_exchange_epoch/2" not in recovery._store.values
    assert "waiting_queue/2" not in recovery._store.values
    executor.dist.broadcast.assert_not_called()
    recovery._moe.health.mark_active.assert_any_call(FAILED)
    assert _replies(recovery) == [Reply.INSTALLED]
    assert not recovery.pauses_forward()


def test_joining_rank_takes_rank0_waiting_queue(monkeypatch):
    recovery = _admitting(monkeypatch, rank=FAILED)
    recovery._store.set("host_exchange_epoch/2", b"41")
    recovery._store.set("waiting_queue/2", pickle.dumps([RequestQueueItem(7)]))
    executor = recovery._executor
    recovery.before_iteration()
    assert recovery._exchange.admits == [(0b1111, 41)]
    assert recovery._exchange.installs == []
    assert [item.id for item in executor.waiting_queue] == [7]
    executor.dist.broadcast.assert_not_called()
    # It ran no step and enters the host exchange only at its admission.
    executor._retire_inflight_batch_for_control.assert_not_called()
    recovery._moe.remap_workspaces.assert_called_once()
    recovery._moe.rearm.assert_called_once()
    recovery._moe.install_mask.assert_called_once()
    assert _replies(recovery) == [Reply.INSTALLED]
    assert not recovery.pauses_forward()


def test_active_ranks_retire_step_before_resize(monkeypatch):
    """Check that the active ranks retire the step in flight at the resume point before the KV cache resize."""
    recovery = _admitting(monkeypatch, rank=0)
    executor = recovery._executor
    executor.previous_batch = _batch([_request(14)])
    device = _Device()
    steps = device.calls

    def retire():
        steps.append(("retire", executor.previous_batch is not None))
        executor.previous_batch = None

    executor._retire_inflight_batch_for_control.side_effect = retire
    recovery._exchange.admit = lambda active_mask, epoch: steps.append("admit")
    recovery._moe.workspace_bytes.return_value = 4 * MIB
    monkeypatch.setattr(recovery_module.torch.cuda, "mem_get_info", lambda: (device.free, 1 << 40))
    executor.kv_pools_suspended = device.suspended
    recovery.before_iteration()
    _vote(recovery)
    recovery.before_iteration()
    # An admission does not abort the step, so its results are applied before the joining rank enters the host
    # exchange. The give-back at the next step boundary retires the step in flight before its resize.
    order = [step for step in steps if step != "resume" and step[0] != "resize"]
    assert order == [("retire", True), "admit", "suspend", ("retire", False), "suspend"]
    assert _replies(recovery) == [Reply.INSTALLED]


# A newer change that replaces the admission of rank 2: rank 3 fails and rank 2 still joins.
_NEWER = _joining_record(MembershipState.PROPOSE, generation=3).model_copy(
    update=dict(active_mask=0b0111)
)
_NEWER_INSTALL = _NEWER.model_copy(
    update=dict(state=MembershipState.INSTALL, resume=ResumeState(iteration=30, epoch=50))
)


def _newer_change(recovery, answers):
    """Set up the admission of rank 2 in generation 2 to meet the newer generation 3.

    The store holds what rank 0 and the other active ranks publish for both. ``answers`` maps each
    (reply, generation) of this rank to the record the authority publishes next.
    """
    store = recovery._store
    for generation in (2, 3):
        store.set(f"host_exchange_epoch/{generation}", b"41")
        store.set(f"waiting_queue/{generation}", pickle.dumps([RequestQueueItem(7)]))
    for r in (0, 1):
        store.set(f"host_exchange_rows/3/0/{r}", pickle.dumps(f"rows-{r}".encode()))
    reports = recovery._client.reports

    def report(**changes):
        reports.append(changes)
        answer = answers.get((changes.get("reply"), changes.get("reply_generation")))
        if answer is not None:
            recovery._on_record(answer)

    recovery._client.report = report


@pytest.mark.parametrize(
    "installed", [False, True], ids=["proposed after the retire", "installed inside the retire"]
)
def test_newer_change_during_resume_keeps_collectives(monkeypatch, installed):
    """Check that a newer change during the resume keeps the active ranks in the same collectives.

    Rank 3 fails while the active ranks retire the step in flight. A rank may install the newer change inside the
    retire while its peers install it at their next round. Only a rank that installed the newer change keeps the
    replaced change's joining rank out of its host exchange. No rank replies to the replaced change.
    """
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


def test_joining_rank_left_out_exits(monkeypatch):
    """Check that a joining rank a newer change leaves out exits, so its supervisor starts another one."""
    _exit_raises(monkeypatch)
    recovery = _admitting(monkeypatch, rank=FAILED)
    without_rank_2 = _NEWER.model_copy(update=dict(active_mask=0b1011, joining_mask=0))
    _newer_change(recovery, {(Reply.INSTALLED, 2): without_rank_2})
    with pytest.raises(SystemExit) as exited:
        recovery.before_iteration()
    assert exited.value.code == 1


def test_joining_rank_missing_commit_stops(monkeypatch):
    """Check that a joining rank that reads a newer PROPOSE instead of its commit stops like an active rank.

    Its membership client hands it the commit and the PROPOSE of generation 3 back to back, and rank 2 is active in
    generation 3. It stops for the PROPOSE in the host exchange instead of replying READY from where it joined.
    """
    _exit_raises(monkeypatch)
    recovery = _admitting(monkeypatch, rank=FAILED)
    committed = _joining_record(MembershipState.COMMITTED, joining=0)
    newer = _NEWER.model_copy(update=dict(joining_mask=0))
    # An abort answers a READY to generation 3, so that a READY ends the test.
    aborted = newer.model_copy(update=dict(state=MembershipState.ABORTED))
    _newer_change(recovery, {(Reply.READY, 3): aborted})
    report = recovery._client.report

    def commit_then_propose(**changes):
        report(**changes)
        if changes.get("reply") == Reply.INSTALLED:
            recovery._on_record(committed)
            recovery._on_record(newer)

    recovery._client.report = commit_then_propose
    recovery.before_iteration()
    assert _replies(recovery) == [Reply.INSTALLED]
    assert recovery._membership_changing()

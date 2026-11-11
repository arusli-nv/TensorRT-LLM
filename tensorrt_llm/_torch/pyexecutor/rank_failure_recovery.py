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
"""Keep a WideEP deployment serving when one of its ranks fails, and let a replacement rank rejoin.

Every rank is its own process, which joins the membership at the entry of its worker (:func:`start_worker`,
:func:`worker_node_main`); the startup collectives go through the rendezvous store (:class:`StartupGroup`). Once
serving, :class:`RankFailureRecovery` takes part in every membership change the authority proposes: it stops the
all-to-all and the host exchange, replies READY, moves the host exchange to the live ranks at INSTALL, and at the
resume point restores the lost experts and the all-to-all workspaces, then replies INSTALLED. See
``docs/source/features/wide-ep-fault-tolerance.md``.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import os
import pickle  # nosec B403
import secrets
import select
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import torch
from mpi4py import MPI

from tensorrt_llm._utils import (
    FAULT_TOLERANCE_RANK_ENV,
    OMPI_COMM_TYPE_HOST,
    RankIdentity,
    check_fault_tolerance_requirements,
    fault_tolerance_mode,
    fault_tolerance_replacement,
    local_mpi_comm,
    mpi_broadcast,
    mpi_rank,
    set_fault_tolerance_mode,
    set_mpi_comm,
    wide_ep_fault_tolerance_rank,
)
from tensorrt_llm.executor.utils import RankFailureError, RequestError
from tensorrt_llm.logger import logger

from ..distributed.host_exchange_epoch import next_epoch
from ..distributed.membership import (
    HOST_WEIGHTS_PREFIX_KEY,
    JOIN_SECRET_ENV,
    LAUNCHER_ENV_PREFIXES,
    PROGRESS_TIMEOUT_S,
    RECORD_KEY,
    RENDEZVOUS_ENV,
    RENDEZVOUS_TIMEOUT_S,
    STORE_POLL_S,
    TRANSACTION_KEY,
    MembershipAgent,
    MembershipRecord,
    MembershipState,
    ReplacementCannotStart,
    Reply,
    ResumeState,
    SignedStore,
    TransactionSummary,
    claimed_incarnation,
    open_store,
    rendezvous_store,
    startup_gather,
    store_gather,
    wait_for_keys,
)
from ..distributed.mnnvl_live_group import (
    LiveGroupComm,
    MnnvlMappingError,
    MnnvlPeerHandleError,
    isolated_gather,
)
from ..distributed.startup_record import startup_record

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
    from tensorrt_llm.mapping import Mapping

    from ..distributed import Distributed
    from .llm_request import LlmRequest
    from .py_executor import PyExecutor
    from .scheduler.adp_router import RankState
    from .scheduler.waiting_queue import WaitingQueue

# How long a rank that leaves the membership waits for the SHUTDOWN record, and rank 0 for the authority to end.
_AUTHORITY_TIMEOUT_S = 10.0
# How long a read of the server's fault-tolerance status waits for the membership store to answer.
_STATUS_READ_TIMEOUT_S = 1.0
# How often a rank whose KV cache pool fell short of its quota at a commit checks whether device memory was freed.
_GIVE_BACK_RETRY_S = 1.0
# The least length of the join secret, which keys the signatures of the membership store's values.
_JOIN_SECRET_BYTES = 32


class _MembershipRestarted(Exception):
    """A newer membership change replaced the one being installed."""


# The sockets that reserve the authority's ports ensure_rendezvous chose.
_rendezvous_ports: List[socket.socket] = []
# From the entry of this process's worker: its membership agent, and on rank 0 the authority process it started.
_agent: Optional[MembershipAgent] = None
_authority: Optional[subprocess.Popen] = None


class MembershipReady(NamedTuple):
    """The READY that rank 0's executor worker sends the proxy with fault tolerance, with the membership store's
    address."""

    signal: bytes
    error: Optional[str]
    worker_process_identities: list
    membership_address: str


def ensure_rendezvous() -> None:
    """Choose and export the join secret and the rendezvous address (this host and two free consecutive ports) when
    unset, for the workers this process starts. The ports stay reserved by sockets bound with SO_REUSEADDR that never
    listen, which still lets the authority bind them later."""
    global _rendezvous_ports
    if not os.environ.get(JOIN_SECRET_ENV):
        os.environ[JOIN_SECRET_ENV] = secrets.token_hex(_JOIN_SECRET_BYTES)
    if RENDEZVOUS_ENV not in os.environ:
        while True:
            reserved = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(2)]
            for sock in reserved:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            reserved[0].bind(("", 0))
            port = reserved[0].getsockname()[1]
            try:
                reserved[1].bind(("", port + 1))
                break
            except (OSError, OverflowError):  # taken, or past the last port: try another pair
                for sock in reserved:
                    sock.close()
        _rendezvous_ports = reserved
        os.environ[RENDEZVOUS_ENV] = f"{socket.gethostbyname(socket.gethostname())}:{port}"


def _host_collectives_requirements(mapping: Mapping) -> List[str]:
    """What the host collectives over MNNVL memory (``MnnvlDist``) need that this deployment lacks."""
    from tensorrt_llm._utils import mpi_disabled

    from ..distributed.host_exchange import MnnvlDist

    if mpi_disabled():
        return ["the MPI orchestrator (TLLM_DISABLE_MPI unset)"]
    return MnnvlDist.unmet_requirements(mapping) or [
        "host collectives over MNNVL memory (MnnvlDist)"
    ]


def check_supported(llm_args: TorchLlmArgs, mapping: Mapping, dist: Distributed) -> None:
    """Check, before the model loads, what WideEP fault tolerance needs beyond the LLM arguments' validation.

    Raises:
        ValueError: The configuration is not supported with fault tolerance.
    """
    from ..distributed.host_exchange import MnnvlDist
    from ..moe.fused_moe.communication.nvlink_one_sided import FORCE_CFT_ENV, get_force_cft
    from ..moe.fused_moe.moe_comm_timeout_guard import (
        SERVING_TIMEOUT_ENV,
        resolve_moe_comm_timeout_budgets,
    )

    # The all-to-all kernels that wait on a hung rank are released when the authority excludes it, up to 1.5 hang
    # timeouts after their wait began: they must not time out before.
    kernel_timeout_s = resolve_moe_comm_timeout_budgets(os.environ).serving_seconds
    if kernel_timeout_s is None:
        kernel_timeout_s = torch.ops.trtllm.moe_a2a_get_timeout()  # the kernels' own default
    missing = [] if isinstance(dist, MnnvlDist) else _host_collectives_requirements(mapping)
    checks = [
        (not missing, ", ".join(missing)),
        (
            mapping.moe_ep_size == mapping.world_size,
            "expert parallelism over every rank (moe_expert_parallel_size equal to the world size)",
        ),
        # Resolved from "auto" as the model loads, after the arguments' validation: checked only here.
        (
            llm_args.kv_cache_config.use_kv_cache_manager_v2 is True,
            "KV cache manager V2 (kv_cache_config.use_kv_cache_manager_v2=True)",
        ),
        # Bound execution (NVLinkOneSided.bind_execution_state) runs on the fence path.
        (
            get_force_cft() is not True,
            f"the fence path of the NVLink one-sided all-to-all ({FORCE_CFT_ENV} unset or 0)",
        ),
        (
            kernel_timeout_s >= 2 * PROGRESS_TIMEOUT_S,
            f"{SERVING_TIMEOUT_ENV} of at least {2 * PROGRESS_TIMEOUT_S:.0f} s (twice the hang timeout)",
        ),
    ]
    check_fault_tolerance_requirements(checks)


def _launched_rank() -> RankIdentity:
    """The rank this process stands for, once the join secret and the rendezvous address are checked.

    Raises:
        ValueError: The launch does not give the process what it needs.
    """
    try:
        secret = bytes.fromhex(os.environ[JOIN_SECRET_ENV])
        int(os.environ[RENDEZVOUS_ENV].rsplit(":", 1)[1])
    except (KeyError, IndexError, ValueError):
        raise ValueError(
            f"WideEP fault tolerance requires on every rank a hex join secret in {JOIN_SECRET_ENV} and the "
            f"rendezvous address in {RENDEZVOUS_ENV}: host:port on rank 0's node, where the rendezvous store "
            "listens. The server sets both for the workers it starts; where it does not start them "
            "(trtllm-llmapi-launch, or an MPI session passed to the LLM API), export them before launching or "
            "creating the session, e.g. the secret from `openssl rand -hex 32` and the first host of the job "
            "with a free port; or turn it off (enable_wide_ep_fault_tolerance=False)"
        ) from None
    if len(secret) < _JOIN_SECRET_BYTES:
        raise ValueError(
            f"WideEP fault tolerance requires a join secret of at least {_JOIN_SECRET_BYTES} bytes "
            f"({2 * _JOIN_SECRET_BYTES} hex digits) in {JOIN_SECRET_ENV}, got {len(secret)}: it keys the "
            "signatures of the rendezvous store's values, which the ranks unpickle. Export e.g. "
            "`openssl rand -hex 32`, and check that it printed the secret (without openssl it yields an empty "
            "string); or turn it off (enable_wide_ep_fault_tolerance=False)"
        )
    return _rank_identity()


def _rank_identity() -> RankIdentity:
    """This process's rank, from its node supervisor.

    Raises:
        ValueError: The process has no rank identity.
    """
    identity = wide_ep_fault_tolerance_rank()
    if identity is None:
        raise ValueError(
            "WideEP fault tolerance starts every rank as its own process, with "
            f"'<rank>,<world size>,<GPUs per node>,<incarnation>' in {FAULT_TOLERANCE_RANK_ENV}: launch the ranks "
            "with trtllm-llmapi-launch, whose node supervisor sets it, or let the server start them on one node"
        )
    return identity


def check_supported_model(model: torch.nn.Module) -> None:
    """Check the built model before any weight loads: its implementation, architecture, MoE communication and the room
    for its host weights.

    Raises:
        ValueError: The model is not supported with fault tolerance.
    """
    from ..moe.fused_moe.communication.execution_state import MoeExecutionState
    from .config_utils import is_hybrid_linear

    target = model.model_config.modeling_v2_target
    if target is not None:
        raise ValueError(
            f"WideEP fault tolerance does not support a model built from a modeling_v2 target ({target.__name__}); "
            "set modeling_v2='off' or turn it off (enable_wide_ep_fault_tolerance=False)"
        )
    if is_hybrid_linear(model.model_config.pretrained_config):
        raise ValueError(
            "WideEP fault tolerance does not support hybrid linear-attention models: their KV cache manager "
            "(MambaHybridCacheManagerV2) commits recurrent-state snapshots for reuse at snapshot points and when a "
            "request is freed, so it cannot hold a step's reuse commits until the step is accepted; turn it off "
            "(enable_wide_ep_fault_tolerance=False)"
        )
    MoeExecutionState.model_comms(model)
    _check_host_weights_fit(model)


def _check_host_weights_fit(model: torch.nn.Module) -> None:
    """Check that /dev/shm has room for the host weights of every expert of ``model`` beyond the node's segments that
    exist already (those a replacement reattaches).

    Raises:
        ValueError: /dev/shm has too little free space.
    """
    from ..moe.fused_moe.eplb_expert_restore import SHARED_MEMORY_DIR, DeploymentHostCopies
    from ..moe.fused_moe.impl_blocks import MoEEplbWeightLayoutMixin

    # Per MoE layer's host tensor sharer, the bytes of one slot of each of the layer's weights of local slots.
    slot_bytes: Dict[DeploymentHostCopies, List[int]] = {}
    for module in model.modules():
        sharer = getattr(getattr(module, "layer_load_balancer", None), "host_tensor_sharer", None)
        if isinstance(module, MoEEplbWeightLayoutMixin) and sharer is not None:
            slot_bytes.setdefault(sharer, []).extend(
                param.numel() // param.shape[0] * param.element_size()
                for _, param in module.slot_parameters()
            )
    needed = sum(
        sharer.node_bytes(sizes) - sharer.node_bytes_held() for sharer, sizes in slot_bytes.items()
    )
    shm = os.statvfs(SHARED_MEMORY_DIR)
    free = shm.f_bavail * shm.f_frsize
    if needed > free:
        raise ValueError(
            f"WideEP fault tolerance keeps the host weights of every expert on every node in {SHARED_MEMORY_DIR}: "
            f"this node needs {needed / 2**30:.1f} GiB ({needed} bytes) more there, but {SHARED_MEMORY_DIR} has "
            f"{free / 2**30:.1f} GiB ({free} bytes) free. Give {SHARED_MEMORY_DIR} more room (`docker run "
            "--shm-size`; on Kubernetes, an emptyDir volume with medium: Memory and a sizeLimit mounted at "
            f"{SHARED_MEMORY_DIR}), or turn it off (enable_wide_ep_fault_tolerance=False)"
        )


# The agreed-startup record in the rendezvous store: the fingerprint and rank 0's startup record (startup_record).
AGREED_STARTUP_KEY = "agreed_startup"
# Rank 0's worker_main arguments, which every other rank runs worker_main with (see start_worker).
WORKER_ARGS_KEY = "worker_args"

_startup_fingerprint: Optional[str] = None
_worker_main_args: Optional[dict] = None
# A replacement rank: the agreed-startup record, once read.
_published_record: Optional[dict] = None


def adopt_tuned_tactics(profiling_cache) -> None:
    """At the final executor's autotuner warmup: rank 0 records the tactics shared by every rank, and a replacement,
    which skipped the warmup of the KV cache estimation, adopts them."""
    tactics = startup_record.agree(
        "tuned_tactics",
        lambda: profiling_cache._serialize_cache_data(
            profiling_cache._partition_cache_by_strategy()[0]
        ),
    )
    if startup_record.replaying:
        profiling_cache.cache.update(profiling_cache._deserialize_cache_data(tactics))


_deployment_nonce: Optional[str] = None


def deployment_nonce() -> str:
    """A random name of this deployment, the same on every rank (a startup collective), which keeps its shared-memory
    segments apart from another deployment's."""
    global _deployment_nonce
    if _deployment_nonce is None:
        _deployment_nonce = startup_record.agree(
            "deployment_nonce",
            lambda: mpi_broadcast(secrets.token_hex(8) if mpi_rank() == 0 else None),
        )
    return _deployment_nonce


def _agreed_startup_record() -> bytes:
    """Rank 0's agreed-startup record: the fingerprint and its startup record."""
    return pickle.dumps({**startup_record.published(), "fingerprint": _startup_fingerprint})


def _agreed_startup() -> dict:
    """A replacement rank: the agreed-startup record the live ranks published."""
    global _published_record
    if _published_record is None:
        # Only processes with the join secret write the store's values (SignedStore).
        _published_record = pickle.loads(_startup_store().get(AGREED_STARTUP_KEY))  # nosec B301
    return _published_record


def startup_fingerprint(worker_main_args: dict) -> str:
    """What a replacement must share with the deployment beyond rank 0's arguments: the TensorRT-LLM version and the
    checkpoint's ``config.json``."""
    from tensorrt_llm.version import __version__

    digest = hashlib.sha256(__version__.encode())
    model_dir = worker_main_args.get("hf_model_dir")
    if model_dir is not None and Path(model_dir, "config.json").is_file():
        digest.update(Path(model_dir, "config.json").read_bytes())
    return digest.hexdigest()


def start_worker(worker_main_args: dict) -> None:
    """Called first in ``worker_main`` with fault tolerance. Rank 0 starts the authority, publishes the arguments every
    other rank runs ``worker_main`` with, joins the membership and records its startup; the other ranks joined in
    :func:`worker_node_main`.

    Raises:
        ValueError: The launch does not give this process what fault tolerance needs (:func:`_launched_rank`).
    """
    global _startup_fingerprint, _worker_main_args, _authority
    _startup_fingerprint = startup_fingerprint(worker_main_args)
    _worker_main_args = worker_main_args
    identity = _launched_rank()
    if identity.rank == 0:
        _authority, _ = _start_authority(identity.world_size, RENDEZVOUS_TIMEOUT_S)
        store = _startup_store()
        store.set(WORKER_ARGS_KEY, _worker_args_for_other_ranks())
        _join_at_entry(store, identity)
        startup_record.record()
    elif _agent is None:
        raise ValueError(
            f"WideEP fault tolerance starts rank {identity.rank} as its own process, which takes rank 0's worker "
            "arguments from the rendezvous store: run `python -m tensorrt_llm.llmapi.mgmn_worker_node`"
        )


def _worker_args_for_other_ranks() -> bytes:
    """Rank 0's ``worker_main`` arguments for the other ranks, without its IPC addresses and keys: the store signs its
    values but does not encrypt them."""
    from tensorrt_llm.executor.utils import WorkerCommIpcAddrs

    return pickle.dumps(
        {
            **_worker_main_args,
            "worker_queues": WorkerCommIpcAddrs(None, None, None),
            "rpc_addr": None,
            "hmac_key": b"",
        }
    )


def worker_node_main() -> None:
    """The entry point of every rank other than 0 with fault tolerance (``python -m
    tensorrt_llm.llmapi.mgmn_worker_node``): joins the membership, then runs ``worker_main`` with the arguments rank 0
    published; a replacement replays rank 0's startup record. Its waits end when the authority ends the deployment.

    Raises:
        ValueError: The launch does not give this process what fault tolerance needs (:func:`_launched_rank`).
        ReplacementCannotStart: A replacement does not match the deployment (TensorRT-LLM version or checkpoint), or
            its startup failed in another way that a restart would repeat.
    """
    # Both before the executor's modules are imported, which may read them.
    set_fault_tolerance_mode(True)
    identity = _launched_rank()
    from tensorrt_llm.executor.worker import worker_main

    if identity.rank == 0:
        raise ValueError(
            "WideEP fault tolerance runs rank 0's worker in the session of the server that starts the deployment, "
            "and every other rank as its own process"
        )
    store = _startup_store()
    _join_at_entry(store, identity)
    try:
        # Only processes with the join secret can write the store's values (SignedStore).
        worker_main_args = pickle.loads(store.get(WORKER_ARGS_KEY))  # nosec B301
        if identity.incarnation > 0:
            if startup_fingerprint(worker_main_args) != _agreed_startup()["fingerprint"]:
                raise ReplacementCannotStart(
                    "The replacement rank does not match the deployment it replaces a rank of (TensorRT-LLM "
                    "version or checkpoint)"
                )
            startup_record.replay(_agreed_startup())
        worker_main(**worker_main_args)
    except ReplacementCannotStart as error:
        logger.error(f"{error}; no replacement of this rank can start")
        raise


@functools.lru_cache(maxsize=1)
def _startup_store() -> SignedStore:
    """This process's client of the rendezvous store for its startup, which may start before rank 0's authority."""
    return rendezvous_store(RENDEZVOUS_TIMEOUT_S)


# The startup collectives this rank completed: its progress while the deployment starts (RankReport.progress).
_startup_steps = 0


def _report_startup_wait(waiting_for: int) -> None:
    """Report the peers a startup collective of this rank waits for; 0 completes a collective, a step of progress."""
    global _startup_steps
    if not waiting_for:
        _startup_steps += 1
    _agent.report(waiting_for=waiting_for, progress=_startup_steps)


# The deployment's mapping, which gives the node of each rank to a split by node (join_startup_group).
_mapping: Optional[Mapping] = None


class StartupGroup(LiveGroupComm):
    """``mpi_comm()`` with fault tolerance, and its subgroups: a group of ``ranks`` (global ranks, in group-rank order)
    exchanges through :func:`startup_gather`. A replacement's groups have only itself live: their exchanges answer at
    once, and its splits come from the startup record. It has the mpi4py methods the startup calls.
    """

    def __init__(self, identity: RankIdentity, ranks: Sequence[int], key: str = "world"):
        ranks = tuple(ranks)
        if identity.incarnation:
            isolated = isolated_gather(identity.rank, identity.world_size)

            def gather(value: Any) -> List[Any]:
                values = isolated(value)
                # A replacement's startup progress: the authority stops holding other joins for one that has none.
                _report_startup_wait(0)
                return values

        else:
            gather = startup_gather(
                _startup_store(),
                key,
                identity.rank,
                ranks,
                identity.world_size,
                _report_startup_wait,
            )
        super().__init__(ranks, ranks.index(identity.rank), gather)
        self._identity = identity
        self._key = key
        self._subgroups = 0

    def subgroup(self, ranks: Sequence[int]) -> StartupGroup:
        """The group of ``ranks``, members of this group with this rank among them, in that order."""
        ranks = tuple(ranks)
        digest = hashlib.sha256(repr(ranks).encode()).hexdigest()[:16]
        self._subgroups += 1
        return StartupGroup(self._identity, ranks, f"{self._key}/{self._subgroups}-{digest}")

    def Split(self, color: int = 0, key: int = 0) -> StartupGroup:
        """The members that give the same ``color``, ordered by ``key``, then by their rank in this group."""
        splits = startup_record.agree("startup_group_split", lambda: self.allgather((color, key)))
        ranks = sorted(
            (split[1], group_rank, rank)
            for group_rank, (rank, split) in enumerate(zip(self.membership, splits))
            if split[0] == color
        )
        return self.subgroup(rank for _, _, rank in ranks)

    def Split_type(self, split_type: int, key: int = 0, info: Any = None) -> StartupGroup:
        """The members on this rank's node (``Mapping.get_node_rank``), for the node split types
        (``MPI.COMM_TYPE_SHARED``, Open MPI's ``COMM_TYPE_HOST``)."""
        if split_type not in (MPI.COMM_TYPE_SHARED, OMPI_COMM_TYPE_HOST):
            raise NotImplementedError(
                f"StartupGroup splits by node only, not by split type {split_type}"
            )
        node = _mapping.get_node_rank(self._identity.rank)
        return self.subgroup(r for r in self.membership if _mapping.get_node_rank(r) == node)

    def Barrier(self) -> None:
        self.barrier()

    def bcast(self, obj: Any, root: int = 0) -> Any:
        """The ``obj`` of group rank ``root`` (None in a replacement rank's group, but for its own)."""
        return self.allgather(obj)[root]

    def Free(self) -> None:
        """A group holds nothing to free: its exchanges share this process's store client."""

    def allreduce(self, obj: Any, op: Any = MPI.SUM) -> Any:
        """The live members' ``obj`` reduced by ``op``: ``MPI.SUM``, ``MPI.MIN`` or ``MPI.MAX``."""
        values = [value for value in self.allgather(obj) if value is not None]
        for mpi_op, reduce in ((MPI.SUM, sum), (MPI.MIN, min), (MPI.MAX, max)):
            if op == mpi_op:
                return reduce(values)
        raise NotImplementedError(f"StartupGroup reduces by MPI.SUM, MPI.MIN or MPI.MAX, not {op}")


def join_startup_group(mapping: Mapping) -> None:
    """Called in ``worker_main`` after :func:`start_worker`: makes the startup group of every rank this rank's
    ``mpi_comm()``, whose splits by node take each rank's node from ``mapping``, and sets ``TLLM_RANK``.

    Raises:
        ValueError: The process has no rank identity, or ``mapping`` places the ranks on other nodes than the node
            supervisor runs them on.
        RuntimeError: The node's communicator was split from MPI before.
    """
    global _mapping
    identity = _rank_identity()
    one_node = identity.world_size <= min(identity.gpus_per_node, mapping.gpus_per_node)
    check_fault_tolerance_requirements(
        [
            (
                mapping.gpus_per_node == identity.gpus_per_node or one_node,
                f"gpus_per_node {identity.gpus_per_node}, the ranks the node supervisor runs on each node (the GPUs "
                f"it sees, or its --gpus-per-node), not {mapping.gpus_per_node}",
            )
        ]
    )
    os.environ["TLLM_RANK"] = str(identity.rank)
    _mapping = mapping
    set_mpi_comm(StartupGroup(identity, range(identity.world_size)))
    if not isinstance(local_mpi_comm(), StartupGroup):
        raise RuntimeError(
            "The node's communicator was split from MPI before the startup group was set"
        )


def _join_at_entry(store: SignedStore, identity: RankIdentity) -> None:
    """Join the membership at the entry of this process's worker. Until the executor takes it over, the process exits
    once the deployment ends or the authority is gone."""
    global _agent
    _agent = MembershipAgent(
        identity.rank,
        store,
        _on_record_at_startup,
        incarnation=identity.incarnation,
        on_error=_on_error_at_startup,
    )
    _agent.start()


# Set once a record is past STARTING: every rank has started.
_every_rank_started = threading.Event()


def _on_record_at_startup(record: MembershipRecord) -> None:
    if record.state in (MembershipState.ABORT, MembershipState.SHUTDOWN):
        status = 0 if record.state == MembershipState.SHUTDOWN else 1
        _end_process(_agent.rank, status, f"the membership record is {record} during the startup")
    if record.state != MembershipState.STARTING:
        _every_rank_started.set()


def _on_error_at_startup(error: Exception) -> None:
    _end_process(_agent.rank, 1, f"the membership authority is gone during the startup ({error!r})")


def wait_for_every_rank() -> None:
    """Report that this original rank's startup is done, and return once every rank's is."""
    _agent.report(started=True)
    _every_rank_started.wait()


def _end_process(rank: int, status: int, reason: str) -> None:
    (logger.info if status == 0 else logger.error)(f"Rank {rank} exits: {reason}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)


def _moe_load_balancer_of(executor: PyExecutor):
    """The executor's EPLB load balancer (a ``MoeLoadBalancer``), or None."""
    from ..moe.fused_moe.moe_load_balancer import MoeLoadBalancer

    balancer = executor.model_engine.moe_load_balancer
    return balancer if isinstance(balancer, MoeLoadBalancer) else None


def _admission_keys(generation: int) -> Tuple[str, str]:
    """The keys of the host exchange's epoch and the waiting queue rank 0 hands the joining ranks of ``generation``."""
    return f"host_exchange_epoch/{generation}", f"waiting_queue/{generation}"


class StartupMemory(NamedTuple):
    """This rank's GPU KV cache pool once the deployment serves, which every membership change resizes."""

    # The pool's quota: it keeps this quota less the device memory fault tolerance holds.
    kv_cache_quota: int

    def free_to_keep(self, kv_cache_manager, held: int) -> int:
        """The device memory serving keeps free at the start of a change: the free memory less what the pool lacks of
        its quota for ``held`` (:meth:`fit_kv_cache`), which is free as well."""
        return torch.cuda.mem_get_info()[0] - (
            self.kv_cache_quota - held - kv_cache_manager.gpu_quota
        )

    def fit_kv_cache(self, kv_cache_manager, held: int, needed: int, free_to_keep: int = 0) -> int:
        """Resize the KV cache pool to its quota once the deployment served less ``held`` (the device memory fault
        tolerance holds, ``EplbExpertRestore.backed_spare_slot_bytes``) and ``needed`` (the memory it allocates next),
        growing only into memory free beyond ``free_to_keep``, and return its quota. Derived afresh at every change, so
        repeated failures do not ratchet it down. Every KV cache must be suspended.

        Raises:
            RuntimeError: The pool cannot take that quota (``KVCacheManagerV2.resize_gpu_quota``), e.g. below its
                minimum.
        """
        return kv_cache_manager.resize_gpu_quota(
            self.kv_cache_quota
            - held
            - self.shortfall(kv_cache_manager, held, free_to_keep)
            - needed
        )

    def shortfall(self, kv_cache_manager, held: int, free_to_keep: int = 0) -> int:
        """How much of its quota for ``held`` (:meth:`fit_kv_cache`) the pool neither holds nor can take from the
        device memory free beyond ``free_to_keep``."""
        return max(
            0,
            self.kv_cache_quota
            - held
            - kv_cache_manager.gpu_quota
            - torch.cuda.mem_get_info()[0]
            + free_to_keep,
        )


@dataclass
class _GiveBackRetry:
    """A give-back at a commit that fell short (:meth:`RankFailureRecovery._give_back_kv_cache`): pages an excluded
    process still maps come free once it ended."""

    # The device memory serving keeps free (StartupMemory.free_to_keep at the start of the change).
    free_to_keep: int
    # Per rank the committed record excludes, the incarnation of its last process.
    excluded: Dict[int, int]
    # The free device memory after the last give-back.
    free_seen: int
    # The time.monotonic() from which the next check is due.
    check_after_s: float
    # Whether every excluded process ended: the next give-back is the last.
    ended: bool = False


def _check_largest_failure_fits(
    executor: PyExecutor, balancer, startup_memory: StartupMemory
) -> None:
    """Check that every original rank can restore the largest failure by the rule every change applies
    (:meth:`StartupMemory.fit_kv_cache`). Collective over the original ranks.

    Raises:
        ValueError: No membership change could restore that failure on some rank.
    """
    needed = (
        balancer.expert_restore.largest_failure_restore_bytes()
        + executor.model_engine.moe_execution_state.workspace_bytes()
    )
    fits = True
    with executor._kv_pools_suspended() as kv_cache_manager:
        try:
            startup_memory.fit_kv_cache(kv_cache_manager, 0, needed)
        except RuntimeError:
            fits = False
        startup_memory.fit_kv_cache(kv_cache_manager, 0, 0)
    short = [rank for rank, rank_fits in enumerate(executor.dist.allgather(fits)) if not rank_fits]
    if short:
        raise ValueError(
            "WideEP fault tolerance cannot recover from the loss of a node other than rank 0's (or of one rank): "
            f"restoring its experts needs more device memory on ranks {short} than their KV cache pools can give "
            f"up above their minimum ({needed} bytes on rank {executor.dist.rank}, whose pool holds "
            f"{startup_memory.kv_cache_quota} bytes); use more ranks, fewer EPLB slots per rank "
            "(moe_config.load_balancer.num_slots) or a larger KV cache (kv_cache_config.free_gpu_memory_fraction)"
        )


def _helper_process_env() -> Dict[str, str]:
    """The environment of the membership authority, a process outside this rank's MPI job without MPI (no mpi4py
    or C++ MPI_Init), the launcher's rank variables or a GPU, at this rank's log level."""
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(LAUNCHER_ENV_PREFIXES)
    }
    env.update(
        TRT_LLM_NO_LIB_INIT="1",
        MPI4PY_RC_INITIALIZE="false",
        CUDA_VISIBLE_DEVICES="",
        TLLM_LOG_LEVEL=logger.level,
    )
    return env


def _start_authority(world_size: int, timeout_s: float) -> Tuple[subprocess.Popen, str]:
    """Start the membership authority, whose store listens at the rendezvous address on this node; it ends with this
    process, which holds the writing end of its standard input.

    Returns the process and the ``host:port`` its store listens on.

    Raises:
        RuntimeError: The authority did not report that address within ``timeout_s`` (it is then killed).
    """
    env = _helper_process_env()
    read_fd, write_fd = os.pipe()
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tensorrt_llm._torch.distributed.membership",
                f"--world-size={world_size}",
                f"--address-fd={write_fd}",
                f"--rendezvous-timeout-s={RENDEZVOUS_TIMEOUT_S}",
            ],
            env=env,
            pass_fds=(write_fd,),
            stdin=subprocess.PIPE,
        )
    finally:
        os.close(write_fd)
    os.register_at_fork(after_in_child=process.stdin.close)
    with os.fdopen(read_fd) as address_pipe:
        ready, _, _ = select.select([address_pipe], [], [], timeout_s)
        address = address_pipe.readline().strip() if ready else ""
    if not address:
        process.kill()
        raise RuntimeError(
            f"The membership authority did not report where its store listens within {timeout_s} s "
            f"(exit status {process.wait()}); {RENDEZVOUS_ENV}={os.environ.get(RENDEZVOUS_ENV)} must be a free "
            "port, followed by another free one, on rank 0's node"
        )
    return process, address


# Per MoE layer, its expert count and per rank the expert in each of its slots (-1: an empty spare slot).
Placement = Tuple[Tuple[int, Tuple[Tuple[int, ...], ...]], ...]


def ready_payload(record: MembershipRecord, rank: int, view: Placement) -> Optional[str]:
    """``rank``'s READY payload for the PROPOSE ``record``: the whole ``view`` from the lowest survivor, its own slots
    from a joining rank, None otherwise."""
    if rank == record.lowest_survivor():
        return json.dumps(view)
    if record.joining_mask >> rank & 1:
        return json.dumps([ranks[rank] for _, ranks in view])
    return None


def agreed_placement(record: MembershipRecord) -> Optional[list]:
    """The placement every live rank plans the restore from: the lowest survivor's view (survivors' views can differ
    after an interrupted EPLB update), with each joining rank's own slots. None if a payload is missing."""
    payloads = dict(record.ready_payloads)
    source = record.lowest_survivor()
    joining = [r for r in range(record.joining_mask.bit_length()) if record.joining_mask >> r & 1]
    if source not in payloads or any(r not in payloads for r in joining):
        return None
    own_slots = {r: json.loads(payloads[r]) for r in joining}
    return [
        (
            num_experts,
            [own_slots[r][layer] if r in own_slots else slots for r, slots in enumerate(ranks)],
        )
        for layer, (num_experts, ranks) in enumerate(json.loads(payloads[source]))
    ]


class RankFailureRecovery:
    """One rank's part in keeping the deployment serving after a rank failure; see the module docstring. The executor
    calls its hooks; on rank 0, the executor worker sets :attr:`outstanding_requests`."""

    def __init__(
        self,
        executor: PyExecutor,
        exchange,
        moe_state,
        store: SignedStore,
        address: str,
        startup_memory: StartupMemory,
        authority: Optional[subprocess.Popen] = None,
        incarnation: int = 0,
    ):
        self._executor = executor
        self._exchange = exchange
        self._moe = moe_state
        self._store = store
        # ``host:port`` of the membership store.
        self.address = address
        self._startup_memory = startup_memory
        self._authority = authority
        self.rank = executor.dist.rank
        self.world_size = executor.dist.world_size
        self.incarnation = incarnation
        # Rank 0: request id -> client id of every request without a final response.
        self.outstanding_requests: Optional[Callable[[], Dict[int, int]]] = None
        self._agent: Optional[MembershipAgent] = None
        self._record = MembershipRecord(
            generation=0, state=MembershipState.COMMITTED, live_mask=(1 << self.world_size) - 1
        )
        self._record_changed = threading.Condition()
        # Whether this process is admitted (MembershipRecord.admits): a replacement waits for its admission.
        self._admitted = incarnation == 0
        # The live ranks of the last COMMITTED record.
        self._committed_mask = self._record.live_mask
        # The INSTALL record this rank installed last, and the last one it tried to finish installing.
        self._installed: Optional[MembershipRecord] = None
        self._attempted: Optional[MembershipRecord] = None
        # Set at INSTALL: the last iteration any survivor had reached when it stopped; None once the change is
        # committed. Until then, no forward runs.
        self._resume_after: Optional[int] = None
        # Mask of the ranks whose value this rank waits for in the store, during a membership change.
        self._waiting_in_store = 0
        # The steps of installing membership changes this rank completed (RankReport.install_steps).
        self._install_steps = 0
        # Rank 0: the generations whose admission keys it set and the store still holds (_admit_joining_ranks).
        self._published_admissions: List[int] = []
        # The requests of the discarded steps; they continue by recompute once the survivors resume.
        self._interrupted: List[LlmRequest] = []
        # Once the KV cache pool gave up memory for a change not committed yet, the device memory serving keeps free
        # (StartupMemory.free_to_keep, _give_back_kv_cache).
        self._kv_cache_free_to_keep: Optional[int] = None
        # A give-back at a commit that fell short, retried while device memory is freed (_retry_give_back).
        self._give_back_retry: Optional[_GiveBackRetry] = None
        # Whether this rank, and whether some rank, asked to retry its give-back in the last forward vote.
        self._give_back_due = False
        self._give_back_agreed = False
        # Whether the current record is a change this rank has not installed yet, under _record_changed.
        self._pending = False
        # Whether some rank held its steps for a pending change in the last forward vote (forward_vote).
        self._steps_quiet = False
        exchange.interrupted = self._membership_changing
        exchange.on_interrupt = self._on_exchange_interrupt

    @classmethod
    def create(cls, executor: PyExecutor) -> Optional[RankFailureRecovery]:
        """Take over, in fault tolerance mode (None otherwise), the membership this process joined at the entry of
        its worker, once its startup is done. An original rank returns once every rank has started, a replacement once
        the change that adds it is installed.

        Raises:
            ValueError: Main's hang detector could end a rank before the authority excludes a peer it waits for.
        """
        if not fault_tolerance_mode():
            return None
        # A rank whose peer hangs waits in a collective until the authority excludes the peer, up to 1.5 hang
        # timeouts after its wait began: main's hang detector must not end the rank first.
        hang_detection_s = executor.hang_detector.timeout
        check_fault_tolerance_requirements(
            [
                (
                    hang_detection_s >= 2 * PROGRESS_TIMEOUT_S,
                    f"an executor hang detection timeout of at least {2 * PROGRESS_TIMEOUT_S:.0f} s (twice the "
                    f"hang timeout), not {hang_detection_s} s",
                )
            ]
        )
        dist = executor.dist
        replacement = fault_tolerance_replacement()
        startup_record.close()
        if torch.distributed.is_initialized():
            raise RuntimeError(
                "With WideEP fault tolerance no rank may create a torch.distributed process group (NCCL or Gloo) "
                "during its startup: every rank is its own process, and a rank's replacement could never join it"
            )
        balancer = _moe_load_balancer_of(executor)
        startup_memory = StartupMemory(executor.kv_cache_manager.gpu_quota)
        if replacement is None and balancer is not None:
            _check_largest_failure_fits(executor, balancer, startup_memory)
        # The deployment serves from here: a lost rank's replacement reattaches the node's host copies, which the
        # node supervisor unlinks once the deployment ended.
        host_weights_prefix = (
            None
            if balancer is None
            else balancer.expert_restore.keep_host_copies_for_the_deployment()
        )
        store = rendezvous_store()
        if dist.rank == 0:
            if host_weights_prefix is not None:
                store.set(HOST_WEIGHTS_PREFIX_KEY, host_weights_prefix.encode())
            store.set(AGREED_STARTUP_KEY, _agreed_startup_record())
        recovery = cls(
            executor,
            dist.host_exchange,
            executor.model_engine.moe_execution_state,
            store,
            os.environ[RENDEZVOUS_ENV],
            startup_memory,
            _authority,
            incarnation=0 if replacement is None else replacement.incarnation,
        )
        if replacement is None:
            wait_for_every_rank()
        recovery.start()
        if replacement is not None:
            recovery._join()
        logger.info(f"Rank {dist.rank} joined the fault-tolerance membership at {recovery.address}")
        return recovery

    def _join(self) -> None:
        """A replacement's admission: report that it is ready to join, take part in the change that adds it, and start
        the executor loop at the iteration the live ranks resume at."""
        # Its READY to the PROPOSE that adds it, which it may never see (the INSTALL may come first): it ends its own
        # online EPLB work here, as the live ranks do before their READY.
        adding = self._record.model_copy(
            update={"state": MembershipState.PROPOSE, "joining_mask": 1 << self.rank}
        )
        self._end_eplb_iterations()
        self._agent.report(joining_ready=True, **self._expert_placement(adding))
        self._change_membership()
        self._executor.iter_counter = self._resume_after + 1

    def start(self) -> None:
        """Take over the membership agent this process started at the entry of its worker."""
        self._agent = _agent
        self._agent.bind(self._on_record, on_tick=self._on_tick, on_error=self._on_agent_error)

    def stop(self, completed: bool) -> None:
        """When the executor loop ends: after a normal end, leave the membership (wait for SHUTDOWN, so the closing
        connection is no failure); then end the process, as a teardown would wait for ranks that failed."""
        if completed:
            if self.rank == 0:
                self._agent.report(shutdown=True)
            with self._record_changed:
                self._record_changed.wait_for(
                    lambda: self._record.state == MembershipState.SHUTDOWN, _AUTHORITY_TIMEOUT_S
                )
            self._agent.stop()
            if self._authority is not None:
                try:
                    self._authority.wait(_AUTHORITY_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    self._authority.kill()
        _end_process(self.rank, 0 if completed else 1, "the executor stopped")

    # Membership agent thread.

    def _on_record(self, record: MembershipRecord) -> None:
        with self._record_changed:
            self._record = record
            if record.admits(self.rank, self.incarnation):
                self._admitted = True
            if record.state == MembershipState.COMMITTED:
                self._committed_mask = record.live_mask
            self._pending = self._change_pending(record)
            self._record_changed.notify_all()
        if not self._admitted:
            return  # a replacement before its admission: _join and before_iteration follow the records
        if record.state in (MembershipState.PROPOSE, MembershipState.ABORT) or self._excluded(
            record
        ):
            if self._interrupts_steps(record):
                # The all-to-all waits end, and the steps in flight are discarded (drop_invalid_step); their requests
                # continue by recompute.
                self._moe.request_abort()
            if self.rank == 0:  # rank 0 may be waiting for requests rather than in a collective
                self._executor.executor_request_queue.wake()
            loop = getattr(self._executor, "worker_thread", None)
            # No executor loop left to take the exit path.
            if (record.state == MembershipState.ABORT or self._excluded(record)) and (
                loop is not None and not loop.is_alive()
            ):
                _end_process(self.rank, 1, f"the membership record is {record!r}")

    def _on_agent_error(self, error: Exception) -> None:
        """The agent lost its connection: serving continues unprotected (no lease). If only this rank lost it, the
        authority takes it for failed, and main's hang detector ends it once it waits for peers that moved on. A wait
        for the next record ends."""
        with self._record_changed:
            self._record_changed.notify_all()
        logger.warning(
            f"WideEP fault tolerance: the membership agent stopped ({error!r}); serving continues unprotected"
        )

    def _on_tick(self) -> None:
        waited_rounds = self._moe.waited_rounds()
        host_waits, epoch = self._exchange.waiting_for, self._exchange.epoch
        self._agent.report(
            progress=self._executor.iter_counter,
            # The executor pauses its hang detector while it waits for requests.
            idle=self._executor.hang_detector.paused(),
            waiting_for=host_waits
            | self._waiting_in_store
            | sum(1 << peer for peer in waited_rounds),
            waited_rounds=tuple(sorted(waited_rounds.items())),
            waited_epochs=tuple(
                (peer, epoch) for peer in range(self.world_size) if host_waits >> peer & 1
            ),
        )

    # Executor thread, inside the host exchange.

    def _excluded(self, record: MembershipRecord) -> bool:
        """Whether ``record`` no longer lists this admitted process as a live member."""
        return self._admitted and not record.is_member(self.rank, self.incarnation)

    def _interrupts_steps(self, record: MembershipRecord) -> bool:
        """Whether ``record`` interrupts the steps in flight: it ends the deployment, leaves this process out, or drops
        a process of the committed membership. A change that only adds replacements stops the live ranks at a step
        boundary instead."""
        survivors = record.live_mask & ~record.joining_mask
        return (
            record.state == MembershipState.ABORT
            or self._excluded(record)
            or bool(self._committed_mask & ~survivors)
        )

    def _change_pending(self, record: MembershipRecord) -> bool:
        """Whether ``record`` is a membership change this rank has not installed yet."""
        return (
            record.state in (MembershipState.PROPOSE, MembershipState.ABORT)
            or (record.state == MembershipState.INSTALL and record != self._installed)
            or self._excluded(record)
        )

    def _membership_changing(self) -> bool:
        """Whether the host exchange must stop for a change this rank has not installed yet. One that does not
        interrupt the steps in flight stops it only once a forward vote had a rank hold for it, or while the steps are
        held: that vote launched no forward, so every live rank stops in the same round, after the same forwards."""
        if not self._pending:
            return False
        return (
            self._steps_quiet
            or self._resume_after is not None
            or self._interrupts_steps(self._record)
        )

    @contextlib.contextmanager
    def _hang_detection_paused(self):
        """Pause main's hang detector during a membership change, which the authority bounds instead; a detector
        paused already stays paused."""
        detector = self._executor.hang_detector
        armed = not detector.paused()
        detector.disarm()
        try:
            yield
        finally:
            if armed:
                detector.checkpoint()

    def _on_exchange_interrupt(self) -> None:
        """The host exchange stopped for a membership change: take part in it. A joining rank inside its admission's
        collective leaves it once it installed a newer change; :meth:`before_iteration` admits it from there.

        Raises:
            _MembershipRestarted: This rank joins in the change it installed.
        """
        self._change_membership()
        if self._resume_after is not None and self._installed.joining_mask >> self.rank & 1:
            raise _MembershipRestarted

    def _change_membership(self) -> None:
        """Take part in the membership change, main's hang detector paused; returns once the host exchange is
        installed for the live ranks (the rest finishes in :meth:`before_iteration`)."""
        with self._hang_detection_paused():
            seen = self._installed  # a change this rank installed already is none to take part in
            listed = seen is not None  # by a record of this change or an earlier one
            while True:
                record = self._next_record(seen)
                seen = record
                if record.state == MembershipState.ABORT:
                    _end_process(
                        self.rank, 1, "the deployment cannot recover from the rank failure"
                    )
                if record.state == MembershipState.SHUTDOWN:
                    _end_process(self.rank, 0, "the deployment is shutting down")
                if not record.is_member(self.rank, self.incarnation):
                    if self._admitted or listed:
                        # A replacement the change left out: its supervisor starts another one.
                        _end_process(self.rank, 1, "the membership no longer includes this rank")
                    continue  # a replacement, until the authority adds it
                listed = True
                if record.state == MembershipState.PROPOSE:
                    self._end_eplb_iterations()
                    if self._record != record:
                        continue  # a newer change replaced it while the updates ended
                    stopped_at = ResumeState(
                        iteration=self._executor.iter_counter, epoch=self._exchange.stopped_epoch
                    )
                    self._agent.report(
                        reply=Reply.READY,
                        reply_generation=record.generation,
                        stopped_at=stopped_at,
                        **self._expert_placement(record),
                    )
                elif record.state == MembershipState.INSTALL:
                    # The INSTALL record carries the READY payloads; a joining rank's readiness stood for its READY
                    # to this change only.
                    self._agent.report(ready_payload=None, joining_ready=False)
                    try:
                        self._install(record)
                    except (_MembershipRestarted, MnnvlPeerHandleError) as error:
                        # A peer exited during the install: the authority finds it failed and proposes again.
                        logger.warning(
                            f"Rank {self.rank} waits for the next membership record: {error!r}"
                        )
                        continue
                    except Exception as error:  # noqa: BLE001 - a failure here must reach the authority
                        self._fail_install(record, error)
                        continue
                    return
                else:
                    return  # no change to take part in

    def _next_record(self, seen: Optional[MembershipRecord]) -> MembershipRecord:
        """The current record once it differs from ``seen``; exits if the authority's connection closes meanwhile."""
        with self._record_changed:
            self._record_changed.wait_for(lambda: self._record != seen or not self._agent.connected)
            if self._record == seen:
                _end_process(
                    self.rank,
                    1,
                    "the membership authority closed its connection during a membership change",
                )
            return self._record

    def _end_eplb_iterations(self) -> None:
        """End the online EPLB work of the iterations so far and restart its statistics and update plan. Before READY,
        so the placement READY carries no longer changes, and again at the resume point, where every live rank stands
        at the same step."""
        balancer = _moe_load_balancer_of(self._executor)
        if balancer is not None:
            balancer.abandon_iterations()

    def _expert_placement(self, record: MembershipRecord) -> dict:
        """This rank's READY payload for the PROPOSE ``record`` (:func:`ready_payload`); empty without EPLB."""
        balancer = _moe_load_balancer_of(self._executor)
        if balancer is None:
            return {}
        view = tuple(
            (num_experts, tuple(tuple(slots) for slots in ranks))
            for num_experts, ranks in balancer.expert_restore.current_placement()
        )
        return {"ready_payload": ready_payload(record, self.rank, view)}

    def _install(self, record: MembershipRecord) -> None:
        """Move the host exchange to fresh rows of every live rank; the joining ranks enter it only at the resume
        point."""
        members = record.live_mask & ~record.joining_mask
        gather = self._gather_through_store(record, "host_exchange_rows")
        if members >> self.rank & 1:
            self._exchange.install(gather, members, record.resume.epoch)
        else:
            self._exchange.rebuild(gather, members, record.resume.epoch)
        with self._record_changed:
            self._installed = record
            self._pending = self._change_pending(self._record)
        self._resume_after = record.resume.iteration
        self._install_step_done()

    def _install_step_done(self) -> None:
        """Report one more install step, which the authority counts as progress."""
        self._install_steps += 1
        self._agent.report(install_steps=self._install_steps)

    def _gather_through_store(
        self, record: MembershipRecord, name: str
    ) -> Callable[[Any], List[Any]]:
        """An allgather over the record's live ranks through the store (:func:`store_gather`), which reports the ranks
        it waits for and raises :class:`_MembershipRestarted` once the record is no longer ``record``."""

        def on_wait(waiting: int) -> None:
            self._waiting_in_store = waiting
            if waiting and self._record != record:
                raise _MembershipRestarted

        return store_gather(
            self._store,
            f"{name}/{record.generation}",
            self.rank,
            [r for r in range(self.world_size) if record.is_live(r)],
            self.world_size,
            on_wait,
        )

    def _fail_install(self, record: MembershipRecord, error: Exception) -> None:
        """Installing ``record`` failed: reply FAILED, unless only other live ranks failed to map MNNVL memory (they
        reply FAILED, and this rank waits for the next record)."""
        if isinstance(error, MnnvlMappingError) and self.rank not in error.ranks:
            logger.warning(f"Rank {self.rank} waits for the next membership record: {error!r}")
            return
        logger.error(
            f"Rank {self.rank} could not install membership generation {record.generation}: {error!r}"
        )
        self._agent.report(reply=Reply.FAILED, reply_generation=record.generation)

    # Executor loop hooks.

    def before_iteration(self) -> None:
        """Finish installing the membership change once, at the first iteration past the resume point, where no forward
        or EPLB iteration is active. A joining rank follows a newer change that still adds it, then finishes its
        admission from that change's install. First, it retries a give-back if the last forward vote asked for it
        (:meth:`_retry_give_back`)."""
        if self._give_back_agreed:
            self._retry_give_back()
        record = self._installed
        if (
            self._resume_after is None
            or record is self._attempted
            or self._executor.iter_counter <= self._resume_after
        ):
            return
        with self._hang_detection_paused():
            while True:
                self._attempted = record
                if self._finish_install(record) or not record.joining_mask >> self.rank & 1:
                    return
                if self._installed is record:  # not installed inside the admission's collective
                    self._change_membership()  # exits if the newer change leaves this rank out
                self._executor.iter_counter = self._resume_after + 1
                record = self._installed

    def _finish_install(self, record: MembershipRecord) -> bool:
        """Recompute the interrupted requests, admit the joining ranks, restore, reply INSTALLED and wait for the
        commit; then rank 0 fails the requests no live rank holds. Returns whether the change was committed."""
        executor = self._executor
        # Milliseconds of the phases before the restore, logged with it.
        times: Dict[str, float] = {}
        start_s = time.perf_counter()
        torch.cuda.synchronize()  # no step is in flight
        try:
            # Every live rank ran the same EPLB iterations since its last restart; it may have run one more since
            # its end before READY (the step it stopped in, finished after INSTALL), which other ranks ran before it.
            self._end_eplb_iterations()
            if not record.joining_mask >> self.rank & 1:
                # As main does before an action on the KV cache, which the restore resizes; collective over the
                # host exchange, which the joining ranks enter only below.
                executor._retire_inflight_batch_for_control()
            recomputed = self._recompute_interrupted_requests()
            if record.joining_mask:
                # A rank that installed a newer change inside those collectives leaves the joining ranks out; the
                # others install it at their next round. Every live rank runs the gather below either way, so all
                # of them stay in the same collective.
                if self._installed is record:
                    self._admit_joining_ranks(record)
                self._install_step_done()
            # Taken before the requests held, so that a request enqueued in between is not taken for a lost one.
            outstanding = (
                self.outstanding_requests() if self.outstanding_requests is not None else {}
            )
            times["prepare"] = (time.perf_counter() - start_s) * 1e3
            start_s = time.perf_counter()
            # Every live rank reaches this gather at the same point of the executor loop; past it, no live rank has
            # an all-to-all kernel in flight, and none runs another host collective before the next iteration.
            gathered = executor.dist.allgather(self._held_request_ids())
            times["request gather"] = (time.perf_counter() - start_s) * 1e3
            self._install_step_done()
            digest = self._restore(record, times)
        except (_MembershipRestarted, MnnvlPeerHandleError) as error:
            # A newer change replaced this one, or a peer exited during the remap: the authority finds it failed and
            # proposes again.
            logger.warning(
                f"Rank {self.rank} stopped installing membership generation {record.generation}: {error!r}"
            )
            return False
        except Exception as error:  # noqa: BLE001 - a failure here must reach the authority
            self._fail_install(record, error)
            # The authority answers with a newer change or ABORT; exits if it stops answering.
            self._next_record(record)
            return False
        self._agent.report(
            reply=Reply.INSTALLED,
            reply_generation=record.generation,
            placement_digest=digest,
        )
        committed = self._next_record(record)
        if committed.state != MembershipState.COMMITTED:
            return False  # a newer change, or ABORT at the next exchange
        # Every live rank, a joining one included, passes here before its next round: the next change that does not
        # interrupt the steps in flight waits for a forward vote in which a rank holds for it.
        self._resume_after = None
        self._steps_quiet = False
        # The joining ranks read them before they replied INSTALLED, and a replaced change's have no reader left: the
        # store keeps no key of a past change.
        for generation in self._published_admissions:
            for key in _admission_keys(generation):
                self._store.store.delete_key(key)
        self._published_admissions.clear()
        given_back = self._give_back_kv_cache()
        # The live ranks, a joining one included, restart the attention-DP balance counters alike.
        executor.adp_ctx_waiting_iters_count = executor.adp_ctx_batching_wait_iters_count = 0
        held = set().union(*(request_ids for request_ids in gathered if request_ids is not None))
        # Rank 0's outstanding requests that no live rank holds (other ranks have none outstanding).
        lost = {r: c for r, c in outstanding.items() if r not in held}
        for request_id, client_id in lost.items():
            self._fail(request_id, client_id, RankFailureError())
        live = [r for r in range(self.world_size) if record.is_live(r)]
        logger.info(
            f"Rank {self.rank} resumed serving with membership generation {record.generation}, ranks {live}: "
            f"{recomputed} interrupted requests continue by recompute"
            + (
                f", {len(lost)} requests no live rank holds ended by the rank failure"
                if self.rank == 0
                else ""
            )
            + given_back
        )
        return True

    def _give_back_kv_cache(self) -> str:
        """Once a change is committed: release the spare slots the placement leaves empty, and let the KV cache pool
        take back all but what fault tolerance still holds, as far as the memory freed since the change started allows.

        Returns:
            What it gave back, for the log ("" if nothing).
        """
        start_s = time.perf_counter()
        balancer = _moe_load_balancer_of(self._executor)
        released = 0 if balancer is None else balancer.expert_restore.release_empty_spare_slots()
        kept, self._kv_cache_free_to_keep = self._kv_cache_free_to_keep, None
        if not (released or kept is not None):
            return ""
        free_to_keep = kept or 0
        held = 0 if balancer is None else balancer.expert_restore.backed_spare_slot_bytes
        with self._executor._kv_pools_suspended() as kv_cache_manager:
            quota = self._startup_memory.fit_kv_cache(kv_cache_manager, held, 0, free_to_keep)
            short = self._startup_memory.shortfall(kv_cache_manager, held, free_to_keep)
        self._give_back_retry = None
        if short:
            record = self._installed
            self._give_back_retry = _GiveBackRetry(
                free_to_keep,
                {
                    r: record.incarnations[r] if r < len(record.incarnations) else 0
                    for r in range(self.world_size)
                    if not record.is_live(r)
                },
                torch.cuda.mem_get_info()[0],
                time.monotonic() + _GIVE_BACK_RETRY_S,
            )
        return (
            f"; {released} bytes of empty spare slots released, KV cache quota {quota} bytes"
            + (f" ({short} bytes short until device memory is freed)" if short else "")
            + f", in {(time.perf_counter() - start_s) * 1e3:.0f} ms"
        )

    def _give_back_retry_due(self) -> bool:
        """Whether to retry the give-back that fell short (:meth:`_retry_give_back`): checked at most once a second,
        when device memory was freed since the last give-back. Once every excluded process ended (its replacement's
        incarnation is claimed), that retry is the last; without memory freed, the retries end at once."""
        retry = self._give_back_retry
        now_s = time.monotonic()
        if now_s < retry.check_after_s:
            return False
        retry.check_after_s = now_s + _GIVE_BACK_RETRY_S
        if not retry.ended:
            retry.ended = all(
                claimed_incarnation(self._store, rank) > incarnation
                for rank, incarnation in retry.excluded.items()
            )
        freed = torch.cuda.mem_get_info()[0] > retry.free_seen
        if retry.ended and not freed:
            self._give_back_retry = None
        return freed

    def _retry_give_back(self) -> None:
        """At the iteration after a forward vote in which a rank asked to retry its give-back: every rank retires its
        step in flight together (doing so runs collectives), then the ranks that asked let the KV cache pool take back
        the device memory freed since (:meth:`StartupMemory.fit_kv_cache`). The retries end once the pool holds its
        quota, or after the first one once every excluded process ended."""
        self._give_back_agreed = False
        torch.cuda.synchronize()
        self._executor._retire_inflight_batch_for_control()
        if not self._give_back_due:
            return
        self._give_back_due = False
        retry = self._give_back_retry
        balancer = _moe_load_balancer_of(self._executor)
        held = 0 if balancer is None else balancer.expert_restore.backed_spare_slot_bytes
        with self._executor._kv_pools_suspended() as kv_cache_manager:
            quota = self._startup_memory.fit_kv_cache(kv_cache_manager, held, 0, retry.free_to_keep)
            short = self._startup_memory.shortfall(kv_cache_manager, held, retry.free_to_keep)
        retry.free_seen = torch.cuda.mem_get_info()[0]
        if not short or retry.ended:
            self._give_back_retry = None
        logger.info(
            f"Rank {self.rank} took back device memory freed since the commit: KV cache quota {quota} bytes"
            + (f", {short} bytes short" if short else "")
        )

    def _restore(self, record: MembershipRecord, times: Optional[Dict[str, float]] = None) -> str:
        """This rank's part of installing ``record``, collective over its live ranks: restore the experts from the
        agreed placement, remap the all-to-all workspaces (or only map the joining ranks' rows), with the KV cache pool
        giving up the memory they need, then reset the all-to-all and install the live mask.

        Args:
            record: The INSTALL record.
            times: The milliseconds of the install's phases so far; the restore adds its own and logs them all.

        Returns:
            The digest of the restored EPLB placement ("" without EPLB).

        Raises:
            _MembershipRestarted: The record changed since INSTALL (checked first, in every wait for a peer, and
                last).
        """
        if self._record != record:
            raise _MembershipRestarted
        began_s = time.perf_counter()
        times = {} if times is None else times
        executor = self._executor
        balancer = _moe_load_balancer_of(executor)
        plan = None
        loaded = 0  # experts this rank loads into its slots
        if balancer is not None:
            current = [np.array(ranks, dtype=np.int32) for _, ranks in agreed_placement(record)]
            plan = balancer.expert_restore.plan_expert_restore(
                [record.is_live(r) for r in range(self.world_size)],
                current,
                rejoining=[bool(record.joining_mask >> r & 1) for r in range(self.world_size)],
            )
            for planned, held in zip(plan.placements, current):
                mine, before = planned[balancer.ep_rank], held[balancer.ep_rank]
                loaded += int(np.count_nonzero((mine != before) & (mine >= 0)))
            self._install_step_done()
        for r in range(self.world_size):
            if record.is_live(r):
                self._moe.health.mark_active(r)
            else:
                self._moe.health.mark_failed(r)
        gather = self._gather_through_store(record, "moe_a2a_workspaces")
        fresh_pages = self._moe.workspaces_need_fresh_pages(gather)
        # The remap allocates the workspaces' fresh pages before the pages they replace are freed; an admission
        # that keeps them allocates nothing.
        workspace_bytes = self._moe.workspace_bytes() if fresh_pages else 0
        needed = workspace_bytes + (0 if plan is None else plan.device_bytes)

        def remap() -> None:
            start_s = time.perf_counter()
            if fresh_pages:
                self._moe.remap_workspaces(gather)
            else:
                self._moe.admit_to_workspaces(gather)
            times["remap" if fresh_pages else "admission"] = (time.perf_counter() - start_s) * 1e3
            self._install_step_done()

        kv_cache_quota = "unchanged"
        # A rank that needs no memory, as at an admission that keeps the workspaces' pages and backs no spare slot,
        # leaves its KV cache pool alone. Each rank decides on its own: suspending the pools runs a collective only
        # with disaggregated serving, which fault tolerance rejects.
        with (
            executor._kv_pools_suspended() if needed else contextlib.nullcontext()
        ) as kv_cache_manager:
            if needed:
                start_s = time.perf_counter()
                before, free = kv_cache_manager.gpu_quota, torch.cuda.mem_get_info()[0]
                backed = 0 if balancer is None else balancer.expert_restore.backed_spare_slot_bytes
                self._kv_cache_free_to_keep = self._startup_memory.free_to_keep(
                    kv_cache_manager, backed
                )
                fitted = self._startup_memory.fit_kv_cache(kv_cache_manager, backed, needed)
                times["KV cache shrink"] = (time.perf_counter() - start_s) * 1e3
                kv_cache_quota = (
                    f"{before} -> {fitted} bytes ({needed} bytes needed, {free} bytes free)"
                )
                self._install_step_done()
            if plan is not None:
                balancer.expert_restore.restore_lost_experts(plan, times)
                self._install_step_done()
            remap()
        # Under the lock the agent thread updates the record with, before it requests the abort of a newer change that
        # interrupts the steps: once this check passes, that abort comes after the rearm and stays.
        start_s = time.perf_counter()
        with self._record_changed:
            if self._record != record:
                raise _MembershipRestarted
            self._moe.rearm()
            self._moe.install_mask()
        times["rearm"] = (time.perf_counter() - start_s) * 1e3
        logger.info(
            f"Rank {self.rank} restored membership generation {record.generation} in "
            f"{(time.perf_counter() - began_s) * 1e3:.0f} ms: {loaded} experts loaded "
            f"({0 if plan is None else plan.device_bytes} bytes), KV cache quota {kv_cache_quota}; phases: "
            + ", ".join(f"{phase} {ms:.0f} ms" for phase, ms in times.items())
        )
        return "" if plan is None else plan.digest

    def _admit_joining_ranks(self, record: MembershipRecord) -> None:
        """Bring the joining ranks into the host exchange at the iteration the live ranks resume at; rank 0 hands them
        the next epoch and its waiting queue, which every live rank holds alike, through the store."""
        epoch_key, queue_key = _admission_keys(record.generation)
        executor = self._executor
        if record.joining_mask >> self.rank & 1:

            def check(_unsigned: Set[str]) -> None:
                self._waiting_in_store = 1  # rank 0 sets both keys
                if self._record != record:
                    raise _MembershipRestarted

            try:
                epoch, waiting = wait_for_keys(
                    self._store, [epoch_key, queue_key], STORE_POLL_S, check
                )
            finally:
                self._waiting_in_store = 0
            self._exchange.admit(record.live_mask, int(epoch))
            # Rank 0's queue replaces the one an earlier attempt of this admission took.
            executor.waiting_queue.remove_by_ids({item.id for item in executor.waiting_queue})
            executor.waiting_queue.add_requests(pickle.loads(waiting))  # nosec B301
            return
        epoch = next_epoch(self._exchange.epoch)
        if self.rank == 0:
            self._store.set(queue_key, pickle.dumps(list(executor.waiting_queue)))
            self._store.set(epoch_key, str(epoch).encode())
            self._published_admissions.append(record.generation)
        self._exchange.admit(record.live_mask, epoch)

    def _held_request_ids(self) -> Set[int]:
        """The requests this rank holds: active, waiting, or (rank 0) not fetched yet."""
        executor = self._executor
        held = {
            request.py_request_id for request in executor.active_requests if not request.is_dummy
        }
        held.update(item.id for item in executor.waiting_queue)
        held.update(item.id for item in executor.request_accumulated)
        request_queue = executor.executor_request_queue.request_queue
        with request_queue.mutex:
            # A cancel item carries the id of the request it cancels: only requests count.
            held.update(item.id for item in request_queue.queue if item.is_normal_request)
        return held

    def holds_steps(self) -> bool:
        """Whether this iteration runs no forward: none runs from a change this rank has seen until it is committed, so
        the live ranks resume at the same iteration."""
        return self._resume_after is not None or self._pending

    def forward_vote(
        self, batch_size: int, gather: Callable[[List[int]], np.ndarray]
    ) -> np.ndarray:
        """The executor's attention-DP forward vote: ``gather`` collects every rank's batch size (0 while it holds its
        steps), whether it holds for a pending change and whether it retries a give-back that fell short
        (:meth:`_give_back_retry_due`); returns the batch sizes. Unless a rank held for a pending change, every rank
        takes part in a retry at the next iteration (:meth:`_retry_give_back`)."""
        pending = self._pending
        held = pending or self._resume_after is not None
        self._give_back_due = (
            not held and self._give_back_retry is not None and self._give_back_retry_due()
        )
        votes = gather([0 if held else batch_size, int(pending), int(self._give_back_due)])
        self._steps_quiet = bool(votes[:, 1].any())
        self._give_back_agreed = bool(votes[:, 2].any()) and not self._steps_quiet
        if held and self.rank == 0:
            self._executor.executor_request_queue.wake()
        return votes[:, 0]

    def drop_invalid_step(self) -> None:
        """Discard the executor's previous step before its results are applied if the all-to-all is aborted (an abort
        may have cut its rounds short); its requests continue by recompute."""
        batch = self._executor.previous_batch
        event = batch.sample_state.sampler_event if batch.sample_state is not None else None
        if event is not None:
            event.synchronize()  # the wait that applying the step starts with
        if not self._moe.aborted():
            return
        self._interrupted.extend(batch.scheduled_requests.all_requests())
        for request, mark in batch.context_rows:
            request.py_result.rewind_context_rows(mark)
        executor = self._executor
        if batch.gpu_forward_events_from_perf_pool:
            executor.perf_manager.release_forward_timing_events(
                batch.gpu_forward_start_event, batch.gpu_forward_end_event
            )
        if executor.drafter is not None and executor.use_spec_decode:
            executor.drafter.cleanup_previous_draft_resources()
        executor.previous_batch = None

    def discard_failed_forward(self, scheduled_requests) -> bool:
        """Discard a forward that raised while the all-to-all is aborted, and the previous step; their requests
        continue by recompute. Returns False if the forward failed by itself."""
        if not self._moe.aborted():
            return False
        self._interrupted.extend(scheduled_requests.all_requests())
        self._executor._accepted_previous_batch()
        return True

    def _recompute_interrupted_requests(self) -> int:
        """The requests of the discarded steps continue from the tokens they have, as after a preemption; returns how
        many."""
        interrupted, self._interrupted = {id(request) for request in self._interrupted}, []
        executor = self._executor
        recomputed = 0
        for request in list(executor.active_requests):
            # A request the scheduler paused for recompute meanwhile holds no sequence slot, nor anything else.
            if (
                id(request) in interrupted
                and not request.is_dummy
                and not request.is_finished
                and request.py_seq_slot is not None
            ):
                executor._free_request_resources(request)
                executor._pause_recompute_request(request)
                recomputed += 1
        return recomputed

    def fail_unservable_requests(
        self, waiting_queue: WaitingQueue, rank_states: List[RankState]
    ) -> None:
        """Before admission, remove the waiting requests pinned strictly to an attention-DP rank that does not serve;
        rank 0 fails them."""
        if len(rank_states) == self._executor.dist.tp_size:
            return  # every rank serves
        serving = {state.rank for state in rank_states}
        unservable = {}
        for item in waiting_queue:
            params = getattr(item.request, "py_scheduling_params", None)
            if (
                params is not None
                and params.attention_dp_rank is not None
                and not params.attention_dp_relax
                and params.attention_dp_rank not in serving
            ):
                unservable[item.id] = item
        if not unservable:
            return
        waiting_queue.remove_by_ids(set(unservable))
        for item in unservable.values():
            self._fail(
                item.id,
                getattr(item.request, "client_id", None),
                f"Attention DP rank {item.request.py_scheduling_params.attention_dp_rank} does not serve; send the "
                "request without attention_dp_rank or with attention_dp_relax",
            )

    def _fail(
        self, request_id: int, client_id: Optional[int], error: Union[str, RequestError]
    ) -> None:
        """Rank 0 fails a request with ``error``, sent with the executor's next responses."""
        from .llm_request import LlmResponse

        if self.rank == 0:
            self._executor._pending_transfer_responses.append(
                (
                    request_id,
                    LlmResponse(
                        request_id=request_id,
                        error_msg=error,
                        client_id=client_id,
                    ),
                )
            )


class FaultToleranceHealth:
    """The membership state of a deployment with WideEP fault tolerance, read from the membership store for the
    server's ``/health`` and status endpoint: ``healthy``, ``degraded`` (some ranks lost), ``recovering`` (a change
    is in progress), ``unprotected`` (the authority does not answer; final) or ``stopped``."""

    def __init__(self, address: str, secret: bytes):
        self._address = address
        self._secret = secret
        self._store: Optional[SignedStore] = None
        self._record: Optional[MembershipRecord] = None  # the last read
        self._next_open = 0.0  # time.monotonic() before which the store is not connected to again
        self._authority_ended = False
        self._lock = threading.Lock()

    def _read(self) -> Tuple[Optional[MembershipRecord], Optional[TransactionSummary], bool]:
        """The record, the last transaction and whether the authority answers. Until the store first answers, a read
        connects at most once per ``_STATUS_READ_TIMEOUT_S``; once it answered, a failed read means the authority
        ended, and later reads answer at once."""
        with self._lock:
            if self._authority_ended:
                return self._record, None, False
            try:
                if self._store is None:
                    if time.monotonic() < self._next_open:
                        return self._record, None, False
                    self._next_open = time.monotonic() + _STATUS_READ_TIMEOUT_S
                    self._store = open_store(self._address, self._secret, _STATUS_READ_TIMEOUT_S)
                record, transaction = (
                    self._store.get_if_set(key) for key in (RECORD_KEY, TRANSACTION_KEY)
                )
            except (RuntimeError, OSError):
                self._authority_ended = self._store is not None
                self._store = None
                return self._record, None, False
            record = self._parse(MembershipRecord, record)
            if record is not None:
                self._record = record
            return self._record, self._parse(TransactionSummary, transaction), True

    @staticmethod
    def _parse(model: type, value: Optional[bytes]):
        """``value`` read from the store as ``model``; None if it is absent or does not parse (logged)."""
        if value is None:
            return None
        try:
            return model.model_validate_json(value)
        except ValueError as error:
            logger.error(
                f"WideEP fault tolerance: cannot parse a value of the membership store: {error!r}"
            )
            return None

    @staticmethod
    def _state(record: Optional[MembershipRecord], authority_alive: bool) -> str:
        if record is None:
            return "unprotected"
        # Final: once the deployment ends, the authority keeps its store up until every live rank has left.
        if record.state in (MembershipState.ABORT, MembershipState.SHUTDOWN):
            return "stopped"
        if not authority_alive:
            return "unprotected"
        if record.state != MembershipState.COMMITTED:
            return "recovering"
        world = (1 << len(record.incarnations)) - 1
        return "healthy" if record.live_mask == world else "degraded"

    def status(self) -> Dict[str, object]:
        """The server's fault-tolerance status: the state, whether the authority answers, and from the last record its
        generation and state, the live and lost ranks, each rank's incarnation and the last transaction that ended
        (None until the store answers)."""
        record, transaction, authority_alive = self._read()
        status = {
            "state": self._state(record, authority_alive),
            "authority_alive": authority_alive,
            "generation": None,
            "membership_state": None,
            "live_ranks": None,
            "lost_ranks": None,
            "incarnations": None,
            "last_transaction": None
            if transaction is None
            else transaction.model_dump(mode="json"),
        }
        if record is not None:
            world_size = len(record.incarnations)
            status.update(
                generation=record.generation,
                membership_state=record.state.name,
                live_ranks=[r for r in range(world_size) if record.is_live(r)],
                lost_ranks=[r for r in range(world_size) if not record.is_live(r)],
                incarnations=list(record.incarnations),
            )
        return status

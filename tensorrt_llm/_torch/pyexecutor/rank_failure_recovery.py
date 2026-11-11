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
"""Keep a WideEP deployment serving when one of its ranks fails.

Every rank is its own process, a one-process MPI world whose rank its launcher gives it. Rank 0's ``worker_main``
first starts the membership authority as a separate process, publishes its arguments in the rendezvous store and
joins the membership (:func:`start_worker`); every other rank's process joins the membership, takes those arguments
from the store and runs ``worker_main`` with them (:func:`worker_node_main`). The startup collectives go through the
store. Every rank runs a :class:`~tensorrt_llm._torch.distributed.membership.MembershipAgent` from the entry of its
worker; the deployment aborts if a rank is lost before every rank has started (:func:`wait_for_every_rank`). When the
authority proposes a membership without a failed rank:

1. The agent thread requests an abort of the MoE all-to-all, so kernels waiting for the failed rank return, and the
   host exchange under the executor's collectives stops at its next wait or before its next round.
2. There, on the executor thread, :meth:`RankFailureRecovery._change_membership` replies READY with where the rank
   stopped and its part of the expert placement (once the online EPLB work of the interrupted iterations has ended),
   and returns once it has moved the host exchange to the survivors at INSTALL.
   The interrupted collective then completes with the same result on every survivor, so the executor loop goes on
   without unwinding, and the survivors keep the same waiting queue and request routing.
3. A step whose results are applied while the all-to-all is aborted has invalid MoE outputs and is discarded
   (:meth:`RankFailureRecovery.drop_invalid_step`). Until the change is committed, no rank runs a forward
   (:meth:`RankFailureRecovery.holds_steps`). Past the last iteration any survivor had reached when it stopped, no
   forward or EPLB iteration is active: there (:meth:`RankFailureRecovery.before_iteration`), every live rank
   restarts its online EPLB work at the same step, the requests of the discarded steps continue by recompute,
   through the executor's preemption path, and each survivor restores the lost experts into spare slots, with KV
   cache memory, and remaps and resets the all-to-all for the survivors, then replies INSTALLED. The authority
   commits once every survivor has; a survivor that fails replies FAILED, and the authority aborts.
4. Rank 0 fails the requests that no survivor holds with :class:`~tensorrt_llm.executor.utils.RankFailureError`; its
   executor worker enqueues such a request again, once, if it has sent none of its output yet, and the client sees
   the error otherwise.

ABORT ends every process, and the frontend then fails the pending requests; a process excluded from the membership
exits. Without the authority, the ranks serve on unprotected; once rank 0 is gone too, node 0's supervisor ends the
job (under ``mpirun``), or main's hang detector ends them.

A replacement rank rejoins (:meth:`RankFailureRecovery.create`): the node supervisor of its node
(``_llmapi_node_supervisor.py``) starts it on the failed rank's GPU like any rank other than 0
(:func:`worker_node_main`); after a node was lost, the node's supervisor started again does. It asks the authority to
join with a newer incarnation at the entry of its worker, then starts alone (isolated startup): its collectives reach
only itself, and it takes the results of its startup collectives from the startup record rank 0 published
(:class:`~tensorrt_llm._torch.distributed.startup_record.StartupRecord`). It reports that it is ready to join (only
then do the live ranks stop for it: no process leaves, so nothing aborts the all-to-all; the live ranks hold their
steps from their next forward vote, and the host exchange stops them in the round after a vote in which one held, so
every step in flight runs on every live rank), and takes part in the membership change that adds it like the live
ranks, which keep the host exchange to themselves until they resume. At the resume point every live rank, the joining
one included, restarts its online EPLB work, brings the joining ranks into the host exchange, then restores the
membership as after a failure: the experts go back to the joining rank, the spare slots that held them return their
memory to the KV cache pool, and the all-to-all workspaces are remapped among every live rank. A newer change before
the commit (another rank failed) that still adds the replacement it follows as it joined; one that leaves it out (it
failed or did not reply in time) ends it, and its supervisor starts another one.
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
    REPORT_INTERVAL_S,
    TRANSACTION_KEY,
    MembershipAgent,
    MembershipRecord,
    MembershipState,
    ReplacementCannotStart,
    Reply,
    ResumeState,
    SignedStore,
    TransactionSummary,
    open_store,
    rendezvous_store,
    startup_gather,
    store_gather,
    wait_for_keys,
)
from ..distributed.mnnvl_memory import (
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

# How long one wait for peers' values in the store during a membership change lasts before the record is checked
# again.
_RECORD_CHECK_S = 0.1
# How long a read of the server's fault-tolerance status waits for the membership store to answer.
_STATUS_READ_TIMEOUT_S = 1.0
# The least length of the join secret, which keys the signatures of the membership store's values.
_JOIN_SECRET_BYTES = 32


class _MembershipRestarted(Exception):
    """A newer membership change replaced the one being installed."""


# The sockets that reserve the authority's ports ensure_rendezvous chose.
_rendezvous_ports: List[socket.socket] = []
# From the entry of this process's worker (start_worker, worker_node_main): its membership agent, and on rank 0 the
# authority process it started.
_agent: Optional[MembershipAgent] = None
_authority: Optional[subprocess.Popen] = None


class MembershipReady(NamedTuple):
    """The READY that rank 0's executor worker sends the proxy with fault tolerance: the fields of the READY without
    it, and the address of the membership store."""

    signal: bytes
    error: Optional[str]
    worker_process_identities: list
    membership_address: str


def ensure_rendezvous() -> None:
    """Choose the join secret and the rendezvous address (this host's IP address and two free consecutive ports,
    :func:`~tensorrt_llm._torch.distributed.membership.authority_ports`, where rank 0 runs) when unset and export
    them, so that the executor workers started from this process find them. Where the server does not start the
    workers (``trtllm-llmapi-launch``), set both before launching. An empty secret counts as unset.

    The ports chosen stay reserved while this process lives, since the authority binds them only once rank 0's
    worker has started: by sockets bound with SO_REUSEADDR, as the authority's own are, that never listen. Linux then
    lets the authority bind and listen on the ports, and keeps them from other binds and outgoing connections."""
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
    """What the host collectives over MNNVL memory (``MnnvlDist``) need that this deployment lacks, when
    ``Distributed.get`` chose another implementation."""
    from tensorrt_llm._utils import mpi_disabled

    from ..distributed.communicator import MnnvlDist

    if mpi_disabled():
        return ["the MPI orchestrator (TLLM_DISABLE_MPI unset)"]
    return MnnvlDist.unmet_requirements(mapping) or [
        "host collectives over MNNVL memory (MnnvlDist)"
    ]


def check_supported(llm_args: TorchLlmArgs, mapping: Mapping, dist: Distributed) -> None:
    """Check, before the model loads, what WideEP fault tolerance needs beyond what the LLM arguments' validation
    checks. The built model is checked before any weight loads (:func:`check_supported_model`).

    Raises:
        ValueError: The configuration is not supported with fault tolerance.
    """
    from ..distributed.communicator import MnnvlDist
    from ..moe.fused_moe.communication.nvlink_one_sided import FORCE_CFT_ENV, get_force_cft
    from ..moe.fused_moe.moe_comm_timeout_guard import (
        SERVING_TIMEOUT_ENV,
        resolve_moe_comm_timeout_budgets,
    )

    # The all-to-all kernels that wait on a hung rank are released when the authority excludes it, up to 1.5 hang
    # timeouts after their wait began (the timeout runs from the first fresh report of the wait): they must not time
    # out before.
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
    """The rank this process stands for, once the launch it needs is checked: its rank from its node supervisor, the
    join secret and the rendezvous address. The first call, at the entry of the worker, starts the deadline of this
    process's startup waits (:func:`_startup_time_left`).

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
    _startup_time_left()  # the startup deadline runs from here
    return _rank_identity()


def _rank_identity() -> RankIdentity:
    """This process's rank, from its node supervisor.

    Raises:
        ValueError: The process has no rank identity.
    """
    identity = wide_ep_fault_tolerance_rank()
    if identity is None:
        raise ValueError(
            "WideEP fault tolerance starts every rank as its own process, with '<rank>,<world size>,<incarnation>' "
            f"in {FAULT_TOLERANCE_RANK_ENV}: launch the ranks with trtllm-llmapi-launch, whose node supervisor sets "
            "it, or let the server start them on one node"
        )
    return identity


def check_supported_model(model: torch.nn.Module) -> None:
    """Check the built model, before any weight loads: that it is the built-in implementation, not a modeling_v2
    target; its architecture, from the configuration it keeps, from which the KV cache manager is chosen (a
    multimodal model may keep its language model's in place of the checkpoint's); the communication of its MoE
    layers (``MoeExecutionState.model_comms``); and the room for its host weights (:func:`_check_host_weights_fit`).

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
    """Check that /dev/shm on this node has room for the host weights of every expert of every MoE layer of
    ``model``, one slot of each weight of local slots per expert (``HostMoeTensorSharer``), beyond what the node's
    segments of them that exist already take there (those a replacement rank reattaches).

    Raises:
        ValueError: /dev/shm has too little free space.
    """
    from ..moe.fused_moe.impl_blocks import MoEEplbWeightLayoutMixin
    from ..moe.fused_moe.moe_load_balancer import SHARED_MEMORY_DIR, HostMoeTensorSharer

    # Per MoE layer's host tensor sharer, the bytes of one slot of each of the layer's weights of local slots.
    slot_bytes: Dict[HostMoeTensorSharer, List[int]] = {}
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
    """At the autotuner warmup of the final executor, while the startup record records or replays: rank 0 records the
    tactics it shares with every rank (those of the ops not tuned per rank), which the original ranks tuned in their
    warmup during the KV cache estimation, and a replacement rank, which skips that warmup, adopts them. The
    replacement then finds the tactics the original ranks find in the rest of the warmup."""
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
    """A random name of this deployment, the same on every rank: rank 0 draws it at its first call, a startup
    collective of the original ranks (a replacement takes it from the startup record). It keeps the names of what
    outlives the deployment's processes on a node (the shared-memory segments of the host weights) apart from another
    deployment's, even one started with the same join secret."""
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
    """What a replacement rank must share with the deployment it joins beyond its configuration, which it receives
    from rank 0: the TensorRT-LLM version and the checkpoint's configuration (``config.json``)."""
    from tensorrt_llm.version import __version__

    digest = hashlib.sha256(__version__.encode())
    model_dir = worker_main_args.get("hf_model_dir")
    if model_dir is not None and Path(model_dir, "config.json").is_file():
        digest.update(Path(model_dir, "config.json").read_bytes())
    return digest.hexdigest()


def start_worker(worker_main_args: dict) -> None:
    """Called first in ``worker_main`` with fault tolerance, with its arguments: keeps them and takes the deployment's
    fingerprint. Rank 0 starts the membership authority, whose store listens at the rendezvous address, publishes there
    the arguments every other rank runs ``worker_main`` with (:func:`worker_node_main`), joins the membership
    (:func:`_join_at_entry`) and records its startup (``startup_record``); every other rank joined it in
    :func:`worker_node_main` already.

    Raises:
        ValueError: The launch does not give this process what fault tolerance needs (:func:`_launched_rank`).
    """
    global _startup_fingerprint, _worker_main_args, _authority
    _startup_fingerprint = startup_fingerprint(worker_main_args)
    _worker_main_args = worker_main_args
    identity = _launched_rank()
    if identity.rank == 0:
        _authority, _ = _start_authority(identity.world_size, _startup_time_left())
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
    """Rank 0's ``worker_main`` arguments as every other rank takes them: without the IPC addresses and keys only rank
    0 uses, since the store signs its values but does not encrypt them."""
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
    """The entry point of every rank other than 0 with fault tolerance, the original process and its replacements
    (``python -m tensorrt_llm.llmapi.mgmn_worker_node``): its node supervisor starts it as its own process, with its
    rank (``FAULT_TOLERANCE_RANK_ENV``), the rendezvous address and the join secret in its environment. It joins the
    membership (:func:`_join_at_entry`), then runs ``worker_main`` with the arguments rank 0 published in the
    rendezvous store; a replacement replays rank 0's startup record (``startup_record``). It waits for the store to
    answer and for rank 0 to publish them until the startup deadline (:func:`_startup_time_left`).

    Raises:
        ValueError: The launch does not give this process what fault tolerance needs (:func:`_launched_rank`).
        TimeoutError: Rank 0 did not publish its arguments in time.
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
    if not store.wait([WORKER_ARGS_KEY], _startup_time_left()):
        raise TimeoutError(
            f"Rank 0 did not publish its worker arguments within {RENDEZVOUS_TIMEOUT_S:.0f} s"
        )
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


# When this process's startup waits give up: RENDEZVOUS_TIMEOUT_S after the entry of its worker, as on every rank.
_startup_deadline_s: Optional[float] = None


def _startup_time_left() -> float:
    """How long this process's startup waits may still last; the first call starts the deadline."""
    global _startup_deadline_s
    if _startup_deadline_s is None:
        _startup_deadline_s = time.monotonic() + RENDEZVOUS_TIMEOUT_S
    return max(0.0, _startup_deadline_s - time.monotonic())


@functools.lru_cache(maxsize=1)
def _startup_store() -> SignedStore:
    """This process's client of the rendezvous store for its startup: rank 0's worker starts the authority, which
    hosts the store, so a rank waits for it until the startup deadline."""
    return rendezvous_store(_startup_time_left())


# The startup collectives this rank completed: its progress while the deployment starts (RankReport.progress).
_startup_steps = 0


def _report_startup_wait(waiting_for: int) -> None:
    """Report the peers a startup collective of this rank waits for, so that the authority finds a peer that hangs;
    once it completed (``waiting_for`` 0), one more step of progress."""
    global _startup_steps
    if not waiting_for:
        _startup_steps += 1
    _agent.report(waiting_for=waiting_for, progress=_startup_steps)


# Per rank, the node it runs on (nodes numbered in the order of their first rank), as the original ranks found it
# (join_startup_group).
_rank_nodes: Optional[Tuple[int, ...]] = None


def rank_nodes() -> Tuple[int, ...]:
    """Per rank, the node it runs on, with fault tolerance (after :func:`join_startup_group`)."""
    return _rank_nodes


class StartupGroup(LiveGroupComm):
    """``mpi_comm()`` with fault tolerance, and its subgroups. Every rank is its own process, a one-process MPI world,
    so the ranks that start together meet in the rendezvous store instead: a group of ``ranks`` (global ranks, in
    group-rank order) exchanges through :func:`startup_gather`. A replacement rank's groups have only itself live
    (isolated startup): their exchanges answer at once, None in place of every other member's value, and its splits
    take every member's color and key from the startup record (``startup_record.agree``).

    It has the mpi4py methods the startup calls; any other one fails as a missing attribute. The members of a group
    create its subgroups in the same order, as they would split an MPI communicator.
    """

    def __init__(self, identity: RankIdentity, ranks: Sequence[int], key: str = "world"):
        ranks = tuple(ranks)
        if identity.incarnation:
            gather = isolated_gather(identity.rank, identity.world_size)
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
        """The members on this rank's node (:func:`rank_nodes`), for the node split types (``MPI.COMM_TYPE_SHARED``,
        Open MPI's ``COMM_TYPE_HOST``)."""
        if split_type not in (MPI.COMM_TYPE_SHARED, OMPI_COMM_TYPE_HOST):
            raise NotImplementedError(
                f"StartupGroup splits by node only, not by split type {split_type}"
            )
        node = _rank_nodes[self._identity.rank]
        return self.subgroup(r for r in self.membership if _rank_nodes[r] == node)

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


def join_startup_group() -> None:
    """Called in ``worker_main`` with fault tolerance right after :func:`start_worker`, which checked this rank's
    identity and, on rank 0, started the authority that hosts the rendezvous store: makes this rank's startup group of
    every rank (:class:`StartupGroup`) its ``mpi_comm()``, sets the C++ logger's rank (``TLLM_RANK``), and takes the
    node of every rank from the ranks' hosts (:func:`rank_nodes`).

    Raises:
        ValueError: The process has no rank identity.
        RuntimeError: The node's communicator was split from MPI before.
    """
    global _rank_nodes
    identity = _rank_identity()
    os.environ["TLLM_RANK"] = str(identity.rank)
    world = StartupGroup(identity, range(identity.world_size))
    hosts = startup_record.agree("rank_hosts", lambda: world.allgather(socket.gethostname()))
    nodes: Dict[str, int] = {}
    _rank_nodes = tuple(nodes.setdefault(host, len(nodes)) for host in hosts)
    set_mpi_comm(world)
    if not isinstance(local_mpi_comm(), StartupGroup):
        raise RuntimeError(
            "The node's communicator was split from MPI before the startup group was set"
        )


def _join_at_entry(store: SignedStore, identity: RankIdentity) -> None:
    """Join the membership at the entry of this process's worker. Until its executor takes the membership over
    (:meth:`RankFailureRecovery.start`), the process exits once the deployment aborts or shuts down, or the authority
    is gone."""
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
    """An original rank whose startup is done: report it, and return once every rank's is (generation 0 is
    COMMITTED). Until then, the process exits if the deployment aborts or the authority is gone
    (:func:`_join_at_entry`)."""
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
    """The keys under which rank 0 hands the joining ranks of the change of ``generation`` the host exchange's epoch
    and its waiting queue (:meth:`RankFailureRecovery._admit_joining_ranks`)."""
    return f"host_exchange_epoch/{generation}", f"waiting_queue/{generation}"


class StartupMemory(NamedTuple):
    """This rank's device memory once the deployment serves, which every membership change resizes the KV cache
    pool from."""

    # The GPU KV cache pool's quota: the most the pool takes back.
    kv_cache_quota: int
    # The free device memory: kept free, as the allocations after the KV cache sizing (main's
    # free_gpu_memory_fraction margin) may need it.
    free_bytes: int

    def fit_kv_cache(self, kv_cache_manager, needed: int) -> int:
        """Resize the KV cache pool so that ``needed`` bytes of device memory are free beyond those free when the
        deployment started serving, to at most its quota then, and return its quota: the one memory rule of every
        membership change, which startup checks the largest failure with (:func:`_check_largest_failure_fits`). The
        pool gives up only what the free memory lacks, and its quota is derived afresh at every change, so repeated
        failures do not ratchet it down. Every KV cache must be suspended (``PyExecutor._kv_pools_suspended``).

        Each rank sizes its own pool, so the ranks' KV cache capacities may differ after a change: each attention-DP
        rank serves its own requests. Attention TP, whose ranks share every request's KV cache, would need them
        equal; fault tolerance rejects it at startup.

        Raises:
            RuntimeError: The pool cannot take that quota (``KVCacheManagerV2.resize_gpu_quota``), e.g. below its
                minimum.
        """
        spare_bytes = max(0, torch.cuda.mem_get_info()[0] - self.free_bytes)
        return kv_cache_manager.resize_gpu_quota(
            min(self.kv_cache_quota, kv_cache_manager.gpu_quota + spare_bytes - needed)
        )


def _check_largest_failure_fits(
    executor: PyExecutor, balancer, startup_memory: StartupMemory
) -> None:
    """Check that every original rank can restore the largest failure fault tolerance recovers from, by the rule
    every membership change applies (:meth:`StartupMemory.fit_kv_cache`): its KV cache pool gives up the memory of
    the spare slots it backs (``MoeLoadBalancer.largest_failure_restore_bytes``) and of the all-to-all workspaces
    the remap maps before it frees the pages they replace, then takes it back. Collective over the original ranks,
    which all decide alike.

    Raises:
        ValueError: No membership change could restore that failure on some rank.
    """
    needed = (
        balancer.largest_failure_restore_bytes()
        + executor.model_engine.moe_execution_state.workspace_bytes()
    )
    fits = True
    with executor._kv_pools_suspended() as kv_cache_manager:
        try:
            startup_memory.fit_kv_cache(kv_cache_manager, needed)
        except RuntimeError:
            fits = False
        startup_memory.fit_kv_cache(kv_cache_manager, 0)
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
    """The environment of a process this rank starts outside its MPI job (the membership authority), which uses
    neither MPI nor a GPU. MPI stays uninitialized: mpi4py does not initialize it at import
    (MPI4PY_RC_INITIALIZE=false), and TRT_LLM_NO_LIB_INIT=1 makes importing tensorrt_llm skip its library
    initialization, which includes the C++ MPI_Init (MpiComm.local_init). The launcher's rank variables are removed
    as well, so nothing in the process takes it for a rank of the launcher's job. It logs at this rank's log level."""
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
    """Start the membership authority as its own process, outside this process's MPI job and without GPUs
    (:func:`_helper_process_env`). Its store listens at the rendezvous address, which must name this node. This
    process holds the writing end of the authority's standard input (closed in a forked child), so the authority ends
    once this process ends.

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
    """``rank``'s READY payload for the PROPOSE ``record``, from the rank's ``view`` of the EPLB placement: only what
    :func:`agreed_placement` reads, the view from the lowest survivor and the rank's own slots from a joining rank,
    so that one payload, not every one, carries the whole placement."""
    if rank == record.lowest_survivor():
        return json.dumps(view)
    if record.joining_mask >> rank & 1:
        return json.dumps([ranks[rank] for _, ranks in view])
    return None


def agreed_placement(record: MembershipRecord) -> Optional[list]:
    """The one placement every live rank plans the expert restore from, out of the READY payloads of the INSTALL
    ``record`` (:func:`ready_payload`): the lowest survivor's view (survivors' views can differ when a failure
    interrupted an online EPLB update), with the slots of each joining rank as that rank sent them (it starts from the
    initial placement). None without a survivor, or if one of these ranks sent no placement."""
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
    """One rank's part in keeping the deployment serving after a rank failure; see the module docstring.

    The executor calls :meth:`before_iteration` at the top of every iteration, :meth:`forward_vote` for its
    attention-DP forward vote, :meth:`drop_invalid_step` before it applies a step's results,
    :meth:`fail_unservable_requests` before attention-DP admission, and :meth:`stop` when its loop ends. On rank 0, the
    executor worker sets :attr:`outstanding_requests`.
    """

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
        # Whether the KV cache pool gave up memory for a change not committed yet (_give_back_kv_cache).
        self._kv_cache_shrunk = False
        # Whether the current record is a change this rank has not installed yet (_change_pending), kept up to date
        # under _record_changed.
        self._pending = False
        # Whether some rank held its steps for a pending change in the last forward vote (forward_vote).
        self._steps_quiet = False
        exchange.interrupted = self._membership_changing
        exchange.on_interrupt = self._on_exchange_interrupt

    @classmethod
    def create(cls, executor: PyExecutor) -> Optional[RankFailureRecovery]:
        """Take over, in fault tolerance mode (None otherwise), the membership this process joined at the entry of
        its worker, once its startup is done, which ends its startup record (a replacement checks there that it
        started as the original ranks did). The original ranks first check that the largest failure's restore fits
        (:func:`_check_largest_failure_fits`); rank 0 then publishes the agreed-startup record a replacement rank
        starts with, and an original rank waits for every rank's startup (:func:`wait_for_every_rank`). The
        configuration was checked before the model loaded (:func:`check_supported`), the executor's hang detection
        here.

        A replacement rank returns once the membership change that adds it is installed (:meth:`_join`).

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
        startup_memory = StartupMemory(
            executor.kv_cache_manager.gpu_quota, torch.cuda.mem_get_info()[0]
        )
        if replacement is None and balancer is not None:
            _check_largest_failure_fits(executor, balancer, startup_memory)
        # The deployment serves from here: a lost rank's replacement reattaches the node's host copies, which the
        # node supervisor unlinks once the deployment ended.
        host_weights_prefix = (
            None if balancer is None else balancer.keep_host_copies_for_the_deployment()
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
        """A replacement rank's admission: report that it is ready to join, with its READY (its startup and
        everything else it can prepare without the live ranks are done), so that the live ranks stop for it only now;
        take part in the membership change that adds it, then start the executor loop at the first iteration the live
        ranks resume at, where :meth:`before_iteration` brings it into the host exchange and restores its part of the
        membership."""
        # Its READY to the PROPOSE that adds it, which depends on that PROPOSE only through its joining: it may never
        # see that PROPOSE (the INSTALL may come first), so it ends its own online EPLB work here, as the live ranks
        # do before their READY.
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
        """When the executor loop ends. After a normal end (``completed``), leave the membership: rank 0 asks the
        authority to stop, and every rank waits for the SHUTDOWN record, so that its closing connection is not taken
        for a failure. Then end the process with the loop's status: a teardown would wait for ranks that failed."""
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
        self._exit(0 if completed else 1, "the executor stopped")

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
            if record.state == MembershipState.ABORT or self._excluded(record):
                if (
                    loop is not None and not loop.is_alive()
                ):  # no executor loop left to take the exit path
                    self._exit(1, f"the membership record is {record!r}")

    def _on_agent_error(self, error: Exception) -> None:
        """The agent lost its connection to the authority. If the authority is gone, every rank sees this and
        serving continues without protection (no lease). If only this rank lost it, the authority takes the rank for
        failed: the survivors' remap leaves this process only memory nobody reads, and main's hang detector ends it
        once it waits for peers that moved on. The authority ends with rank 0's process, so it is gone while rank 0
        serves only after a fault of its own; if rank 0 then ends too, node 0's supervisor ends the job (under
        ``mpirun``), or main's hang detector ends the ranks that wait for it. A wait for the next record
        (:meth:`_next_record`) ends."""
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
        """Whether ``record`` no longer lists this process as a live member (a replacement is not excluded before a
        record first lists it)."""
        return self._admitted and not record.is_member(self.rank, self.incarnation)

    def _interrupts_steps(self, record: MembershipRecord) -> bool:
        """Whether ``record`` interrupts the steps in flight: it ends the deployment, leaves this process out, or drops
        a process of the committed membership (a lost rank, or a live rank whose replacement takes it over), which the
        all-to-all of those steps may wait for. A change that only adds replacements keeps every process: the live
        ranks stop at a step boundary instead (:meth:`_membership_changing`)."""
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
        """Whether the host exchange must stop for the membership change: one this rank has not installed yet. A change
        that does not interrupt the steps in flight (:meth:`_interrupts_steps`) stops it only once a forward vote had
        a rank hold its steps for a pending change (:meth:`forward_vote`), or while the steps are held until a change
        commits. That vote launched no forward, and every rank that holds stops at the start of the next round,
        without sending: no live rank completes that round, so every one of them stops in it, after the same forwards,
        which complete."""
        if not self._pending:
            return False
        return (
            self._steps_quiet
            or self._resume_after is not None
            or self._interrupts_steps(self._record)
        )

    @contextlib.contextmanager
    def _hang_detection_paused(self):
        """Pause main's hang detector while this rank takes part in a membership change: the authority bounds each
        phase by the hang timeout instead, and the detector's kill would end a survivor of a long change. A change that
        interrupts a wait the detector is already paused for (``HangDetector.pause``) leaves it paused for that wait."""
        detector = self._executor.hang_detector
        armed = not detector.paused()
        detector.disarm()
        try:
            yield
        finally:
            if armed:
                detector.checkpoint()

    def _on_exchange_interrupt(self) -> None:
        """The host exchange stopped for a membership change: take part in it (:meth:`_change_membership`). A joining
        rank uses the host exchange before the commit only in the collective of its admission
        (:meth:`_finish_install`), which a newer change has the live ranks finish without it: it leaves that
        collective once it installed the newer change, and :meth:`before_iteration` admits it from there.

        Raises:
            _MembershipRestarted: This rank joins in the change it installed.
        """
        self._change_membership()
        if self._resume_after is not None and self._installed.joining_mask >> self.rank & 1:
            raise _MembershipRestarted

    def _change_membership(self) -> None:
        """Take part in the membership change with main's hang detector paused (:meth:`_follow_change`)."""
        with self._hang_detection_paused():
            self._follow_change()

    def _follow_change(self) -> None:
        """Take part in the membership change; returns once the host exchange is installed for its survivors (the
        rest of the change finishes in :meth:`before_iteration`)."""
        seen = self._installed  # a change this rank installed already is none to take part in
        listed = seen is not None  # by a record of this change or an earlier one
        while True:
            record = self._next_record(seen)
            seen = record
            if record.state == MembershipState.ABORT:
                self._exit(1, "the deployment cannot recover from the rank failure")
            if record.state == MembershipState.SHUTDOWN:
                self._exit(0, "the deployment is shutting down")
            if not record.is_member(self.rank, self.incarnation):
                if self._admitted or listed:
                    # A replacement the change left out: its supervisor starts another one.
                    self._exit(1, "the membership no longer includes this rank")
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
                # The INSTALL record carries the READY payloads: the reports the agent keeps sending no longer carry
                # this rank's, nor a joining rank's readiness, which stood for its READY to this change only.
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
                self._exit(
                    1, "the membership authority closed its connection during a membership change"
                )
            return self._record

    def _end_eplb_iterations(self) -> None:
        """End the online EPLB work of the iterations so far and restart its statistics and update plan; waits for
        every kernel launched. Before READY, so that the placement READY carries no longer changes, and again at the
        resume point (:meth:`_finish_install`), where every live rank, a joining one included, stands at the same step:
        a rank may run the step it stopped in after its end before READY, which a peer ran before its own, and update
        plans that advanced by different numbers of iterations would update different layers on different ranks. Blocks
        as long as the updates already started run: the authority waits on a rank that has not replied, so a rank whose
        updates do not end is found hung and left out, like any rank that stops."""
        balancer = _moe_load_balancer_of(self._executor)
        if balancer is not None:
            balancer.abandon_iterations()

    def _expert_placement(self, record: MembershipRecord) -> dict:
        """What this rank's READY reply to the PROPOSE ``record`` gives the agreed placement: its READY payload
        (:func:`ready_payload`). Empty without EPLB."""
        balancer = _moe_load_balancer_of(self._executor)
        if balancer is None:
            return {}
        view = tuple(
            (num_experts, tuple(tuple(slots) for slots in ranks))
            for num_experts, ranks in balancer.current_placement()
        )
        return {"ready_payload": ready_payload(record, self.rank, view)}

    def _install(self, record: MembershipRecord) -> None:
        """Move the host exchange to fresh rows of every live rank, used by the live ranks but the joining ones,
        which enter it at the resume point (:meth:`_admit_joining_ranks`): the round the live ranks stopped in is not
        one a joining rank takes part in."""
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
        """Report that this rank completed one more step of installing a membership change: the authority counts it
        as progress, so an install that takes long but keeps completing steps is not taken for a hang."""
        self._install_steps += 1
        self._agent.report(install_steps=self._install_steps)

    def _gather_through_store(
        self, record: MembershipRecord, name: str, mask: Optional[int] = None
    ) -> Callable[[Any], List[Any]]:
        """An allgather over the ranks in ``mask`` (default: the record's live ranks) through the store, under keys of
        ``name``, the record's generation and the call: every rank's value, None for a rank outside ``mask``. Only
        processes with the join secret can write the values (:class:`SignedStore`)."""
        mask = record.live_mask if mask is None else mask
        return store_gather(
            self._store,
            f"{name}/{record.generation}",
            self.rank,
            [r for r in range(self.world_size) if mask >> r & 1],
            self.world_size,
            lambda keys, ranks, missing: self._wait_for_keys(record, keys, ranks, missing),
        )

    def _wait_for_keys(
        self,
        record: MembershipRecord,
        keys: List[str],
        ranks: List[Optional[int]],
        missing: Optional[Callable[[], int]] = None,
    ) -> List[bytes]:
        """The values of ``keys`` once all are set; ``ranks[i]`` sets ``keys[i]`` (None: the authority). A value not
        signed with the join secret counts as not set (:class:`SignedStore`). Until then, this rank reports that it
        waits for the ranks whose values are missing, which it reads once per report interval with ``missing()``
        (default: one store call per key), and raises :class:`_MembershipRestarted` once the record is no longer
        ``record``."""
        if missing is None:

            def missing() -> int:
                return sum(
                    1 << rank
                    for key, rank in zip(keys, ranks)
                    if rank is not None and not self._store.store.check([key])
                )

        read_s = None

        def check(unsigned: Set[str]) -> None:
            nonlocal read_s
            self._check_record(record)
            now_s = time.monotonic()
            if read_s is None or now_s - read_s >= REPORT_INTERVAL_S:
                read_s = now_s
                self._waiting_in_store = missing() | sum(
                    1 << rank
                    for key, rank in zip(keys, ranks)
                    if rank is not None and key in unsigned
                )

        try:
            return wait_for_keys(self._store, keys, _RECORD_CHECK_S, check)
        finally:
            self._waiting_in_store = 0

    def _check_record(self, record: MembershipRecord) -> None:
        """Raises :class:`_MembershipRestarted` once the membership record is no longer ``record``: a newer change
        replaced it, or the deployment aborts."""
        if self._record != record:
            raise _MembershipRestarted

    def _fail_install(self, record: MembershipRecord, error: Exception) -> bool:
        """Installing ``record`` failed with ``error``: reply FAILED, so that the authority aborts, or goes on without
        a joining rank that failed. If only other live ranks failed to map MNNVL memory, they reply FAILED, and this
        rank waits for the authority's next record instead. Returns whether this rank replied FAILED."""
        if isinstance(error, MnnvlMappingError) and self.rank not in error.ranks:
            logger.warning(f"Rank {self.rank} waits for the next membership record: {error!r}")
            return False
        logger.error(
            f"Rank {self.rank} could not install membership generation {record.generation}: {error!r}"
        )
        self._agent.report(reply=Reply.FAILED, reply_generation=record.generation)
        return True

    def _exit(self, status: int, reason: str) -> None:
        _end_process(self.rank, status, reason)

    # Executor loop hooks.

    def before_iteration(self) -> None:
        """Finish installing the membership change once, at the first iteration past the last one any survivor had
        reached when it stopped, where no forward or EPLB iteration is active: apply the results of the step in flight
        (or drop it, :meth:`drop_invalid_step`), continue the interrupted requests by recompute, bring the joining ranks
        into the host exchange (:meth:`_admit_joining_ranks`), restore this rank's part of the membership
        (:meth:`_restore`), reply INSTALLED and wait for the commit; then rank 0 fails the requests that no live rank
        holds. A failure of this rank replies FAILED, so the authority aborts, or goes on without a joining rank that
        failed. Under a newer change, the forwards stay held until its own install. A joining rank, which has run no
        step with the live ranks yet, follows a newer change that still adds it as it joined (:meth:`_join`), then
        finishes its admission from that change's install."""
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
        """:meth:`before_iteration` at the iteration that finishes installing ``record``. Returns whether the change
        was committed."""
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
            if self._fail_install(record, error):
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
        """Once a change is committed: release the device memory of the spare slots the committed placement leaves
        empty (a joining rank's experts went back to it), and, if that freed memory or the change took KV cache memory
        (:meth:`_restore`), let the KV cache pool take back what is free again (:meth:`StartupMemory.fit_kv_cache`),
        the pages the restore's remap replaced included. No live rank waits for this in the transaction; routing has
        not reached these slots since the install.

        Returns:
            What it gave back, for the log ("" if nothing).
        """
        start_s = time.perf_counter()
        balancer = _moe_load_balancer_of(self._executor)
        released = 0 if balancer is None else balancer.release_empty_spare_slots()
        if not (released or self._kv_cache_shrunk):
            return ""
        self._kv_cache_shrunk = False
        with self._executor._kv_pools_suspended() as kv_cache_manager:
            quota = self._startup_memory.fit_kv_cache(kv_cache_manager, 0)
        return (
            f"; {released} bytes of empty spare slots released, KV cache quota {quota} bytes, in "
            f"{(time.perf_counter() - start_s) * 1e3:.0f} ms"
        )

    def _restore(self, record: MembershipRecord, times: Optional[Dict[str, float]] = None) -> str:
        """This rank's part of installing ``record``, collective over its live ranks: place the experts as planned
        from the agreed placement (:func:`agreed_placement`), a lost rank's
        into spare slots and a joining rank's back to it, remap the all-to-all workspaces among the live ranks (or,
        when every process that maps their pages stays live, only map the joining ranks' rows), then reset the
        all-to-all and install the live mask. The KV cache pool gives up the device memory the backed spare slots and
        the remap need beyond the free memory (:meth:`StartupMemory.fit_kv_cache`); it takes back what is free again,
        the pages the remap replaced and the spare slots the placement empties, once the change is committed
        (:meth:`_give_back_kv_cache`).

        Args:
            record: The INSTALL record.
            times: The milliseconds of the install's phases so far; the restore adds its own and logs them all.

        Returns:
            The digest of the restored EPLB placement ("" without EPLB).

        Raises:
            _MembershipRestarted: The record changed since INSTALL (checked first, in every wait for a peer, and
                last).
        """
        self._check_record(record)
        began_s = time.perf_counter()
        times = {} if times is None else times
        executor = self._executor
        balancer = _moe_load_balancer_of(executor)
        plan = None
        loaded = 0  # experts this rank loads into its slots
        if balancer is not None:
            current = [np.array(ranks, dtype=np.int32) for _, ranks in agreed_placement(record)]
            plan = balancer.plan_expert_restore(
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
                fitted = self._startup_memory.fit_kv_cache(kv_cache_manager, needed)
                self._kv_cache_shrunk = True
                times["KV cache shrink"] = (time.perf_counter() - start_s) * 1e3
                kv_cache_quota = (
                    f"{before} -> {fitted} bytes ({needed} bytes needed, {free} bytes free)"
                )
                self._install_step_done()
            if plan is not None:
                balancer.restore_lost_experts(plan, times)
                self._install_step_done()
            remap()
        # Under the lock the agent thread updates the record with, before it requests the abort of a newer change that
        # interrupts the steps: once this check passes, that abort comes after the rearm and stays.
        start_s = time.perf_counter()
        with self._record_changed:
            self._check_record(record)
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
        """Bring the joining ranks into the host exchange, whose rows every live rank mapped at INSTALL, at the first
        iteration the live ranks resume at, where they all stand between the same two rounds, so each of them
        continues from its own next epoch. Through the store, rank 0 hands that epoch and its waiting queue, which
        every rank holds alike, to the joining ranks."""
        epoch_key, queue_key = _admission_keys(record.generation)
        executor = self._executor
        if record.joining_mask >> self.rank & 1:
            epoch, waiting = self._wait_for_keys(record, [epoch_key, queue_key], [0, 0])
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
        """Whether this iteration runs no forward: none runs from a membership change this rank has seen until it is
        committed, so the survivors resume at the same iteration, as a forward of an earlier one would pair its
        all-to-all rounds with those of other iterations."""
        return self._resume_after is not None or self._pending

    def forward_vote(
        self, batch_size: int, gather: Callable[[List[int]], np.ndarray]
    ) -> np.ndarray:
        """The executor's attention-DP forward vote: ``gather`` (its allgather of int64 vectors) collects from every
        rank its batch size, 0 while it holds its steps (:meth:`holds_steps`), and whether it holds them for a pending
        membership change; returns the batch sizes. Every live rank learns alike whether one held for a pending
        change, from which on the host exchange stops for a change that does not interrupt the steps in flight
        (:meth:`_membership_changing`). While this rank holds its steps, rank 0 does not wait for requests: the host
        exchange stops at its next round."""
        pending = self._pending
        held = pending or self._resume_after is not None
        votes = gather([0 if held else batch_size, int(pending)])
        self._steps_quiet = bool(votes[:, 1].any())
        if held and self.rank == 0:
            self._executor.executor_request_queue.wake()
        return votes[:, 0]

    def drop_invalid_step(self) -> None:
        """Discard the executor's previous step before its results are applied if the all-to-all is aborted: an
        abort may have cut one of its all-to-all rounds short. Its requests continue by recompute later, and the
        context rows it stored are dropped, so the recompute stores those positions again."""
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
        """A forward that raised while the all-to-all is aborted failed for the membership change, not for its
        requests: discard its step, and the previous one, whose all-to-all rounds the abort may have cut short too
        (:meth:`drop_invalid_step`); their requests continue by recompute once the live ranks resume. Returns whether
        it discarded them; False: the forward failed by itself.
        """
        if not self._moe.aborted():
            return False
        self._interrupted.extend(scheduled_requests.all_requests())
        self._executor._accepted_previous_batch()
        return True

    def _recompute_interrupted_requests(self) -> int:
        """The requests of the discarded steps continue from the tokens they have, as after a preemption; returns how
        many. The executor commits a context chunk's KV for reuse only once its step is accepted, so the discarded
        steps left nothing reusable."""
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
        """Before admission, remove the waiting requests pinned (strictly) to an attention-DP rank that does not serve
        (``rank_states`` holds the serving ranks); rank 0 fails them. Every rank holds the same waiting queue, so all of
        them remove the same requests."""
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
    """The membership record of a deployment with WideEP fault tolerance, read from the membership store when the
    server's ``/health`` or its status endpoint is served.

    ``healthy``: every rank serves. ``degraded``: the deployment serves without some ranks. ``recovering``: a
    membership change is in progress. ``unprotected``: serving, but the authority does not answer (its process, which
    hosts the store, ended), so a rank failure would not be recovered; final, as the authority is not restarted.
    ``stopped``: the deployment aborted or shuts down.

    A record older than one already read counts as absent, as does a value not signed with the join secret or one that
    does not parse (logged).
    """

    def __init__(self, address: str, secret: bytes):
        self._address = address
        self._secret = secret
        self._store: Optional[SignedStore] = None
        self._record: Optional[MembershipRecord] = None  # the newest read
        self._next_open = 0.0  # time.monotonic() before which the store is not connected to again
        self._authority_ended = False
        self._lock = threading.Lock()

    def _read(self) -> Tuple[Optional[MembershipRecord], Optional[TransactionSummary], bool]:
        """The record, the last transaction and whether the authority answers: its store does; the newest record
        read, no transaction and False while it does not.

        Until the store first answers, a read waits for it ``_STATUS_READ_TIMEOUT_S`` at most and connects to it at
        most once per ``_STATUS_READ_TIMEOUT_S``, so reads while it does not answer do not each wait for it. Once it
        answered, a failed read means the authority's process ended: later reads answer at once without the store.
        """
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
            if record is not None and (
                self._record is None or record.sequence >= self._record.sequence
            ):
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
        world = (1 << len(record.incarnations)) - 1 if record.incarnations else record.live_mask
        return "healthy" if record.live_mask == world else "degraded"

    def status(self) -> Dict[str, object]:
        """The server's fault-tolerance status; see :meth:`describe`."""
        return self.describe(*self._read())

    @classmethod
    def describe(
        cls,
        record: Optional[MembershipRecord],
        transaction: Optional[TransactionSummary],
        authority_alive: bool,
    ) -> Dict[str, object]:
        """The state (see the class), whether the authority answers, and from the last record the authority
        published: its generation and state, the live and lost ranks and each rank's incarnation, and the last
        transaction that ended, with its phases in milliseconds and when each ended in milliseconds since the Unix epoch
        (``TransactionSummary``). The record's fields are None until the store answers."""
        status = {
            "state": cls._state(record, authority_alive),
            "authority_alive": authority_alive,
            "generation": None,
            "membership_state": None,
            "live_ranks": None,
            "lost_ranks": None,
            "incarnations": None,
            "last_transaction": None,
        }
        if record is not None:
            world_size = len(record.incarnations) or record.live_mask.bit_length()
            status.update(
                generation=record.generation,
                membership_state=record.state.name,
                live_ranks=[r for r in range(world_size) if record.is_live(r)],
                lost_ranks=[r for r in range(world_size) if not record.is_live(r)],
                incarnations=list(record.incarnations),
            )
        if transaction is not None:
            status["last_transaction"] = {
                "generation": transaction.generation,
                "outcome": transaction.state.name,
                "removed_ranks": list(transaction.removed_ranks),
                "added_ranks": list(transaction.added_ranks),
                "duration_s": transaction.duration_s,
                "phases_ms": {
                    "detect_to_propose": transaction.detect_to_propose_ms,
                    "propose_to_ready": transaction.propose_to_ready_ms,
                    "ready_to_installed": transaction.ready_to_installed_ms,
                },
                "times_unix_ms": transaction.times_unix_ms(),
            }
        return status


if __name__ == "__main__":
    worker_node_main()

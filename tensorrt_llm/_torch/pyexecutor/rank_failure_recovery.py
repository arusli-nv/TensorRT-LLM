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
"""Keep a Wide-EP deployment serving when one of its ranks fails, and let a replacement rank rejoin.

See ``docs/source/features/wide-ep-fault-tolerance.md``.
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
from typing import TYPE_CHECKING, Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Union

import numpy as np
import torch

from tensorrt_llm._utils import (
    local_mpi_comm,
    mpi_broadcast,
    mpi_rank,
    set_mpi_comm,
    set_mpi_world_comm,
)
from tensorrt_llm._wide_ep_fault_tolerance import (
    FAULT_TOLERANCE_RANK_ENV,
    RankIdentity,
    check_fault_tolerance_requirements,
    fault_tolerance_replacement,
    set_wide_ep_fault_tolerance_enabled,
    wide_ep_fault_tolerance_enabled,
    wide_ep_fault_tolerance_rank,
)
from tensorrt_llm.executor.utils import RankFailureError, RequestError
from tensorrt_llm.logger import logger

from ..distributed.host_exchange_epoch import next_epoch
from ..distributed.membership import (
    DISCONNECTED_KEY,
    HANG_TIMEOUT_S,
    HOST_WEIGHTS_PREFIX_KEY,
    LAUNCHER_ENV_PREFIXES,
    RECORD_KEY,
    RENDEZVOUS_ENV,
    RENDEZVOUS_TIMEOUT_S,
    SHARED_SECRET_ENV,
    STORE_POLL_S,
    TRANSACTION_KEY,
    MembershipClient,
    MembershipRecord,
    MembershipState,
    RendezvousStore,
    ReplacementCannotStart,
    Reply,
    ResumeState,
    StoreGroup,
    TransactionSummary,
    open_store,
    rendezvous_store,
    wait_for_keys,
)
from ..distributed.mnnvl_active_group import MnnvlMappingError, MnnvlPeerHandleError
from ..distributed.startup_record import startup_record

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
    from tensorrt_llm.mapping import Mapping

    from ..distributed import Distributed
    from ..moe.fused_moe.communication.execution_state import MoeExecutionState
    from .llm_request import LlmRequest
    from .py_executor import PyExecutor
    from .scheduler.adp_router import RankState
    from .scheduler.waiting_queue import WaitingQueue

_AUTHORITY_TIMEOUT_S = 10.0
_STATUS_READ_TIMEOUT_S = 1.0
_GIVE_BACK_RETRY_S = 1.0
_SHARED_SECRET_BYTES = 32


class _MembershipRestarted(Exception):
    """A newer membership change replaced the one being installed."""


_rendezvous_ports: List[socket.socket] = []
_client: Optional[MembershipClient] = None
_authority: Optional[subprocess.Popen] = None


def ensure_rendezvous() -> None:
    """Export the shared secret and the rendezvous address when unset.

    SO_REUSEADDR sockets that never listen reserve the two ports, so the authority can still bind them.
    """
    global _rendezvous_ports
    if not os.environ.get(SHARED_SECRET_ENV):
        os.environ[SHARED_SECRET_ENV] = secrets.token_hex(_SHARED_SECRET_BYTES)
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
            except (OSError, OverflowError):
                for sock in reserved:
                    sock.close()
        _rendezvous_ports = reserved
        os.environ[RENDEZVOUS_ENV] = f"{socket.gethostbyname(socket.gethostname())}:{port}"


def check_supported(llm_args: TorchLlmArgs, mapping: Mapping, dist: Distributed) -> None:
    """Check the Wide-EP fault tolerance requirements that the LLM argument validation cannot check.

    This runs before the model loads.

    Raises:
        ValueError: The configuration is not supported with fault tolerance.
    """
    from tensorrt_llm._utils import mpi_disabled

    from ..distributed.host_exchange import MnnvlDist
    from ..moe.fused_moe.communication.nvlink_one_sided import FORCE_CFT_ENV, get_force_cft
    from ..moe.fused_moe.moe_comm_timeout_guard import (
        SERVING_TIMEOUT_ENV,
        resolve_moe_comm_timeout_budgets,
    )

    # The authority excludes a hung rank up to 1.5 hang timeouts after a wait on it began. The all-to-all kernels
    # that wait on the rank must not time out before that.
    kernel_timeout_s = resolve_moe_comm_timeout_budgets(os.environ).serving_seconds
    if kernel_timeout_s is None:
        kernel_timeout_s = torch.ops.trtllm.moe_a2a_get_timeout()
    if isinstance(dist, MnnvlDist):
        missing = []
    elif mpi_disabled():
        missing = ["the MPI orchestrator (TLLM_DISABLE_MPI unset)"]
    else:
        missing = MnnvlDist.unmet_requirements(mapping) or [
            "host collectives over MNNVL memory (MnnvlDist)"
        ]
    checks = [
        (not missing, ", ".join(missing)),
        # The executor creator turns the overlap scheduler off after the arguments' validation in some modes.
        (
            not llm_args.disable_overlap_scheduler,
            "the overlap scheduler, which speculative decoding modes without overlap support (such as two-model "
            "Eagle3 or draft-target) and multimodal encoder-only mode turn off, because Wide-EP fault tolerance "
            "accepts or discards each step when the next one starts, which only the overlap scheduler's executor "
            "loop does yet",
        ),
        (
            mapping.moe_ep_size == mapping.world_size,
            "expert parallelism over every rank (moe_expert_parallel_size equal to the world size), so that a "
            "failed rank holds whole experts, which the survivors restore into their spare slots",
        ),
        # Resolved from "auto" only as the model loads, after the arguments' validation.
        (
            llm_args.kv_cache_config.use_kv_cache_manager_v2 is True,
            "KV cache manager V2 (kv_cache_config.use_kv_cache_manager_v2=True), the only KV cache manager "
            "that holds the KV reuse commits of a step until the step is accepted",
        ),
        (
            get_force_cft() is not True,
            f"the fence path of the NVLink one-sided all-to-all ({FORCE_CFT_ENV} unset or 0), the only path "
            "that follows a membership change on the same CUDA graphs",
        ),
        (
            kernel_timeout_s >= 2 * HANG_TIMEOUT_S,
            f"{SERVING_TIMEOUT_ENV} of at least {2 * HANG_TIMEOUT_S:.0f} s (twice the hang timeout), so "
            "that an all-to-all waiting on a hung rank lasts until the rank is dropped",
        ),
    ]
    check_fault_tolerance_requirements(checks)


def _launched_rank() -> RankIdentity:
    """Return this process's rank after checking the shared secret and the rendezvous address."""
    try:
        secret = bytes.fromhex(os.environ[SHARED_SECRET_ENV])
        int(os.environ[RENDEZVOUS_ENV].rsplit(":", 1)[1])
    except (KeyError, IndexError, ValueError):
        raise ValueError(
            f"Wide-EP fault tolerance requires on every rank a hex shared secret in {SHARED_SECRET_ENV} and the "
            f"rendezvous address in {RENDEZVOUS_ENV}: host:port on rank 0's node, where the rendezvous store "
            "listens. The server sets both for the workers it starts; where it does not start them "
            "(trtllm-llmapi-launch, or an MPI session passed to the LLM API), export them before launching or "
            "creating the session, e.g. the secret from `openssl rand -hex 32` and the first host of the job "
            "with a free port; or turn it off (enable_wide_ep_fault_tolerance=False)"
        ) from None
    if len(secret) < _SHARED_SECRET_BYTES:
        raise ValueError(
            f"Wide-EP fault tolerance requires a shared secret of at least {_SHARED_SECRET_BYTES} bytes "
            f"({2 * _SHARED_SECRET_BYTES} hex digits) in {SHARED_SECRET_ENV}, got {len(secret)}: it keys the "
            "signatures of the rendezvous store's values, which the ranks unpickle. Export e.g. "
            "`openssl rand -hex 32`, and check that it printed the secret (without openssl it yields an empty "
            "string); or turn it off (enable_wide_ep_fault_tolerance=False)"
        )
    return _rank_identity()


def _rank_identity() -> RankIdentity:
    identity = wide_ep_fault_tolerance_rank()
    if identity is None:
        raise ValueError(
            "Wide-EP fault tolerance starts every rank as its own process, with "
            f"'<rank>,<world size>,<GPUs per node>,<replacement>' in {FAULT_TOLERANCE_RANK_ENV}: launch the ranks "
            "with trtllm-llmapi-launch, whose node supervisor sets it, or let the server start them on one node"
        )
    return identity


def prepare_model(model: torch.nn.Module) -> None:
    """Check that the built model supports fault tolerance and bind its MoE layers before any weight loads.

    Raises:
        ValueError: The model is not supported with fault tolerance.
    """
    from ..moe.fused_moe.communication.execution_state import MoeExecutionState
    from .config_utils import is_hybrid_linear

    if is_hybrid_linear(model.model_config.pretrained_config):
        raise ValueError(
            "Wide-EP fault tolerance does not support hybrid linear-attention models: their KV cache manager "
            "(MambaHybridCacheManagerV2) commits recurrent-state snapshots for reuse at snapshot points and when a "
            "request is freed, so it cannot hold a step's reuse commits until the step is accepted; turn it off "
            "(enable_wide_ep_fault_tolerance=False)"
        )
    MoeExecutionState.for_model(model)
    _check_host_weights_fit(model)


def _check_host_weights_fit(model: torch.nn.Module) -> None:
    """Check that /dev/shm has room for the host weights of every expert beyond the segments already on the node."""
    from ..moe.fused_moe.eplb_expert_restore import SHARED_MEMORY_DIR, DeploymentHostWeights
    from ..moe.fused_moe.impl_blocks import MoEEplbWeightLayoutMixin

    slot_bytes: Dict[DeploymentHostWeights, List[int]] = {}
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
            f"Wide-EP fault tolerance keeps the host weights of every expert on every node in {SHARED_MEMORY_DIR}: "
            f"this node needs {needed / 2**30:.1f} GiB ({needed} bytes) more there, but {SHARED_MEMORY_DIR} has "
            f"{free / 2**30:.1f} GiB ({free} bytes) free. Give {SHARED_MEMORY_DIR} more room (`docker run "
            "--shm-size`; on Kubernetes, an emptyDir volume with medium: Memory and a sizeLimit mounted at "
            f"{SHARED_MEMORY_DIR}), or turn it off (enable_wide_ep_fault_tolerance=False)"
        )


AGREED_STARTUP_KEY = "agreed_startup"
WORKER_ARGS_KEY = "worker_args"

_startup_fingerprint: Optional[str] = None
_deployment_nonce: Optional[str] = None


def deployment_nonce() -> str:
    """Return a random deployment name that is the same on every rank.

    The name keeps the shared-memory segments of different deployments apart.
    """
    global _deployment_nonce
    if _deployment_nonce is None:
        _deployment_nonce = startup_record.agree(
            "deployment_nonce",
            lambda: mpi_broadcast(secrets.token_hex(8) if mpi_rank() == 0 else None),
        )
    return _deployment_nonce


def startup_fingerprint(worker_main_args: dict) -> str:
    """Return a digest of the TensorRT-LLM version and the checkpoint's ``config.json``."""
    from tensorrt_llm.version import __version__

    digest = hashlib.sha256(__version__.encode())
    model_dir = worker_main_args.get("hf_model_dir")
    if model_dir is not None and Path(model_dir, "config.json").is_file():
        digest.update(Path(model_dir, "config.json").read_bytes())
    return digest.hexdigest()


def start_worker(worker_main_args: dict) -> None:
    """Start the authority on rank 0, publish its worker arguments, join the membership and record the startup."""
    global _startup_fingerprint, _authority
    _startup_fingerprint = startup_fingerprint(worker_main_args)
    identity = _launched_rank()
    if identity.rank == 0:
        _authority, _ = _start_authority(identity.world_size, RENDEZVOUS_TIMEOUT_S)
        store = _startup_store()
        store.set(WORKER_ARGS_KEY, _worker_args_for_other_ranks(worker_main_args))
        _join_at_entry(store, identity)
        startup_record.record()
    elif _client is None:
        raise ValueError(
            f"Wide-EP fault tolerance starts rank {identity.rank} as its own process, which takes rank 0's worker "
            "arguments from the rendezvous store: start it with worker_node_main, as trtllm-llmapi-launch does"
        )


def _worker_args_for_other_ranks(worker_main_args: dict) -> bytes:
    """Return rank 0's worker arguments without its IPC addresses and keys.

    The rendezvous store does not encrypt its values.
    """
    from tensorrt_llm.executor.utils import WorkerCommIpcAddrs

    return pickle.dumps(
        {
            **worker_main_args,
            "worker_queues": WorkerCommIpcAddrs(None, None, None),
            "rpc_addr": None,
            "hmac_key": b"",
        }
    )


def worker_node_main() -> None:
    """Run a rank other than 0 with fault tolerance, as trtllm-llmapi-launch does.

    Raises:
        ReplacementCannotStart: A replacement does not match the deployment, or its startup failed in a way that a
            restart would repeat.
    """
    set_wide_ep_fault_tolerance_enabled(True)
    identity = _launched_rank()
    from tensorrt_llm.executor.worker import worker_main

    if identity.rank == 0:
        raise ValueError(
            "Wide-EP fault tolerance runs rank 0's worker in the session of the server that starts the deployment, "
            "and every other rank as its own process"
        )
    store = _startup_store()
    _join_at_entry(store, identity)
    try:
        # Only processes with the shared secret can write the store's values (RendezvousStore).
        worker_main_args = pickle.loads(store.get(WORKER_ARGS_KEY))  # nosec B301
        if identity.replacement:
            agreed_startup = pickle.loads(store.get(AGREED_STARTUP_KEY))  # nosec B301
            if startup_fingerprint(worker_main_args) != agreed_startup["fingerprint"]:
                raise ReplacementCannotStart(
                    "The replacement rank does not match the deployment it replaces a rank of (TensorRT-LLM "
                    "version or checkpoint)"
                )
            startup_record.replay(agreed_startup)
        worker_main(**worker_main_args)
    except ReplacementCannotStart as error:
        logger.error(f"No replacement of this rank can start: {error}")
        raise


@functools.lru_cache(maxsize=1)
def _startup_store() -> RendezvousStore:
    return rendezvous_store(RENDEZVOUS_TIMEOUT_S)


_startup_steps = 0
_startup_waiting_for = 0


def _report_startup_wait(waiting_for: int) -> None:
    """Report each change in the peers that a startup collective waits for.

    A ``waiting_for`` of 0 means that the collective completed.
    """
    global _startup_steps, _startup_waiting_for
    if waiting_for and waiting_for == _startup_waiting_for:
        return
    _startup_waiting_for = waiting_for
    if not waiting_for:
        _startup_steps += 1
    _client.report(waiting_for=waiting_for, progress=_startup_steps)


def join_startup_group(mapping: Mapping) -> None:
    """Make the startup group of every rank this rank's ``mpi_comm()`` and MPI world, and set ``TLLM_RANK``."""
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
    startup_group = StoreGroup(
        _startup_store(),
        "startup/world",
        identity.rank,
        range(identity.world_size),
        _report_startup_wait,
        active=[identity.rank] if identity.replacement else None,
        gpus_per_node=identity.gpus_per_node,
        startup=True,
    )
    set_mpi_comm(startup_group)
    set_mpi_world_comm(startup_group)
    if not isinstance(local_mpi_comm(), StoreGroup):
        raise RuntimeError(
            "The node's communicator was split from MPI before the startup group was set"
        )


def _join_at_entry(store: RendezvousStore, identity: RankIdentity) -> None:
    """Join the membership, exiting the process if the deployment ends before the executor takes the client over."""
    global _client
    _client = MembershipClient(
        identity.rank,
        store,
        _on_record_at_startup,
        replacement=identity.replacement,
        on_error=_on_error_at_startup,
    )
    _client.start()


_every_rank_started = threading.Event()


def _on_record_at_startup(record: MembershipRecord) -> None:
    if record.state in (MembershipState.ABORTED, MembershipState.SHUTDOWN):
        status = 0 if record.state == MembershipState.SHUTDOWN else 1
        _end_process(_client.rank, status, f"the membership record is {record} during the startup")
    if record.state != MembershipState.STARTING:
        _every_rank_started.set()


def _on_error_at_startup(error: Exception) -> None:
    _end_process(
        _client.rank, 1, f"the membership authority is gone during the startup ({error!r})"
    )


def wait_for_every_rank() -> None:
    """Report this rank started and wait until every rank has."""
    _client.report(started=True)
    _every_rank_started.wait()


def _end_process(rank: int, status: int, reason: str) -> None:
    (logger.info if status == 0 else logger.error)(f"Rank {rank} exits: {reason}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)


def _moe_load_balancer_of(executor: PyExecutor):
    from ..moe.fused_moe.moe_load_balancer import MoeLoadBalancer

    balancer = executor.model_engine.moe_load_balancer
    return balancer if isinstance(balancer, MoeLoadBalancer) else None


def _admission_keys(generation: int) -> Tuple[str, str]:
    return f"host_exchange_epoch/{generation}", f"waiting_queue/{generation}"


class StartupMemory(NamedTuple):
    """The GPU KV cache quota of this rank once the deployment serves.

    Every membership change resizes the KV cache pool from this quota.
    """

    kv_cache_quota: int

    def free_to_keep(self, kv_cache_manager, held: int) -> int:
        """Return the free device memory less what the pool lacks of its quota for ``held``."""
        return torch.cuda.mem_get_info()[0] - (
            self.kv_cache_quota - held - kv_cache_manager.gpu_quota
        )

    def fit_kv_cache(self, kv_cache_manager, held: int, needed: int, free_to_keep: int = 0) -> int:
        """Resize the suspended KV cache pool to its startup quota less ``held`` and ``needed`` and return the quota.

        The quota is derived afresh at every change so that repeated failures do not ratchet it down. The pool grows
        only into memory free beyond ``free_to_keep``.
        """
        return kv_cache_manager.resize_gpu_quota(
            self.kv_cache_quota
            - held
            - self.shortfall(kv_cache_manager, held, free_to_keep)
            - needed
        )

    def shortfall(self, kv_cache_manager, held: int, free_to_keep: int = 0) -> int:
        """Return how much of its quota the pool neither holds nor can take from memory free beyond ``free_to_keep``."""
        return max(0, free_to_keep - self.free_to_keep(kv_cache_manager, held))


@dataclass
class _GiveBackRetry:
    """A KV cache give-back that fell short after a commit.

    The give-back is retried as the excluded processes end and free their pages.
    """

    free_to_keep: int
    excluded: Dict[int, int]
    free_seen: int
    check_after_s: float
    ended: bool = False


def _moe_execution_state_of(executor: PyExecutor) -> MoeExecutionState:
    from ..moe.fused_moe.communication.execution_state import MoeExecutionState

    return MoeExecutionState.model_comms(executor.model_engine.model)[0].execution_state


def _check_largest_failure_fits(
    executor: PyExecutor,
    balancer,
    startup_memory: StartupMemory,
    moe_execution_state: MoeExecutionState,
) -> None:
    """Check on every original rank that the KV cache pool can give up the memory to restore the largest failure."""
    needed = (
        balancer.expert_restore.largest_failure_restore_bytes()
        + moe_execution_state.workspace_bytes()
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
            "Wide-EP fault tolerance cannot recover from the loss of a node other than rank 0's (or of one rank): "
            f"restoring its experts needs more device memory on ranks {short} than their KV cache pools can give "
            f"up above their minimum ({needed} bytes on rank {executor.dist.rank}, whose pool holds "
            f"{startup_memory.kv_cache_quota} bytes); use more ranks, fewer EPLB slots per rank "
            "(moe_config.load_balancer.num_slots) or a larger KV cache (kv_cache_config.free_gpu_memory_fraction)"
        )


def _start_authority(world_size: int, timeout_s: float) -> Tuple[subprocess.Popen, str]:
    """Start the membership authority and return it with the ``host:port`` its store listens on.

    The authority ends with this process, which holds the writing end of its standard input. It runs without MPI,
    launcher variables or a GPU.
    """
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


# For each MoE layer, the expert count and, for each rank, the expert in each slot. An empty spare slot holds -1.
Placement = Tuple[Tuple[int, Tuple[Tuple[int, ...], ...]], ...]


def ready_payload(record: MembershipRecord, rank: int, view: Placement) -> Optional[str]:
    """Return the READY payload of ``rank``.

    The lowest survivor sends the whole placement, and a joining rank sends its own slots.
    """
    if rank == record.lowest_survivor():
        return json.dumps(view)
    if record.joining_mask >> rank & 1:
        return json.dumps([ranks[rank] for _, ranks in view])
    return None


def agreed_placement(record: MembershipRecord) -> Optional[list]:
    """Return the placement every active rank plans the restore from, or None if a READY payload is missing.

    The lowest survivor's view is the source because survivors' views can differ after an interrupted EPLB update.
    """
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
    """Take part in every membership change on one rank, so that the deployment keeps serving after a rank failure."""

    def __init__(
        self,
        executor: PyExecutor,
        exchange,
        moe_state,
        store: RendezvousStore,
        address: str,
        startup_memory: StartupMemory,
        authority: Optional[subprocess.Popen] = None,
        incarnation: int = 0,
    ):
        self._executor = executor
        self._exchange = exchange
        self._moe = moe_state
        self._store = store
        self.address = address
        self._startup_memory = startup_memory
        self._authority = authority
        self.rank = executor.dist.rank
        self.world_size = executor.dist.world_size
        self.incarnation = incarnation
        # On rank 0, returns the client id of every request without a final response, by request id. The executor
        # worker attaches it before it reports READY, so before the proxy sends any request.
        self._outstanding_requests: Optional[Callable[[], Dict[int, int]]] = None
        self._client: Optional[MembershipClient] = None
        self._record = MembershipRecord(
            generation=0, state=MembershipState.COMMITTED, active_mask=(1 << self.world_size) - 1
        )
        self._record_changed = threading.Condition()
        self._admitted = incarnation == 0
        self._committed_mask = self._record.active_mask
        # Per rank, the newest process that an INSTALL or COMMITTED record listed as active, which may map its peers'
        # pages until it ends.
        self._listed_incarnations = [0] * self.world_size
        self._installed: Optional[MembershipRecord] = None
        self._attempted: Optional[MembershipRecord] = None
        # The last iteration that any survivor reached. It is set from INSTALL until the commit, while no forward runs.
        self._resume_after: Optional[int] = None
        self._waiting_in_store = 0
        self._install_steps = 0
        self._published_admissions: List[int] = []
        self._interrupted: List[LlmRequest] = []
        self._kv_cache_free_to_keep: Optional[int] = None
        self._give_back_pending = False
        self._give_back_retry: Optional[_GiveBackRetry] = None
        self._give_back_due = False
        self._give_back_agreed = False
        # Guarded by _record_changed.
        self._pending = False
        self._steps_quiet = False
        exchange.interrupted = self._membership_changing
        exchange.on_interrupt = self._on_exchange_interrupt

    @classmethod
    def create(cls, executor: PyExecutor) -> Optional[RankFailureRecovery]:
        """Take over the membership this process joined at startup, or return None without fault tolerance.

        An original rank returns once every rank has started, a replacement once the change that adds it is installed.
        """
        if not wide_ep_fault_tolerance_enabled():
            return None
        # A rank waits on a hung peer for up to 1.5 hang timeouts, until the authority excludes the peer. The
        # executor's hang detector must not end the rank before that.
        hang_detection_s = executor.hang_detector.timeout
        check_fault_tolerance_requirements(
            [
                (
                    hang_detection_s >= 2 * HANG_TIMEOUT_S,
                    f"an executor hang detection timeout of at least {2 * HANG_TIMEOUT_S:.0f} s (twice the "
                    f"hang timeout), not {hang_detection_s} s",
                )
            ]
        )
        dist = executor.dist
        replacement = fault_tolerance_replacement()
        startup_record.close()
        if torch.distributed.is_initialized():
            raise RuntimeError(
                "With Wide-EP fault tolerance no rank may create a torch.distributed process group (NCCL or Gloo) "
                "during its startup: every rank is its own process, and a rank's replacement could never join it"
            )
        balancer = _moe_load_balancer_of(executor)
        startup_memory = StartupMemory(executor.kv_cache_manager.gpu_quota)
        moe_execution_state = _moe_execution_state_of(executor)
        if replacement is None and balancer is not None:
            _check_largest_failure_fits(executor, balancer, startup_memory, moe_execution_state)
        host_weights_prefix = (
            None
            if balancer is None
            else balancer.expert_restore.keep_host_weights_for_the_deployment()
        )
        store = rendezvous_store()
        if dist.rank == 0:
            if host_weights_prefix is not None:
                store.set(HOST_WEIGHTS_PREFIX_KEY, host_weights_prefix.encode())
            agreed_startup = {**startup_record.published(), "fingerprint": _startup_fingerprint}
            store.set(AGREED_STARTUP_KEY, pickle.dumps(agreed_startup))
        recovery = cls(
            executor,
            dist.host_exchange,
            moe_execution_state,
            store,
            os.environ[RENDEZVOUS_ENV],
            startup_memory,
            _authority,
            incarnation=_client.incarnation,
        )
        if replacement is None:
            wait_for_every_rank()
        recovery.start()
        if replacement is not None:
            recovery._join()
        logger.info(f"Rank {dist.rank} joined the fault-tolerance membership at {recovery.address}")
        return recovery

    def set_outstanding_requests(self, outstanding_requests: Callable[[], Dict[int, int]]) -> None:
        """Attach the executor worker's map from request id to client id of the requests without a final response.

        When it resumes after a membership change, rank 0 fails the requests that no active rank holds.
        """
        self._outstanding_requests = outstanding_requests

    def _join(self) -> None:
        """Admit this replacement and start the executor loop at the iteration the active ranks resume at."""
        # The replacement may never see the PROPOSE that adds it, so it reports its READY payload up front.
        adding = self._record.model_copy(
            update={"state": MembershipState.PROPOSE, "joining_mask": 1 << self.rank}
        )
        self._end_eplb_iterations()
        self._client.report(joining_ready=True, **self._expert_placement(adding))
        self._change_membership()
        self._executor.iter_counter = self._resume_after + 1

    def start(self) -> None:
        """Take over the membership client this process started at the entry of its worker."""
        self._client = _client
        self._client.bind(self._on_record, on_tick=self._on_tick, on_error=self._on_client_error)

    def stop(self, completed: bool) -> None:
        """Leave the membership after a normal end, then end the process.

        The process ends without a teardown, which would wait for ranks that failed.
        """
        if completed:
            if self.rank == 0:
                self._client.report(shutdown=True)
            with self._record_changed:
                self._record_changed.wait_for(
                    lambda: self._record.state == MembershipState.SHUTDOWN, _AUTHORITY_TIMEOUT_S
                )
            self._client.stop()
            if self._authority is not None:
                try:
                    self._authority.wait(_AUTHORITY_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    self._authority.kill()
        _end_process(self.rank, 0 if completed else 1, "the executor stopped")

    def _on_record(self, record: MembershipRecord) -> None:
        with self._record_changed:
            self._record = record
            if record.admits(self.rank, self.incarnation):
                self._admitted = True
            if record.state == MembershipState.COMMITTED:
                self._committed_mask = record.active_mask
            if record.state in (MembershipState.INSTALL, MembershipState.COMMITTED):
                self._listed_incarnations = [
                    record.incarnations[r]
                    if record.is_active(r) and r < len(record.incarnations)
                    else incarnation
                    for r, incarnation in enumerate(self._listed_incarnations)
                ]
            self._pending = self._change_pending(record)
            self._record_changed.notify_all()
        if not self._admitted:
            return
        if record.state in (MembershipState.PROPOSE, MembershipState.ABORTED) or self._excluded(
            record
        ):
            if self._interrupts_steps(record):
                self._moe.request_abort()
            if self.rank == 0:  # rank 0 may be waiting for requests rather than in a collective
                self._executor.executor_request_queue.wake()
            loop = getattr(self._executor, "worker_thread", None)
            if (record.state == MembershipState.ABORTED or self._excluded(record)) and (
                loop is not None and not loop.is_alive()
            ):
                _end_process(self.rank, 1, f"the membership record is {record!r}")

    def _on_client_error(self, error: Exception) -> None:
        """Keep serving unprotected after the client lost its connection, and end any wait for the next record."""
        with self._record_changed:
            self._record_changed.notify_all()
        logger.warning(f"The membership client stopped ({error!r}). Serving continues unprotected.")

    def _on_tick(self) -> None:
        waited_rounds = self._moe.waited_rounds()
        host_waits, epoch = self._exchange.waiting_for, self._exchange.epoch
        self._client.report(
            progress=self._executor.iter_counter,
            idle=self._executor.hang_detector.paused(),
            waiting_for=host_waits
            | self._waiting_in_store
            | sum(1 << peer for peer in waited_rounds),
            waited_rounds=tuple(sorted(waited_rounds.items())),
            waited_epochs=tuple(
                (peer, epoch) for peer in range(self.world_size) if host_waits >> peer & 1
            ),
        )

    def _excluded(self, record: MembershipRecord) -> bool:
        return self._admitted and not record.is_member(self.rank, self.incarnation)

    def _interrupts_steps(self, record: MembershipRecord) -> bool:
        """Return whether ``record`` interrupts the steps in flight rather than waiting for a step boundary."""
        survivors = record.active_mask & ~record.joining_mask
        return (
            record.state == MembershipState.ABORTED
            or self._excluded(record)
            or bool(self._committed_mask & ~survivors)
        )

    def _change_pending(self, record: MembershipRecord) -> bool:
        return (
            record.state in (MembershipState.PROPOSE, MembershipState.ABORTED)
            or (record.state == MembershipState.INSTALL and record != self._installed)
            or self._excluded(record)
        )

    def _membership_changing(self) -> bool:
        """Return whether the host exchange must stop for a change this rank has not installed yet.

        A change that does not interrupt the steps waits for a forward vote that launched no forward. This way every
        active rank stops in the same round and after the same forwards.
        """
        if not self._pending:
            return False
        return (
            self._steps_quiet
            or self._resume_after is not None
            or self._interrupts_steps(self._record)
        )

    @contextlib.contextmanager
    def _hang_detection_paused(self):
        """Pause the hang detector during a membership change, whose duration the authority bounds instead."""
        detector = self._executor.hang_detector
        armed = not detector.paused()
        detector.disarm()
        try:
            yield
        finally:
            if armed:
                detector.checkpoint()

    def _on_exchange_interrupt(self) -> None:
        """Take part in the membership change the host exchange stopped for."""
        self._change_membership()
        if self._resume_after is not None and self._installed.joining_mask >> self.rank & 1:
            raise _MembershipRestarted

    def _change_membership(self) -> None:
        """Take part in the membership change until the host exchange is installed for the active ranks."""
        with self._hang_detection_paused():
            seen = self._installed
            listed = seen is not None
            while True:
                record = self._next_record(seen)
                seen = record
                if record.state == MembershipState.ABORTED:
                    _end_process(
                        self.rank, 1, "the deployment cannot recover from the rank failure"
                    )
                if record.state == MembershipState.SHUTDOWN:
                    _end_process(self.rank, 0, "the deployment is shutting down")
                if not record.is_member(self.rank, self.incarnation):
                    if self._admitted or listed:
                        _end_process(self.rank, 1, "the membership no longer includes this rank")
                    continue
                listed = True
                if record.state == MembershipState.PROPOSE:
                    self._end_eplb_iterations()
                    if self._record != record:
                        continue
                    paused_at = ResumeState(
                        iteration=self._executor.iter_counter, epoch=self._exchange.stopped_epoch
                    )
                    self._client.report(
                        reply=Reply.READY,
                        reply_generation=record.generation,
                        paused_at=paused_at,
                        **self._expert_placement(record),
                    )
                elif record.state == MembershipState.INSTALL:
                    self._client.report(ready_payload=None, joining_ready=False)
                    try:
                        self._install(record)
                    except (_MembershipRestarted, MnnvlPeerHandleError) as error:
                        logger.warning(
                            f"Rank {self.rank} waits for the next membership record: {error!r}"
                        )
                        continue
                    except Exception as error:
                        self._fail_install(record, error)
                        continue
                    return
                else:
                    return

    def _next_record(self, seen: Optional[MembershipRecord]) -> MembershipRecord:
        """Wait for a record other than ``seen``, exiting if the authority closes its connection meanwhile."""
        with self._record_changed:
            self._record_changed.wait_for(
                lambda: self._record != seen or not self._client.connected
            )
            if self._record == seen:
                _end_process(
                    self.rank,
                    1,
                    "the membership authority closed its connection during a membership change",
                )
            return self._record

    def _end_eplb_iterations(self) -> None:
        """Abandon the online EPLB iterations in progress so that the placement stays fixed until the resume point."""
        balancer = _moe_load_balancer_of(self._executor)
        if balancer is not None:
            balancer.abandon_iterations()

    def _expert_placement(self, record: MembershipRecord) -> dict:
        balancer = _moe_load_balancer_of(self._executor)
        if balancer is None:
            return {}
        view = tuple(
            (num_experts, tuple(tuple(slots) for slots in ranks))
            for num_experts, ranks in balancer.expert_restore.current_placement()
        )
        return {"ready_payload": ready_payload(record, self.rank, view)}

    def _install(self, record: MembershipRecord) -> None:
        """Map the host exchange rows among the active ranks.

        The rows move to fresh pages if a process that maps them may be gone.
        """
        gather = self._active_group(record, "host_exchange_rows").allgather
        fresh_rows = self._exchange.rows.stale_rows(gather)
        self._exchange.install(
            gather, record.active_mask & ~record.joining_mask, record.resume.epoch, fresh_rows
        )
        with self._record_changed:
            self._installed = record
            self._pending = self._change_pending(self._record)
        self._resume_after = record.resume.iteration
        self._install_step_done()

    def _install_step_done(self) -> None:
        self._install_steps += 1
        self._client.report(install_steps=self._install_steps)

    def _active_group(self, record: MembershipRecord, name: str) -> StoreGroup:
        """Return a store group of the active ranks that stops waiting once ``record`` is replaced."""

        def on_wait(waiting: int) -> None:
            self._waiting_in_store = waiting
            if waiting and self._record != record:
                raise _MembershipRestarted

        return StoreGroup(
            self._store,
            f"{name}/{record.generation}",
            self.rank,
            range(self.world_size),
            on_wait,
            active=[r for r in range(self.world_size) if record.is_active(r)],
        )

    def _fail_install(self, record: MembershipRecord, error: Exception) -> None:
        """Reply FAILED, unless only other ranks failed to map MNNVL memory and reply FAILED themselves."""
        if isinstance(error, MnnvlMappingError) and self.rank not in error.ranks:
            logger.warning(f"Rank {self.rank} waits for the next membership record: {error!r}")
            return
        logger.error(
            f"Rank {self.rank} could not install membership generation {record.generation}: {error!r}"
        )
        self._client.report(reply=Reply.FAILED, reply_generation=record.generation)

    def before_iteration(self) -> None:
        """Finish installing the membership change at the first iteration past the resume point."""
        if self._give_back_agreed:
            self._give_back_at_step_boundary()
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
                # A joining rank whose admission was committed stops for a newer change in the host exchange, as the
                # other active ranks do.
                if self._finish_install(record) or self._admitted:
                    return
                if self._installed is record:
                    self._change_membership()
                self._executor.iter_counter = self._resume_after + 1
                record = self._installed

    def _finish_install(self, record: MembershipRecord) -> bool:
        """Restore the active ranks, reply INSTALLED and return whether the change was committed."""
        executor = self._executor
        times: Dict[str, float] = {}
        start_s = time.perf_counter()
        torch.cuda.synchronize()
        try:
            # A rank may have run one more EPLB iteration since READY, in the step it stopped in.
            self._end_eplb_iterations()
            if not record.joining_mask >> self.rank & 1:
                # The retire runs host exchange collectives. The joining ranks enter the host exchange only below.
                executor._retire_inflight_batch_for_control()
            recomputed = self._recompute_interrupted_requests()
            if record.joining_mask:
                # A rank that installed a newer change meanwhile skips the admission, but every active rank still runs
                # the gather below, so that all of them stay in the same collective.
                if self._installed is record:
                    self._admit_joining_ranks(record)
                self._install_step_done()
            # Read the outstanding requests before the held ones so that a request enqueued in between does not count
            # as lost.
            outstanding = (
                self._outstanding_requests() if self._outstanding_requests is not None else {}
            )
            times["prepare"] = (time.perf_counter() - start_s) * 1e3
            start_s = time.perf_counter()
            # Past this gather no active rank has an all-to-all kernel in flight or runs another host collective before
            # the next iteration.
            gathered = executor.dist.allgather(self._held_request_ids())
            times["request gather"] = (time.perf_counter() - start_s) * 1e3
            self._install_step_done()
            digest = self._restore(record, times)
        except (_MembershipRestarted, MnnvlPeerHandleError) as error:
            logger.warning(
                f"Rank {self.rank} stopped installing membership generation {record.generation}: {error!r}"
            )
            return False
        except Exception as error:
            self._fail_install(record, error)
            self._next_record(record)
            return False
        self._client.report(
            reply=Reply.INSTALLED,
            reply_generation=record.generation,
            placement_digest=digest,
        )
        committed = self._next_record(record)
        if committed.state != MembershipState.COMMITTED:
            return False
        self._resume_after = None
        self._steps_quiet = False
        # The joining ranks read these keys before they replied INSTALLED.
        for generation in self._published_admissions:
            for key in _admission_keys(generation):
                self._store.store.delete_key(key)
        self._published_admissions.clear()
        # Serving resumes first. Every rank gives the memory back at the step boundary of the next forward vote.
        self._give_back_pending = True
        executor.adp_ctx_waiting_iters_count = executor.adp_ctx_batching_wait_iters_count = 0
        held = set().union(*(request_ids for request_ids in gathered if request_ids is not None))
        lost = {r: c for r, c in outstanding.items() if r not in held}
        for request_id, client_id in lost.items():
            self._fail(request_id, client_id, RankFailureError())
        active = [r for r in range(self.world_size) if record.is_active(r)]
        logger.info(
            f"Rank {self.rank} resumed serving with membership generation {record.generation}, ranks {active}. "
            f"{recomputed} interrupted requests continue by recompute"
            + (
                f", {len(lost)} requests no active rank holds ended by the rank failure"
                if self.rank == 0
                else ""
            )
        )
        return True

    def _give_back_kv_cache(self) -> str:
        """Release the empty spare slots, return their memory to the KV cache pool and return a log suffix."""
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
                    r: self._listed_incarnations[r]
                    for r in range(self.world_size)
                    if not record.is_active(r)
                },
                torch.cuda.mem_get_info()[0],
                time.monotonic() + _GIVE_BACK_RETRY_S,
            )
        return (
            f". Released {released} bytes of empty spare slots, KV cache quota {quota} bytes"
            + (f" ({short} bytes short until device memory is freed)" if short else "")
            + f", in {(time.perf_counter() - start_s) * 1e3:.0f} ms"
        )

    def _give_back_retry_due(self) -> bool:
        """Return whether device memory was freed since the last give-back, checked at most once a second.

        The retries end once every excluded process has ended and no more memory was freed.
        """
        retry = self._give_back_retry
        now_s = time.monotonic()
        if now_s < retry.check_after_s:
            return False
        retry.check_after_s = now_s + _GIVE_BACK_RETRY_S
        if not retry.ended:
            disconnected = json.loads(self._store.get_if_set(DISCONNECTED_KEY) or "[]")
            retry.ended = all(
                rank < len(disconnected) and disconnected[rank] >= incarnation
                for rank, incarnation in retry.excluded.items()
            )
        freed = torch.cuda.mem_get_info()[0] > retry.free_seen
        if retry.ended and not freed:
            self._give_back_retry = None
        return freed

    def _give_back_at_step_boundary(self) -> None:
        """Give back the memory of the last committed change, or retry a give-back that fell short.

        Every rank retires its step in flight, which runs collectives, even if only another rank gives memory back.
        """
        self._give_back_agreed = False
        torch.cuda.synchronize()
        self._executor._retire_inflight_batch_for_control()
        if not self._give_back_due:
            return
        self._give_back_due = False
        if self._give_back_pending:
            self._give_back_pending = False
            given_back = self._give_back_kv_cache()
            if given_back:
                logger.info(
                    f"Rank {self.rank} gave back the memory of membership generation "
                    f"{self._installed.generation}{given_back}"
                )
            return
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
            f"Rank {self.rank} took back the device memory freed since the commit, KV cache quota {quota} bytes"
            + (f", {short} bytes short" if short else "")
        )

    def _restore(self, record: MembershipRecord, times: Optional[Dict[str, float]] = None) -> str:
        """Restore the lost experts and the all-to-all workspaces.

        This is collective over the active ranks of ``record``.

        Returns:
            The digest of the restored EPLB placement, or "" without EPLB.
        """
        if self._record != record:
            raise _MembershipRestarted
        began_s = time.perf_counter()
        times = {} if times is None else times
        executor = self._executor
        balancer = _moe_load_balancer_of(executor)
        plan = None
        loaded = 0
        if balancer is not None:
            current = [np.array(ranks, dtype=np.int32) for _, ranks in agreed_placement(record)]
            plan = balancer.expert_restore.plan_expert_restore(
                [record.is_active(r) for r in range(self.world_size)],
                current,
                rejoining=[bool(record.joining_mask >> r & 1) for r in range(self.world_size)],
            )
            for planned, held in zip(plan.placements, current):
                mine, before = planned[balancer.ep_rank], held[balancer.ep_rank]
                loaded += int(np.count_nonzero((mine != before) & (mine >= 0)))
            self._install_step_done()
        for r in range(self.world_size):
            if record.is_active(r):
                self._moe.health.mark_active(r)
            else:
                self._moe.health.mark_failed(r)
        gather = self._active_group(record, "moe_a2a_workspaces").allgather
        fresh_rows = self._moe.stale_workspace_rows(gather)
        # The remap allocates fresh pages for this rank's rows before the pages they replace are freed.
        workspace_bytes = self._moe.workspace_bytes() if self.rank in fresh_rows else 0
        needed = workspace_bytes + (0 if plan is None else plan.device_bytes)

        kv_cache_quota = "unchanged"
        # Each rank decides on its own because suspending the pools runs a collective only with disaggregated serving,
        # which fault tolerance rejects.
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
            start_s = time.perf_counter()
            self._moe.remap_workspaces(gather, fresh_rows)
            times["remap"] = (time.perf_counter() - start_s) * 1e3
            self._install_step_done()
        # The client thread updates the record under this lock before it requests the abort of a newer change. Once
        # this check passes, that abort comes after the rearm and stays in effect.
        start_s = time.perf_counter()
        with self._record_changed:
            if self._record != record:
                raise _MembershipRestarted
            self._moe.rearm()
            self._moe.install_mask()
        times["rearm"] = (time.perf_counter() - start_s) * 1e3
        logger.info(
            f"Rank {self.rank} restored membership generation {record.generation} in "
            f"{(time.perf_counter() - began_s) * 1e3:.0f} ms. Loaded {loaded} experts "
            f"({0 if plan is None else plan.device_bytes} bytes), KV cache quota {kv_cache_quota}. Phases: "
            + ", ".join(f"{phase} {ms:.0f} ms" for phase, ms in times.items())
        )
        return "" if plan is None else plan.digest

    def _admit_joining_ranks(self, record: MembershipRecord) -> None:
        """Bring the joining ranks into the host exchange with rank 0's next epoch and waiting queue."""
        epoch_key, queue_key = _admission_keys(record.generation)
        executor = self._executor
        if record.joining_mask >> self.rank & 1:

            def check(_unsigned: Set[str]) -> None:
                self._waiting_in_store = 1
                if self._record != record:
                    raise _MembershipRestarted

            try:
                epoch, waiting = wait_for_keys(
                    self._store, [epoch_key, queue_key], STORE_POLL_S, check
                )
            finally:
                self._waiting_in_store = 0
            self._exchange.admit(record.active_mask, int(epoch))
            # Rank 0's queue replaces the one an earlier attempt of this admission took.
            executor.waiting_queue.remove_by_ids({item.id for item in executor.waiting_queue})
            executor.waiting_queue.add_requests(pickle.loads(waiting))  # nosec B301
            return
        epoch = next_epoch(self._exchange.epoch)
        if self.rank == 0:
            self._store.set(queue_key, pickle.dumps(list(executor.waiting_queue)))
            self._store.set(epoch_key, str(epoch).encode())
            self._published_admissions.append(record.generation)
        self._exchange.admit(record.active_mask, epoch)

    def _held_request_ids(self) -> Set[int]:
        executor = self._executor
        held = {
            request.py_request_id for request in executor.active_requests if not request.is_dummy
        }
        held.update(item.id for item in executor.waiting_queue)
        held.update(item.id for item in executor.request_accumulated)
        request_queue = executor.executor_request_queue.request_queue
        with request_queue.mutex:
            # A cancel item carries the id of the request it cancels.
            held.update(item.id for item in request_queue.queue if item.is_normal_request)
        return held

    def pauses_forward(self) -> bool:
        """Return whether this iteration runs no forward because a membership change is not committed yet."""
        return self._resume_after is not None or self._pending

    def forward_vote(
        self, batch_size: int, gather: Callable[[List[int]], np.ndarray]
    ) -> np.ndarray:
        """Run the attention-DP forward vote with the pending change and give-back flags, and return the batch sizes."""
        pending = self._pending
        held = pending or self._resume_after is not None
        self._give_back_due = not held and (
            self._give_back_pending
            or (self._give_back_retry is not None and self._give_back_retry_due())
        )
        votes = gather([0 if held else batch_size, int(pending), int(self._give_back_due)])
        self._steps_quiet = bool(votes[:, 1].any())
        self._give_back_agreed = bool(votes[:, 2].any()) and not self._steps_quiet
        if held and self.rank == 0:
            self._executor.executor_request_queue.wake()
        return votes[:, 0]

    def drop_invalid_step(self) -> None:
        """Discard the previous step if its all-to-all was aborted.

        An abort may have cut the rounds of the step short.
        """
        batch = self._executor.previous_batch
        event = batch.sample_state.sampler_event if batch.sample_state is not None else None
        if event is not None:
            event.synchronize()
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
        """Discard a forward that raised because its all-to-all was aborted.

        Returns:
            Whether the forward was discarded. False means that the forward failed for another reason.
        """
        if not self._moe.aborted():
            return False
        self._interrupted.extend(scheduled_requests.all_requests())
        self._executor._accepted_previous_batch()
        return True

    def _recompute_interrupted_requests(self) -> int:
        """Pause the requests of the discarded steps for recompute, as after a preemption, and return how many."""
        interrupted, self._interrupted = {id(request) for request in self._interrupted}, []
        executor = self._executor
        recomputed = 0
        for request in list(executor.active_requests):
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
        """Fail the waiting requests pinned strictly to an attention-DP rank that does not serve."""
        if len(rank_states) == self._executor.dist.tp_size:
            return
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
        from .llm_request import LlmResponse

        if self.rank == 0:
            response = LlmResponse(request_id=request_id, error_msg=error, client_id=client_id)
            self._executor._pending_transfer_responses.append((request_id, response))


class FaultToleranceHealth:
    """Read the membership state for the server's ``/health`` and status endpoint."""

    def __init__(self, address: str, secret: bytes):
        self._address = address
        self._secret = secret
        self._store: Optional[RendezvousStore] = None
        self._record: Optional[MembershipRecord] = None
        self._next_open = 0.0
        self._authority_ended = False
        self._lock = threading.Lock()

    def _read(self) -> Tuple[Optional[MembershipRecord], Optional[TransactionSummary], bool]:
        """Return the record, the last transaction and whether the authority answers.

        Once the store has answered, a failed read means the authority ended, so later reads return at once.
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
            if record is not None:
                self._record = record
            return self._record, self._parse(TransactionSummary, transaction), True

    @staticmethod
    def _parse(model: type, value: Optional[bytes]):
        if value is None:
            return None
        try:
            return model.model_validate_json(value)
        except ValueError as error:
            logger.error(f"Cannot parse a value of the rendezvous store: {error!r}")
            return None

    @staticmethod
    def _state(record: Optional[MembershipRecord], authority_alive: bool) -> str:
        if record is None:
            return "unprotected"
        # The authority keeps its store up after the deployment ends until every active rank has left.
        if record.state in (MembershipState.ABORTED, MembershipState.SHUTDOWN):
            return "failed"
        if not authority_alive:
            return "unprotected"
        if record.state != MembershipState.COMMITTED:
            return "recovering"
        world = (1 << len(record.incarnations)) - 1
        return "healthy" if record.active_mask == world else "degraded"

    def status(self) -> Dict[str, object]:
        """Return the server's fault-tolerance status from the last membership record and transaction."""
        record, transaction, authority_alive = self._read()
        status = {
            "state": self._state(record, authority_alive),
            "authority_alive": authority_alive,
            "generation": None,
            "active_ranks": None,
            "failed_ranks": None,
            "incarnations": None,
            "last_transaction": None
            if transaction is None
            else transaction.model_dump(mode="json"),
        }
        if record is not None:
            world_size = len(record.incarnations)
            status.update(
                generation=record.generation,
                active_ranks=[r for r in range(world_size) if record.is_active(r)],
                failed_ranks=[r for r in range(world_size) if not record.is_active(r)],
                incarnations=list(record.incarnations),
            )
        return status

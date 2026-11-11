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
    wide_ep_fault_tolerance_enabled,
    wide_ep_fault_tolerance_rank,
)
from tensorrt_llm.executor.utils import RankFailureError, RequestError
from tensorrt_llm.logger import logger

from ..distributed.host_exchange_epoch import next_epoch
from ..distributed.membership import (
    DISCONNECTED_KEY,
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
    StoreGroup,
    TransactionSummary,
    open_store,
    rendezvous_store,
    wait_for_keys,
)
from ..distributed.mnnvl_live_group import MnnvlMappingError, MnnvlPeerHandleError
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
_JOIN_SECRET_BYTES = 32


class _MembershipRestarted(Exception):
    """A newer membership change replaced the one being installed."""


_rendezvous_ports: List[socket.socket] = []
_agent: Optional[MembershipAgent] = None
_authority: Optional[subprocess.Popen] = None
_moe_execution_state: Optional[MoeExecutionState] = None


def ensure_rendezvous() -> None:
    """Export the join secret and the rendezvous address when unset.

    The two ports stay reserved by SO_REUSEADDR sockets that never listen, so the authority can still bind them."""
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
            except (OSError, OverflowError):
                for sock in reserved:
                    sock.close()
        _rendezvous_ports = reserved
        os.environ[RENDEZVOUS_ENV] = f"{socket.gethostbyname(socket.gethostname())}:{port}"


def _host_collectives_requirements(mapping: Mapping) -> List[str]:
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

    # All-to-all kernels waiting on a hung rank are released up to 1.5 hang timeouts after their wait began, when the
    # authority excludes the rank, so they must not time out first.
    kernel_timeout_s = resolve_moe_comm_timeout_budgets(os.environ).serving_seconds
    if kernel_timeout_s is None:
        kernel_timeout_s = torch.ops.trtllm.moe_a2a_get_timeout()
    missing = [] if isinstance(dist, MnnvlDist) else _host_collectives_requirements(mapping)
    checks = [
        (not missing, ", ".join(missing)),
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
            kernel_timeout_s >= 2 * PROGRESS_TIMEOUT_S,
            f"{SERVING_TIMEOUT_ENV} of at least {2 * PROGRESS_TIMEOUT_S:.0f} s (twice the hang timeout), so "
            "that an all-to-all waiting on a hung rank lasts until the rank is dropped",
        ),
    ]
    check_fault_tolerance_requirements(checks)


def _launched_rank() -> RankIdentity:
    """Return this process's rank after checking the join secret and the rendezvous address."""
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
    identity = wide_ep_fault_tolerance_rank()
    if identity is None:
        raise ValueError(
            "WideEP fault tolerance starts every rank as its own process, with "
            f"'<rank>,<world size>,<GPUs per node>,<replacement>' in {FAULT_TOLERANCE_RANK_ENV}: launch the ranks "
            "with trtllm-llmapi-launch, whose node supervisor sets it, or let the server start them on one node"
        )
    return identity


def prepare_model(model: torch.nn.Module) -> None:
    """Check, before any weight loads, that the built model supports fault tolerance, and bind its MoE layers.

    Raises:
        ValueError: The model is not supported with fault tolerance.
    """
    from ..moe.fused_moe.communication.execution_state import MoeExecutionState
    from .config_utils import is_hybrid_linear

    global _moe_execution_state
    if is_hybrid_linear(model.model_config.pretrained_config):
        raise ValueError(
            "WideEP fault tolerance does not support hybrid linear-attention models: their KV cache manager "
            "(MambaHybridCacheManagerV2) commits recurrent-state snapshots for reuse at snapshot points and when a "
            "request is freed, so it cannot hold a step's reuse commits until the step is accepted; turn it off "
            "(enable_wide_ep_fault_tolerance=False)"
        )
    _moe_execution_state = MoeExecutionState.for_model(model)
    _check_host_weights_fit(model)


def _check_host_weights_fit(model: torch.nn.Module) -> None:
    """Check that /dev/shm has room for the host weights of every expert beyond the segments already on the node."""
    from ..moe.fused_moe.eplb_expert_restore import SHARED_MEMORY_DIR, DeploymentHostCopies
    from ..moe.fused_moe.impl_blocks import MoEEplbWeightLayoutMixin

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


AGREED_STARTUP_KEY = "agreed_startup"
WORKER_ARGS_KEY = "worker_args"

_startup_fingerprint: Optional[str] = None
_worker_main_args: Optional[dict] = None
_published_record: Optional[dict] = None


_deployment_nonce: Optional[str] = None


def deployment_nonce() -> str:
    """Return a random deployment name, the same on every rank, that keeps its shared-memory segments apart."""
    global _deployment_nonce
    if _deployment_nonce is None:
        _deployment_nonce = startup_record.agree(
            "deployment_nonce",
            lambda: mpi_broadcast(secrets.token_hex(8) if mpi_rank() == 0 else None),
        )
    return _deployment_nonce


def _agreed_startup_record() -> bytes:
    return pickle.dumps({**startup_record.published(), "fingerprint": _startup_fingerprint})


def _agreed_startup() -> dict:
    global _published_record
    if _published_record is None:
        # Only processes with the join secret write the store's values (SignedStore).
        _published_record = pickle.loads(_startup_store().get(AGREED_STARTUP_KEY))  # nosec B301
    return _published_record


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
            "arguments from the rendezvous store: run `python -m tensorrt_llm._torch.pyexecutor.rank_failure_recovery`"
        )


def _worker_args_for_other_ranks() -> bytes:
    """Return rank 0's worker arguments without its IPC addresses and keys, which the store does not encrypt."""
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
    """Run a rank other than 0 with fault tolerance (this module's ``python -m`` entry).

    Raises:
        ReplacementCannotStart: A replacement does not match the deployment, or its startup failed in a way that a
            restart would repeat.
    """
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
        if identity.replacement:
            if startup_fingerprint(worker_main_args) != _agreed_startup()["fingerprint"]:
                raise ReplacementCannotStart(
                    "The replacement rank does not match the deployment it replaces a rank of (TensorRT-LLM "
                    "version or checkpoint)"
                )
            startup_record.replay(_agreed_startup())
        worker_main(**worker_main_args)
    except ReplacementCannotStart as error:
        logger.error(f"No replacement of this rank can start: {error}")
        raise


@functools.lru_cache(maxsize=1)
def _startup_store() -> SignedStore:
    return rendezvous_store(RENDEZVOUS_TIMEOUT_S)


_startup_steps = 0
_startup_waiting_for = 0


def _report_startup_wait(waiting_for: int) -> None:
    """Report each change of the peers a startup collective waits for, where 0 completes a collective."""
    global _startup_steps, _startup_waiting_for
    if waiting_for and waiting_for == _startup_waiting_for:
        return
    _startup_waiting_for = waiting_for
    if not waiting_for:
        _startup_steps += 1
    _agent.report(waiting_for=waiting_for, progress=_startup_steps)


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
        live=[identity.rank] if identity.replacement else None,
        gpus_per_node=identity.gpus_per_node,
    )
    set_mpi_comm(startup_group)
    set_mpi_world_comm(startup_group)
    if not isinstance(local_mpi_comm(), StoreGroup):
        raise RuntimeError(
            "The node's communicator was split from MPI before the startup group was set"
        )


def _join_at_entry(store: SignedStore, identity: RankIdentity) -> None:
    """Join the membership, exiting the process if the deployment ends before the executor takes the agent over."""
    global _agent
    _agent = MembershipAgent(
        identity.rank,
        store,
        _on_record_at_startup,
        replacement=identity.replacement,
        on_error=_on_error_at_startup,
    )
    _agent.start()


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
    """Report this rank started and wait until every rank has."""
    _agent.report(started=True)
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
    """This rank's GPU KV cache quota once the deployment serves, which every membership change resizes from."""

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
    """A KV cache give-back that fell short at a commit, retried as excluded processes end and free their pages."""

    free_to_keep: int
    excluded: Dict[int, int]
    free_seen: int
    check_after_s: float
    ended: bool = False


def _check_largest_failure_fits(
    executor: PyExecutor, balancer, startup_memory: StartupMemory
) -> None:
    """Check on every original rank that the KV cache pool can give up the memory to restore the largest failure."""
    needed = (
        balancer.expert_restore.largest_failure_restore_bytes()
        + _moe_execution_state.workspace_bytes()
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
    """Return the environment of the membership authority, which runs without MPI, launcher variables or a GPU."""
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
    """Start the membership authority and return it with the ``host:port`` its store listens on.

    The authority ends with this process, which holds the writing end of its standard input.
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
    """Return ``rank``'s READY payload: the whole placement from the lowest survivor, own slots from a joining rank."""
    if rank == record.lowest_survivor():
        return json.dumps(view)
    if record.joining_mask >> rank & 1:
        return json.dumps([ranks[rank] for _, ranks in view])
    return None


def agreed_placement(record: MembershipRecord) -> Optional[list]:
    """Return the placement every live rank plans the restore from, or None if a READY payload is missing.

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
        self.address = address
        self._startup_memory = startup_memory
        self._authority = authority
        self.rank = executor.dist.rank
        self.world_size = executor.dist.world_size
        self.incarnation = incarnation
        # Rank 0: request id -> client id of every request without a final response; None (no request) until the
        # executor worker attaches it, before it reports READY and so before the proxy sends any request.
        self._outstanding_requests: Optional[Callable[[], Dict[int, int]]] = None
        self._agent: Optional[MembershipAgent] = None
        self._record = MembershipRecord(
            generation=0, state=MembershipState.COMMITTED, live_mask=(1 << self.world_size) - 1
        )
        self._record_changed = threading.Condition()
        self._admitted = incarnation == 0
        self._committed_mask = self._record.live_mask
        self._installed: Optional[MembershipRecord] = None
        self._attempted: Optional[MembershipRecord] = None
        # The last iteration any survivor reached, from INSTALL until the commit. No forward runs meanwhile.
        self._resume_after: Optional[int] = None
        self._waiting_in_store = 0
        self._install_steps = 0
        self._published_admissions: List[int] = []
        self._interrupted: List[LlmRequest] = []
        self._kv_cache_free_to_keep: Optional[int] = None
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
        # A rank waits on a hung peer until the authority excludes it, up to 1.5 hang timeouts after its wait began,
        # so the executor's hang detector must not end the rank first.
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
            _moe_execution_state,
            store,
            os.environ[RENDEZVOUS_ENV],
            startup_memory,
            _authority,
            incarnation=_agent.incarnation,
        )
        if replacement is None:
            wait_for_every_rank()
        recovery.start()
        if replacement is not None:
            recovery._join()
        logger.info(f"Rank {dist.rank} joined the fault-tolerance membership at {recovery.address}")
        return recovery

    def set_outstanding_requests(self, outstanding_requests: Callable[[], Dict[int, int]]) -> None:
        """Attach the executor worker's request id -> client id of every request without a final response; rank 0
        fails those no live rank holds when it resumes after a membership change."""
        self._outstanding_requests = outstanding_requests

    def _join(self) -> None:
        """Admit this replacement and start the executor loop at the iteration the live ranks resume at."""
        # The replacement may never see the PROPOSE that adds it, so it reports its READY payload up front.
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
        """Leave the membership after a normal end, then end the process.

        The process ends without a teardown, which would wait for ranks that failed.
        """
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
            return
        if record.state in (MembershipState.PROPOSE, MembershipState.ABORT) or self._excluded(
            record
        ):
            if self._interrupts_steps(record):
                self._moe.request_abort()
            if self.rank == 0:  # rank 0 may be waiting for requests rather than in a collective
                self._executor.executor_request_queue.wake()
            loop = getattr(self._executor, "worker_thread", None)
            if (record.state == MembershipState.ABORT or self._excluded(record)) and (
                loop is not None and not loop.is_alive()
            ):
                _end_process(self.rank, 1, f"the membership record is {record!r}")

    def _on_agent_error(self, error: Exception) -> None:
        """Keep serving unprotected after the agent lost its connection, and end any wait for the next record."""
        with self._record_changed:
            self._record_changed.notify_all()
        logger.warning(f"The membership agent stopped ({error!r}). Serving continues unprotected.")

    def _on_tick(self) -> None:
        waited_rounds = self._moe.waited_rounds()
        host_waits, epoch = self._exchange.waiting_for, self._exchange.epoch
        self._agent.report(
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
        survivors = record.live_mask & ~record.joining_mask
        return (
            record.state == MembershipState.ABORT
            or self._excluded(record)
            or bool(self._committed_mask & ~survivors)
        )

    def _change_pending(self, record: MembershipRecord) -> bool:
        return (
            record.state in (MembershipState.PROPOSE, MembershipState.ABORT)
            or (record.state == MembershipState.INSTALL and record != self._installed)
            or self._excluded(record)
        )

    def _membership_changing(self) -> bool:
        """Return whether the host exchange must stop for a change this rank has not installed yet.

        A change that does not interrupt the steps waits for a forward vote that launched no forward, so that every
        live rank stops in the same round, after the same forwards.
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
        """Pause the hang detector during a membership change, which the authority bounds instead."""
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
        """Take part in the membership change until the host exchange is installed for the live ranks."""
        with self._hang_detection_paused():
            seen = self._installed
            listed = seen is not None
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
                        _end_process(self.rank, 1, "the membership no longer includes this rank")
                    continue
                listed = True
                if record.state == MembershipState.PROPOSE:
                    self._end_eplb_iterations()
                    if self._record != record:
                        continue
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
                    self._agent.report(ready_payload=None, joining_ready=False)
                    try:
                        self._install(record)
                    except (_MembershipRestarted, MnnvlPeerHandleError) as error:
                        logger.warning(
                            f"Rank {self.rank} waits for the next membership record: {error!r}"
                        )
                        continue
                    except Exception as error:  # noqa: BLE001 - a failure here must reach the authority
                        self._fail_install(record, error)
                        continue
                    return
                else:
                    return

    def _next_record(self, seen: Optional[MembershipRecord]) -> MembershipRecord:
        """Wait for a record other than ``seen``, exiting if the authority closes its connection meanwhile."""
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
        """End the online EPLB work so far, so that the placement no longer changes until the resume point."""
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
        """Map the host exchange's rows among the live ranks, on fresh pages if a process that maps them may be gone."""
        gather = self._live_group(record, "host_exchange_rows").allgather
        fresh_rows = self._exchange.rows.stale_rows(gather)
        self._exchange.install(
            gather, record.live_mask & ~record.joining_mask, record.resume.epoch, fresh_rows
        )
        with self._record_changed:
            self._installed = record
            self._pending = self._change_pending(self._record)
        self._resume_after = record.resume.iteration
        self._install_step_done()

    def _install_step_done(self) -> None:
        self._install_steps += 1
        self._agent.report(install_steps=self._install_steps)

    def _live_group(self, record: MembershipRecord, name: str) -> StoreGroup:
        """Return a store group of the live ranks that stops waiting once ``record`` is replaced."""

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
            live=[r for r in range(self.world_size) if record.is_live(r)],
        )

    def _fail_install(self, record: MembershipRecord, error: Exception) -> None:
        """Reply FAILED, unless only other ranks failed to map MNNVL memory and reply FAILED themselves."""
        if isinstance(error, MnnvlMappingError) and self.rank not in error.ranks:
            logger.warning(f"Rank {self.rank} waits for the next membership record: {error!r}")
            return
        logger.error(
            f"Rank {self.rank} could not install membership generation {record.generation}: {error!r}"
        )
        self._agent.report(reply=Reply.FAILED, reply_generation=record.generation)

    def before_iteration(self) -> None:
        """Finish installing the membership change at the first iteration past the resume point."""
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
                if self._installed is record:
                    self._change_membership()
                self._executor.iter_counter = self._resume_after + 1
                record = self._installed

    def _finish_install(self, record: MembershipRecord) -> bool:
        """Restore the live ranks, reply INSTALLED and return whether the change was committed."""
        executor = self._executor
        times: Dict[str, float] = {}
        start_s = time.perf_counter()
        torch.cuda.synchronize()
        try:
            # A rank may have run one more EPLB iteration since READY, in the step it stopped in.
            self._end_eplb_iterations()
            if not record.joining_mask >> self.rank & 1:
                # Collective over the host exchange, which the joining ranks enter only below.
                executor._retire_inflight_batch_for_control()
            recomputed = self._recompute_interrupted_requests()
            if record.joining_mask:
                # A rank that installed a newer change meanwhile skips the admission, but every live rank still runs
                # the gather below, so that all of them stay in the same collective.
                if self._installed is record:
                    self._admit_joining_ranks(record)
                self._install_step_done()
            # Taken before the held requests, so that a request enqueued in between is not taken for a lost one.
            outstanding = (
                self._outstanding_requests() if self._outstanding_requests is not None else {}
            )
            times["prepare"] = (time.perf_counter() - start_s) * 1e3
            start_s = time.perf_counter()
            # Past this gather no live rank has an all-to-all kernel in flight or runs another host collective before
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
        except Exception as error:  # noqa: BLE001 - a failure here must reach the authority
            self._fail_install(record, error)
            self._next_record(record)
            return False
        self._agent.report(
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
        given_back = self._give_back_kv_cache()
        executor.adp_ctx_waiting_iters_count = executor.adp_ctx_batching_wait_iters_count = 0
        held = set().union(*(request_ids for request_ids in gathered if request_ids is not None))
        lost = {r: c for r, c in outstanding.items() if r not in held}
        for request_id, client_id in lost.items():
            self._fail(request_id, client_id, RankFailureError())
        live = [r for r in range(self.world_size) if record.is_live(r)]
        logger.info(
            f"Rank {self.rank} resumed serving with membership generation {record.generation}, ranks {live}. "
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
        """Release the empty spare slots and give the KV cache pool back its memory, returning a log suffix."""
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

    def _retry_give_back(self) -> None:
        """Retry a KV cache give-back that fell short.

        Every rank retires its step in flight, which runs collectives, even if only another rank asked for the retry.
        """
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
            f"Rank {self.rank} took back the device memory freed since the commit, KV cache quota {quota} bytes"
            + (f", {short} bytes short" if short else "")
        )

    def _restore(self, record: MembershipRecord, times: Optional[Dict[str, float]] = None) -> str:
        """Restore the lost experts and the all-to-all workspaces, collective over the live ranks of ``record``.

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
        gather = self._live_group(record, "moe_a2a_workspaces").allgather
        fresh_rows = self._moe.stale_workspace_rows(gather)
        # The remap allocates fresh pages for this rank's rows before the pages they replace are freed.
        workspace_bytes = self._moe.workspace_bytes() if self.rank in fresh_rows else 0
        needed = workspace_bytes + (0 if plan is None else plan.device_bytes)

        def remap() -> None:
            start_s = time.perf_counter()
            self._moe.remap_workspaces(gather, fresh_rows)
            times["remap"] = (time.perf_counter() - start_s) * 1e3
            self._install_step_done()

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
            remap()
        # The agent thread updates the record under this lock before it requests the abort of a newer change, so once
        # this check passes, that abort comes after the rearm and stays.
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

    def holds_steps(self) -> bool:
        """Return whether this iteration runs no forward because a membership change is not committed yet."""
        return self._resume_after is not None or self._pending

    def forward_vote(
        self, batch_size: int, gather: Callable[[List[int]], np.ndarray]
    ) -> np.ndarray:
        """Run the attention-DP forward vote with the pending change and give-back flags, and return the batch sizes."""
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
        """Discard the previous step if the all-to-all was aborted, which may have cut its rounds short."""
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
        """Discard a forward that raised because the all-to-all was aborted, and return False if it failed by itself."""
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
    """Read the membership state for the server's ``/health`` and status endpoint."""

    def __init__(self, address: str, secret: bytes):
        self._address = address
        self._secret = secret
        self._store: Optional[SignedStore] = None
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
            logger.error(f"Cannot parse a value of the membership store: {error!r}")
            return None

    @staticmethod
    def _state(record: Optional[MembershipRecord], authority_alive: bool) -> str:
        if record is None:
            return "unprotected"
        # The authority keeps its store up after the deployment ends until every live rank has left.
        if record.state in (MembershipState.ABORT, MembershipState.SHUTDOWN):
            return "stopped"
        if not authority_alive:
            return "unprotected"
        if record.state != MembershipState.COMMITTED:
            return "recovering"
        world = (1 << len(record.incarnations)) - 1
        return "healthy" if record.live_mask == world else "degraded"

    def status(self) -> Dict[str, object]:
        """Return the server's fault-tolerance status from the last membership record and transaction."""
        record, transaction, authority_alive = self._read()
        status = {
            "state": self._state(record, authority_alive),
            "authority_alive": authority_alive,
            "generation": None,
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
                live_ranks=[r for r in range(world_size) if record.is_live(r)],
                lost_ranks=[r for r in range(world_size) if not record.is_live(r)],
                incarnations=list(record.incarnations),
            )
        return status


if __name__ == "__main__":
    # Run in the imported module, whose state the executor shares.
    from tensorrt_llm._torch.pyexecutor import rank_failure_recovery

    rank_failure_recovery.worker_node_main()

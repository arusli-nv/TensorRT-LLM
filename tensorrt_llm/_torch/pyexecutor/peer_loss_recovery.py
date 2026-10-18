# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep a WideEP PyExecutor serving after one peer process or a whole node dies."""

import base64
import collections
import dataclasses
import functools
import os
import threading
import time
from typing import (
    TYPE_CHECKING,
    Callable,
    Deque,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
)

import numpy as np
import torch

from tensorrt_llm._utils import peer_loss_recovery_enabled
from tensorrt_llm.logger import logger

from ..distributed.communicator import HostCollectiveInterrupted, MPIDist, PeerFailureWatch
from .error_classification import classify_error
from .kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from .llm_request import LlmRequest, LlmRequestState, LlmResponse
from .resource_manager import ResourceManagerType
from .sampler.penalties import has_occurrence_penalty
from .scheduler import ScheduledRequests
from .scheduler.adp_router import DefaultADPRouter

if TYPE_CHECKING:
    from tensorrt_llm.executor.recovery.client import RecoveryClient
    from tensorrt_llm.executor.recovery.protocol import Commit, Install

    from ..moe.fused_moe.communication.execution_state import MoeExecutionState
    from ..moe.fused_moe.moe_load_balancer import MoeLoadBalancer
    from .executor_request_queue import RequestQueueItem
    from .hang_evidence import HangEvidenceReporter
    from .py_executor import PyExecutor
    from .sampler import SampleState

_RECOVERY_TIMEOUT_ENV = "TLLM_FAULT_TOLERANCE_RECOVERY_TIMEOUT_S"
# A recovery takes milliseconds; the bound covers a survivor that must first finish a long step, and
# decides how long a recovery that cannot complete (e.g. without its authority) holds the job.
_RECOVERY_TIMEOUT_DEFAULT_S = 30.0
# Workers register while every rank constructs its first executor.
_REGISTRATION_TIMEOUT_S = 600.0

Responses = List[Tuple[int, LlmResponse]]
# A set of ranks lost together, as a sorted tuple: one rank or a whole node.
FailureDomain = Tuple[int, ...]
# A request and the context position and state it had before a step changed them.
StepChange = Tuple[LlmRequest, int, LlmRequestState]
_T = TypeVar("_T")
_GATHER_HISTORY = 8


def _recovery_timeout_s() -> float:
    return float(os.environ.get(_RECOVERY_TIMEOUT_ENV, _RECOVERY_TIMEOUT_DEFAULT_S))


def _encode_placements(placements: List[np.ndarray]) -> str:
    """Per-layer ``[ep_size, slots_per_rank]`` expert ids as compact text for the recovery protocol."""
    return base64.b64encode(np.asarray(placements, dtype=np.int16).tobytes()).decode()


def _decode_placements(text: str, like: List[np.ndarray]) -> List[np.ndarray]:
    """Placements encoded by ``_encode_placements``, shaped as ``like``.

    Raises:
        ValueError: ``text`` does not hold placements of that shape.
    """
    shape = np.asarray(like).shape
    flat = np.frombuffer(base64.b64decode(text, validate=True), dtype=np.int16)
    if flat.size != int(np.prod(shape)):
        raise ValueError(f"rank 0's placement has {flat.size} slots, expected shape {shape}")
    return list(flat.reshape(shape).astype(np.int32))


def failure_domains(world_size: int, gpus_per_node: int) -> List[FailureDomain]:
    """The rank sets whose loss recovery can cover: every rank but 0, then every node but node 0.

    Rank ``r`` runs on node ``r // gpus_per_node``.
    """
    nodes = [
        tuple(range(start, min(start + gpus_per_node, world_size)))
        for start in range(gpus_per_node, world_size, gpus_per_node)
    ]
    return list(dict.fromkeys([(rank,) for rank in range(1, world_size)] + nodes))


@functools.cache
def _register(
    rank: int, recoverable_domains: Tuple[FailureDomain, ...]
) -> Tuple["RecoveryClient", frozenset]:
    """Register this process with the recovery authority, once per process.

    A process constructs more than one PyExecutor (KV cache estimation, then
    serving) but registers exactly once.

    Returns:
        The client and the failure domains (rank sets) whose loss the deployment recovers from.
    """
    from tensorrt_llm.executor.recovery.client import RecoveryClient

    client = RecoveryClient.from_env(rank)
    logger.info(
        f"Rank {rank}: registering with the WideEP recovery authority; "
        f"waiting up to {_REGISTRATION_TIMEOUT_S:.0f} s for every rank"
    )
    return client, client.register(recoverable_domains, time.monotonic() + _REGISTRATION_TIMEOUT_S)


@dataclasses.dataclass
class PreparedStep:
    """A scheduled step that holds resources but whose results are not accepted yet.

    ``context_capacities`` are its context requests' KV cache capacities before it (the KV cache manager's
    ``context_capacities``); ``changes`` lists, in order, what each request had before the step changed it.
    """

    batch: ScheduledRequests
    context_capacities: Dict[int, Optional[int]]
    changes: List[StepChange] = dataclasses.field(default_factory=list)


class RecoveryHook:
    """The executor's recovery hook; this base, used when peer-loss recovery is off, does nothing."""

    enabled = False
    peer_failed = False

    def run(self, event_loop: Callable[[], None]) -> None:
        event_loop()

    def exits_alone(self, error: Exception) -> bool:
        return False

    def fetching_requests(self) -> None:
        pass

    def broadcast_requests(
        self, broadcast: Callable[[List["RequestQueueItem"]], _T], items: List["RequestQueueItem"]
    ) -> _T:
        return broadcast(items)

    def step_prepared(self, batch: ScheduledRequests) -> Optional[List[StepChange]]:
        return None

    def step_released(self) -> None:
        pass

    def commit_step(self, batch: ScheduledRequests, sample_state: Optional["SampleState"]) -> None:
        pass

    def validate_request(self, request: LlmRequest) -> None:
        pass

    def requests_routed(self, routed: Dict[int, List["RequestQueueItem"]]) -> None:
        pass

    def gathering_responses(self, responses: Iterable[Tuple[int, LlmResponse]]):
        return responses

    def responses_gathered(self, responses: Iterable[Tuple[int, LlmResponse]]) -> None:
        pass

    def defer_terminations(self, requests: Iterable[LlmRequest]) -> None:
        pass


class PeerLossRecovery(RecoveryHook):
    """Recovers a PyExecutor from the loss of one failure domain: a peer process or a whole node.

    Survivors discard the steps the loss interrupted (two under the overlap
    scheduler), agree on the new
    membership through the recovery authority (``Stopped`` -> ``Propose`` ->
    ``Ready`` -> ``Install`` -> ``Installed`` -> ``Commit``) and resume the
    same requests on their original CUDA graphs: the departed ranks are masked
    out of host collectives, EPLB routing and the MoE dispatch/combine
    descriptor. Requests the departed attention-DP ranks were serving fail.
    A failure raises into the executor's terminal path.

    The executor calls :meth:`step_prepared` after reserving a step's
    resources and :meth:`commit_step` before applying its results; a step in
    between is discarded when a peer fails. Response gathers report through
    :meth:`gathering_responses` and :meth:`responses_gathered` so that the
    responses of an interrupted gather are delivered after recovery. The rest of
    that iteration is skipped.

    The interruption is not a request error: ``HostCollectiveInterrupted``
    bypasses the executor's error handling, so it fails no request and is not
    charged to the error budget. Errors of a recovery itself are terminal.
    """

    enabled = True

    def __init__(
        self, executor: "PyExecutor", client: "RecoveryClient", admitted: frozenset
    ) -> None:
        self._executor = executor
        self._client = client
        # Failure domains (rank sets) whose loss the survivors recover from.
        self._recoverable = admitted
        self._admitted = frozenset({executor.dist.rank}) in admitted
        # Whether the survivors can recover if this rank exits: only while the
        # serving loop watches peers, and never during or after a recovery.
        self.survives_own_loss = False
        self._timeout_s = _recovery_timeout_s()
        # Steps prepared and not yet committed, oldest first: two while the overlap scheduler runs a step
        # ahead of the one whose results it applies.
        self._uncommitted_steps: Deque[PreparedStep] = collections.deque()
        self._peer_failed = threading.Event()
        self._watch: Optional[PeerFailureWatch] = None
        # Response gathers are numbered identically on every rank. A rank can
        # complete the rooted gathers of an iteration before rank 0 does, but
        # not pass the iteration's allgather, so each rank keeps its last
        # contributions, more than one iteration makes.
        self._gather_count = 0
        self._gathers: Deque[Tuple[int, Responses]] = collections.deque(maxlen=_GATHER_HISTORY)
        self._last_processed_gather = 0
        # Finished requests whose termination waits for their responses.
        self._undelivered_terminations: List[LlmRequest] = []
        # On rank 0: request id -> (attention-DP rank, routed item) of every
        # routed request that has not finished.
        self._request_owners: Optional[Dict[int, Tuple[int, "RequestQueueItem"]]] = None
        # On rank 0: id of each extra sequence of an ``n > 1`` request -> request id.
        self._parent_ids: Dict[int, int] = {}
        if executor.dist.rank == 0:
            self._request_owners = {}
        # Requests rank 0 fetched but failed to broadcast because a peer loss interrupted it.
        self._unbroadcast_requests: List["RequestQueueItem"] = []
        # On rank 0: the control items of the last request broadcast, until a later collective
        # shows that every rank received it.
        self._broadcast_control_items: List["RequestQueueItem"] = []

    @classmethod
    def create(cls, executor: "PyExecutor") -> RecoveryHook:
        """Return the recovery hook of ``executor``: the do-nothing :class:`RecoveryHook` when it is off.

        The configuration must make a discarded step re-runnable: no
        speculative decoding, guided decoding, beam search, KV connector or
        PEFT. Requests with occurrence penalties are rejected; a discarded
        step can change the samples of a seeded request.

        Raises:
            ValueError: Recovery is enabled for a configuration it does not
                support.
            RecoveryError: The recovery authority rejected or did not answer
                the registration.
        """
        if not peer_loss_recovery_enabled():
            return RecoveryHook()
        engine = executor.model_engine
        dist = executor.dist
        load_balancer = engine.moe_load_balancer
        kv_cache_manager = executor.kv_cache_manager
        # MPIDist.enable_peer_loss_recovery() checks the pure attention-DP layout.
        requirements = {
            "MPI executor": isinstance(dist, MPIDist),
            "expert TP 1": dist.mapping.moe_tp_size == 1,
            "DefaultADPRouter": type(executor.adp_router) is DefaultADPRouter,
            "no speculative decoding": executor.drafter is None
            and executor.draft_model_engine is None
            and engine.spec_config is None,
            "no guided decoding": executor.guided_decoder is None,
            "no beam search": executor.max_beam_width == 1,
            "no KV cache transceiver": executor.kv_cache_transceiver is None,
            "no KV connector": executor.kv_connector_manager is None,
            "KV Cache Manager V2": type(kv_cache_manager) is KVCacheManagerV2,
            # Discarding a later context chunk after earlier chunks committed reusable blocks
            # is not tested yet.
            "no KV block reuse": not getattr(kv_cache_manager, "enable_block_reuse", False),
            "no SWA scratch reuse": not getattr(
                kv_cache_manager, "enable_swa_scratch_reuse", False
            ),
            "no PEFT": executor.resource_manager.get_resource_manager(
                ResourceManagerType.PEFT_CACHE_MANAGER
            )
            is None,
            "no encoder-decoder cross KV cache": executor.resource_manager.get_resource_manager(
                ResourceManagerType.CROSS_KV_CACHE_MANAGER
            )
            is None,
            "no LM head TP in attention DP": not dist.mapping.enable_lm_head_tp_in_adp,
            "no conversation manager": getattr(kv_cache_manager, "conversation_manager", None)
            is None,
            # One recovery runs under one hang checkpoint and waits up to the
            # recovery timeout twice: for the device, then for the authority.
            "hang detection timeout above twice the recovery timeout": executor.hang_detector.timeout
            > 2 * _recovery_timeout_s(),
            "no sleep/wakeup": executor.llm_args.sleep_config is None,
            # Rebalancing applies a step's results outside the step loop and
            # votes with collectives that do not expect a departed rank.
            "no KV pool rebalance": not executor.enable_kv_pool_rebalance,
            "static EPLB": load_balancer is not None and load_balancer.is_static_routing(),
            "MoE execution state": engine.moe_execution_state is not None,
        }
        unmet = [name for name, met in requirements.items() if not met]
        if unmet:
            raise ValueError(
                "WideEP peer-loss recovery is enabled but the configuration "
                f"lacks: {', '.join(unmet)}"
            )
        recoverable = tuple(
            domain
            for domain in failure_domains(dist.world_size, dist.mapping.gpus_per_node)
            if load_balancer.covers_loss_of(domain)
        )
        dist.enable_peer_loss_recovery()
        client, admitted = _register(dist.rank, recoverable)
        if dist.rank == 0:
            logger.info(
                "WideEP peer-loss recovery covers the loss of one of "
                f"{sorted(sorted(domain) for domain in admitted)}"
            )
        return cls(executor, client, admitted)

    def validate_request(self, request: LlmRequest) -> None:
        """Reject a request whose state a discarded step would change.

        Raises:
            ValueError: The request uses occurrence penalties, whose counts a
                discarded step updates, or carries multimodal data under the
                overlap scheduler, whose next step releases it before that
                step is accepted, or has logits post-processors, which may keep
                state and would run again on a replayed step.
        """
        if has_occurrence_penalty(request):
            raise ValueError(
                "Repetition, presence and frequency penalties are not supported "
                "with WideEP peer-loss recovery"
            )
        if request.py_multimodal_data and not self._executor.disable_overlap_scheduler:
            raise ValueError(
                "Multimodal requests are not supported with WideEP peer-loss "
                "recovery and the overlap scheduler"
            )
        if getattr(request, "py_logits_post_processors", None):
            raise ValueError(
                "Logits post-processors are not supported with WideEP peer-loss recovery"
            )

    @property
    def peer_failed(self) -> bool:
        """Whether a peer failure was detected since the last recovery."""
        return self._peer_failed.is_set() or (self._watch is not None and self._watch.failed)

    def exits_alone(self, error: Exception) -> bool:
        """Whether this rank should exit alone after its serving loop failed with ``error``.

        Only after an error that corrupted its CUDA context, and only while the survivors can
        recover from its loss. Other errors, such as a bug every rank hits or a timeout a healthy
        rank sees when a peer dies, end the job as they do without recovery.
        """
        return self.survives_own_loss and classify_error(str(error)) == "immediate_fatal"

    # Event loop

    def run(self, event_loop: Callable[[], None]) -> None:
        """Run ``event_loop``, recovering from each peer loss it raises.

        Peers are watched only while a serving loop runs; a loss during KV
        cache estimation, which gathers responses on every rank, is terminal.
        """
        if self._executor.gather_all_responses:
            event_loop()
            return
        self._watch_peer_failures()
        self.survives_own_loss = self._admitted
        reporter = None
        try:
            reporter = self._start_hang_evidence()
            while True:
                try:
                    event_loop()
                    return
                except HostCollectiveInterrupted as error:
                    # The hang detector bounds recovery: a survivor that stops
                    # making progress, for example in Shrink, hard-kills the job.
                    with self._executor.hang_detector as hang_detector:
                        hang_detector.checkpoint()
                        self.survives_own_loss = False
                        self._recover(error)
        finally:
            if reporter is not None:
                reporter.stop()
                self._hang_client.close()
            self._watch.stop()

    def _start_hang_evidence(self) -> Optional["HangEvidenceReporter"]:
        """Start reporting heartbeats and long all-to-all waits to the authority, if enabled."""
        from tensorrt_llm.executor.recovery.client import RecoveryClient

        from .hang_evidence import HangEvidenceReporter, ProgressRowReader

        if not HangEvidenceReporter.enabled():
            return None
        executor = self._executor
        comm = executor.model_engine.moe_execution_state.progress_comm
        reader = ProgressRowReader(comm.workspace, comm.ep_rank, *comm.progress_row_offsets())
        # Its own connection: the recovery client is used only by the serving thread.
        client = self._hang_client = RecoveryClient.from_env(executor.dist.rank)
        queue = executor.executor_request_queue
        suspect_s, heartbeat_s = HangEvidenceReporter.intervals_from_env()
        reporter = HangEvidenceReporter(
            rank=executor.dist.rank,
            send=client.send,
            read_row=reader.read,
            active_ranks=lambda: executor.dist.live_ranks,
            iteration=lambda: executor.iter_counter,
            waiting=lambda: executor.dist.in_host_collective or queue.waiting_for_requests,
            suspect_s=suspect_s,
            heartbeat_s=heartbeat_s,
            receive=client.receive,
        )
        reporter.start()
        return reporter

    def _watch_peer_failures(self) -> None:
        # The watch thread sets its failed flag, then the kernels' abort
        # request, natively: a step whose device work finished while the flag
        # was clear never observed the abort.
        abort = self._executor.model_engine.moe_execution_state.abort
        self._watch = self._executor.dist.watch_peer_failures(
            self._on_peer_failure, abort.data_ptr()
        )

    def _on_peer_failure(self) -> None:
        """Release the host side; runs on the watch thread after the abort request.

        Revoking the host communicators releases every survivor blocked in a
        host collective; waking its request queue releases rank 0 waiting for
        new requests.
        """
        self._peer_failed.set()
        logger.warning(
            f"Rank {self._executor.dist.rank}: a peer failed; aborting in-flight MoE collectives"
        )
        executor = self._executor
        try:
            executor.dist.revoke()
        finally:
            if executor.dist.rank == 0:
                executor.executor_request_queue.wake()

    # Step hooks

    def step_prepared(self, batch: ScheduledRequests) -> List[StepChange]:
        """Record that ``batch`` holds resources but has no results yet.

        Returns:
            The list into which the executor records each request change the step makes.
        """
        step = PreparedStep(batch, self._executor.kv_cache_manager.context_capacities(batch))
        self._uncommitted_steps.append(step)
        return step.changes

    def step_released(self) -> None:
        """Record that the newest prepared step returned its resources without running."""
        self._uncommitted_steps.pop()

    def commit_step(self, batch: ScheduledRequests, sample_state: Optional["SampleState"]) -> None:
        """Accept the oldest prepared step, ``batch``, unless a peer failed while it ran.

        Waits for the step's device work. A peer that stops responding without
        dying is bounded by the MoE kernels' device timeout, which fails stop.

        Survivors may decide differently on the same step: one that finished it
        before the loss was reported commits it, another discards it. Both are
        correct because a step that finished without an abort received every
        active peer's completion flag, so its data came from every peer.

        Raises:
            HostCollectiveInterrupted: The step's MoE collectives may have been
                aborted; its results must be discarded.
        """
        if not self._uncommitted_steps:
            return
        if self._uncommitted_steps[0].batch is not batch:
            raise RuntimeError("Steps must be committed in the order they were prepared")
        if sample_state is not None and sample_state.sampler_event is not None:
            sample_state.sampler_event.synchronize()
        else:
            torch.cuda.current_stream().synchronize()
        if self.peer_failed:
            raise HostCollectiveInterrupted("MoE all-to-all", frozenset())
        self._uncommitted_steps.popleft()

    # Request and response hooks

    def fetching_requests(self) -> None:
        """The rank-state exchange before a fetch showed every rank received the last broadcast."""
        self._broadcast_control_items = []

    def broadcast_requests(
        self, broadcast: Callable[[List["RequestQueueItem"]], _T], items: List["RequestQueueItem"]
    ) -> _T:
        """Run the request ``broadcast`` of ``items``, keeping what a peer loss would need again.

        The requests of a broadcast a loss interrupts are admitted again after recovery. On rank
        0, the control items of a broadcast are kept until a later collective shows that every
        rank received them.
        """
        try:
            result = broadcast(items)
        except HostCollectiveInterrupted:
            self._unbroadcast_requests = list(items)
            raise
        if self._request_owners is not None:
            self._broadcast_control_items = [
                item
                for item in items
                if not item.is_normal_request
                and not item.is_profile_start_request
                and not item.is_profile_stop_request
            ]
        return result

    def requests_routed(self, routed: Dict[int, List["RequestQueueItem"]]) -> None:
        """Record the attention-DP rank serving each new request (rank 0)."""
        if self._request_owners is None:
            return
        for rank, items in routed.items():
            for item in items:
                self._request_owners[item.id] = (rank, item)
                for child_id in item.child_req_ids or ():
                    self._parent_ids[child_id] = item.id

    def gathering_responses(self, responses: Iterable[Tuple[int, LlmResponse]]) -> Responses:
        """Keep this rank's part of a response gather for redelivery; returns it as a list."""
        responses = list(responses)
        self._gather_count += 1
        self._gathers.append((self._gather_count, responses))
        return responses

    def responses_gathered(self, responses: Iterable[Tuple[int, LlmResponse]]) -> None:
        """Note that rank 0 processed the last gather; forget finished requests.

        A result is final once every sequence of the request finished, and an
        error of any sequence ends the whole request.
        """
        self._last_processed_gather = self._gather_count
        self._forget_finished(responses)

    def _forget_finished(self, responses: Iterable[Tuple[int, LlmResponse]]) -> None:
        """Forget the owners of the requests these responses finish (rank 0)."""
        if self._request_owners is None:
            return
        for _, response in responses:
            if isinstance(response, LlmResponse) and (
                response.has_error() or (response.result is not None and response.result.is_final)
            ):
                # Results carry the request id; an error can carry a sequence's own id.
                self._forget(self._parent_ids.get(response.request_id, response.request_id))

    def _answered_request_ids(self, buffered: Responses) -> Set[int]:
        """Ids of the requests this rank answered in responses not yet delivered."""
        pending = [response for _, contribution in self._gathers for response in contribution]
        return {request_id for request_id, _ in pending + buffered}

    def _forget(self, request_id: int) -> None:
        _, item = self._request_owners.pop(request_id, (None, None))
        if item is not None:
            for child_id in item.child_req_ids or ():
                self._parent_ids.pop(child_id, None)

    def defer_terminations(self, requests: Iterable[LlmRequest]) -> None:
        """Terminate ``requests`` after recovery delivers their responses."""
        self._undelivered_terminations.extend(requests)

    # Recovery

    def _recover(self, error: HostCollectiveInterrupted) -> None:
        """Continue serving without the departed peers, or raise.

        The failed set is the one ULFM's shrink agrees on; it must be an admitted failure
        domain. A peer that fails after the shrink, before ``Commit``, is not recovered:
        the survivors' collectives or the authority's deadline end the job.
        """
        from tensorrt_llm.executor.recovery.protocol import (
            Commit,
            Install,
            Installed,
            Propose,
            Ready,
            Stopped,
        )

        executor = self._executor
        dist, engine, client = executor.dist, executor.model_engine, self._client
        started = time.monotonic()
        deadline = started + self._timeout_s
        rank = dist.rank
        logger.warning(
            f"Rank {rank}: host collective interrupted ({error}); attempting peer-loss recovery"
        )
        # Release MoE collectives that may still wait for the lost peer.
        engine.moe_execution_state.request_abort()
        self._watch.stop()
        dist.revoke()
        # Shrink is a ULFM agreement among survivors, so every survivor reports
        # the same failed set even if it only observed the revocation.
        failed = dist.shrink()
        if failed not in self._recoverable:
            raise RuntimeError(
                f"The loss of ranks {sorted(failed)} is not recoverable: it is not exactly one "
                "rank or whole node, other than rank 0's, whose experts the survivors can serve"
            )
        failed_ranks = tuple(sorted(failed))
        self._join_device_work(deadline)
        # Only now can no kernel of the steps still write their KV blocks.
        discarded = len(self._uncommitted_steps)
        executor.discard_interrupted_steps(reversed(self._uncommitted_steps))
        self._uncommitted_steps.clear()

        load_balancer = engine.moe_load_balancer
        # Online EPLB: survivors may hold different placements (an update fed by an interrupted step's
        # statistics, or one more update on a rank that ran one more iteration). Once no update can still
        # publish, every survivor starts from rank 0's placement.
        online = not load_balancer.is_static_routing()
        if online and not load_balancer.quiesce(max(0.0, deadline - time.monotonic())):
            raise RuntimeError("EPLB weight updates did not finish within the recovery timeout")
        placement = (
            _encode_placements(load_balancer.current_placements()) if online and rank == 0 else ""
        )
        client.send(Stopped(rank=rank, failed_ranks=failed_ranks, placement=placement))
        # The authority's deadline starts no later than this Stopped arrives and is shorter, so a recovery it cannot
        # complete reaches every survivor as Abort rather than as a timeout here.
        deadline = time.monotonic() + self._timeout_s
        proposal = client.wait(Propose, deadline)
        if (
            proposal.failed_ranks != failed_ranks
            or rank not in proposal.survivors
            or not failed.isdisjoint(proposal.survivors)
        ):
            raise RuntimeError(
                f"Rank {rank} observed the loss of ranks {list(failed_ranks)} but the "
                f"authority proposed {proposal}"
            )
        generation = proposal.generation

        base = None
        if online:
            base = _decode_placements(proposal.base_placement, load_balancer.current_placements())
        placements = load_balancer.recovery_placements(failed, base)
        placement_digest = load_balancer.placement_digest(failed, placements)
        client.send(Ready(rank=rank, generation=generation, placement_digest=placement_digest))
        _expect_generation(client.wait(Install, deadline), generation)

        dist.install_survivors()
        install_recovery_placement(
            engine.moe_execution_state, load_balancer, failed, placements, own_rank=rank
        )
        if online:
            # Every survivor updates the same layers from the same statistics from here on.
            load_balancer.reset_update_schedule()
        failed_requests = self._departed_requests(failed)
        client.send(Installed(rank=rank, generation=generation, failed_requests=failed_requests))
        _expect_generation(client.wait(Commit, deadline), generation)

        # What each survivor holds: the requests it serves and those it already answered (for example
        # rejected ones); none of them may be admitted again. A loss can stop survivors one iteration apart, so
        # they also resume from the same iteration count, which iteration-gated collectives assume.
        buffered = executor.take_buffered_responses()
        gathered = dist.allgather(
            (
                executor.served_request_ids() | self._answered_request_ids(buffered[0]),
                executor.iter_counter,
            ),
            departed=None,
        )
        held = [entry and entry[0] for entry in gathered]
        executor.iter_counter = max(entry[1] for entry in gathered if entry)
        self._requeue_unscheduled_requests(held)
        self._deliver_responses(failed, buffered)
        self._readmit_undelivered_requests(failed, held)
        self._peer_failed.clear()
        self._watch_peer_failures()
        logger.warning(
            f"Rank {rank}: resumed serving without ranks {list(failed_ranks)} at membership "
            f"generation {generation}, {(time.monotonic() - started) * 1e3:.0f} ms after the "
            f"interruption, discarding {discarded} steps; {len(dist.live_ranks)} ranks remain and "
            "a further loss is not recoverable"
        )

    def _join_device_work(self, deadline: float) -> None:
        """Wait for all work on this device, on any stream, up to ``deadline``."""
        device = torch.cuda.current_device()
        done = threading.Event()
        errors: List[BaseException] = []

        def synchronize() -> None:
            try:
                torch.cuda.synchronize(device)
            except BaseException as error:  # noqa: BLE001 - re-raised on the recovering thread
                errors.append(error)
            done.set()

        threading.Thread(target=synchronize, name="peer_loss_device_join", daemon=True).start()
        if not done.wait(max(0.0, deadline - time.monotonic())):
            raise TimeoutError("Timed out joining device work for peer-loss recovery")
        if errors:
            raise errors[0]

    def _departed_requests(self, failed: frozenset) -> int:
        """How many requests the departed ranks were serving; known on rank 0 only (0 elsewhere)."""
        if self._request_owners is None:
            return 0
        return sum(owner in failed for owner, _ in self._request_owners.values())

    def _requeue_unscheduled_requests(self, served: List[Optional[Set[int]]]) -> None:
        """Admit again the requests that were received but never scheduled.

        A peer loss can interrupt the broadcast of new requests, leaving them on some ranks and
        not others. Rank 0 holds every such request, so it broadcasts again the ones no survivor
        started, in arrival order, and records the owner of the ones a survivor did start; the
        other ranks drop theirs. Cancellations and shutdown of that broadcast are sent again after
        them; applying them twice has no further effect. Profiler items are not.

        Args:
            served: Per rank, the ids of the requests it serves (None for a departed rank).

        Raises:
            RuntimeError: A control request may not have reached every rank.
        """
        unscheduled = [*self._executor.take_waiting_requests(), *self._unbroadcast_requests]
        self._unbroadcast_requests = []
        control_items, self._broadcast_control_items = self._broadcast_control_items, []
        if any(item.is_control_request for item in control_items):
            raise RuntimeError("A peer loss interrupted the broadcast of a control request")
        if self._request_owners is None:
            return
        started = {
            rank: items
            for rank, ids in enumerate(served)
            if ids and (items := [item for item in unscheduled if item.id in ids])
        }
        self.requests_routed(started)
        started_ids = {item.id for items in started.values() for item in items}
        self._executor.readmit_requests(
            [*(item for item in unscheduled if item.id not in started_ids), *control_items]
        )

    def _deliver_responses(
        self, failed: frozenset, buffered: Tuple[Responses, List[LlmRequest]]
    ) -> None:
        """Deliver what the interrupted gathers owed their clients.

        Collective over the survivors. Every survivor resends its parts of the
        gathers rank 0 did not process, and the ``buffered`` error responses it
        held for the next synchronized flush, then terminates the requests they
        finished. Rank 0 also fails the requests that were running on a departed
        attention-DP rank: their KV cache left with the rank.
        """
        executor = self._executor
        processed = executor.dist.broadcast(self._last_processed_gather, root=0)
        # Every rank numbers the gathers alike because each calls them in the
        # same sequence, and rank 0 processes a gather only once every rank sent.
        if self._gather_count < processed:
            raise RuntimeError(
                f"Rank 0 processed response gather {processed}, but this rank "
                f"made only {self._gather_count}"
            )
        if self._gathers and self._gathers[0][0] > processed + 1:
            raise RuntimeError(
                f"Response gather {processed + 1} is owed but no longer held for redelivery"
            )
        responses = [
            response
            for number, contribution in self._gathers
            if number > processed
            for response in contribution
        ]
        self._gathers.clear()
        # A rank may have completed gathers rank 0 did not: number them like rank 0 from here on.
        self._gather_count = self._last_processed_gather = processed
        buffered_responses, buffered_terminations = buffered
        responses += buffered_responses
        self._undelivered_terminations += buffered_terminations
        if self._request_owners is not None:
            for request_id, (owner, item) in list(self._request_owners.items()):
                if owner in failed:
                    self._forget(request_id)
                    responses.append(
                        (
                            request_id,
                            LlmResponse(
                                request_id=request_id,
                                error_msg=(
                                    f"Attention DP rank {owner} failed while "
                                    "serving this request; retry it"
                                ),
                                client_id=getattr(item.request, "client_id", None),
                            ),
                        )
                    )
        terminations, self._undelivered_terminations = self._undelivered_terminations, []
        executor.publish_responses(responses, terminations)
        self._forget_finished(responses)

    def _readmit_undelivered_requests(
        self, failed: frozenset, held: Sequence[Optional[Set[int]]]
    ) -> None:
        """Re-admit requests that rank 0 routed to a survivor which never received them.

        Collective over the survivors. A loss can interrupt the broadcast of new
        requests after rank 0 completed its part, so rank 0 can route a request
        that its owner never received. The owner never started it, so rank 0
        admits it again. Runs after redelivery, which retires finished requests.

        Args:
            held: Per rank, the ids of the requests it serves or answered (None for a departed rank).
        """
        if self._request_owners is None:
            return
        missing = [
            item
            for request_id, (owner, item) in self._request_owners.items()
            if owner not in failed and request_id not in held[owner]
        ]
        for item in missing:
            self._forget(item.id)
        if missing:
            logger.warning(f"Re-admitting {len(missing)} requests that never reached their rank")
            self._executor.readmit_requests(missing)


def install_recovery_placement(
    state: "MoeExecutionState",
    load_balancer: "MoeLoadBalancer",
    failed_ranks: Iterable[int],
    placements: List["np.ndarray"],
    own_rank: int,
) -> None:
    """Serve without ``failed_ranks``: route MoE traffic by ``placements``, then mask the ranks.

    ``placements`` come from ``MoeLoadBalancer.recovery_placements``: resident
    replicas, plus experts restored into survivors' slots from host weights.
    Only while the model is idle. Routing must exclude the failed ranks before
    their mask is installed, because the MoE kernels silently skip routes to a
    masked rank. The caller has validated the placements (their digest, sent
    with Ready); the ranks are validated here before anything changes. An error
    after that (from the native routing update, a slot load or the device)
    leaves the state mixed and ends the job.

    Raises:
        RuntimeError: A rank is ``own_rank`` or had already failed.
        ValueError: A rank is out of range.
    """
    failed = sorted(set(failed_ranks))
    for rank in failed:
        if rank == own_rank:
            raise RuntimeError(f"EP rank {rank} is this rank")
        if not state.health.is_active(rank):
            raise RuntimeError(f"EP rank {rank} is already failed")
    load_balancer.install_recovery_placement(failed, placements)
    for rank in failed:
        state.health.mark_failed(rank)
    state.rearm()
    state.install_mask()


def _expect_generation(message: "Install | Commit", generation: int) -> None:
    if message.generation != generation:
        raise RuntimeError(f"Expected recovery generation {generation}, received {message}")

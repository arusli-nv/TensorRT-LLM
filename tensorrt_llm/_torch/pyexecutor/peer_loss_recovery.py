# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep a WideEP PyExecutor serving after peer processes or whole nodes die, one loss after another."""

import atexit
import collections
import contextlib
import dataclasses
import functools
import socket
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
    Union,
)

import numpy as np
import torch

from tensorrt_llm._utils import (
    FAULT_TOLERANCE_AUTHORITY_ENV,
    FAULT_TOLERANCE_MODE_ENV,
    fault_tolerance_mode,
    is_mpi_peer_failure,
    peer_loss_recovery_enabled,
)
from tensorrt_llm.executor.recovery.config import CONFIG_ENV as FAULT_TOLERANCE_CONFIG_ENV
from tensorrt_llm.executor.recovery.config import fault_tolerance_config
from tensorrt_llm.logger import logger

from ..distributed.communicator import (
    Distributed,
    HostCollectiveInterrupted,
    MPIDist,
    PeerFailureWatch,
)
from .error_classification import classify_error
from .kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from .llm_request import (
    LlmRequest,
    LlmRequestState,
    LlmResponse,
    PyResult,
)
from .resource_manager import ResourceManagerType
from .sampler.penalties import has_occurrence_penalty
from .scheduler import ScheduledRequests
from .scheduler.adp_router import DefaultADPRouter

if TYPE_CHECKING:
    from mpi4py import MPI

    from tensorrt_llm.executor.recovery.client import RecoveryClient
    from tensorrt_llm.executor.recovery.protocol import Commit, Install, Propose

    from ..moe.fused_moe.communication.execution_state import MoeExecutionState
    from ..moe.fused_moe.moe_load_balancer import MoeLoadBalancer
    from .executor_request_queue import RequestQueueItem
    from .hang_evidence import HangEvidenceReporter, ProgressRowReader
    from .py_executor import PyExecutor
    from .sampler import SampleState

# Workers register while every rank constructs its first executor.
_REGISTRATION_TIMEOUT_S = 600.0

Responses = List[Tuple[int, LlmResponse]]
# One rank's part of a response gather: ``(responses, finished, streaming, progressed)``, or ``()`` when it has
# no responses and no progress to report. ``finished``: ids of the responses that are an error or a final result.
# ``streaming``: ids of the other responses whose request this rank had not answered yet. ``progressed``: ids of
# the interrupted requests that committed a step since this rank's last gather. Rank 0 reads these instead of
# every response it gathers. A plain tuple keeps rank 0's unpickling of every part cheap.
ResponseContribution = Union[Tuple[()], Tuple[Responses, List[int], List[int], List[int]]]
# A set of ranks lost together, as a sorted tuple: one rank or a whole node.
FailureDomain = Tuple[int, ...]
# A request and the context position, state, draft tokens and exclusion of its last generation logits it had
# before a step changed them.
StepChange = Tuple[LlmRequest, int, LlmRequestState, List[int], bool]
_T = TypeVar("_T")
# Free device memory left after ghost slots are mapped: the KV cache rounds its quota to its chunk size per pool.
_GHOST_MARGIN_BYTES = 256 << 20
# How often an idle executor checks whether the failed ranks' replaced workspace pages were freed.
_REMAP_RETURN_INTERVAL_S = 1.0
# Iterations online EPLB only gathers statistics after a recovery, so its first placements rest on many steps.
_EPLB_WARM_UP_ITERS_AFTER_RECOVERY = 20


def _free_device_bytes() -> int:
    return torch.cuda.mem_get_info()[0]


def _tested_spec_dec(spec_config) -> bool:
    """No speculative decoding, or one-model MTP with a linear draft and no in-graph draft state."""
    if spec_config is None:
        return True
    mode = getattr(spec_config, "spec_dec_mode", None)
    return (
        mode is not None
        and mode.is_mtp_eagle_one_model()
        and not getattr(spec_config, "use_dynamic_tree", False)
        and not getattr(spec_config, "use_relaxed_acceptance_for_thinking", False)
        and getattr(spec_config, "sa_config", None) is None
    )


def failure_domains(hosts: Sequence[str], failure_domain: str) -> List[FailureDomain]:
    """The rank sets to recover from: every rank but 0, then for ``node`` every node but rank 0's.

    Args:
        hosts: Per rank, the host it runs on.
        failure_domain: ``rank`` or ``node`` (FaultToleranceConfig.failure_domain).
    """
    domains = [(rank,) for rank in range(1, len(hosts))]
    if failure_domain == "node":
        nodes: Dict[str, List[int]] = {}
        for rank, host in enumerate(hosts):
            nodes.setdefault(host, []).append(rank)
        domains += [tuple(ranks) for host, ranks in nodes.items() if host != hosts[0]]
    return list(dict.fromkeys(domains))


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


# Ends the job on a peer failure until a serving loop watches peers.
_startup_watch: Optional[PeerFailureWatch] = None


def watch_startup_peer_failures(dist: Distributed) -> None:
    """End the job on a peer failure from now until a serving loop watches peers.

    Under ULFM the MPI runtime no longer ends the job when a process dies, and no loss before
    serving (model loading, warmup, CUDA graph capture, KV cache estimation) is recovered: without
    this, the survivors would wait out their timeouts. Collective over the world; does nothing
    when peer-loss recovery is off.

    Raises:
        ValueError, RuntimeError: See :meth:`MPIDist.enable_peer_loss_recovery`.
    """
    global _startup_watch
    if _startup_watch is None and peer_loss_recovery_enabled() and isinstance(dist, MPIDist):
        from tensorrt_llm._utils import mpi_comm

        # Checks ULFM and MPI_THREAD_MULTIPLE, which the watch thread needs.
        dist.enable_peer_loss_recovery()
        # The watch thread has no thread-local communicator of its own.
        _startup_watch = dist.watch_peer_failures(
            functools.partial(_end_job_before_serving, mpi_comm())
        )
        # Python's exit handlers run before MPI finalizes, which must not race the watch thread.
        atexit.register(stop_startup_watch)


def _end_job_before_serving(communicator: "MPI.Comm") -> None:
    from .hang_detector import propagate_hard_kill

    logger.error("A peer failed before serving started: ending the job")
    propagate_hard_kill(diagnostics=False, communicator=communicator)


def stop_startup_watch() -> None:
    """Stop the watch of :func:`watch_startup_peer_failures`, if it runs."""
    global _startup_watch
    if _startup_watch is not None:
        _startup_watch.stop()
        _startup_watch = None


@dataclasses.dataclass
class PreparedStep:
    """A scheduled step that holds resources but whose results are not accepted yet.

    ``context_capacities`` are its context requests' KV cache capacities before it (the KV cache manager's
    ``context_capacities``); ``changes`` lists, in order, what each request had before the step changed it;
    ``results`` pairs each result the step appends logits or additional outputs to with its checkpoint.
    """

    batch: ScheduledRequests
    context_capacities: Dict[int, Optional[int]]
    changes: List[StepChange] = dataclasses.field(default_factory=list)
    results: List[Tuple[PyResult, tuple]] = dataclasses.field(default_factory=list)

    @staticmethod
    def change(request: LlmRequest) -> StepChange:
        """What ``request`` has before a step changes it."""
        return (
            request,
            request.context_current_position,
            request.state,
            request.py_draft_tokens,
            request.py_result.exclude_last_generation_logits,
        )


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

    def step_outputs_appending(self, batch: ScheduledRequests) -> None:
        pass

    def commit_step(self, batch: ScheduledRequests, sample_state: Optional["SampleState"]) -> None:
        pass

    def validate_request(self, request: LlmRequest) -> None:
        pass

    def requests_routed(self, routed: Dict[int, List["RequestQueueItem"]]) -> None:
        pass

    def requests_cancelled(self, request_ids: Set[int]) -> None:
        pass

    def gathering_responses(self, responses: Iterable[Tuple[int, LlmResponse]]):
        return responses

    def contribution_responses(self, contributions: Sequence[object]) -> Sequence[object]:
        return contributions

    def responses_gathered(
        self,
        responses: Iterable[Tuple[int, LlmResponse]],
        contributions: Optional[Sequence[Optional[Responses]]] = None,
    ) -> None:
        pass

    def defer_terminations(self, requests: Iterable[LlmRequest]) -> None:
        pass

    def serving_idle(self) -> None:
        pass


class PeerLossRecovery(RecoveryHook):
    """Recovers a PyExecutor from the loss of failure domains (a peer process or a whole node), one after another.

    Survivors discard the steps the loss interrupted, agree on the new membership with the recovery
    authority and resume the same requests on their original CUDA graphs without the departed ranks.
    Requests of the departed attention-DP ranks run again or fail (see ``_deliver_responses``).

    The executor calls :meth:`step_prepared` after reserving a step's resources and :meth:`commit_step`
    before applying its results; :meth:`gathering_responses` and :meth:`responses_gathered` let an
    interrupted gather be delivered after recovery. ``HostCollectiveInterrupted`` bypasses the executor's
    error handling; errors of a recovery itself are terminal.
    """

    enabled = True

    def __init__(
        self,
        executor: "PyExecutor",
        client: "RecoveryClient",
        admitted: frozenset,
    ) -> None:
        self._executor = executor
        self._client = client
        # Failure domains (rank sets) whose loss the survivors recover from.
        self._recoverable = admitted
        self._admitted = frozenset({executor.dist.rank}) in admitted
        # Whether the survivors can recover if this rank exits: only while the
        # serving loop watches peers, never during a recovery, and after one only
        # while their slots cover its loss too.
        self.survives_own_loss = False
        self._timeout_s = fault_tolerance_config().recovery_timeout_s
        self._shrink_timeout_s = fault_tolerance_config().shrink_timeout_s()
        # KV cache bytes given to the ghost slots of the plan being prepared (returned if it is abandoned).
        self._kv_lent = 0
        # KV cache bytes given to remapped all-to-all workspaces while the failed ranks' pages were still held
        # (returned once the memory is free again: at the commit, or later when the executor is idle).
        self._remap_lent = 0
        # When an idle executor next checks whether that memory is free.
        self._next_remap_return = 0.0
        # Hang detection's reader of this rank's all-to-all progress row, paused while the row is remapped.
        self._row_reader: Optional["ProgressRowReader"] = None
        # During a recovery, the log records it defers: a log write can stall (a slow file system), and every survivor
        # waits for the slowest at each collective.
        self._deferred_logs: Optional[List[Tuple[Callable[[str], None], str]]] = None
        # The last committed membership generation, and whether rank 0 still logs a recovery's first step.
        self._generation = 0
        self._first_step_pending = False
        # Steps prepared and not yet committed, oldest first: two while the overlap scheduler runs a step
        # ahead of the one whose results it applies.
        self._uncommitted_steps: Deque[PreparedStep] = collections.deque()
        # Whether a request admitted so far returns logits or additional outputs, which steps then checkpoint.
        self._outputs_requested = False
        self._peer_failed = threading.Event()
        self._watch: Optional[PeerFailureWatch] = None
        # The native heartbeat (HeartbeatSender) while hang detection reports.
        self._heartbeat = None
        # Response gathers are numbered identically on every rank. A rank can
        # complete the rooted gathers of an iteration before rank 0 processes
        # them, but not pass the next rank-state exchange: each rank keeps its
        # contributions since that exchange (see fetching_requests).
        self._gather_count = 0
        self._gathers: Deque[Tuple[int, Responses]] = collections.deque()
        self._last_processed_gather = 0
        # Finished requests whose termination waits for their responses.
        self._undelivered_terminations: List[LlmRequest] = []
        # On rank 0: request id -> (attention-DP rank, routed item) of every
        # routed request that has not finished.
        self._request_owners: Optional[Dict[int, Tuple[int, "RequestQueueItem"]]] = None
        # On rank 0: id of each extra sequence of an ``n > 1`` request -> request id.
        self._parent_ids: Dict[int, int] = {}
        # On rank 0: the owned requests that delivered a response.
        self._delivered: Set[int] = set()
        # On rank 0: the owned requests a loss interrupted (in a discarded step, or on the lost rank) and that have
        # not committed a step since; another interruption fails them.
        self._interrupted: Set[int] = set()
        # This rank's requests a loss interrupted, until one of their steps commits; then the ones that did: until
        # the next response gather tells rank 0, and until the next recovery, in case that gather was interrupted.
        self._unprogressed: Set[int] = set()
        self._progress_report: List[int] = []
        self._progressed: Set[int] = set()
        # The interrupted requests rank 0 re-admitted to run again, until routed or cancelled: the rank they are
        # routed to adds them to its ``_unprogressed``.
        self._rerouting: Set[int] = set()
        # On rank 0: the owned requests whose cancellation was broadcast, until their final response.
        self._cancelled: Set[int] = set()
        # Ids of the requests this rank answered, until their error or final result.
        self._answering: Set[int] = set()
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

        The configuration must make a discarded step re-runnable: no guided decoding, beam search, KV
        connector or PEFT. Any speculative decoding method is accepted: a discard restores the draft tokens
        and reverts the draft KV, and target verification keeps the output exact; a method other than
        linear one-model MTP is untested and logs a warning. Requests with occurrence penalties are
        rejected; a discarded step can change the samples of a seeded request.

        Raises:
            ValueError: Recovery is enabled for a configuration it does not
                support.
            RecoveryError: The recovery authority rejected or did not answer
                the registration.
        """
        if not peer_loss_recovery_enabled():
            if fault_tolerance_mode() and executor.dist.rank == 0:
                logger.warning(
                    f"{FAULT_TOLERANCE_MODE_ENV}=1 without {FAULT_TOLERANCE_AUTHORITY_ENV}: the "
                    "WideEP rank mask and its cost are on, but a peer loss is not recovered; set "
                    f"{FAULT_TOLERANCE_AUTHORITY_ENV} to the recovery authority's address to recover"
                )
            return RecoveryHook()
        from ..attention.backends.sparse.dsa import DSACacheManagerV2

        engine = executor.model_engine
        dist = executor.dist
        load_balancer = engine.moe_load_balancer
        kv_cache_manager = executor.kv_cache_manager
        config = fault_tolerance_config()
        min_hang_timeout_s = config.shrink_timeout_s() + 2 * config.recovery_timeout_s
        # MPIDist.enable_peer_loss_recovery() checks the pure attention-DP layout.
        requirements = {
            "MPI executor": isinstance(dist, MPIDist),
            "expert TP 1": dist.mapping.moe_tp_size == 1,
            "DefaultADPRouter": type(executor.adp_router) is DefaultADPRouter,
            "no guided decoding": executor.guided_decoder is None,
            "no beam search": executor.max_beam_width == 1,
            "no KV cache transceiver": executor.kv_cache_transceiver is None,
            "no KV connector": executor.kv_connector_manager is None,
            # Managers whose step allocation and revert are KVCacheManagerV2's own (DSA keeps its indexer K
            # cache as another buffer of the same pages).
            "KV Cache Manager V2": type(kv_cache_manager) in (KVCacheManagerV2, DSACacheManagerV2)
            and isinstance(executor.draft_kv_cache_manager, (type(None), KVCacheManagerV2)),
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
            # Each shrink of a recovery starts a hang checkpoint: the survivors agree on the failed ranks (a hung
            # peer stalls them until it is confirmed and fenced), then wait up to the recovery timeout twice:
            # for the device, then for the authority. The executor's own hang detector must not end them first.
            f"hang_detection_timeout ({executor.hang_detector.timeout} s) above {min_hang_timeout_s:g} s, the "
            "shrink timeout plus twice recovery_timeout_s (lower the timeouts in "
            f"{FAULT_TOLERANCE_CONFIG_ENV})": executor.hang_detector.timeout > min_hang_timeout_s,
            "no sleep/wakeup": executor.llm_args.sleep_config is None,
            # commit_step waits for a step's sampler event; only generation samplers record one.
            "a generation sampler": executor.sampler.is_generation_model(),
            # Rebalancing applies a step's results outside the step loop and
            # votes with collectives that do not expect a departed rank.
            "no KV pool rebalance": not executor.enable_kv_pool_rebalance,
            # The event flush gathers over every attention-DP rank without a departed rank.
            "no KV cache events": not executor.enable_kv_cache_events,
            "an EPLB load balancer": load_balancer is not None,
            "MoE execution state": engine.moe_execution_state is not None,
        }
        unmet = [name for name, met in requirements.items() if not met]
        if unmet:
            raise ValueError(
                "WideEP peer-loss recovery is enabled but the configuration "
                f"lacks: {', '.join(unmet)}"
            )
        if dist.rank == 0 and not (
            executor.drafter is None
            and executor.draft_model_engine is None
            and _tested_spec_dec(engine.spec_config)
        ):
            logger.warning(
                "WideEP peer-loss recovery is untested with "
                f"{type(engine.spec_config).__name__}: recovery discards drafts and reverts draft KV; "
                "method-specific in-graph draft state (relaxed-acceptance delta, suffix automaton, "
                "dynamic tree) is not reset and may lower acceptance after a recovery"
            )
        # Nodes from the hosts the ranks run on, not from the rank order (a launcher may map ranks round-robin).
        hosts = dist.allgather(socket.gethostname())
        domains = failure_domains(hosts, config.failure_domain)
        uncovered = [domain for domain in domains if not load_balancer.covers_loss_of(domain)]
        if uncovered:
            raise ValueError(
                f"WideEP peer-loss recovery with failure_domain={config.failure_domain} needs EPLB slots that "
                f"serve every expert without each domain; the survivors cannot after losing {uncovered[0]} "
                "(with s slots per rank, N ranks and K ranks lost, s * (N - K) must reach the expert count)"
            )
        recoverable = tuple(domains)
        dist.enable_peer_loss_recovery()
        client, admitted = _register(dist.rank, recoverable)
        if dist.rank == 0:
            logger.info(
                "WideEP peer-loss recovery covers the loss of one of "
                f"{sorted(sorted(domain) for domain in admitted)}"
            )
        return cls(executor, client, admitted)

    def validate_request(self, request: LlmRequest) -> None:
        """Reject a request whose state a discarded step would change; note one that returns logits or
        additional outputs.

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
        if (
            request.py_return_context_logits
            or request.py_return_generation_logits
            or request.py_additional_outputs
        ):
            self._outputs_requested = True

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

        Peers are watched for recovery only while a serving loop runs; a loss
        before (for example during KV cache estimation, which gathers responses
        on every rank) ends the job (see :func:`watch_startup_peer_failures`).
        """
        if self._executor.gather_all_responses:
            event_loop()
            return
        self._watch_peer_failures()
        stop_startup_watch()
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
                        self._deferred_logs = []
                        try:
                            self._recover(error)
                        finally:
                            deferred, self._deferred_logs = self._deferred_logs, None
                            for log, message in deferred:
                                log(message)
                        self.survives_own_loss = self._own_loss_covered()
        finally:
            # A reporter thread that did not stop may still use its connection.
            if reporter is not None and reporter.stop():
                self._hang_client.close()
            if self._heartbeat is not None:
                self._heartbeat.stop()
                self._heartbeat = None
            self._watch.stop()

    def _own_loss_covered(self) -> bool:
        """Whether the survivors could recover if this rank were lost too, after the ranks lost so far."""
        if not self._admitted:
            return False
        dist = self._executor.dist
        lost = frozenset(range(dist.world_size)) - frozenset(dist.live_ranks) | {dist.rank}
        return self._executor.model_engine.moe_load_balancer.covers_loss_of(lost)

    def _start_hang_evidence(self) -> Optional["HangEvidenceReporter"]:
        """Start reporting heartbeats and long all-to-all waits to the authority, if enabled."""
        from tensorrt_llm._utils import mpi_comm
        from tensorrt_llm.executor.recovery.client import RecoveryClient

        from ..distributed.pmix_fence import pmix_ranks
        from .hang_evidence import HangEvidenceReporter, ProgressRowReader, start_native_heartbeat

        config = fault_tolerance_config()
        if not config.hang_detection:
            return None
        executor = self._executor
        graphs = executor.model_engine.cuda_graph_runner
        if graphs is not None and graphs._capture_allowed:
            raise RuntimeError("CUDA graphs are captured only before serving")
        comm = executor.model_engine.moe_execution_state.progress_comm
        reader = self._row_reader = ProgressRowReader(
            comm.workspace, comm.ep_rank, *comm.progress_row_offsets()
        )
        # Its own connection: the recovery client is used only by the serving thread.
        client = self._hang_client = RecoveryClient.from_env(executor.dist.rank)
        queue = executor.executor_request_queue
        reporter = HangEvidenceReporter(
            rank=executor.dist.rank,
            send=client.send,
            read_row=reader.read,
            active_ranks=lambda: executor.dist.live_ranks,
            iteration=lambda: executor.iter_counter,
            waiting=lambda: executor.dist.in_host_collective or queue.waiting_for_requests,
            suspect_s=config.hang_suspect_s,
            heartbeat_s=config.heartbeat_s,
            receive=client.receive,
            pmix_ranks=pmix_ranks(mpi_comm()),
        )
        # The reporter's first heartbeat names this process, whose native heartbeats the authority then accepts.
        reporter.start()
        self._heartbeat = start_native_heartbeat(executor.dist.rank, config.heartbeat_s)
        if self._heartbeat is not None:
            self._heartbeat.publish(executor.iter_counter)
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
        executor = self._executor
        try:
            executor.dist.revoke()
        finally:
            if executor.dist.rank == 0:
                executor.executor_request_queue.wake()
        logger.warning(
            f"Rank {self._executor.dist.rank}: a peer failed; aborting in-flight MoE collectives"
        )

    # Step hooks

    def step_prepared(self, batch: ScheduledRequests) -> List[StepChange]:
        """Record that ``batch`` holds resources but has no results yet.

        Returns:
            The list into which the executor records each request change the step makes.
        """
        # Sampling sets a finishing context request's draft tokens before the step is accepted.
        changes = [PreparedStep.change(request) for request in batch.context_requests]
        step = PreparedStep(
            batch, self._executor.kv_cache_manager.context_capacities(batch), changes
        )
        self._uncommitted_steps.append(step)
        return step.changes

    def step_released(self) -> None:
        """Record that the newest prepared step returned its resources without running."""
        self._uncommitted_steps.pop()

    def step_outputs_appending(self, batch: ScheduledRequests) -> None:
        """Checkpoint the results to which ``batch``, the newest prepared step, appends logits or additional
        outputs, so that a discard drops them."""
        if (
            not self._outputs_requested
            or not self._uncommitted_steps
            or self._uncommitted_steps[-1].batch is not batch
        ):
            return
        self._uncommitted_steps[-1].results = [
            (request.py_result, request.py_result.checkpoint())
            for request in batch.all_requests()
            if request.py_return_context_logits
            or request.py_return_generation_logits
            or request.py_additional_outputs
        ]

    def commit_step(self, batch: ScheduledRequests, sample_state: Optional["SampleState"]) -> None:
        """Accept the oldest prepared step, ``batch``, unless a peer failed while it ran.

        Waits for the step's sampler event, which ``update_requests`` waits on
        next: it follows the step's MoE collectives on the stream (``create``
        requires a generation sampler, which records one every step). A step
        without a sample state failed before sampling and has no event; for it,
        the commit waits for the stream, so that the step's MoE collectives
        finished before the abort record is read and before the executor
        publishes the step's context blocks. A peer that stops responding
        without dying is bounded by the MoE kernels' device timeout, which
        fails stop.

        Survivors may decide differently on the same step: one that finished it
        before the loss was reported commits it, another discards it. Both are
        correct because the step is committed only if this rank's abort record
        is clear: every collective of the step then read every active source's
        complete data, and a peer whose dispatch was released publishes no
        further round flag, so none of its outputs reached this rank (see
        ``MoeA2AAbortState``).

        Raises:
            HostCollectiveInterrupted: The step's MoE collectives may have been
                aborted; its results must be discarded.
        """
        if not self._uncommitted_steps:
            return
        if self._uncommitted_steps[0].batch is not batch:
            raise RuntimeError("Steps must be committed in the order they were prepared")
        if sample_state is not None:
            # Not the stream: with the overlap scheduler it holds the next step's forward.
            sample_state.sampler_event.synchronize()
        else:
            torch.cuda.current_stream().synchronize()
        if self.peer_failed or self._executor.model_engine.moe_execution_state.aborted():
            raise HostCollectiveInterrupted("MoE all-to-all", frozenset())
        self._uncommitted_steps.popleft()
        if self._unprogressed:
            self._note_progress(batch)
        if self._first_step_pending:
            self._first_step_pending = False
            self._log_phase("first_step", self._generation)
        if self._heartbeat is not None:
            self._heartbeat.publish(self._executor.iter_counter)

    def fetching_requests(self) -> None:
        """The rank-state exchange before a fetch showed every rank received the last broadcast.

        Rank 0 joined it after processing every earlier response gather, so none is owed any more.
        """
        self._broadcast_control_items = []
        self._gathers.clear()

    def _recover(self, error: HostCollectiveInterrupted) -> None:
        """Continue serving without the departed peers, or raise.

        The failed set is the one ULFM's shrink agrees on: every rank lost so far. A peer that fails before
        ``Install`` joins it and the survivors plan again; one that fails after ``Install`` ends the job.
        The first shrink gets ``shrink_timeout_s``; every later round shares one ``recovery_timeout_s``
        deadline. An MPI call that never returns is ended by the executor's hang detector (see ``run``).
        """
        from tensorrt_llm.executor.recovery.client import RecoveryInterrupted
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
        rank = dist.rank
        self._log_phase("detect", self._generation + 1)
        self._log(
            logger.warning,
            f"Rank {rank}: host collective interrupted ({error}); attempting peer-loss recovery",
        )
        # Release MoE collectives that may still wait for the lost peer.
        engine.moe_execution_state.request_abort()
        self._watch.stop()
        dist.revoke()
        # Shrink is a ULFM agreement among survivors, so every survivor reports
        # the same failed set even if it only observed the revocation.
        failed = self._shrink()
        _check_deadline(
            started + self._shrink_timeout_s, "the survivors agreed on the failed ranks"
        )
        deadline = time.monotonic() + self._timeout_s
        self._check_recoverable(failed)
        failed_ranks = tuple(sorted(failed))
        self._join_device_work(deadline)
        # Only now can no kernel of the steps still write their KV blocks.
        discarded = len(self._uncommitted_steps)
        interrupted = {
            request.parent_request_id if request.is_child else request.py_request_id
            for step in self._uncommitted_steps
            for request in step.batch.all_requests()
            if not request.is_dummy
        }
        executor.discard_interrupted_steps(reversed(self._uncommitted_steps))
        self._uncommitted_steps.clear()

        load_balancer = engine.moe_load_balancer
        # Online EPLB: survivors may hold different placements (an update fed by an interrupted step's
        # statistics, or one more update on a rank that ran one more iteration). Once no update can still
        # publish, every survivor starts from rank 0's placement.
        online = not load_balancer.is_static_routing()
        if online and not load_balancer.quiesce(max(0.0, deadline - time.monotonic())):
            raise RuntimeError("EPLB weight updates did not finish within the recovery timeout")
        # Until Install, a further loss restarts the plan with every rank lost so far. A loss during the workspace
        # remap can fail the remap instead.
        replans = 0
        # What the authority sent while this rank served (a refusal) belongs to no recovery.
        client.discard_pending()
        while True:
            if replans:
                _check_deadline(deadline, "planning the recovery again")
            failed_ranks = tuple(sorted(failed))
            further_loss = threading.Event()
            survivor_watch = self._watch_survivors(further_loss)

            def stale(message: object) -> bool:
                """A Propose of an earlier plan, sent before the authority learned of the further loss."""
                return isinstance(message, Propose) and message.failed_ranks != failed_ranks

            # In the first round a Propose for other ranks is a protocol violation, not a leftover.
            skip = stale if replans else None

            try:
                client.send(Stopped(rank=rank, failed_ranks=failed_ranks))
                self._log_phase("stopped", self._generation + 1)
                # The authority's deadline starts no later than this Stopped arrives and is shorter, so a recovery
                # it cannot complete reaches every survivor as Abort rather than as a timeout here.
                if not replans:
                    deadline = time.monotonic() + self._timeout_s
                # The restore's memory is mapped while the authority gathers the survivors' Stopped. Rank 0's
                # placement travels among the survivors, not through the authority: it can exceed a message.
                base = self._rank0_placements(load_balancer) if online else None
                placements = self._restore_placements(load_balancer, failed, base)
                proposal = self._wait_for_proposal(deadline, further_loss, skip)
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
                self._log_phase("proposed", generation)
                self._remap_workspaces(failed, deadline)
                copy_plan = (
                    None
                    if load_balancer.copy_nodes is None
                    else self._stage_ring_restore(load_balancer, failed, placements)
                )
                placement_digest = load_balancer.placement_digest(failed, placements, copy_plan)
                ready = Ready(rank=rank, generation=generation, placement_digest=placement_digest)
                client.send(ready)
                self._log_phase("ready", generation)
                install = client.wait(Install, deadline, interrupt=further_loss)
                _expect_generation(install, generation)
                self._log_phase("install", generation)
            except RecoveryInterrupted:
                failed = self._shrink_again(survivor_watch, failed, deadline)
                replans += 1
                continue
            finally:
                if survivor_watch is not None:
                    survivor_watch.stop()
            break

        dist.install_survivors()
        installing = time.monotonic()
        restored = install_recovery_placement(
            engine.moe_execution_state, load_balancer, failed, placements, own_rank=rank
        )
        if restored:
            self._log(
                logger.info,
                f"Restored {restored} expert slots without ranks {list(failed_ranks)} in "
                f"{(time.monotonic() - installing) * 1e3:.1f} ms",
            )
        # The KV cache cut now serves the installed slots for good.
        kv_lent, self._kv_lent = self._kv_lent, 0
        if online:
            # Every survivor updates the same layers from the same statistics from here on.
            load_balancer.reset_update_schedule()
            load_balancer.set_warm_up_iter_count(_EPLB_WARM_UP_ITERS_AFTER_RECOVERY)
        failed_requests = self._departed_requests(failed)
        client.send(Installed(rank=rank, generation=generation, failed_requests=failed_requests))
        self._log_phase("installed", generation)
        _expect_generation(client.wait(Commit, deadline), generation)
        self._generation = generation
        self._log_phase("committed", generation)
        self._return_remap_room()

        # What each survivor holds: the requests it serves and those it already answered (for example
        # rejected ones); none of them may be admitted again. A loss can stop survivors one iteration apart, so
        # they also resume from the same iteration count, which iteration-gated collectives assume. And the
        # requests the loss interrupted, and those that committed a step since an earlier loss interrupted them.
        buffered = executor.take_buffered_responses()
        gathered = dist.allgather(
            (
                executor.served_request_ids() | self._answered_request_ids(buffered[0]),
                executor.iter_counter,
                interrupted,
                self._progressed,
            ),
            departed=None,
        )
        self._progressed = set()
        self._progress_report = []
        if self._request_owners is not None:
            self._interrupted.difference_update(*(entry[3] for entry in gathered if entry))
        held = [entry and entry[0] for entry in gathered]
        executor.iter_counter = max(entry[1] for entry in gathered if entry)
        self._requeue_unscheduled_requests(held)
        rerun, failing = self._deliver_responses(
            failed, buffered, set().union(*(entry[2] for entry in gathered if entry))
        )
        # Rank 0's re-run requests not routed yet, including those an earlier recovery re-admitted.
        self._rerouting = set(dist.broadcast([*self._rerouting, *rerun], root=0))
        self._unprogressed = (self._unprogressed | interrupted) & executor.served_request_ids()
        self._readmit_undelivered_requests(failed, held)
        self._log_phase("resumed", generation)
        self._first_step_pending = rank == 0
        self._peer_failed.clear()
        self._watch_peer_failures()
        requests = f"re-ran {len(rerun)} requests and failed {failing}; " if rank == 0 else ""
        self._log(
            logger.warning,
            f"Rank {rank}: resumed serving without ranks {list(failed_ranks)} at membership "
            f"generation {generation}, {(time.monotonic() - started) * 1e3:.0f} ms after the "
            f"interruption, discarding {discarded} steps; {len(dist.live_ranks)} ranks remain; "
            f"{requests}lent {kv_lent} bytes of the KV cache",
        )

    def _log(self, log: Callable[[str], None], message: str) -> None:
        """``log(message)``, deferred to the end of a recovery during one."""
        if self._deferred_logs is None:
            log(message)
        else:
            self._deferred_logs.append((log, message))

    def _log_phase(self, name: str, generation: int) -> None:
        """Log on rank 0 the wall-clock time the recovery to ``generation`` reached phase ``name``."""
        if self._executor.dist.rank == 0:
            self._log(
                logger.info, f"WideEP recovery gen={generation} phase={name} t={time.time():.3f}"
            )

    def _wait_for_proposal(
        self,
        deadline: float,
        interrupt: threading.Event,
        skip: Optional[Callable[[object], bool]],
    ) -> "Propose":
        """Wait for the authority's plan; on rank 0, first end each failed rank it asks to fence.

        The authority asks for every failed rank it does not know to be dead: the survivors' agreement may
        rest on a failure detector's suspicion, and a stopped process must not run on beside the survivors.
        """
        from tensorrt_llm.executor.recovery.protocol import Fence, Fenced, Propose

        while True:
            message = self._client.wait((Propose, Fence), deadline, interrupt=interrupt, skip=skip)
            if isinstance(message, Propose):
                return message
            error = self._end_ranks([message.rank])[message.rank]
            self._client.send(
                Fenced(rank=self._executor.dist.rank, target=message.rank, error=error)
            )

    def _end_ranks(self, ranks: Iterable[int]) -> Dict[int, str]:
        """Have the MPI runtime kill each of ``ranks`` at once, together bounded by ``RUNTIME_FENCE_CALL_S``.

        Returns why each rank could not be ended, or '' for one that was (also a no-op for one that had ended).
        """
        from tensorrt_llm._utils import mpi_comm
        from tensorrt_llm.executor.recovery.config import RUNTIME_FENCE_CALL_S

        from ..distributed.pmix_fence import PmixUnavailable, kill_rank, pmix_ranks

        ranks = list(ranks)
        if not ranks:
            return {}
        runtime_ranks = pmix_ranks(mpi_comm())
        errors: Dict[int, str] = {}

        def kill(rank: int) -> None:
            try:
                kill_rank(runtime_ranks[rank])
            except (PmixUnavailable, RuntimeError) as error:
                errors[rank] = str(error)

        # The runtime may never answer; such a thread is left blocked.
        threads = {
            rank: threading.Thread(target=kill, args=(rank,), name="rank_fence", daemon=True)
            for rank in ranks
        }
        for thread in threads.values():
            thread.start()
        deadline = time.monotonic() + RUNTIME_FENCE_CALL_S
        outcome = {}
        for rank, thread in threads.items():
            thread.join(max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                outcome[rank] = f"the MPI runtime did not answer within {RUNTIME_FENCE_CALL_S}s"
            else:
                outcome[rank] = errors.get(rank, "")
                if not outcome[rank]:
                    self._log(
                        logger.warning,
                        f"Rank {self._executor.dist.rank}: the MPI runtime ended failed rank {rank}",
                    )
        return outcome

    def _shrink(self) -> frozenset:
        """Agree with the other survivors on the failed ranks.

        With hang detection, the ranks this rank sees failed are ended first: a rank Open MPI's failure detector
        declared failed may still be alive (a stopped process), and the agreement does not complete until it
        ends, so the detector's verdict counts as its failure. Every survivor does so for what it sees: the kills
        are idempotent, and one that fails leaves the authority's fence before Propose to fail closed. Failed
        ranks are always ended through the MPI runtime before a plan is proposed (rank 0, at the authority's
        request); hang detection additionally confirms and fences a hung (alive) rank. Without it, a stopped
        rank holds this agreement until it ends.
        """
        dist = self._executor.dist
        if fault_tolerance_config().hang_detection:
            reported = dist.reported_failed_ranks()
            for rank, error in self._end_ranks(sorted(reported)).items():
                if error:
                    self._log(
                        logger.warning,
                        f"Rank {dist.rank}: could not end failed rank {rank}: {error}",
                    )
        return dist.shrink()

    def _watch_survivors(self, further_loss: threading.Event) -> Optional[PeerFailureWatch]:
        """Set ``further_loss`` when another survivor fails before Install (at once if one already did).

        The watch also revokes, which releases every survivor blocked in a survivor collective.
        """
        dist = self._executor.dist

        def on_failure() -> None:
            further_loss.set()
            dist.revoke()

        try:
            return dist.watch_survivor_failures(on_failure)
        except Exception as error:  # noqa: BLE001 - only a peer failure is expected here
            if not is_mpi_peer_failure(error):
                raise
            further_loss.set()
            return None

    def _check_recoverable(self, lost: frozenset) -> None:
        """Raise unless the survivors can serve without every rank in ``lost``.

        Recoverable: an admitted domain (one rank or a whole node), or a union of admitted domains whose experts
        the survivors' slots can still serve, whether those ranks failed together or one during the other's
        recovery. Online EPLB placements can differ among survivors until rank 0's is agreed, so there
        ``_restore_placements`` checks the slots, from that placement.

        Raises:
            RuntimeError: ``lost`` is not recoverable.
        """
        if lost in self._recoverable:
            return
        load_balancer = self._executor.model_engine.moe_load_balancer
        if (
            not lost
            or not lost <= frozenset().union(*self._recoverable)
            or (load_balancer.is_static_routing() and not load_balancer.covers_loss_of(lost))
        ):
            raise RuntimeError(
                f"The loss of ranks {sorted(lost)} is not recoverable: it includes a rank outside the "
                "admitted domains (e.g. rank 0), or the survivors cannot serve every expert"
            )

    def _shrink_again(
        self, watch: Optional[PeerFailureWatch], failed: frozenset, deadline: float
    ) -> frozenset:
        """Agree on every rank lost so far once another one failed during the recovery.

        Raises:
            RecoveryTimeout: ``deadline`` passed before the shrink started.
            RuntimeError: The survivors' slots cannot serve the larger loss, it includes rank 0, or the KV
                cache could not take back an abandoned plan's cut.
        """
        if watch is not None:
            watch.stop()
        _check_deadline(deadline, "shrinking the survivors again")
        # The checkpoint taken at the interruption bounds one shrink; this one gets its own.
        self._executor.hang_detector.checkpoint()
        self._executor.dist.revoke()
        lost = self._shrink()
        self._check_recoverable(lost)
        self._log(
            logger.warning,
            f"Rank {self._executor.dist.rank}: ranks {sorted(lost - failed)} failed during the recovery; "
            f"planning again without ranks {sorted(lost)}",
        )
        self._release_uninstalled_backing()
        return lost

    def _release_uninstalled_backing(self) -> None:
        """Undo the ghost-slot memory and KV cache cut of an abandoned plan; the next plan maps its own.

        Raises:
            RuntimeError: The KV cache could not take back what it gave, as when it could not give it.
        """
        released = self._executor.model_engine.moe_load_balancer.release_uninstalled_backing()
        lent, self._kv_lent = self._kv_lent, 0
        if released or lent:
            self._log(logger.info, f"Released {released} bytes of an abandoned plan's ghost slots")
        if lent and not self._executor.resize_kv_cache_quota(lent):
            raise RuntimeError(
                f"The KV cache could not take back {lent} bytes after a plan was abandoned"
            )

    def _rank0_placements(self, load_balancer: "MoeLoadBalancer") -> List[np.ndarray]:
        """Rank 0's current placements, broadcast among the survivors (online EPLB).

        Raises:
            RecoveryInterrupted: Another survivor failed during the broadcast.
            RuntimeError: Rank 0's placements do not have this rank's shape.
        """
        dist = self._executor.dist
        own = load_balancer.current_placements()
        base = self._among_survivors(
            "rank 0 sent its placement",
            lambda: dist.broadcast_among_survivors(own if dist.rank == 0 else None, root=0),
        )
        if np.asarray(base).shape != np.asarray(own).shape:
            raise RuntimeError(
                f"Rank 0's placement has shape {np.asarray(base).shape}, expected {np.asarray(own).shape}"
            )
        return base

    def _restore_placements(
        self, load_balancer: "MoeLoadBalancer", failed: frozenset, base: Optional[List[np.ndarray]]
    ) -> List[np.ndarray]:
        """The placements to serve without ``failed``, with memory mapped behind the ghost slots they load.

        Before Ready, so a rank that cannot get the memory aborts the recovery before anything changes.

        Raises:
            RuntimeError: The survivors' slots cannot serve every expert without ``failed``.
        """
        try:
            placements = load_balancer.recovery_placements(failed, base)
        except ValueError as error:
            raise RuntimeError(
                f"The loss of ranks {sorted(failed)} is not recoverable: {error}"
            ) from error
        self._back_restored_slots(load_balancer, placements)
        return placements

    def _back_restored_slots(
        self, load_balancer: "MoeLoadBalancer", placements: List[np.ndarray]
    ) -> None:
        """Map device memory behind the ghost slots the restore loads, taken from the KV cache if need be."""
        need = load_balancer.restore_backing_bytes(placements)
        if not need:
            return
        start = time.monotonic()
        self._kv_lent += self._make_room(need, "the restored expert slots")
        mapping = time.monotonic()
        load_balancer.back_restored_slots(placements)
        end = time.monotonic()
        self._log(
            logger.info,
            f"Mapped {need} bytes behind ghost expert slots in {(end - start) * 1e3:.1f} ms "
            f"({(mapping - start) * 1e3:.1f} ms making room, {(end - mapping) * 1e3:.1f} ms mapping)",
        )

    def _make_room(self, need: int, what: str) -> int:
        """Free ``need`` device bytes for ``what``, taken from the KV cache if need be; return the bytes taken.

        Raises:
            RuntimeError: The KV cache could not give them up.
        """
        short = need + _GHOST_MARGIN_BYTES - _free_device_bytes()
        if short > 0:
            # Memory the caching allocator holds for no tensor goes first (empty_cache synchronizes and is slow).
            torch.cuda.empty_cache()
            short = need + _GHOST_MARGIN_BYTES - _free_device_bytes()
        if short <= 0:
            return 0
        if not self._executor.resize_kv_cache_quota(-short):
            raise RuntimeError(f"The KV cache could not give up {short} bytes for {what}")
        self._log(logger.warning, f"Gave {short} bytes of the KV cache to {what}")
        return short

    def _remap_workspaces(self, failed: frozenset, deadline: float) -> None:
        """Back the all-to-all workspaces with fresh pages that ``failed`` never mapped, before anything uses them.

        The failed ranks' kernels may still write the pages they mapped; those stay allocated for them alone,
        and every survivor maps fresh pages at the same addresses, so captured graphs stay valid. Pages no
        failed rank maps are freed once every survivor released them, so room is made only after that, from
        the KV cache if need be; the failed ranks' pages are freed only when their processes end, so a commit
        returns that share once the memory is free (:meth:`_return_remap_room`). Without the room, page
        creation fails on this rank and the exchange fails the recovery on every survivor.

        Its exchanges block without a deadline, so ``deadline`` is checked before and after them.

        Raises:
            RecoveryInterrupted: Another survivor failed during the remap.
            RecoveryTimeout: ``deadline`` passed before or during the remap.
            RuntimeError: A survivor could not remap.
        """
        _check_deadline(deadline, "remapping the all-to-all workspaces")
        dist = self._executor.dist
        state = self._executor.model_engine.moe_execution_state
        start = time.monotonic()
        lent = 0
        reader = self._row_reader
        with reader.paused() if reader is not None else contextlib.nullcontext():
            need = state.release_workspaces(failed)
            self._among_survivors(
                "the all-to-all workspaces were released", dist.barrier_among_survivors
            )
            try:
                lent = self._make_room(need, "the remapped all-to-all workspaces")
                self._remap_lent += lent
            except RuntimeError as error:
                self._log(logger.error, f"Rank {dist.rank}: {error}")
            self._among_survivors(
                "the all-to-all workspaces were remapped",
                lambda: state.remap_workspaces(dist.allgather_among_survivors),
            )
            # The reset zeroes this rank's rows: no peer may write into them (ring restore staging) before it.
            self._among_survivors(
                "the all-to-all workspaces were reset", dist.barrier_among_survivors
            )
        _check_deadline(deadline, "continuing after the all-to-all workspace remap")
        self._log(
            logger.info,
            f"Remapped {need} bytes of all-to-all workspace without ranks {sorted(failed)} in "
            f"{(time.monotonic() - start) * 1e3:.1f} ms, {lent} bytes of them from the KV cache",
        )

    def serving_idle(self) -> None:
        """The executor has no step in flight, so the KV cache can resize: return what remaps still hold.

        Checks at most every ``_REMAP_RETURN_INTERVAL_S``, and only while a remap's share is lent.
        """
        if not self._remap_lent:
            return
        now = time.monotonic()
        if now < self._next_remap_return:
            return
        self._next_remap_return = now + _REMAP_RETURN_INTERVAL_S
        self._return_remap_room()

    def _return_remap_room(self) -> None:
        """Give the KV cache back what remaps took from it, as far as the device has that memory free again.

        The failed ranks' pages that the remapped workspaces replaced are freed once those processes end, which
        may be after their recovery. What stays short is returned when the executor is idle
        (:meth:`serving_idle`) or at a later commit.
        """
        lent = self._remap_lent
        if not lent:
            return
        free = min(lent, _free_device_bytes() - _GHOST_MARGIN_BYTES)
        if free <= 0 or not self._executor.resize_kv_cache_quota(free):
            return
        self._remap_lent -= free
        self._log(
            logger.info,
            f"Returned {free} bytes the all-to-all workspace remap took to the KV cache "
            f"({self._remap_lent} bytes still lent)",
        )

    def _stage_ring_restore(
        self,
        load_balancer: "MoeLoadBalancer",
        failed: frozenset,
        placements: List[np.ndarray],
    ) -> np.ndarray:
        """Copy the experts ``placements`` restores from the ring host copies into their slots; return the copy plan.

        Runs after :meth:`_remap_workspaces`, through the staging workspace's fresh pages, which no failed rank
        maps; no all-to-all runs until the install rearms the workspaces.
        """
        start = time.monotonic()
        workspace = self._executor.model_engine.moe_execution_state.staging_workspace()
        copy_plan = load_balancer.restore_copy_plan(failed, placements, workspace.size(1))
        barrier = self._executor.dist.barrier_among_survivors
        self._among_survivors(
            "restored experts were copied",
            lambda: load_balancer.stage_restored_slots(copy_plan, workspace, barrier),
        )
        self._log(
            logger.info,
            f"Staged {len(copy_plan)} expert weight pieces in {(time.monotonic() - start) * 1e3:.1f} ms",
        )
        return copy_plan

    def _among_survivors(self, what: str, collective: Callable[[], _T]) -> _T:
        """``collective()`` among the survivors.

        Raises:
            RecoveryInterrupted: Another survivor failed while ``what``.
        """
        from mpi4py import MPI

        from tensorrt_llm.executor.recovery.client import RecoveryInterrupted

        try:
            return collective()
        except MPI.Exception as error:
            if not is_mpi_peer_failure(error):
                raise
            raise RecoveryInterrupted(
                f"Rank {self._executor.dist.rank}: a survivor failed while {what}"
            ) from error

    def _join_device_work(self, deadline: float) -> None:
        """Wait for all work on this device, on any stream, up to ``deadline``.

        Raises:
            RecoveryTimeout: ``deadline`` passed first.
        """
        from tensorrt_llm.executor.recovery.client import RecoveryTimeout

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
            raise RecoveryTimeout("The recovery deadline passed before the device work joined")
        if errors:
            raise errors[0]

def install_recovery_placement(
    state: "MoeExecutionState",
    load_balancer: "MoeLoadBalancer",
    failed_ranks: Iterable[int],
    placements: List["np.ndarray"],
    own_rank: int,
) -> int:
    """Serve without ``failed_ranks`` (every rank lost so far): route MoE traffic by ``placements``, then mask them.

    ``placements`` come from ``MoeLoadBalancer.recovery_placements``: resident
    replicas, plus experts restored into survivors' slots from host weights.
    Only while the model is idle. Routing must exclude the failed ranks before
    their mask is installed, because the MoE kernels silently skip routes to a
    masked rank. The caller has validated the placements (their digest, sent
    with Ready); the ranks are validated here before anything changes. An error
    after that (from the native routing update, a slot load or the device)
    leaves the state mixed and ends the job.

    Returns:
        The number of slots restored.

    Raises:
        RuntimeError: A rank is ``own_rank``, or every rank had already failed.
        ValueError: A rank is out of range.
    """
    failed = sorted(set(failed_ranks))
    if own_rank in failed:
        raise RuntimeError(f"EP rank {own_rank} is this rank")
    if not any(state.health.is_active(rank) for rank in failed):
        raise RuntimeError(f"EP ranks {failed} are already failed")
    restored = load_balancer.install_recovery_placement(failed, placements)
    for rank in failed:
        state.health.mark_failed(rank)
    state.rearm()
    state.install_mask()
    return restored


def _check_deadline(deadline: float, action: str) -> None:
    """Raise ``RecoveryTimeout`` instead of ``action`` once ``deadline`` passed."""
    from tensorrt_llm.executor.recovery.client import RecoveryTimeout

    if time.monotonic() >= deadline:
        raise RecoveryTimeout(f"The recovery deadline passed before {action}")


def _expect_generation(message: "Install | Commit", generation: int) -> None:
    if message.generation != generation:
        raise RuntimeError(f"Expected recovery generation {generation}, received {message}")

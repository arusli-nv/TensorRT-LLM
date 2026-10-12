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
"""Keep a WideEP PyExecutor serving after one peer process dies."""

import collections
import os
import threading
import time
from typing import TYPE_CHECKING, Callable, Deque, Dict, Iterable, List, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

from ..distributed.communicator import HostCollectiveInterrupted, PeerFailureWatch
from ..moe.fused_moe.wide_ep_ft import recovery_enabled
from .kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from .llm_request import LlmRequest, LlmResponse
from .resource_manager import KVCacheManager, ResourceManagerType
from .scheduler import ScheduledRequests
from .scheduler.adp_router import DefaultADPRouter

if TYPE_CHECKING:
    from tensorrt_llm.executor.recovery.client import RecoveryClient
    from tensorrt_llm.executor.recovery.protocol import Commit, Install

    from .executor_request_queue import RequestQueueItem
    from .py_executor import PyExecutor
    from .sampler import SampleState

_RECOVERY_TIMEOUT_ENV = "TLLM_FAULT_TOLERANCE_RECOVERY_TIMEOUT_S"
_RECOVERY_TIMEOUT_DEFAULT_S = 120.0
# Workers register while every rank constructs its first executor.
_REGISTRATION_TIMEOUT_S = 600.0

Responses = List[Tuple[int, LlmResponse]]

_registration: Optional[Tuple["RecoveryClient", frozenset]] = None


def _recovery_timeout_s() -> float:
    return float(os.environ.get(_RECOVERY_TIMEOUT_ENV, _RECOVERY_TIMEOUT_DEFAULT_S))


def _register(rank: int, recoverable_ranks: Tuple[int, ...]) -> Tuple["RecoveryClient", frozenset]:
    """Register this process with the recovery authority, once per process.

    A process constructs more than one PyExecutor (KV cache estimation, then
    serving) but registers exactly once.

    Returns:
        The client and the ranks whose loss the deployment recovers from.
    """
    global _registration
    if _registration is None:
        from tensorrt_llm.executor.recovery.client import RecoveryClient

        client = RecoveryClient.from_env(rank)
        logger.info(
            f"Rank {rank}: registering with the WideEP recovery authority; "
            f"waiting up to {_REGISTRATION_TIMEOUT_S:.0f} s for every rank"
        )
        admitted = client.register(recoverable_ranks, time.monotonic() + _REGISTRATION_TIMEOUT_S)
        _registration = (client, admitted)
    return _registration


class PeerLossRecovery:
    """Recovers a PyExecutor from the loss of one peer process.

    Survivors discard the step the loss interrupted, agree on the new
    membership through the recovery authority (``Stopped`` -> ``Propose`` ->
    ``Ready`` -> ``Install`` -> ``Installed`` -> ``Commit``) and resume the
    same requests on their original CUDA graphs: the departed rank is masked
    out of host collectives, EPLB routing and the MoE dispatch/combine
    descriptor. Requests the departed attention-DP rank was serving fail.
    A failure raises into the executor's terminal path.

    The executor calls :meth:`step_prepared` after reserving a step's
    resources and :meth:`commit_step` before applying its results; a step in
    between is discarded when a peer fails. Response gathers report through
    :meth:`gathering_responses` and :meth:`responses_gathered` so that the
    responses of an interrupted gather are delivered after recovery. The rest of
    that iteration is skipped.
    """

    def __init__(
        self, executor: "PyExecutor", client: "RecoveryClient", admitted: frozenset
    ) -> None:
        self._executor = executor
        self._client = client
        self._admitted = executor.dist.rank in admitted
        # Whether the survivors can recover if this rank exits: only while the
        # serving loop watches peers, and never during or after a recovery.
        self.survives_own_loss = False
        self._timeout_s = _recovery_timeout_s()
        self._uncommitted_step: Optional[ScheduledRequests] = None
        self._peer_failed = threading.Event()
        self._watch: Optional[PeerFailureWatch] = None
        # Response gathers are numbered identically on every rank. Each rank
        # keeps its last two contributions: a rank can complete one gather
        # more than rank 0 before the loss stops it.
        self._gather_count = 0
        self._gathers: Deque[Tuple[int, Responses]] = collections.deque(maxlen=2)
        self._last_processed_gather = 0
        # Finished requests whose termination waits for their responses.
        self._undelivered_terminations: List[LlmRequest] = []
        # On rank 0: request id -> (attention-DP rank, routed item) of every
        # routed request that has not finished.
        self._request_owners: Optional[Dict[int, Tuple[int, "RequestQueueItem"]]] = None
        if executor.dist.rank == 0:
            self._request_owners = {}

    @classmethod
    def create(cls, executor: "PyExecutor") -> Optional["PeerLossRecovery"]:
        """Return the recovery of ``executor``, or None when it is disabled.

        The configuration must make a discarded step re-runnable: no
        speculative decoding, guided decoding, beam search, KV connector or
        PEFT. Requests are expected to sample greedily; penalties, seeded
        sampling and multi-token stop words are not rolled back with a
        discarded step.

        Raises:
            ValueError: Recovery is enabled for a configuration it does not
                support.
            RecoveryError: The recovery authority rejected or did not answer
                the registration.
        """
        if not recovery_enabled():
            return None
        engine = executor.model_engine
        dist = executor.dist
        load_balancer = engine.moe_load_balancer
        kv_cache_manager = executor.kv_cache_manager
        # MPIDist.enable_peer_loss_recovery() checks the pure attention-DP layout.
        requirements = {
            "MPI executor": not executor._disable_mpi,
            "expert TP 1": dist.mapping.moe_tp_size == 1,
            "no attention DP balancing": not executor.attention_dp_enable_balance,
            "DefaultADPRouter": type(executor.adp_router) is DefaultADPRouter,
            "no speculative decoding": executor.drafter is None
            and executor.draft_model_engine is None
            and engine.spec_config is None,
            "no guided decoding": executor.guided_decoder is None,
            "no beam search": executor.max_beam_width == 1,
            "no KV cache transceiver": executor.kv_cache_transceiver is None,
            "no KV connector": executor.kv_connector_manager is None,
            "KV cache manager V1 or V2": type(kv_cache_manager)
            in (KVCacheManager, KVCacheManagerV2),
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
            # An abort request must not queue behind the kernels it stops.
            "CUDA_DEVICE_MAX_CONNECTIONS > 1": os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") != "1",
            "no overlap scheduler": executor.disable_overlap_scheduler,
            "no conversation manager": getattr(kv_cache_manager, "conversation_manager", None)
            is None,
            "hang detection timeout above the recovery timeout": executor.hang_detector.timeout
            > _recovery_timeout_s(),
            "no sleep/wakeup": executor.llm_args.sleep_config is None,
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
            rank for rank in range(1, dist.world_size) if load_balancer.covers_loss_of(rank)
        )
        dist.enable_peer_loss_recovery()
        client, admitted = _register(dist.rank, recoverable)
        if dist.rank == 0:
            logger.info(f"WideEP peer-loss recovery covers the loss of ranks {sorted(admitted)}")
        return cls(executor, client, admitted)

    @property
    def peer_failed(self) -> bool:
        """Whether a peer failure was detected since the last recovery."""
        return self._peer_failed.is_set()

    # Event loop

    def run(self, event_loop: Callable[[], None]) -> None:
        """Run ``event_loop``, recovering from each peer loss it raises.

        Peers are watched only while a serving loop runs; a loss during KV
        cache estimation, which gathers responses on every rank, is terminal.
        """
        if self._executor.gather_all_responses:
            event_loop()
            return
        self._watch = self._executor.dist.watch_peer_failures(self._on_peer_failure)
        self.survives_own_loss = self._admitted
        try:
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
            self._watch.stop()

    def _on_peer_failure(self) -> None:
        """Stop the in-flight step; runs on the peer-failure watcher thread.

        The flag is set before the abort request, so a step whose device work
        finished with the flag clear never observed the abort. Revoking the
        host communicators releases every survivor blocked in a host collective.
        """
        self._peer_failed.set()
        logger.warning(
            f"Rank {self._executor.dist.rank}: a peer failed; aborting in-flight MoE collectives"
        )
        try:
            self._executor.model_engine.moe_execution_state.request_abort()
        finally:
            self._executor.dist.revoke()

    # Step hooks

    def step_prepared(self, batch: ScheduledRequests) -> None:
        """Record that ``batch`` holds resources but has no results yet."""
        self._uncommitted_step = batch

    def step_released(self) -> None:
        """Record that the prepared step returned its resources without running."""
        self._uncommitted_step = None

    def commit_step(self, sample_state: Optional["SampleState"]) -> None:
        """Accept the prepared step unless a peer failed while it ran.

        Waits for the step's device work. A peer that stops responding without
        dying is bounded by the MoE kernels' device timeout, which fails stop.

        Raises:
            HostCollectiveInterrupted: The step's MoE collectives may have been
                aborted; its results must be discarded.
        """
        if self._uncommitted_step is None:
            return
        if sample_state is not None and sample_state.sampler_event is not None:
            sample_state.sampler_event.synchronize()
        else:
            torch.cuda.current_stream().synchronize()
        if self._peer_failed.is_set():
            raise HostCollectiveInterrupted("MoE all-to-all", frozenset())
        self._uncommitted_step = None

    # Request and response hooks

    def requests_routed(self, routed: Dict[int, List["RequestQueueItem"]]) -> None:
        """Record the attention-DP rank serving each new request (rank 0)."""
        if self._request_owners is None:
            return
        for rank, items in routed.items():
            for item in items:
                self._request_owners[item.id] = (rank, item)

    def gathering_responses(self, responses: Responses) -> None:
        """Keep this rank's part of a response gather for redelivery."""
        self._gather_count += 1
        self._gathers.append((self._gather_count, responses))

    def responses_gathered(self, responses: Iterable[Tuple[int, LlmResponse]]) -> None:
        """Note that rank 0 processed the last gather; forget finished requests.

        A result is final once every sequence of the request finished, and an
        error ends the whole request, so ``n > 1`` requests leave together.
        """
        self._last_processed_gather = self._gather_count
        if self._request_owners is None:
            return
        for request_id, response in responses:
            if isinstance(response, LlmResponse) and (
                response.has_error() or (response.result is not None and response.result.is_final)
            ):
                self._request_owners.pop(request_id, None)

    def defer_terminations(self, requests: Iterable[LlmRequest]) -> None:
        """Terminate ``requests`` after recovery delivers their responses."""
        self._undelivered_terminations.extend(requests)

    # Recovery

    def _recover(self, error: HostCollectiveInterrupted) -> None:
        """Continue serving without one departed peer, or raise."""
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
        deadline = time.monotonic() + self._timeout_s
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
        if len(failed) != 1:
            raise RuntimeError(
                f"Peer-loss recovery requires exactly one departed rank, got {sorted(failed)}"
            )
        (failed_rank,) = failed
        self._join_device_work(deadline)
        # Only now can no kernel of the step still write its KV blocks.
        executor._discard_uncommitted_step(self._uncommitted_step)
        self._uncommitted_step = None

        client.send(Stopped(rank=rank, failed_rank=failed_rank))
        proposal = client.wait(Propose, deadline)
        if (
            proposal.failed_rank != failed_rank
            or rank not in proposal.survivors
            or failed_rank in proposal.survivors
        ):
            raise RuntimeError(
                f"Rank {rank} observed the loss of rank {failed_rank} but the "
                f"authority proposed {proposal}"
            )
        generation = proposal.generation

        placement_digest = engine.moe_load_balancer.masked_placement_digest(failed)
        client.send(Ready(rank=rank, generation=generation, placement_digest=placement_digest))
        _expect_generation(client.wait(Install, deadline), generation)

        dist.install_survivors(failed)
        engine.install_resident_placement(failed, placement_digest)
        client.send(Installed(rank=rank, generation=generation))
        _expect_generation(client.wait(Commit, deadline), generation)

        executor._requeue_unscheduled_requests()
        self._deliver_responses(failed)
        self._readmit_undelivered_requests(failed)
        self._peer_failed.clear()
        self._watch = dist.watch_peer_failures(self._on_peer_failure)
        logger.warning(
            f"Rank {rank}: resumed serving without rank {failed_rank} at "
            f"membership generation {generation}"
        )

    def _join_device_work(self, deadline: float) -> None:
        """Wait for work enqueued on the executor streams, up to ``deadline``."""
        events = []
        for stream in (torch.cuda.current_stream(), self._executor.execution_stream):
            event = torch.cuda.Event()
            event.record(stream)
            events.append(event)
        while not all(event.query() for event in events):
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out joining device work for peer-loss recovery")
            time.sleep(0.001)

    def _deliver_responses(self, failed: frozenset) -> None:
        """Deliver what the interrupted gathers owed their clients.

        Collective over the survivors. Every survivor resends its parts of the
        gathers rank 0 did not process, and the error responses it buffered for
        the next synchronized flush, then terminates the requests they finished.
        Rank 0 also fails the requests that were running on a departed
        attention-DP rank: their KV cache left with the rank.
        """
        executor = self._executor
        processed = executor.dist.broadcast(self._last_processed_gather, root=0)
        responses = [
            response
            for number, contribution in self._gathers
            if number > processed
            for response in contribution
        ]
        self._gathers.clear()
        responses += executor._pending_transfer_responses
        self._undelivered_terminations += executor._pending_response_terminations
        executor._pending_transfer_responses = []
        executor._pending_response_terminations = []
        if self._request_owners is not None:
            for request_id, (owner, item) in list(self._request_owners.items()):
                if owner in failed:
                    del self._request_owners[request_id]
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
        executor._publish_responses(responses, terminations)

    def _readmit_undelivered_requests(self, failed: frozenset) -> None:
        """Re-admit requests that rank 0 routed to a survivor which never received them.

        Collective over the survivors. A loss can interrupt the broadcast of new
        requests after rank 0 completed its part, so rank 0 can route a request
        that its owner never received. The owner never started it, so rank 0
        admits it again. Runs after redelivery, which retires finished requests.
        """
        held = self._executor.dist.allgather(
            {
                request.parent_request_id if request.is_child else request.py_request_id
                for request in self._executor.active_requests
                if not request.is_attention_dp_dummy
            }
        )
        if self._request_owners is None:
            return
        missing = [
            item
            for request_id, (owner, item) in self._request_owners.items()
            if owner not in failed and request_id not in held[owner]
        ]
        for item in missing:
            del self._request_owners[item.id]
        if missing:
            logger.warning(f"Re-admitting {len(missing)} requests that never reached their rank")
            self._executor.request_accumulated[:0] = missing


def _expect_generation(message: "Install | Commit", generation: int) -> None:
    if message.generation != generation:
        raise RuntimeError(f"Expected recovery generation {generation}, received {message}")

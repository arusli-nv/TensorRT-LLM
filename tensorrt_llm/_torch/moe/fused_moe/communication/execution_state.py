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

"""Own fixed execution descriptors, graphs and mapped communication storage."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Hashable

import torch

from ..ep_group_health import EP_MASK_NUM_WORDS, EPGroupHealth
from .execution_ticket import ExecutionTicket, wait_cuda_event

if TYPE_CHECKING:
    from tensorrt_llm._torch.alltoall_watchdog import (
        ActiveRankMaskSnapshot,
        AlltoAllWatchdogTimeout,
    )
    from tensorrt_llm._torch.mnnvl_alltoall_workspace import _MnnvlAlltoAllWorkspaceLifecycle
    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner
    from tensorrt_llm._torch.pyexecutor.model_engine import ModelEngine
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager, ResourceManager

    from .nvlink_one_sided import NVLinkOneSided

# Failed teardown must retain CUDA storage independently of engine collection.
_UNCERTAIN_STATES = {}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _signed_word(word: int) -> int:
    return word if word < (1 << 63) else word - (1 << 64)


@dataclass(frozen=True)
class LocalExecutionJoin:
    owner: "ExecutionState"
    generation: int
    execution_sequence: int
    inventory: tuple
    events: tuple
    ticket_retired: bool
    watchdog_retired: bool
    stopped: bool
    previous: LocalExecutionJoin | None = None


@dataclass(frozen=True)
class CapturedTensorSpan:
    role: str
    pointer: int
    byte_size: int
    shape: tuple[int, ...]
    dtype: str
    origin: str


@dataclass(frozen=True)
class CapturedExecutionFootprint:
    """Original allocation and graph identity; not remote-write admission."""

    graph_key: Hashable
    graph_owner_id: int
    raw_graph: int
    executable: int
    capture_generation: int
    execution_sequence: int
    workspace_geometry: tuple
    mapping_token: object
    descriptor_addresses: tuple[int, ...]
    schedule: tuple[tuple[str, int], ...]
    spans: tuple[tuple[CapturedTensorSpan, ...], ...]


class ExecutionState:
    def __init__(
        self,
        lifecycle: _MnnvlAlltoAllWorkspaceLifecycle,
        *,
        ep_group_health: EPGroupHealth,
        logical_rank: int,
        clients: tuple[NVLinkOneSided, ...],
        streams: tuple[torch.cuda.Stream, ...],
        model_owner: ModelEngine,
    ) -> None:
        self._lock = threading.RLock()
        self._callback_lock = threading.Lock()
        self._ticket = None
        self._execution_sequence = 0
        self._watchdog = None
        self._join_events = []
        self._graph_schedules = {}
        self._capture_footprints: dict[int, CapturedExecutionFootprint] = {}
        self.lifecycle = lifecycle
        self.ep_group_health = ep_group_health
        self.logical_rank = logical_rank
        self.memory = lifecycle.memory
        self._clients = tuple(clients)
        self._phase = "constructing"
        self._health_token = None
        self._claimed_lifecycle = False
        self._graphs = {}
        self._model_owner = model_owner
        self._graph_resource_owners = []
        self._resource_owners = None
        self._stop_receipt = None
        self._idle_join = None
        self._pending_join = None
        self._local_join_owned = False
        self._resource_reset_ready = False
        self._streams = []
        self._frame = None
        self._started = False
        self._reset_previous_phase = None
        _require(isinstance(ep_group_health, EPGroupHealth), "Canonical EPGroupHealth required")
        _require(
            clients and len({id(c) for c in clients}) == len(clients),
            "Distinct owned frontends required",
        )
        _require(0 <= logical_rank < ep_group_health.moe_world_size, "Invalid logical rank")
        _require(model_owner is not None, "A retained model resource owner is required")
        for stream in streams:
            self._register_stream(stream)
        self._geometry = clients[0].execution_geometry()
        for client in clients:
            self.validate_frontend(client)
        _require(
            all(c.execution_watchdog() is None for c in clients),
            "Pending binding requires deferred watchdog startup",
        )
        watchdog_options = lifecycle.execution_watchdog_options()
        timeout_s, poll_interval_s, callback = watchdog_options or (None, 0.1, None)
        _require(callback is None, "External watchdog callback ownership is not qualified")
        _require(
            (timeout_s is None or (math.isfinite(timeout_s) and timeout_s > 0))
            and math.isfinite(poll_interval_s)
            and poll_interval_s > 0,
            "Finite positive watchdog timing required before ownership claims",
        )
        self._join_timeout_s = (timeout_s or 5.0) + 2 * poll_interval_s + 1.0
        self._coordinator = clients[0]._watchdog_coordinator
        _require(
            all(c._watchdog_coordinator is self._coordinator for c in clients),
            "One actual shared watchdog coordinator required",
        )
        _require(
            callable(getattr(lifecycle, "retain_execution_state", None))
            and callable(getattr(lifecycle, "release_execution_state", None)),
            "Atomic complete frontend lifecycle claim is not implemented",
        )
        self._health_token = ep_group_health._claim_commit_owner()
        try:
            self._installed = ep_group_health.snapshot()
            self._construction_snapshot = self._installed
            n = ep_group_health.moe_world_size
            _require(n <= 64 * EP_MASK_NUM_WORDS, "Membership exceeds native descriptor ABI")
            _require(
                self._installed.mask == (1 << n) - 1 and not self._installed.failed_ranks,
                "Healthy construction requires the original full membership",
            )
            lifecycle.retain_execution_state(self, self._clients)
            self._claimed_lifecycle = True
        except (RuntimeError, ValueError, TypeError):
            ep_group_health._release_commit_owner(self._health_token)
            self._health_token = None
            raise
        # From the first CUDA allocation onward, any construction error retains
        # the entire owner until process containment; no uncertain implicit free.
        _UNCERTAIN_STATES[id(self)] = self
        n = ep_group_health.moe_world_size
        words = [
            (self._installed.mask >> (64 * i)) & ((1 << 64) - 1) for i in range(EP_MASK_NUM_WORDS)
        ]
        device = clients[0].workspace.device
        self.abort = torch.zeros(2, dtype=torch.int32, device=device)
        self.descriptor = torch.tensor(
            [_signed_word(w) for w in words] + [_signed_word(self.abort.data_ptr())],
            dtype=torch.int64,
            device=device,
        )
        self.published = torch.tensor(
            [_signed_word(self.descriptor.data_ptr())], dtype=torch.int64, device=device
        )
        self.slot = torch.zeros_like(self.published)
        self._register_stream(torch.cuda.current_stream(device))
        self._join()
        for client in clients:
            client.bind_execution_state(self)
        self._phase = "ready"
        self._watchdog = lifecycle.start_execution_watchdog(self, self._on_timeout)
        self._release_uncertain_if_ready()

    def bind_resource_manager(
        self, resource_manager: ResourceManager, kv_cache_manager: KVCacheManager
    ) -> None:
        with self._lock:
            _require(self._phase == "ready", "Resource binding outside an idle execution boundary")
            _require(
                resource_manager is not None and kv_cache_manager is not None,
                "Actual resource and KV cache owners required",
            )
            if self._resource_owners is not None:
                if self._resource_owners == (resource_manager, kv_cache_manager):
                    return
                _require(
                    self._resource_reset_ready and not self._graphs and self._ticket is None,
                    "Resource replacement requires an acknowledged graph-only reset",
                )
                _require(
                    self.ep_group_health.snapshot() == self._construction_snapshot,
                    "Resource replacement requires the original healthy membership",
                )
                self._check_epoch()
                self._join()
                self._healthy_device()
                self._check_epoch()
            self._resource_owners = (resource_manager, kv_cache_manager)
            self._resource_reset_ready = False

    def _release_uncertain_if_ready(self) -> None:
        if (
            self._phase == "ready"
            and not self._local_join_owned
            and all(record[3] for record in self._graphs.values())
        ):
            _UNCERTAIN_STATES.pop(id(self), None)

    @property
    def generation(self) -> int:
        return self.ep_group_health.snapshot().generation

    def _block(self, message: str) -> None:
        self._phase = "blocked"
        _UNCERTAIN_STATES[id(self)] = self
        raise RuntimeError(message)

    def _register_stream(self, stream: torch.cuda.Stream) -> None:
        key = (stream.device, int(stream.cuda_stream))
        if all((existing.device, int(existing.cuda_stream)) != key for existing in self._streams):
            self._streams.append(stream)

    def _check_epoch(self) -> None:
        if self.ep_group_health.snapshot() != self._installed:
            self._block("Health differs from installed descriptor and static placement")

    def _join(self) -> None:
        _require(
            not torch.cuda.is_current_stream_capturing(), "Join inside graph capture forbidden"
        )
        try:
            self._join_events = []
            for stream in self._streams:
                event = torch.cuda.Event(blocking=False)
                self._join_events.append(event)
                event.record(stream)
            for event in self._join_events:
                wait_cuda_event(event, self._join_timeout_s)
            self._join_events.clear()
        except (RuntimeError, TimeoutError):
            self._phase = "blocked"
            _UNCERTAIN_STATES[id(self)] = self
            raise

    def _healthy_device(self) -> None:
        if self.abort.tolist() != [0, 0]:
            self._block("Native abort requires issuer-terminal retention acknowledgement")
        if self._started and self.slot.item() != _signed_word(self.descriptor.data_ptr()):
            self._block("Execution latch differs from owned descriptor")

    def _on_timeout(self, event: AlltoAllWatchdogTimeout) -> None:
        with self._callback_lock:
            ticket = self._ticket
        if ticket is not None:
            ticket.on_timeout(event)

    def request_local_abort(self) -> bool:
        """Request local cancellation only; this does not commit STOP or membership."""
        with self._callback_lock:
            ticket = self._ticket
        return ticket is not None and ticket.request_abort()

    def _open_ticket(self, schedule: tuple[tuple[str, NVLinkOneSided], ...]) -> None:
        if self._watchdog is None:
            return
        _require(self._ticket is None, "Previous execution ticket remains owned")
        before = self._coordinator.read_current_flag_val()
        watches = tuple(
            (phase, (before + offset + 1) & 0xFFFFFFFF)
            for offset, (phase, frontend) in enumerate(schedule)
        )
        ticket = ExecutionTicket(abort=self.abort, watches=watches, timeout_s=self._join_timeout_s)
        with self._callback_lock:
            self._ticket = ticket

    def _finish_ticket(self, *, deadline_monotonic: float | None = None) -> None:
        with self._callback_lock:
            ticket = self._ticket
        if ticket is not None:
            if deadline_monotonic is None:
                ticket.close_and_join()
            else:
                ticket.close_and_join(deadline_monotonic=deadline_monotonic)
            remaining = (
                self._join_timeout_s
                if deadline_monotonic is None
                else deadline_monotonic - time.monotonic()
            )
            if remaining <= 0:
                raise TimeoutError("STOP deadline expired before watchdog retirement")
            _require(
                self._watchdog.wait_until_idle(remaining),
                "Watchdog still owns outstanding phase reads",
            )
            _require(self._watchdog.last_error is None, "Watchdog polling failed")
            with self._callback_lock:
                _require(self._ticket is ticket, "Execution ticket ownership changed")
                self._ticket = None

    def watch_collective(
        self, frontend: NVLinkOneSided, phase: str, active_rank_mask: ActiveRankMaskSnapshot | None
    ) -> None:
        self.validate_frontend_epoch(frontend)
        if self._watchdog is not None and not torch.cuda.is_current_stream_capturing():
            self._coordinator.watch_collective(self._watchdog, phase, active_rank_mask)

    def validate_frontend(self, frontend: NVLinkOneSided) -> None:
        _require(any(frontend is c for c in self._clients), "Unowned frontend")
        _require(
            frontend.ep_group_health is self.ep_group_health
            and frontend.execution_lifecycle() is self.lifecycle
            and frontend.ep_rank == self.logical_rank
            and frontend.ep_size == self.ep_group_health.moe_world_size
            and frontend.execution_geometry() == self._geometry,
            "Frontend health, original logical layout or mapping differs",
        )

    def validate_frontend_epoch(self, frontend: NVLinkOneSided) -> None:
        with self._lock:
            self.validate_frontend(frontend)
            self._check_epoch()
            _require(self._phase == "forward", "Frontend outside owned model forward")

    def begin_forward(self) -> None:
        with self._lock:
            _require(self._phase == "ready", "Forward blocked or outstanding")
            _require(
                self._resource_owners is not None, "Resource owners must bind before execution"
            )
            self._resource_reset_ready = False
            self._check_epoch()
            if not torch.cuda.is_current_stream_capturing():
                self._execution_sequence += 1
            self._frame = []
            self._phase = "forward"
            self._register_stream(torch.cuda.current_stream(self._clients[0].workspace.device))
            # Register uncertainty before issuing GPU work; exceptions cannot drop owners.
            _UNCERTAIN_STATES[id(self)] = self
            if not torch.cuda.is_current_stream_capturing():
                schedule = tuple(
                    (phase, client) for client in self._clients for phase in ("dispatch", "combine")
                )
                self._open_ticket(schedule)

    def latch_descriptor(self) -> None:
        with self._lock:
            _require(self._phase == "forward", "Latch outside forward")
            torch.ops.trtllm.moe_a2a_fence_latch_descriptor(self.published, self.slot)
            self._started = True

    def dispatch_kwargs(
        self, frontend: NVLinkOneSided, payloads: tuple[torch.Tensor, ...]
    ) -> dict[str, object]:
        with self._lock:
            self.validate_frontend_epoch(frontend)
            position = len(self._frame)
            if (
                position % 2
                or position // 2 >= len(self._clients)
                or self._clients[position // 2] is not frontend
            ):
                self._block("Unsupported chunking or incomplete dispatch/combine order")
            kwargs = frontend.execution_dispatch_kwargs(self.slot, payloads)
            self._frame.append(("dispatch", frontend))
            return kwargs

    def combine_kwargs(self, frontend: NVLinkOneSided, output: torch.Tensor) -> dict[str, object]:
        with self._lock:
            self.validate_frontend_epoch(frontend)
            if not self._frame or self._frame[-1] != ("dispatch", frontend):
                self._block("Combine does not match the owned dispatch")
            kwargs = frontend.execution_combine_kwargs(self.slot, output)
            self._frame.append(("combine", frontend))
            return kwargs

    def end_forward(self) -> None:
        with self._lock:
            self._check_epoch()
            _require(self._phase == "forward", "Forward no longer executable")
            if len(self._frame) != 2 * len(self._clients):
                self._block("Model forward did not complete every frontend")
            if not torch.cuda.is_current_stream_capturing():
                self._join()
                self._finish_ticket()
                self._healthy_device()
                self._check_epoch()
            self._completed_schedule = tuple(self._frame)
            self._frame = None
            self._phase = "ready"
            self._release_uncertain_if_ready()

    def capture_started(
        self,
        graph: torch.cuda.CUDAGraph,
        pool: tuple[int, int],
        *,
        graph_key: Hashable,
        resource_owner: CUDAGraphRunner,
    ) -> None:
        with self._lock:
            _require(self._phase == "ready", "Capture blocked")
            _require(self._resource_owners is not None, "Resource owners must bind before capture")
            self._resource_reset_ready = False
            _require(resource_owner is not None, "Actual graph resource owner required")
            if all(resource_owner is not owner for owner in self._graph_resource_owners):
                self._graph_resource_owners.append(resource_owner)
            _require(id(graph) not in self._graphs, "Graph already belongs to this owner")
            self._graphs[id(graph)] = (graph, pool, graph_key, False)
            _UNCERTAIN_STATES[id(self)] = self

    def capture_finished(
        self, graph: torch.cuda.CUDAGraph, pool: tuple[int, int], *, graph_key: Hashable
    ) -> None:
        with self._lock:
            _require(self._phase == "ready", "Capture did not complete healthy forward")
            self._check_epoch()
            _require(
                id(graph) in self._graphs
                and self._graphs[id(graph)][0] is graph
                and self._graphs[id(graph)][2] == graph_key,
                "Unregistered capture or changed capture key",
            )
            self._join()
            self._healthy_device()
            self._check_epoch()
            raw_graph, executable = int(graph.raw_cuda_graph()), int(graph.raw_cuda_graph_exec())
            _require(
                raw_graph > 0 and executable > 0, "Original graph template or executable is missing"
            )
            spans = []
            for phase, frontend in self._completed_schedule:
                sequence, recorded = frontend.captured_allocation_spans(phase)
                _require(
                    sequence == self._execution_sequence, "Captured allocation spans are stale"
                )
                spans.append(recorded)
            inventory = self._retained_inventory()
            footprint = CapturedExecutionFootprint(
                graph_key=graph_key,
                graph_owner_id=id(graph),
                raw_graph=raw_graph,
                executable=executable,
                capture_generation=self.generation,
                execution_sequence=self._execution_sequence,
                workspace_geometry=self._geometry,
                mapping_token=inventory[0][2],
                descriptor_addresses=tuple(
                    int(tensor.data_ptr())
                    for tensor in (self.abort, self.descriptor, self.published, self.slot)
                ),
                schedule=tuple(
                    (phase, id(frontend)) for phase, frontend in self._completed_schedule
                ),
                spans=tuple(spans),
            )
            self._graphs[id(graph)] = (graph, pool, graph_key, True)
            self._graph_schedules[id(graph)] = self._completed_schedule
            self._capture_footprints[id(graph)] = footprint
            self._release_uncertain_if_ready()

    @property
    def captured_footprints(self) -> tuple[CapturedExecutionFootprint, ...]:
        with self._lock:
            _require(
                bool(self._graphs)
                and set(self._capture_footprints) == set(self._graphs)
                and all(record[3] for record in self._graphs.values()),
                "Original capture footprints are incomplete",
            )
            mapping_token = self._retained_inventory()[0][2]
            addresses = tuple(
                int(tensor.data_ptr())
                for tensor in (self.abort, self.descriptor, self.published, self.slot)
            )
            for graph_id, footprint in self._capture_footprints.items():
                graph = self._graphs[graph_id][0]
                _require(
                    footprint.graph_owner_id == id(graph)
                    and footprint.graph_key == self._graphs[graph_id][2]
                    and footprint.raw_graph == int(graph.raw_cuda_graph())
                    and footprint.executable == int(graph.raw_cuda_graph_exec())
                    and footprint.mapping_token is mapping_token
                    and footprint.descriptor_addresses == addresses
                    and all(
                        client.execution_geometry() == footprint.workspace_geometry
                        for client in self._clients
                    ),
                    "Original captured graph or allocation identity changed",
                )
            return tuple(self._capture_footprints.values())

    def before_replay(self, graph: torch.cuda.CUDAGraph, *, graph_key: Hashable) -> None:
        with self._lock:
            _require(self._phase == "ready", "Replay blocked or outstanding")
            record = self._graphs.get(id(graph))
            _require(
                record is not None and record[0] is graph and record[2] == graph_key and record[3],
                "Replay graph/key was not captured by this owner",
            )
            self._check_epoch()
            self._register_stream(torch.cuda.current_stream(self._clients[0].workspace.device))
            self._execution_sequence += 1
            self._phase = "replay"
            _UNCERTAIN_STATES[id(self)] = self
            schedule = self._graph_schedules[id(graph)]
            self._open_ticket(schedule)
            if self._watchdog is not None:
                for phase, frontend in schedule:
                    self._coordinator.watch_collective(self._watchdog, phase, None)

    def replay_finished(self) -> None:
        with self._lock:
            self._check_epoch()
            _require(self._phase == "replay", "Replay no longer executable")
            self._join()
            self._finish_ticket()
            self._healthy_device()
            self._check_epoch()
            self._phase = "ready"
            self._release_uncertain_if_ready()

    @property
    def execution_sequence(self) -> int:
        with self._lock:
            return self._execution_sequence

    def _retained_inventory(self) -> tuple:
        memory = self.memory
        with type(memory)._locked_record(memory.ptr) as record:
            _require(
                self.lifecycle._execution_state is self
                and record.retained_backing is self.lifecycle._execution_backing_token
                and record.retained_backing is not None,
                "Original mapped backing is not retained",
            )
            backing = (
                memory,
                record,
                record.retained_backing,
                record.state,
                tuple(record.mem_handles),
                record.comm_membership,
                record.start_address,
                record.address_offset,
                record.rank_stride,
                memory.ptr,
                memory.segment_size,
            )
        tensors = tuple(
            (tensor, tensor.data_ptr())
            for tensor in (self.abort, self.descriptor, self.published, self.slot)
        )
        return (
            backing,
            tensors,
            tuple(self._graphs.items()),
            tuple(self._graph_schedules.items()),
            tuple(client.execution_geometry() for client in self._clients),
            (self._model_owner, self._resource_owners),
            tuple(self._streams),
        )

    def retain_failed_stop(self) -> None:
        """Keep mapped storage and graph roots after an incomplete local STOP."""
        with self._lock:
            _require(self._phase != "closed", "Cannot retain an already released owner")
            self._phase = "blocked"
            _UNCERTAIN_STATES[id(self)] = self

    def begin_local_join(self) -> None:
        """Retain model and auxiliary owners before the scheduler waits on their work."""
        with self._lock:
            _require(
                (self._local_join_owned and self._phase != "closed")
                or self._phase in ("ready", "idle-held"),
                "Local join has no idle owner",
            )
            self._local_join_owned = True
            _UNCERTAIN_STATES[id(self)] = self

    def join_idle(self, *, streams: tuple, deadline_monotonic: float) -> LocalExecutionJoin:
        """Join local GPU issuers without closing admission or authorizing replay."""
        return self._join_owned(streams, deadline_monotonic, stop=False)

    def hold_idle(self, receipt: LocalExecutionJoin) -> None:
        """Seal local GPU issuance after the original scheduler joins its idle owners."""
        with self._lock:
            self.validate_join(receipt)
            _require(not receipt.stopped, "An idle boundary cannot reuse STOP")
            self._idle_join = receipt
            self._phase = "idle-held"
            _UNCERTAIN_STATES[id(self)] = self

    def stop(self, *, streams: tuple, deadline_monotonic: float) -> LocalExecutionJoin:
        """Close local issuance and retire callbacks; retain mapped storage and graphs."""
        return self._join_owned(streams, deadline_monotonic, stop=True)

    def finish_stop(
        self, receipt: LocalExecutionJoin, *, deadline_monotonic: float
    ) -> LocalExecutionJoin:
        """Join work queued by the native bridge without reopening local issuance."""
        self.validate_join(receipt)
        _require(
            receipt.stopped and receipt.watchdog_retired,
            "Final join requires the original callback-retired STOP",
        )
        return self._join_owned((), deadline_monotonic, stop=True, previous=receipt)

    def _join_owned(
        self,
        streams: tuple,
        deadline: float,
        *,
        stop: bool,
        previous: LocalExecutionJoin | None = None,
    ) -> LocalExecutionJoin:
        _require(
            math.isfinite(deadline) and time.monotonic() < deadline,
            "Execution join requires a live monotonic deadline",
        )
        with self._lock:
            _require(
                (
                    self._phase == ("stopped" if previous is not None else "ready")
                    or (stop and previous is None and self._phase == "idle-held")
                )
                and self._ticket is None
                and self._frame is None
                and self._stop_receipt is previous
                and self._graphs
                and all(record[3] for record in self._graphs.values()),
                "Idle execution join requires completed original graphs",
            )
            self._check_epoch()
            _require(
                self.ep_group_health.snapshot() == self._construction_snapshot,
                "Idle execution join requires original membership",
            )
            for stream in streams:
                _require(
                    stream.device == self._clients[0].workspace.device,
                    "Issuer stream belongs to a different device",
                )
                self._register_stream(stream)
            inventory = self._retained_inventory()
            sequence = self._execution_sequence
            events = []
            self._pending_join = (inventory, events, previous)
            if stop:
                self._phase = "stopping"
            _UNCERTAIN_STATES[id(self)] = self
        try:
            for stream in inventory[-1]:
                event = torch.cuda.Event(blocking=False)
                events.append(event)
                event.record(stream)
            for event in events:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Local execution join exceeded its deadline")
                wait_cuda_event(event, remaining)
            self._finish_ticket(deadline_monotonic=deadline)
            if stop and previous is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Watchdog retirement exceeded STOP deadline")
                self.lifecycle.stop_execution_watchdog(self, timeout_s=remaining)
            with self._lock:
                _require(
                    time.monotonic() < deadline
                    and self._phase == ("stopping" if stop else "ready")
                    and self._execution_sequence == sequence
                    and self._retained_inventory() == inventory,
                    "Execution ownership changed during local join",
                )
                self._check_epoch()
                _require(
                    self._watchdog is None or self._watchdog.last_error is None,
                    "Watchdog completion failed",
                )
                receipt = LocalExecutionJoin(
                    self,
                    self._installed.generation,
                    sequence,
                    inventory,
                    tuple(events),
                    self._ticket is None,
                    stop,
                    stop,
                    previous,
                )
                self._pending_join = None
                if stop:
                    self._watchdog = None
                    self._phase = "stopped"
                    self._stop_receipt = receipt
                else:
                    self._release_uncertain_if_ready()
                return receipt
        except (RuntimeError, TimeoutError):
            with self._lock:
                self._phase = "blocked"
                _UNCERTAIN_STATES[id(self)] = self
            raise

    def validate_join(self, receipt: LocalExecutionJoin) -> LocalExecutionJoin:
        """Check retained host ownership; this does not prove remote write drainage."""
        _require(self._lock.acquire(blocking=False), "Execution owner is busy")
        try:
            _require(
                receipt.owner is self
                and (
                    self._phase == ("stopped" if receipt.stopped else "ready")
                    or (
                        not receipt.stopped
                        and self._phase == "idle-held"
                        and receipt is self._idle_join
                    )
                )
                and self._ticket is None
                and self._frame is None
                and self._execution_sequence == receipt.execution_sequence
                and self._installed.generation == receipt.generation
                and self._retained_inventory() == receipt.inventory,
                "Execution join is stale",
            )
            if receipt.stopped:
                _require(
                    receipt is self._stop_receipt and self._watchdog is None,
                    "STOP receipt or callback owner changed",
                )
            self._check_epoch()
            return receipt
        finally:
            self._lock.release()

    def before_graph_reset(self) -> None:
        with self._lock:
            _require(
                self._stop_receipt is None
                and self._idle_join is None
                and self._phase not in ("closed", "stopping", "stopped"),
                "Closed or scheduler-held state cannot reset executable graphs",
            )
            self._reset_previous_phase = self._phase
            self._phase = "resetting"
            _UNCERTAIN_STATES[id(self)] = self
            self._join()

    def graphs_reset(self, graphs: tuple[torch.cuda.CUDAGraph, ...]) -> None:
        with self._lock:
            _require(self._phase == "resetting", "Reset acknowledgement outside reset")
            self._join()
            for graph in graphs:
                record = self._graphs.get(id(graph))
                _require(record is not None and record[0] is graph, "Unknown reset graph")
                del self._graphs[id(graph)]
                self._graph_schedules.pop(id(graph), None)
                self._capture_footprints.pop(id(graph), None)
            if self._reset_previous_phase == "ready":
                self._check_epoch()
                self._healthy_device()
                self._phase = "ready"
                if not self._graphs:
                    self._resource_reset_ready = True
                    self._release_uncertain_if_ready()
            else:
                self._phase = "blocked"

    def before_frontend_destroy(self, frontend: NVLinkOneSided) -> None:
        with self._lock:
            _require(self._phase == "closed", "Execution owner retains frontend and mapped backing")

    def close(self) -> None:
        with self._lock:
            if self._phase == "closed":
                return
            _require(self._phase != "closing", "Final close already outstanding; retain ownership")
            healthy = (
                self._phase == "ready"
                and self.ep_group_health.snapshot() == self._construction_snapshot
                and all(record[3] for record in self._graphs.values())
            )
            self.before_graph_reset()
            graphs = tuple(record[0] for record in self._graphs.values())
            for graph in graphs:
                graph.reset()
            self.graphs_reset(graphs)
            self._join()
            self._finish_ticket()
            if not healthy or self._phase != "ready":
                self._block(
                    "Failed/uncertain execution needs authentic issuer-terminal retention acknowledgement"
                )
            self._check_epoch()
            self._healthy_device()
            _require(not self._graphs, "Executable graph owners remain")
            self._phase = "closing"
            _UNCERTAIN_STATES[id(self)] = self
        # Callback/thread retirement cannot run under the model execution lock.
        self.lifecycle.stop_execution_watchdog(self, timeout_s=self._join_timeout_s)
        with self._lock:
            _require(self._phase == "closing", "Final close ownership changed")
            self._watchdog = None
            self.lifecycle.release_execution_state(self)
            self._claimed_lifecycle = False
            self.ep_group_health._release_commit_owner(self._health_token)
            self._health_token = None
            self.abort = self.descriptor = self.published = self.slot = None
            self._clients = ()
            self._streams.clear()
            self._phase = "closed"
            self._model_owner = None
            self._graph_resource_owners.clear()
            self._resource_owners = None
            _UNCERTAIN_STATES.pop(id(self), None)

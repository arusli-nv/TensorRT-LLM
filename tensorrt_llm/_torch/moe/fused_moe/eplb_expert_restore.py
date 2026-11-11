# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Restore a lost rank's experts into EPLB spare slots under WideEP fault tolerance."""

import hashlib
import os
import stat
import time
from dataclasses import dataclass
from multiprocessing import resource_tracker, shared_memory
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import tensorrt_llm.bindings.internal.runtime as _tbr
from tensorrt_llm._utils import fault_tolerance_replacement
from tensorrt_llm.logger import logger

from .moe_load_balancer import HostMoeTensorSharer

if TYPE_CHECKING:
    from .moe_load_balancer import MoeLoadBalancer


def plan_layer_placement(
    current: np.ndarray,
    home: np.ndarray,
    live: np.ndarray,
    active_slot_count: int,
    expert_count: int,
) -> np.ndarray:
    """Return a [ranks, slots] placement that holds every expert on the ``live`` ranks.

    The plan is deterministic, so every rank plans the same placement from the same arguments.

    Raises:
        ValueError: The live ranks' spare slots cannot hold every expert.
    """
    placement = np.full_like(current, -1)
    placement[live, :active_slot_count] = home[live, :active_slot_count]
    replicas = np.bincount(placement[placement >= 0], minlength=expert_count)
    live_ranks = np.flatnonzero(live)
    spare_slots = range(active_slot_count, current.shape[1])
    for rank in live_ranks:
        for slot in spare_slots:
            expert_id = current[rank, slot]
            if expert_id >= 0 and replicas[expert_id] == 0:
                placement[rank, slot] = expert_id
                replicas[expert_id] = 1
    for expert_id in np.flatnonzero(replicas == 0):
        empty = [
            (
                np.count_nonzero(placement[rank, active_slot_count:] >= 0),
                rank,
                current[rank, slot] < 0,
                slot,
            )
            for rank in live_ranks
            for slot in spare_slots
            if placement[rank, slot] < 0
        ]
        if not empty:
            raise ValueError(
                f"The spare slots of live ranks {live_ranks.tolist()} cannot hold expert {expert_id}"
            )
        _, rank, _, slot = min(empty)
        placement[rank, slot] = expert_id
        replicas[expert_id] = 1
    return placement


@dataclass(frozen=True, eq=False)
class ExpertRestorePlan:
    """Placement of every MoE layer for a membership, identical on every live rank except ``device_bytes``."""

    live_ranks: Tuple[bool, ...]
    placements: Tuple[np.ndarray, ...]
    digest: str
    device_bytes: int


def _phase_done(times: Optional[Dict[str, float]], phase: str, start_s: float) -> float:
    now_s = time.perf_counter()
    if times is not None:
        times[phase] = (now_s - start_s) * 1e3
    return now_s


def _outlive_this_process(shm: shared_memory.SharedMemory) -> None:
    """Keep ``shm`` after this process exits, since other ranks on the node still use it."""
    resource_tracker.unregister(shm._name, "shared_memory")


def host_copies_prefix(shared_memory_base_name: str) -> str:
    """Return the name prefix of the host-copy segments under ``shared_memory_base_name``."""
    return f"{shared_memory_base_name}_"


_FINGERPRINT_BYTES = 32
SHARED_MEMORY_DIR = "/dev/shm"
_HOST_COPIES_CHECK_S = 1.0


def host_copies_key(rank: int) -> str:
    """Return the store key a replacement of ``rank`` sets once its node holds the rank's host copies."""
    return f"eplb_host_copies/{rank}"


class DeploymentHostCopies(HostMoeTensorSharer):
    """Host tensor sharer whose segments outlive their processes so that a replacement rank can reattach them."""

    data_offset = HostMoeTensorSharer.align_size(_FINGERPRINT_BYTES)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.process_owned_shms = []

    def _own_segment(self, size: int) -> Tuple[shared_memory.SharedMemory, bool]:
        if fault_tolerance_replacement() is not None:
            shm = self._attach(self.local_rank)
            if shm is not None:
                _outlive_this_process(shm)
                return shm, True
        shm, _ = super()._own_segment(size)
        self.process_owned_shms.append(shm)
        return shm, False

    def finalize_layer_weights(self):
        super().finalize_layer_weights()
        if self.own_shm in self.process_owned_shms:
            # The fingerprint is written last, so a process that finds it also finds the host weights.
            self.own_shm.buf[:_FINGERPRINT_BYTES] = self._fingerprint(self.local_rank)

    def _peer_segment(self, rank: int) -> shared_memory.SharedMemory:
        shm = self._attach(rank)
        if shm is None:
            raise RuntimeError(
                f"EPLB shared memory {self.get_shared_memory_name(rank)} is missing or does not hold its host "
                "weights"
            )
        if fault_tolerance_replacement() is None:
            self.process_owned_shms.append(shm)
        else:
            _outlive_this_process(shm)
        return shm

    def pre_shutdown_cleanup(self):
        # A segment closes only once no view of it is left.
        self.host_weights.clear()
        for shm in self.imported_shms:
            if shm not in self.process_owned_shms:
                shm.close()
        self.imported_shms = [shm for shm in self.imported_shms if shm in self.process_owned_shms]
        super().pre_shutdown_cleanup()

    def post_shutdown_cleanup(self):
        if self.own_shm not in self.process_owned_shms:
            self.own_shm = None
        super().post_shutdown_cleanup()

    def node_bytes(self, slot_bytes: List[int]) -> int:
        """Return the bytes this layer's segments of every local rank take in /dev/shm."""
        return (
            self.expert_count * sum(self.align_size(size) for size in slot_bytes)
            + self.local_size * self.data_offset
        )

    def node_bytes_held(self) -> int:
        """Return the bytes this layer's existing segments on the node take in /dev/shm."""
        held = 0
        for rank in range(self.local_size):
            try:
                path = os.path.join(SHARED_MEMORY_DIR, self.get_shared_memory_name(rank))
                held += os.stat(path).st_blocks * 512
            except FileNotFoundError:
                pass
        return held

    def _fingerprint(self, rank: int) -> bytes:
        layout = (
            self.layer_id,
            self.expert_count,
            self.local_size,
            rank,
            [
                (name, str(dtype), tuple(shape))
                for name, (dtype, shape) in sorted(self.name_info.items())
            ],
        )
        return hashlib.sha256(repr(layout).encode()).digest()

    def _attach(self, rank: int) -> Optional[shared_memory.SharedMemory]:
        """Return the segment of local ``rank`` if this user owns it with mode 0600 and its fingerprint matches."""
        try:
            shm = shared_memory.SharedMemory(name=self.get_shared_memory_name(rank))
        except FileNotFoundError:
            return None
        status = os.fstat(shm._fd)
        if (
            status.st_uid == os.getuid()
            and stat.S_IMODE(status.st_mode) == 0o600
            and bytes(shm.buf[:_FINGERPRINT_BYTES]) == self._fingerprint(rank)
        ):
            return shm
        shm.close()
        resource_tracker.unregister(shm._name, "shared_memory")
        return None

    def holds_host_copies(self, rank: int) -> bool:
        """Return whether the segment of local ``rank`` holds that rank's host weights."""
        shm = self._attach(rank)
        if shm is None:
            return False
        shm.close()
        if all(owned._name != shm._name for owned in self.process_owned_shms):
            resource_tracker.unregister(shm._name, "shared_memory")
        return True

    def keep_for_the_deployment(self):
        """Let the segments this process created or attached outlive it until the node supervisor unlinks them."""
        for shm in self.process_owned_shms:
            _outlive_this_process(shm)
        self.process_owned_shms = []


class EplbExpertRestore:
    """Size the spare slots of a ``MoeLoadBalancer`` and restore a lost rank's experts into them."""

    def __init__(self, balancer: "MoeLoadBalancer"):
        self.balancer = balancer
        self.largest_failure = self._largest_failure_size(self._rank_nodes())
        if (
            self.largest_failure
            and balancer.layer_updates_per_iter > 0
            and not _tbr.host_accessible_device_memory_uses_gdrcopy()
        ):
            raise ValueError(
                "WideEP fault tolerance keeps the spare slots of online EPLB host accessible through GDRCopy, "
                "which this system does not use for EPLB slots (GPU memory is a NUMA node, or GDRCopy is "
                "unavailable): use static EPLB (layer_updates_per_iter 0) or turn fault tolerance off "
                "(enable_wide_ep_fault_tolerance=False)"
            )
        self.live_ranks: Tuple[bool, ...] = (True,) * balancer.ep_size
        self.backed_spare_slot_bytes = 0

    def spare_slot_count(self, expert_count: int, eplb_slot_count: int) -> int:
        """Return the spare slots per rank that hold every expert the largest recoverable failure can lose."""
        lost_experts = min(self.largest_failure * eplb_slot_count, expert_count)
        live_ranks = self.balancer.ep_size - self.largest_failure
        return -(-lost_experts // live_ranks)

    def _rank_nodes(self) -> List[int]:
        return [self.balancer.mapping.get_node_rank(rank) for rank in range(self.balancer.ep_size)]

    def _node_ranks(self) -> List[int]:
        nodes = self._rank_nodes()
        return [rank for rank, node in enumerate(nodes) if node == nodes[self.balancer.ep_rank]]

    @staticmethod
    def _recoverable_failures(nodes: List[int]) -> List[List[int]]:
        """Return the ranks of each recoverable failure: a node other than rank 0's, or one rank on rank 0's node."""
        node_ranks: Dict[int, List[int]] = {}
        for rank, node in enumerate(nodes):
            node_ranks.setdefault(node, []).append(rank)
        rank_0_node = node_ranks.pop(nodes[0], [])
        return list(node_ranks.values()) + [[rank] for rank in rank_0_node[1:]]

    @staticmethod
    def _largest_failure_size(nodes: List[int]) -> int:
        return max(map(len, EplbExpertRestore._recoverable_failures(nodes)), default=0)

    def largest_failure_restore_bytes(self) -> int:
        """Return the most device memory this rank backs to restore the experts of any recoverable failure."""
        ep_rank, ep_size = self.balancer.ep_rank, self.balancer.ep_size
        current = [np.array(ranks, dtype=np.int32) for _, ranks in self.current_placement()]
        return max(
            (
                self.plan_expert_restore(
                    [rank not in failure for rank in range(ep_size)], current
                ).device_bytes
                for failure in self._recoverable_failures(self._rank_nodes())
                if ep_rank not in failure
            ),
            default=0,
        )

    def current_placement(self) -> List[Tuple[int, List[List[int]]]]:
        """Return, per layer, its expert count and this rank's view of the placement."""
        return [
            (
                layer.expert_count,
                [list(rank_slots) for rank_slots in layer.get_old_rank_expert_ids()],
            )
            for layer in self.balancer.single_layer_load_balancers
        ]

    def _host_tensor_sharers(self) -> List[DeploymentHostCopies]:
        return [
            layer.host_tensor_sharer
            for layer in self.balancer.single_layer_load_balancers
            if layer.need_load_shared_weights()
        ]

    def wait_for_node_host_copies(self):
        """Wait until every other rank on this replacement rank's node holds its host copies.

        Raises:
            ReplacementCannotStart: The host copies of a live rank, or of a rank whose replacement announced them,
                are missing or broken.
            RuntimeError: The deployment ended.
        """
        from tensorrt_llm._torch.distributed.membership import (
            RECORD_KEY,
            MembershipRecord,
            MembershipState,
            ReplacementCannotStart,
            rendezvous_store,
            wait_for_keys,
        )

        ep_rank = self.balancer.ep_rank
        store = rendezvous_store()
        store.set(host_copies_key(ep_rank), b"")
        sharers = self._host_tensor_sharers()

        def missing() -> List[int]:
            return [
                rank
                for local_rank, rank in enumerate(self._node_ranks())
                if rank != ep_rank
                and not all(sharer.holds_host_copies(local_rank) for sharer in sharers)
            ]

        def record() -> MembershipRecord:
            return MembershipRecord.model_validate_json(store.get(RECORD_KEY))

        def check_running(unsigned) -> None:
            state = record().state
            if state in (MembershipState.SHUTDOWN, MembershipState.ABORT):
                raise RuntimeError(
                    f"The deployment ended ({state.name}) before the EPLB host copies of ranks {waited} were built "
                    "again"
                )

        waited = missing()
        if not waited:
            return
        live = [rank for rank in waited if record().is_live(rank)]
        if live:
            raise ReplacementCannotStart(
                f"The EPLB host copies of live ranks {live} on this node are missing or broken"
            )
        logger.info(
            f"Waiting for the replacements of ranks {waited} to build their EPLB host copies again"
        )
        wait_for_keys(
            store, [host_copies_key(rank) for rank in waited], _HOST_COPIES_CHECK_S, check_running
        )
        broken = missing()
        if broken:
            raise ReplacementCannotStart(
                f"The EPLB host copies the replacements of ranks {broken} on this node built are missing or broken"
            )

    def keep_host_copies_for_the_deployment(self) -> str:
        """Let the node's host copies outlive this process and return their name prefix."""
        for sharer in self._host_tensor_sharers():
            sharer.keep_for_the_deployment()
        return host_copies_prefix(self.balancer.shared_memory_base_name)

    def plan_expert_restore(
        self,
        live_ranks: Sequence[bool],
        current: Sequence[np.ndarray],
        rejoining: Optional[Sequence[bool]] = None,
    ) -> ExpertRestorePlan:
        """Plan every layer's placement for ``live_ranks``.

        Args:
            live_ranks: Per EP rank, whether it is live.
            current: Per layer, the placement the live ranks agreed on, so every live rank plans the same.
            rejoining: Per EP rank, whether it rejoins. The active slots then take the initial placement.

        Raises:
            RuntimeError: An iteration is active.
            ValueError: The live ranks cannot hold every expert, or a layer would need to load an expert without
                host weights.
        """
        balancer = self.balancer
        live = np.asarray(live_ranks, dtype=bool)
        if live.shape != (balancer.ep_size,) or not live[balancer.ep_rank]:
            raise ValueError(f"EP rank {balancer.ep_rank} plans for live ranks {live.tolist()}")
        if balancer.in_iter:
            raise RuntimeError("Cannot plan an expert restore while an iteration is active")
        rejoined = rejoining is not None and bool((live & np.asarray(rejoining, dtype=bool)).any())
        digest = hashlib.sha256(live.tobytes())
        placements = []
        device_bytes = 0
        for layer, layer_current in zip(balancer.single_layer_load_balancers, current, strict=True):
            layer_current = np.asarray(layer_current, dtype=np.int32)
            home = (
                layer.initial_placement.reshape(layer_current.shape) if rejoined else layer_current
            )
            placement = plan_layer_placement(
                layer_current, home, live, layer.active_slot_count_per_rank, layer.expert_count
            )
            loaded = (placement >= 0) & (placement != layer_current)
            if loaded.any() and not layer.need_load_shared_weights():
                raise ValueError(
                    f"Layer {layer.get_layer_idx()} has no host weights to load experts "
                    f"{sorted(set(placement[loaded].tolist()))} from"
                )
            device_bytes += layer.spare_slot_bytes_to_back(
                layer.spare_slots_where(placement[balancer.ep_rank], filled=True)
            )
            digest.update(placement.tobytes())
            placements.append(placement)
        return ExpertRestorePlan(
            tuple(bool(rank) for rank in live),
            tuple(placements),
            digest.hexdigest()[:16],
            device_bytes,
        )

    def set_spare_slots_backed(self, layer_slots: Sequence[Sequence[int]], backed: bool) -> int:
        """Back spare slots with device memory or map them back to the shared scratch; return the bytes changed."""
        tensors, slots, slot_counts = [], [], []
        for layer, spare_slots in zip(
            self.balancer.single_layer_load_balancers, layer_slots, strict=True
        ):
            if not spare_slots:
                continue
            for tensor in layer.unbacked_slot_tensors.values():
                tensors.append(tensor)
                slots.extend(spare_slots)
                slot_counts.append(len(spare_slots))
        if not tensors:
            return 0
        changed = torch.ops.trtllm.moe_unbacked_slots_set_backed(
            tensors, slots, slot_counts, backed
        )
        self.backed_spare_slot_bytes += changed if backed else -changed
        return changed

    def release_empty_spare_slots(self) -> int:
        """Map the spare slots the installed placement leaves empty back to the shared scratch; return the bytes."""
        balancer = self.balancer
        if balancer.in_iter:
            raise RuntimeError("Cannot release spare slots while an iteration is active")
        return self.set_spare_slots_backed(
            [
                layer.spare_slots_where(
                    np.asarray(layer.get_old_rank_expert_ids()[balancer.ep_rank]), filled=False
                )
                for layer in balancer.single_layer_load_balancers
            ],
            False,
        )

    def restore_lost_experts(
        self, plan: ExpertRestorePlan, times: Optional[Dict[str, float]] = None
    ) -> int:
        """Install ``plan`` on this rank and return the device bytes mapped.

        Every live rank must restore the same plan before any rank routes with it.
        """
        balancer = self.balancer
        if balancer.in_iter:
            raise RuntimeError("Cannot restore experts while an iteration is active")
        layers = balancer.single_layer_load_balancers
        if plan.live_ranks == self.live_ranks and all(
            np.array_equal(placement, layer.get_old_rank_expert_ids())
            for placement, layer in zip(plan.placements, layers)
        ):
            return 0
        start_s = time.perf_counter()
        # A spare slot is backed before an expert is loaded into it and before routing can reach it.
        mapped = self.set_spare_slots_backed(
            [
                layer.spare_slots_where(placement[balancer.ep_rank], filled=True)
                for layer, placement in zip(layers, plan.placements)
            ],
            True,
        )
        start_s = _phase_done(times, "back", start_s)
        balancer.load_balancer_impl.restore_placement(
            [rank for rank, live in enumerate(plan.live_ranks) if not live],
            [placement.tolist() for placement in plan.placements],
        )
        _phase_done(times, "load and install", start_s)
        self.live_ranks = plan.live_ranks
        return mapped

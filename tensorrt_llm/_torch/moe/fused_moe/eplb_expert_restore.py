# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WideEP fault tolerance on EPLB: the spare slots, the restore of a lost rank's experts into them, and the node's
host copies of every expert that the restore loads from and that outlive their processes while the deployment
serves."""

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
    """The placement ([ranks, slots] expert ids, -1 = empty) that holds every expert on the ``live`` ranks.

    The live ranks' active slots (the first ``active_slot_count``) take their experts of ``home``. A spare slot
    keeps its expert of ``current`` if no slot holds that expert yet. Each expert still missing goes, in id order,
    to an empty spare slot of the live rank with the fewest filled spare slots (then the lowest rank, a slot that
    held an expert, the lowest slot). Every rank plans the same placement from the same arguments.

    Raises:
        ValueError: The live ranks' spare slots cannot hold every expert: the one rule for whether a membership
            keeps every expert (an install whose plan fails makes the membership authority abort).
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
    """Where every MoE layer's experts go for a membership; the same on every live rank but ``device_bytes``.

    Attributes:
        live_ranks: Per EP rank, whether it is live.
        placements: Per layer, the placement to restore ([ranks, slots] expert ids, -1 = an empty slot).
        digest: Identifies ``live_ranks`` and ``placements``: equal on ranks that planned the same.
        device_bytes: The device memory the spare slots this rank backs take.
    """

    live_ranks: Tuple[bool, ...]
    placements: Tuple[np.ndarray, ...]
    digest: str
    device_bytes: int


def _phase_done(times: Optional[Dict[str, float]], phase: str, start_s: float) -> float:
    """Records in ``times`` (if given) the milliseconds since ``start_s`` as those of ``phase``; returns now."""
    now_s = time.perf_counter()
    if times is not None:
        times[phase] = (now_s - start_s) * 1e3
    return now_s


def _outlive_this_process(shm: shared_memory.SharedMemory) -> None:
    """Keep ``shm`` if this process dies (Python's resource tracker would unlink it): the node's live ranks and a
    replacement rank keep using it."""
    resource_tracker.unregister(shm._name, "shared_memory")


def host_copies_prefix(shared_memory_base_name: str) -> str:
    """The prefix of every host-copy segment name under ``shared_memory_base_name``
    (``HostMoeTensorSharer.get_shared_memory_name``), by which the node supervisor unlinks them."""
    return f"{shared_memory_base_name}_"


# A segment of host copies starts with a fingerprint of what it holds, written last.
_FINGERPRINT_BYTES = 32
# Where the segments (POSIX shared memory) live on Linux.
SHARED_MEMORY_DIR = "/dev/shm"
# How often a replacement rank waiting for the host copies a replacement of its node builds checks that the
# deployment still runs.
_HOST_COPIES_CHECK_S = 1.0


def host_copies_key(rank: int) -> str:
    """The store key a replacement of EP ``rank`` sets once its node holds the rank's host copies of every layer."""
    return f"eplb_host_copies/{rank}"


class DeploymentHostCopies(HostMoeTensorSharer):
    """The host tensor sharer of WideEP fault tolerance: the host copies of a node outlive their processes while
    the deployment serves, and a replacement rank reattaches those of the rank it replaces if they are intact."""

    data_offset = HostMoeTensorSharer.align_size(_FINGERPRINT_BYTES)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The segments that end with this process (its shutdown, or Python's resource tracker if it dies). They
        # become the deployment's once it serves (keep_for_the_deployment).
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
            # Last: a process that finds the fingerprint finds the host weights.
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
        # Views of the shared memory left if sharing was never finalized: a segment closes only once none is left.
        self.host_weights.clear()
        # The segments of the deployment close without leaving Python's resource tracker, which no longer has them.
        for shm in self.imported_shms:
            if shm not in self.process_owned_shms:
                shm.close()
        self.imported_shms = [shm for shm in self.imported_shms if shm in self.process_owned_shms]
        super().pre_shutdown_cleanup()

    def post_shutdown_cleanup(self):
        # The segment of the deployment stays: the node supervisor unlinks it.
        if self.own_shm not in self.process_owned_shms:
            self.own_shm = None
        super().post_shutdown_cleanup()

    def node_bytes(self, slot_bytes: List[int]) -> int:
        """The bytes the host weights of this layer take in /dev/shm on the node, the segments of every local rank,
        as ``finalize_layer_weights`` lays them out; ``slot_bytes``: per weight, the bytes of one expert's slot."""
        return (
            self.expert_count * sum(self.align_size(size) for size in slot_bytes)
            + self.local_size * self.data_offset
        )

    def node_bytes_held(self) -> int:
        """The bytes the segments of this layer that exist on the node already take in /dev/shm."""
        held = 0
        for rank in range(self.local_size):
            try:
                path = os.path.join(SHARED_MEMORY_DIR, self.get_shared_memory_name(rank))
                held += os.stat(path).st_blocks * 512
            except FileNotFoundError:
                pass
        return held

    def _fingerprint(self, rank: int) -> bytes:
        """Names what the segment of local ``rank`` holds: its experts' host weights of this layer."""
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
        """The segment of local ``rank`` (tracked by Python's resource tracker) if this user's processes alone can
        access it (its owner, mode 0600, as ``SharedMemory`` creates it) and its fingerprint matches; None if it is
        missing, someone else's, incomplete or holds other host weights."""
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
        """Whether the segment of local ``rank`` holds that rank's host weights (see ``_attach``). Python's
        resource tracker keeps tracking a segment this process owns."""
        shm = self._attach(rank)
        if shm is None:
            return False
        shm.close()
        if all(owned._name != shm._name for owned in self.process_owned_shms):
            resource_tracker.unregister(shm._name, "shared_memory")
        return True

    def keep_for_the_deployment(self):
        """Once the deployment serves, the segments this process created or attached outlive it (a replacement
        rank reattaches them) until the node supervisor unlinks them."""
        for shm in self.process_owned_shms:
            _outlive_this_process(shm)
        self.process_owned_shms = []


class EplbExpertRestore:
    """WideEP fault tolerance on a ``MoeLoadBalancer``: sizes its spare slots for the largest recoverable failure,
    plans and installs the placement that restores a lost rank's experts into them, and handles the node's host
    copies the restore loads from."""

    def __init__(self, balancer: "MoeLoadBalancer"):
        """
        Raises:
            ValueError: Online EPLB's spare slots cannot be host accessible through GDRCopy on this system.
        """
        self.balancer = balancer
        # The most ranks one recoverable failure takes (it sizes the spare slots).
        self.largest_failure = self._largest_failure_size(self._rank_nodes())
        # The C++ load balancer made the choice of host accessible device memory.
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
        # Per EP rank, whether it is live in the placement restore_lost_experts installed last.
        self.live_ranks: Tuple[bool, ...] = (True,) * balancer.ep_size
        # The device bytes the spare slots backed with memory of their own take (set_spare_slots_backed).
        self.backed_spare_slot_bytes = 0

    def spare_slot_count(self, expert_count: int, eplb_slot_count: int) -> int:
        """The spare slots per rank of a layer of ``expert_count`` experts in ``eplb_slot_count`` EPLB slots per
        rank: the live ranks' spare slots take every expert the largest failure can lose, whatever the placement."""
        lost_experts = min(self.largest_failure * eplb_slot_count, expert_count)
        live_ranks = self.balancer.ep_size - self.largest_failure
        return -(-lost_experts // live_ranks)

    def _rank_nodes(self) -> List[int]:
        """Per EP rank, its node (WideEP fault tolerance requires the EP group to be every rank)."""
        return [self.balancer.mapping.get_node_rank(rank) for rank in range(self.balancer.ep_size)]

    def _node_ranks(self) -> List[int]:
        """The EP ranks on this rank's node, in rank order."""
        nodes = self._rank_nodes()
        return [rank for rank, node in enumerate(nodes) if node == nodes[self.balancer.ep_rank]]

    @staticmethod
    def _recoverable_failures(nodes: List[int]) -> List[List[int]]:
        """The ranks each recoverable failure takes (``nodes``: per rank, its node): every rank of a node other
        than rank 0's, or one rank of rank 0's node other than rank 0."""
        node_ranks: Dict[int, List[int]] = {}
        for rank, node in enumerate(nodes):
            node_ranks.setdefault(node, []).append(rank)
        rank_0_node = node_ranks.pop(nodes[0], [])
        return list(node_ranks.values()) + [[rank] for rank in rank_0_node[1:]]

    @staticmethod
    def _largest_failure_size(nodes: List[int]) -> int:
        """The most ranks one recoverable failure takes (``nodes``: per rank, its node)."""
        return max(map(len, EplbExpertRestore._recoverable_failures(nodes)), default=0)

    def largest_failure_restore_bytes(self) -> int:
        """The most device memory this rank backs to restore, from its current placement, the experts of a
        recoverable failure it survives (``plan_expert_restore``)."""
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
        """Per layer, its expert count and this rank's view of the placement ([ranks][slots] expert ids, -1 =
        empty); this rank's own row is what its slots hold."""
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
        """A replacement rank: announces that its node holds its own host copies (``host_copies_key``), then waits
        until every other rank of its node holds its host copies of every layer; a rank that is not live (per the
        membership record) has them built again by its own replacement, which announces them.

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
        """Once the deployment serves: the node's host copies outlive this process (see
        ``DeploymentHostCopies.keep_for_the_deployment``).

        Returns:
            The prefix of the host copies' names (``host_copies_prefix``), for the node supervisor.
        """
        for sharer in self._host_tensor_sharers():
            sharer.keep_for_the_deployment()
        return host_copies_prefix(self.balancer.shared_memory_base_name)

    def plan_expert_restore(
        self,
        live_ranks: Sequence[bool],
        current: Sequence[np.ndarray],
        rejoining: Optional[Sequence[bool]] = None,
    ) -> ExpertRestorePlan:
        """Plans every layer's placement for ``live_ranks`` (per EP rank, whether it is live) with
        ``plan_layer_placement``: a lost rank's experts go into spare slots of the live ranks. ``current`` is, per
        layer, the [ranks, slots] placement the live ranks agreed on (one rank's ``current_placement``), so every
        live rank plans the same placements and digest. The live ranks' active slots keep their current experts;
        when a rank rejoins (``rejoining``, per EP rank; None: none) they take their experts of the initial
        placement, which the rejoining rank loaded at startup. With online EPLB, call it once
        ``MoeLoadBalancer.abandon_iterations`` returned.

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
        """Backs spare slots of every layer (``layer_slots``: per layer, its slots) with device memory of their own
        or, with ``backed`` false, maps them to the shared scratch again; synchronizes the device once if a mapping
        changes. Slots already in that state change nothing; the others' contents are undefined afterwards.

        Returns:
            The device bytes mapped or released.
        """
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
        """Maps the spare slots that the installed placement leaves empty, which routing does not reach, to the
        shared scratch again while no iteration runs (see ``set_spare_slots_backed``).

        Returns:
            The device bytes released.
        """
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
        """Installs ``plan`` on this rank while no iteration runs: backs the spare slots it fills, then the C++ load
        balancer validates it, loads every expert the local slots do not hold yet from the host weights and restores
        the placement routing reads. The spare slots it leaves empty keep their memory until
        ``release_empty_spare_slots``.

        Every live rank must restore the same plan before any routes with it (the caller's barrier), with
        ``plan.device_bytes`` of device memory free (``KVCacheManagerV2.resize_gpu_quota``).

        Args:
            plan: The plan of ``plan_expert_restore``.
            times: If given, receives the milliseconds each phase took: "back" and "load and install".

        Returns:
            The device bytes mapped.
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

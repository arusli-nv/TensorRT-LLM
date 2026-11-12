# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the spare EPLB slots that let the active ranks hold every expert after a rank failure."""

import os
import secrets
from contextlib import nullcontext
from multiprocessing import resource_tracker
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from tensorrt_llm import _wide_ep_fault_tolerance
from tensorrt_llm._torch.distributed import membership
from tensorrt_llm._torch.distributed.membership import (
    RECORD_KEY,
    REPLACEMENT_CANNOT_START,
    MembershipRecord,
    MembershipState,
    ReplacementCannotStart,
)
from tensorrt_llm._torch.moe.fused_moe.eplb_expert_restore import (
    DeploymentHostWeights,
    EplbExpertRestore,
    FaultTolerantMoeLoadBalancer,
    host_weights_key,
    plan_layer_placement,
)
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import MoeLoadBalancer
from tensorrt_llm._torch.pyexecutor import rank_failure_recovery
from tensorrt_llm._wide_ep_fault_tolerance import FAULT_TOLERANCE_RANK_ENV
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

_MLB = "tensorrt_llm._torch.moe.fused_moe.moe_load_balancer"
_ER = "tensorrt_llm._torch.moe.fused_moe.eplb_expert_restore"
_USES_GDRCOPY = "tensorrt_llm.bindings.internal.runtime.host_accessible_device_memory_uses_gdrcopy"
WORLD_SIZE = 4
EXPERTS = 4


@pytest.fixture(autouse=True)
def deployment_nonce(monkeypatch):
    """Give each test its own deployment nonce, which names its host weights."""
    monkeypatch.setattr(rank_failure_recovery, "_deployment_nonce", secrets.token_hex(8))


def _fault_tolerance(monkeypatch, largest_failure):
    """Turn fault tolerance on, with nodes [0, 1] and [2, 3] and a largest failure of ``largest_failure`` ranks."""
    monkeypatch.setattr(f"{_MLB}.wide_ep_fault_tolerance_enabled", lambda: True)
    monkeypatch.setattr(EplbExpertRestore, "_rank_nodes", lambda self: [0, 0, 1, 1])
    monkeypatch.setattr(
        EplbExpertRestore, "_largest_failure_size", staticmethod(lambda hosts: largest_failure)
    )


def _load_balancer(
    monkeypatch, *, largest_failure, eplb_slots, layer_updates_per_iter=1, ep_rank=0
):
    """Return a MoeLoadBalancer with Wide-EP fault tolerance over one layer of ``eplb_slots`` per rank."""
    _fault_tolerance(monkeypatch, largest_failure)
    with (
        patch("tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer"),
        patch(_USES_GDRCOPY, return_value=True),
        patch(f"{_ER}.DeploymentHostWeights"),
        patch("torch.cuda.Event"),
    ):
        load_balancer = FaultTolerantMoeLoadBalancer(
            ep_rank, WORLD_SIZE, layer_updates_per_iter, mapping=None
        )
        load_balancer.add_layer(EXPERTS, 2, eplb_slots, aux_stream=Mock())
    return load_balancer


@pytest.mark.parametrize(
    "largest_failure, layer_updates_per_iter, uses_gdrcopy, refused",
    [
        (2, 1, False, True),
        (2, 1, True, False),
        # Static EPLB writes no slot from the host, and without a rank to lose there is no spare slot.
        (2, 0, False, False),
        (0, 1, False, False),
    ],
)
def test_online_spare_slots_need_gdrcopy(
    monkeypatch, largest_failure, layer_updates_per_iter, uses_gdrcopy, refused
):
    """Refuse online EPLB with spare slots at creation unless GDRCopy maps the slots for host writes."""
    _fault_tolerance(monkeypatch, largest_failure)
    with (
        patch("tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer"),
        patch(_USES_GDRCOPY, return_value=uses_gdrcopy),
        pytest.raises(ValueError, match="through GDRCopy") if refused else nullcontext(),
    ):
        FaultTolerantMoeLoadBalancer(0, WORLD_SIZE, layer_updates_per_iter, mapping=None)


@pytest.mark.parametrize(
    "largest_failure, eplb_slots, spare_slot_count",
    [
        # After losing a node of two, the two active ranks need two slots each for four experts.
        (2, 1, 1),
        # After losing one rank, the three active ranks still need two slots each.
        (1, 1, 1),
        # A node may hold both replicas of up to four experts, which the spare slots take.
        (2, 2, 2),
        # Without a rank to lose, no spare slot is added.
        (0, 1, 0),
    ],
)
def test_spare_slots_cover_largest_failure(
    monkeypatch, largest_failure, eplb_slots, spare_slot_count
):
    load_balancer = _load_balancer(
        monkeypatch, largest_failure=largest_failure, eplb_slots=eplb_slots
    )
    (layer,) = load_balancer.single_layer_load_balancers
    load_balancer.load_balancer_impl.add_layer.assert_called_once_with(
        EXPERTS, 2, eplb_slots + spare_slot_count
    )
    assert layer.spare_slot_count == spare_slot_count
    assert layer.active_slot_count_per_rank == eplb_slots


def test_no_spare_slots_without_fault_tolerance(monkeypatch):
    def nodes(self):
        raise AssertionError("placed the ranks on nodes without rank failure recovery")

    monkeypatch.setattr(f"{_MLB}.wide_ep_fault_tolerance_enabled", lambda: False)
    monkeypatch.setattr(EplbExpertRestore, "_rank_nodes", nodes)
    with (
        patch("tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer"),
        patch(f"{_MLB}.HostMoeTensorSharer"),
        patch("torch.cuda.Event"),
    ):
        load_balancer = MoeLoadBalancer(0, WORLD_SIZE, 1)
        layer = load_balancer.add_layer(EXPERTS, 2, 1, aux_stream=Mock())
    load_balancer.load_balancer_impl.add_layer.assert_called_once_with(EXPERTS, 2, 1)
    assert layer.spare_slot_count == 0
    assignments = [0, 1, 2, 3]
    assert layer.with_spare_slots(assignments) is assignments
    layer.set_initial_weight_assignments(assignments)
    assert layer.initial_placement is None


@pytest.mark.parametrize("storage_kept", [False, True], ids=["storage_replaced", "no_host_weights"])
def test_spare_storage_misuse_refused(monkeypatch, storage_kept):
    """Refuse a weight whose spare slots a restore could not load, for lost storage or no host weights."""
    load_balancer = _load_balancer(
        monkeypatch, largest_failure=2, eplb_slots=1, layer_updates_per_iter=0
    )
    (layer,) = load_balancer.single_layer_load_balancers
    with patch(f"{_MLB}.torch") as mock_torch:
        mock_torch.ops.trtllm.moe_unbacked_slots_save_memory.return_value = True
        mock_torch.ops.trtllm.moe_unbacked_slot_tensor.return_value = torch.zeros(2, 4)
        layer.create_spare_slot_storage("w", torch.zeros(2, 4))
    if storage_kept:
        message = r"Weights \['w'\] of MoE layer .* no host weights"
    else:
        layer.add_to_migrate_weight_fn(layer.use_spare_slot_storage, (torch.zeros(2, 4), "w"))
        message = "Weight w of MoE layer .* lost the storage"
    with pytest.raises(RuntimeError, match=message):
        load_balancer.register_weight_slots_after_to_cuda()


@pytest.mark.parametrize("fault_tolerance", [True, False], ids=["fault_tolerance", "off"])
def test_shutdown_node_wait_without_fault_tolerance(monkeypatch, fault_tolerance):
    """Release the host weights after the C++ shutdown, without a node collective under fault tolerance.

    A rank of the node may be gone, so an iteration in progress keeps only its own rank from releasing.
    """
    load_balancer = _load_balancer(monkeypatch, largest_failure=2, eplb_slots=1)
    (layer,) = load_balancer.single_layer_load_balancers
    monkeypatch.setattr(f"{_MLB}.wide_ep_fault_tolerance_enabled", lambda: fault_tolerance)
    node = Mock(allgather=Mock(side_effect=lambda safe: [safe, True]))
    load_balancer.shared_mpi_comm = node
    events = []
    load_balancer.load_balancer_impl.shutdown.side_effect = lambda: events.append("shutdown")
    layer.host_tensor_sharer.pre_shutdown_cleanup.side_effect = lambda: events.append("release")
    load_balancer.in_iter = True
    assert not load_balancer.shutdown()
    assert events == []
    load_balancer.in_iter = False
    assert load_balancer.shutdown()
    assert events == ["shutdown", "release"]
    assert node.allgather.call_count == (0 if fault_tolerance else 2)
    assert node.barrier.call_count == (0 if fault_tolerance else 2)


def _placement(*rows):
    return np.array(rows, dtype=np.int32)


def test_failed_rank_experts_move_and_return():
    """Run static routing at four ranks of two experts and two spare slots each."""
    initial = _placement([0, 1, -1, -1], [2, 3, -1, -1], [4, 5, -1, -1], [6, 7, -1, -1])
    # Rank 3 fails, and its experts go to the spare slots of the ranks with the fewest filled.
    without_3 = plan_layer_placement(initial, initial, np.array([1, 1, 1, 0], bool), 2, 8)
    np.testing.assert_array_equal(
        without_3, _placement([0, 1, 6, -1], [2, 3, 7, -1], [4, 5, -1, -1], [-1, -1, -1, -1])
    )
    # Rank 2 fails too, and the experts restored before stay where they are.
    without_2_3 = plan_layer_placement(without_3, initial, np.array([1, 1, 0, 0], bool), 2, 8)
    np.testing.assert_array_equal(
        without_2_3, _placement([0, 1, 6, 4], [2, 3, 7, 5], [-1] * 4, [-1] * 4)
    )
    # When rank 2 rejoins, the spare slots holding its experts are emptied.
    np.testing.assert_array_equal(
        plan_layer_placement(without_2_3, initial, np.array([1, 1, 1, 0], bool), 2, 8), without_3
    )
    # With every rank back, the placement is the initial one.
    np.testing.assert_array_equal(
        plan_layer_placement(without_3, initial, np.ones(4, bool), 2, 8), initial
    )
    # Only empty spare slots take a lost expert. The active slots of the active ranks keep their experts.
    replicated = _placement([0, 1], [2, 3], [0, 1])
    with pytest.raises(ValueError, match=r"active ranks \[0, 2\] cannot hold expert 2"):
        plan_layer_placement(replicated, replicated, np.array([1, 0, 1], bool), 2, 4)


def _online_balancer(monkeypatch, current, ep_rank=0):
    """Return an online EPLB load balancer at four ranks of two EPLB slots and two spare slots.

    Its placement is ``current``, and the unbacked slots of its weight "w" take 4 MiB each to back.
    """
    load_balancer = _load_balancer(monkeypatch, largest_failure=2, eplb_slots=2, ep_rank=ep_rank)
    (layer,) = load_balancer.single_layer_load_balancers
    layer.set_initial_weight_assignments(layer.with_spare_slots([0, 1, 2, 3, 0, 1, 2, 3]))
    layer.single_layer_load_balancer_impl.get_old_rank_expert_ids.return_value = current.tolist()
    layer.unbacked_slot_tensors = {"w": Mock()}
    monkeypatch.setattr(
        torch.ops.trtllm,
        "moe_unbacked_slots_bytes_to_back",
        lambda tensor, slots: len(slots) * (4 << 20),
        raising=False,
    )
    return load_balancer, layer


_INITIAL = _placement([0, 1, -1, -1], [2, 3, -1, -1], [0, 1, -1, -1], [2, 3, -1, -1])
# Online EPLB moved the experts away from the initial placement.
_ONLINE = _placement([1, 0, -1, -1], [3, 2, -1, -1], [2, 0, -1, -1], [1, 2, -1, -1])
_ONLINE_WITHOUT_1 = _placement([1, 0, 3, -1], [-1] * 4, [2, 0, -1, -1], [1, 2, -1, -1])


def test_active_ranks_plan_the_same_restore(monkeypatch):
    active = [True, False, True, True]
    # An interrupted online update left rank 2 with another view of the placement than rank 0's.
    views = {
        0: _ONLINE,
        2: _placement([1, 0, -1, -1], [3, 2, -1, -1], [2, 0, -1, -1], [3, 2, -1, -1]),
    }
    plans = [
        _online_balancer(monkeypatch, view, ep_rank)[0].expert_restore.plan_expert_restore(
            active, [_ONLINE]
        )
        for ep_rank, view in views.items()
    ]
    for plan in plans:
        # The active slots of the active ranks keep their experts, and expert 3 goes to a spare slot.
        np.testing.assert_array_equal(plan.placements[0], _ONLINE_WITHOUT_1)
    assert plans[0].digest == plans[1].digest
    # Rank 0 backs one 4 MiB slot and rank 2 backs none.
    assert [plan.device_bytes for plan in plans] == [4 << 20, 0]
    other = _online_balancer(monkeypatch, _ONLINE)[0].expert_restore.plan_expert_restore(
        [True, True, False, True], [_ONLINE]
    )
    assert other.digest != plans[0].digest


def _replacement(monkeypatch, rank, layout="8,4,1"):
    monkeypatch.setattr(_wide_ep_fault_tolerance, "_enabled", True)
    monkeypatch.setenv(FAULT_TOLERANCE_RANK_ENV, f"{rank},{layout}")


@pytest.fixture
def tracked(monkeypatch):
    """Yield the names of the segments Python's resource tracker would unlink when this process ends.

    The tracker gets POSIX names, which start with '/'. Segments the test leaves are unlinked afterwards.
    """
    names = set()
    monkeypatch.setattr(
        resource_tracker, "register", lambda name, rtype: names.add(name.removeprefix("/"))
    )
    monkeypatch.setattr(
        resource_tracker, "unregister", lambda name, rtype: names.discard(name.removeprefix("/"))
    )
    monkeypatch.setattr(_wide_ep_fault_tolerance, "_enabled", True)
    monkeypatch.delenv(FAULT_TOLERANCE_RANK_ENV, raising=False)
    yield names
    for name in _segments():
        os.unlink(f"/dev/shm/{name}")


def _sharer(rank, value, node_size=1, dtype=torch.float32, shape=(4,), base_name=None):
    """Return the host weights of local ``rank``'s two experts of a layer.

    Each weight holds ``value`` plus its expert id, under ``base_name`` or the test's own base name.
    """
    sharer = DeploymentHostWeights(
        0,
        2 * node_size,
        Mock(Get_rank=Mock(return_value=rank), Get_size=Mock(return_value=node_size)),
    )
    sharer.set_shared_memory_base_name(base_name or f"test_host_weights_{os.getpid()}")
    for expert_id in range(2 * node_size):
        if sharer.expert_start <= expert_id < sharer.expert_end:
            sharer.share_host_tensor_with_shape(
                expert_id, "w", torch.full(shape, value + expert_id, dtype=dtype)
            )
        else:
            sharer.pre_register_host_tensor_with_shape(expert_id, "w", dtype, torch.Size(shape))
    return sharer


def _segments():
    return sorted(
        name
        for name in os.listdir("/dev/shm")
        if name.startswith(f"test_host_weights_{os.getpid()}")
    )


def test_host_weights_round_trip_by_name(monkeypatch, tracked):
    """Create, keep and reattach the host weights under the load balancer's one base name."""
    monkeypatch.setenv("TRTLLM_EPLB_SHM_NAME", f"test_host_weights_{os.getpid()}")
    monkeypatch.setattr(EplbExpertRestore, "_rank_nodes", lambda self: [0, 0, 1, 1])
    with patch("tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer"):
        load_balancer = FaultTolerantMoeLoadBalancer(0, WORLD_SIZE, 0, mapping=None)
    base_name = load_balancer.shared_memory_base_name
    original = _sharer(0, 1.0, base_name=base_name)
    original.finalize_layer_weights()
    load_balancer.single_layer_load_balancers = [
        SimpleNamespace(need_load_shared_weights=lambda: True, host_tensor_sharer=original)
    ]
    prefix = load_balancer.expert_restore.keep_host_weights_for_the_deployment()
    assert original.get_shared_memory_name().startswith(prefix)
    assert prefix.startswith(base_name)
    _replacement(monkeypatch, 0)
    replacement = _sharer(0, -1.0, base_name=base_name)  # What it loaded must not overwrite them.
    replacement.finalize_layer_weights()
    assert replacement.own_shm.name == original.own_shm.name
    assert replacement.host_weights[(1, "w")].tolist() == [2.0] * 4
    # They belong to the deployment, even if the replacement fails.
    assert not tracked
    original.pre_shutdown_cleanup()
    replacement.pre_shutdown_cleanup()


@pytest.mark.parametrize(
    "found", ["missing", "incomplete", "other host weights", "open to other users"]
)
def test_replacement_rebuilds_damaged_host_weights(monkeypatch, tracked, found):
    """Fill a replacement's host weights again unless its rank's are intact and its deployment's own."""
    if found != "missing":
        # Other host weights of the same size.
        other = found == "other host weights"
        original = _sharer(
            0, 1.0, dtype=torch.float64 if other else torch.float32, shape=(2 if other else 4,)
        )
        original.finalize_layer_weights()
        if found == "incomplete":
            # The process died before writing the fingerprint.
            original.own_shm.buf[:32] = bytes(32)
        elif found == "open to other users":
            os.fchmod(original.own_shm._fd, 0o666)
            assert not original.holds_host_weights(0)
        original.pre_shutdown_cleanup()
    _replacement(monkeypatch, 0)
    replacement = _sharer(0, -1.0)
    replacement.finalize_layer_weights()
    assert replacement.host_weights[(1, "w")].tolist() == [0.0] * 4
    assert replacement.holds_host_weights(0)
    # They stay the replacement's own until it joins, so a later replacement builds them again if it fails.
    assert tracked == {replacement.get_shared_memory_name()}
    replacement.pre_shutdown_cleanup()
    replacement.post_shutdown_cleanup()
    assert _segments() == []


class _Store:
    """Hold ``record`` and other values by key, as the load balancer uses the rendezvous store."""

    def __init__(self, record):
        self.values = {RECORD_KEY: record.model_dump_json().encode()}
        self.store = SimpleNamespace(check=lambda keys: all(key in self.values for key in keys))

    def set(self, key, value):
        self.values[key] = value

    def get(self, key):
        return self.values[key]

    def multi_get(self, keys):
        return [self.values[key] for key in keys]


def _replacement_of_rank_3(monkeypatch, record):
    """Set up rank 3's replacement on the node of ranks 2 and 3, with ``record`` in the rendezvous store.

    Returns:
        The host weights of rank 3 it built, its load balancer, the store, and the host weights the load balancer
        gets by expert.
    """
    _replacement(monkeypatch, 3, "4,2,1")
    store = _Store(record)
    monkeypatch.setattr(membership, "rendezvous_store", lambda: store)
    monkeypatch.setattr(f"{_ER}._HOST_WEIGHTS_CHECK_S", 0.01)
    sharer = _sharer(1, 0.0, node_size=2)
    sharer.finalize_layer_weights()
    host_weights = {}
    load_balancer = MoeLoadBalancer.__new__(MoeLoadBalancer)
    load_balancer.ep_rank, load_balancer.ep_size = 3, 4
    load_balancer.mapping = Mapping(world_size=4, tp_size=4, gpus_per_node=2)
    load_balancer.layer_updates_per_iter = 0
    load_balancer.expert_restore = EplbExpertRestore(load_balancer)
    load_balancer.shared_mpi_comm = Mock()
    load_balancer.load_balancer_impl = Mock()
    load_balancer.single_layer_load_balancers = [
        SimpleNamespace(
            spare_slot_count=0,
            need_load_shared_weights=lambda: True,
            host_tensor_sharer=sharer,
            py_finalize_model=lambda: sharer.finalize_host_tensor_sharing(
                lambda expert_id, name, t: host_weights.update({expert_id: t.tolist()})
            ),
        )
    ]
    return sharer, load_balancer, store, host_weights


def _record(active_mask, incarnations=(0, 0, 0, 0), state=MembershipState.COMMITTED):
    return MembershipRecord(
        generation=1, state=state, active_mask=active_mask, incarnations=incarnations
    )


@pytest.mark.parametrize(
    "record, announced, error, message",
    [
        # Rank 2 is active, so nobody builds its host weights again.
        (_record(0b0111, (0, 0, 0, 1)), False, ReplacementCannotStart, r"active ranks \[2\]"),
        # Rank 2's replacement announced host weights the node does not hold.
        (
            _record(0b0011, (0, 0, 1, 1)),
            True,
            ReplacementCannotStart,
            r"replacements of ranks \[2\] .* built are missing",
        ),
        (_record(0b0011, state=MembershipState.ABORTED), False, RuntimeError, "deployment ended"),
    ],
    ids=["active", "announced", "aborted"],
)
def test_replacement_without_node_host_weights_fails(
    monkeypatch, tracked, record, announced, error, message
):
    replacement, load_balancer, store, _ = _replacement_of_rank_3(monkeypatch, record)
    if announced:
        store.set(host_weights_key(2), b"")
    with pytest.raises(error, match=message) as raised:
        load_balancer.finalize_model()
    if error is ReplacementCannotStart:
        assert raised.value.code == REPLACEMENT_CANNOT_START
    replacement.pre_shutdown_cleanup()
    replacement.post_shutdown_cleanup()

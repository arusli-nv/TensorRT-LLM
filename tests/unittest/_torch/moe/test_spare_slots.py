# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The spare slots WideEP fault tolerance adds to EPLB so that the live ranks after a failure can hold every
expert."""

import os
import secrets
import threading
from contextlib import nullcontext
from multiprocessing import resource_tracker
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from mpi4py import MPI

from tensorrt_llm import _utils
from tensorrt_llm._torch.distributed.membership import (
    RECORD_KEY,
    REPLACEMENT_CANNOT_START,
    MembershipRecord,
    MembershipState,
    ReplacementCannotStart,
)
from tensorrt_llm._torch.moe.fused_moe.eplb_expert_restore import (
    DeploymentHostCopies,
    EplbExpertRestore,
    host_copies_key,
    plan_layer_placement,
)
from tensorrt_llm._torch.moe.fused_moe.impl_blocks import MoEEplbWeightLayoutMixin
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import HostMoeTensorSharer, MoeLoadBalancer
from tensorrt_llm._torch.pyexecutor import rank_failure_recovery
from tensorrt_llm._utils import FAULT_TOLERANCE_RANK_ENV
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

_MLB = "tensorrt_llm._torch.moe.fused_moe.moe_load_balancer"
_ER = "tensorrt_llm._torch.moe.fused_moe.eplb_expert_restore"
_USES_GDRCOPY = "tensorrt_llm.bindings.internal.runtime.host_accessible_device_memory_uses_gdrcopy"
WORLD_SIZE = 4
EXPERTS = 4


@pytest.fixture(autouse=True)
def deployment_nonce(monkeypatch):
    """Each test's own deployment nonce, which names its host copies."""
    monkeypatch.setattr(rank_failure_recovery, "_deployment_nonce", secrets.token_hex(8))


def _load_balancer(
    monkeypatch, *, largest_failure, eplb_slots, layer_updates_per_iter=1, ep_rank=0
):
    """A MoeLoadBalancer with WideEP fault tolerance over one native layer of ``eplb_slots`` per rank, whose
    largest failure takes ``largest_failure`` ranks."""
    monkeypatch.setattr(f"{_MLB}.fault_tolerance_mode", lambda: True)
    monkeypatch.setattr(EplbExpertRestore, "_rank_nodes", lambda self: [0, 0, 1, 1])
    monkeypatch.setattr(
        EplbExpertRestore, "_largest_failure_size", staticmethod(lambda hosts: largest_failure)
    )
    with (
        patch("tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer"),
        patch(_USES_GDRCOPY, return_value=True),
        patch(f"{_ER}.DeploymentHostCopies"),
        patch("torch.cuda.Event"),
    ):
        load_balancer = MoeLoadBalancer(ep_rank, WORLD_SIZE, layer_updates_per_iter)
        load_balancer.add_layer(EXPERTS, 2, eplb_slots, aux_stream=Mock())
    return load_balancer


@pytest.mark.parametrize(
    "largest_failure, layer_updates_per_iter, uses_gdrcopy, refused",
    [
        (2, 1, False, True),
        (2, 1, True, False),
        # Static EPLB writes no slot from the host; without a rank to lose there is no spare slot.
        (2, 0, False, False),
        (0, 1, False, False),
    ],
)
def test_online_eplb_spare_slots_without_gdrcopy_are_refused_at_startup(
    monkeypatch, largest_failure, layer_updates_per_iter, uses_gdrcopy, refused
):
    """Online EPLB writes the spare slots from the host through GDRCopy mappings: where host accessible device memory
    takes another form (GPU memory is a NUMA node, or GDRCopy is missing), the load balancer is refused when it is
    created, before any weight is allocated."""
    monkeypatch.setattr(f"{_MLB}.fault_tolerance_mode", lambda: True)
    monkeypatch.setattr(EplbExpertRestore, "_rank_nodes", lambda self: [0, 0, 1, 1])
    monkeypatch.setattr(
        EplbExpertRestore, "_largest_failure_size", staticmethod(lambda hosts: largest_failure)
    )
    with (
        patch("tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer"),
        patch(_USES_GDRCOPY, return_value=uses_gdrcopy),
        pytest.raises(ValueError, match="through GDRCopy") if refused else nullcontext(),
    ):
        MoeLoadBalancer(0, WORLD_SIZE, layer_updates_per_iter)


def test_the_largest_failure_is_a_node_other_than_rank_0_s_or_one_rank():
    largest = EplbExpertRestore._largest_failure_size
    assert largest([0] * 4 + [1] * 4) == 4
    assert largest([0, 1, 0, 1, 2, 2, 2]) == 3
    # One node: one rank other than rank 0.
    assert largest([0] * 4) == 1
    assert largest([0]) == 0


def test_the_nodes_of_the_ranks_follow_gpus_per_node():
    """No gather over MPI: a replacement rank's MPI world is only itself."""
    expert_restore = EplbExpertRestore.__new__(EplbExpertRestore)
    expert_restore.balancer = SimpleNamespace(
        ep_size=6, mapping=Mapping(world_size=6, tp_size=6, gpus_per_node=4)
    )
    assert expert_restore._rank_nodes() == [0, 0, 0, 0, 1, 1]


@pytest.mark.parametrize(
    "largest_failure, eplb_slots, spare_slot_count",
    [
        # A node of two: the two live ranks need two slots each for four experts.
        (2, 1, 1),
        # One rank: the three live ranks still need two slots each.
        (1, 1, 1),
        # Replicas: the node may hold both replicas of up to four experts, which the live ranks' spare slots take.
        (2, 2, 2),
        # No rank to lose: no spare slot is added.
        (0, 1, 0),
    ],
)
def test_spare_slots_hold_every_expert_the_largest_failure_can_take(
    monkeypatch, largest_failure, eplb_slots, spare_slot_count
):
    load_balancer = _load_balancer(
        monkeypatch, largest_failure=largest_failure, eplb_slots=eplb_slots
    )
    (layer,) = load_balancer.single_layer_load_balancers
    load_balancer.load_balancer_impl.add_layer.assert_called_once_with(
        EXPERTS, 2, eplb_slots + spare_slot_count
    )
    assert (layer.slot_count_per_rank, layer.spare_slot_count) == (
        eplb_slots + spare_slot_count,
        spare_slot_count,
    )
    assert layer.active_slot_count_per_rank == eplb_slots


def test_without_rank_failure_recovery_no_spare_slot_is_added(monkeypatch):
    def nodes(self):
        raise AssertionError("placed the ranks on nodes without rank failure recovery")

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
    layer.single_layer_load_balancer_impl.set_initial_weight_assignments.assert_called_once_with(
        assignments
    )


@pytest.mark.parametrize("layer_updates_per_iter", [1, 0], ids=["online", "static"])
def test_spare_slots_start_empty_and_without_memory(monkeypatch, layer_updates_per_iter):
    load_balancer = _load_balancer(
        monkeypatch,
        largest_failure=2,
        eplb_slots=1,
        layer_updates_per_iter=layer_updates_per_iter,
    )
    (layer,) = load_balancer.single_layer_load_balancers
    impl = layer.single_layer_load_balancer_impl

    # The spare slots hold no expert: the weight loaders fill the active slots only, and routing and EPLB's updates
    # start with the spare slots empty.
    assignments = layer.with_spare_slots([0, 1, 2, 3])
    assert assignments == [0, -1, 1, -1, 2, -1, 3, -1]
    layer.set_initial_weight_assignments(assignments)
    impl.set_initial_weight_assignments.assert_called_once_with(assignments)
    # In both routing modes the spare slots map no memory, from the creation of the weights on.
    assert list(layer.spare_slots) == [1]
    storage = torch.zeros(2, 4)
    with patch(f"{_MLB}.torch") as mock_torch:
        mock_torch.ops.trtllm.moe_unbacked_slots_save_memory.return_value = True
        mock_torch.ops.trtllm.moe_unbacked_slot_tensor.return_value = storage
        assert layer.create_spare_slot_storage("w", torch.full((2, 4), 5.0)) is storage
    mock_torch.ops.trtllm.moe_unbacked_slots_save_memory.assert_called_once_with(16, 2, [1])
    # Online EPLB writes the storage with CPU copies, as the slots of main's online EPLB: it is host accessible.
    mock_torch.ops.trtllm.moe_unbacked_slot_tensor.assert_called_once_with(
        [2, 4], torch.float32, [1], bool(layer_updates_per_iter)
    )
    # It takes the values of the weight's active slot.
    assert storage.tolist() == [[5.0] * 4, [0.0] * 4]
    # The loaded weight still has that storage.
    weights = storage.detach()
    layer.add_to_migrate_weight_fn(layer.use_spare_slot_storage, (weights, "w"))
    layer.host_tensor_sharer.name_info = {"w": (torch.float32, torch.Size([4]))}
    migrate = Mock()
    monkeypatch.setattr(torch.ops.trtllm, "migrate_to_host_accessible", migrate, raising=False)
    load_balancer.register_weight_slots_after_to_cuda()
    assert (
        list(layer.unbacked_slot_tensors) == ["w"] and layer.unbacked_slot_tensors["w"] is storage
    )
    migrate.assert_not_called()
    # The C++ weight updater gets every slot, which online EPLB's slot functions register and static routing's
    # storage migration, and every expert's host weights: restoring a lost rank's experts loads them with GPU copies.
    registered = [
        (call.args[0], call.args[1], call.args[2].weight_ptr)
        for call in impl.add_single_weight_slot.call_args_list
    ]
    if layer_updates_per_iter:
        assert registered == []
    else:
        assert registered == [(slot, "w", weights[slot].data_ptr()) for slot in range(2)]
    assert layer.need_load_shared_weights()
    load_balancer.finalize_model()
    # The updates never copy with the GPU: a GPU copy can wait in the driver for the GPU stage that waits for it.
    load_balancer.load_balancer_impl.set_use_gpu_memcpy.assert_not_called()
    layer.host_tensor_sharer.finalize_host_tensor_sharing.assert_called_once_with(
        layer._add_host_weight_from_tensor
    )


@pytest.mark.parametrize("layer_updates_per_iter", [1, 0], ids=["online", "static"])
def test_slots_whose_spare_slots_fill_no_whole_chunk_keep_their_storage(
    monkeypatch, layer_updates_per_iter
):
    """A per-slot scale ([slots]) would take a whole VMM chunk in the unbacked storage: it keeps the storage it has
    without spare slots, backing its spare slot maps nothing, and its slots are registered for restores all the
    same."""
    load_balancer = _load_balancer(
        monkeypatch,
        largest_failure=2,
        eplb_slots=1,
        layer_updates_per_iter=layer_updates_per_iter,
    )
    (layer,) = load_balancer.single_layer_load_balancers
    load_balancer.register_weight_slots_after_to_cuda()
    scales = torch.zeros(2)
    ops = {
        name: Mock()
        for name in (
            "moe_unbacked_slots_save_memory",
            "moe_unbacked_slot_tensor",
            "migrate_to_host_accessible",
            "moe_unbacked_slots_bytes_to_back",
            "moe_unbacked_slots_set_backed",
        )
    }
    ops["moe_unbacked_slots_save_memory"].return_value = False
    for name, op in ops.items():
        monkeypatch.setattr(torch.ops.trtllm, name, op, raising=False)
    monkeypatch.setattr(torch.cuda, "empty_cache", Mock())
    assert layer.create_spare_slot_storage("alpha", scales) is None
    ops["moe_unbacked_slot_tensor"].assert_not_called()
    layer.use_spare_slot_storage(scales, "alpha")
    assert layer.unbacked_slot_tensors == {}
    registered = [
        (call.args[0], call.args[1], call.args[2].weight_ptr)
        for call in layer.single_layer_load_balancer_impl.add_single_weight_slot.call_args_list
    ]
    if layer_updates_per_iter:
        ops["migrate_to_host_accessible"].assert_called_once_with(scales)
        assert registered == []  # online EPLB's slot functions register them
    else:
        ops["migrate_to_host_accessible"].assert_not_called()
        assert registered == [(slot, "alpha", scales[slot].data_ptr()) for slot in range(2)]
    assert layer.spare_slot_bytes_to_back([1]) == 0
    assert load_balancer.expert_restore.set_spare_slots_backed([[1]], True) == 0
    ops["moe_unbacked_slots_bytes_to_back"].assert_not_called()
    ops["moe_unbacked_slots_set_backed"].assert_not_called()


def _layer_with_spare_slot_storage(monkeypatch, name):
    """The layer of a load balancer whose weight ``name`` got storage without spare slot memory at creation."""
    load_balancer = _load_balancer(
        monkeypatch, largest_failure=2, eplb_slots=1, layer_updates_per_iter=0
    )
    (layer,) = load_balancer.single_layer_load_balancers
    with patch(f"{_MLB}.torch") as mock_torch:
        mock_torch.ops.trtllm.moe_unbacked_slots_save_memory.return_value = True
        mock_torch.ops.trtllm.moe_unbacked_slot_tensor.return_value = torch.zeros(2, 4)
        layer.create_spare_slot_storage(name, torch.zeros(2, 4))
    return load_balancer, layer


def test_a_weight_that_lost_its_spare_slot_storage_is_refused(monkeypatch):
    """A transform after loading that replaced the storage would give the spare slots device memory again."""
    load_balancer, layer = _layer_with_spare_slot_storage(monkeypatch, "w")
    layer.add_to_migrate_weight_fn(layer.use_spare_slot_storage, (torch.zeros(2, 4), "w"))
    with pytest.raises(RuntimeError, match="Weight w of MoE layer .* lost the storage"):
        load_balancer.register_weight_slots_after_to_cuda()


def test_spare_slot_storage_without_host_weights_is_refused(monkeypatch):
    """A restore could not load such a weight's spare slots."""
    load_balancer, _ = _layer_with_spare_slot_storage(monkeypatch, "w")
    with pytest.raises(RuntimeError, match=r"Weights \['w'\] of MoE layer .* no host weights"):
        load_balancer.register_weight_slots_after_to_cuda()


def test_weights_of_slots_get_spare_slot_storage_at_creation():
    """Every weight of local slots, and only those, gets the storage its layer makes, where that saves memory."""

    class _Moe(MoEEplbWeightLayoutMixin, torch.nn.Module):
        pass

    moe = _Moe()
    moe.expert_size_per_partition = 3
    moe.register_parameter("weights", torch.nn.Parameter(torch.zeros(3, 4), requires_grad=False))
    moe.register_parameter("scales", torch.nn.Parameter(torch.zeros(3), requires_grad=False))
    moe.register_parameter("rows", torch.nn.Parameter(torch.zeros(4, 3), requires_grad=False))
    moe.register_parameter("scale", torch.nn.Parameter(torch.tensor(1.0), requires_grad=False))
    storage = torch.ones(3, 4)
    moe.layer_load_balancer = Mock(spare_slot_count=1)
    moe.layer_load_balancer.create_spare_slot_storage.side_effect = lambda name, slots: (
        storage if name == "weights" else None
    )
    moe._create_spare_slot_storage()
    assert [
        c.args[0] for c in moe.layer_load_balancer.create_spare_slot_storage.call_args_list
    ] == [
        "weights",
        "scales",
    ]
    assert moe.weights.data_ptr() == storage.data_ptr() and not moe.weights.requires_grad
    assert moe.scales.eq(0).all()


def test_online_eplb_without_fault_tolerance_keeps_no_restore_state(monkeypatch):
    monkeypatch.setattr(f"{_MLB}.fault_tolerance_mode", lambda: False)
    with (
        patch("tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer"),
        patch(f"{_MLB}.HostMoeTensorSharer"),
        patch("torch.cuda.Event"),
    ):
        layer = MoeLoadBalancer(0, WORLD_SIZE, 1).add_layer(EXPERTS, 2, 1, aux_stream=Mock())
    layer.set_initial_weight_assignments([0, 1, 2, 3])
    assert layer.initial_placement is None
    layer.py_finalize_model()
    layer.host_tensor_sharer.finalize_host_tensor_sharing.assert_called_once_with(
        layer._add_host_weight_from_tensor
    )


def test_static_routing_without_fault_tolerance_shares_no_host_weights(monkeypatch):
    monkeypatch.setattr(EplbExpertRestore, "_rank_nodes", Mock(side_effect=AssertionError))
    with (
        patch("tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer"),
        patch(f"{_MLB}.HostMoeTensorSharer") as sharer,
    ):
        load_balancer = MoeLoadBalancer(0, WORLD_SIZE, 0)
        layer = load_balancer.add_layer(EXPERTS, 2, 1)
    assert not layer.need_load_shared_weights()
    assert layer.host_tensor_sharer is None
    sharer.assert_not_called()


def test_online_eplb_without_spare_slots_copies_as_without_fault_tolerance(monkeypatch):
    # No rank to lose: no spare slot.
    load_balancer = _load_balancer(monkeypatch, largest_failure=0, eplb_slots=2)
    (layer,) = load_balancer.single_layer_load_balancers
    load_balancer.register_weight_slots_after_to_cuda()
    weights = torch.zeros(2, 4)
    with patch(f"{_MLB}.torch") as mock_torch:
        layer.make_tensor_host_accessible(weights)
    assert not layer.spare_slots
    mock_torch.ops.trtllm.migrate_to_host_accessible.assert_called_once_with(weights)
    load_balancer.load_balancer_impl.set_use_gpu_memcpy.assert_not_called()
    layer.host_tensor_sharer.host_weights = {(0, "w"): torch.zeros(4)}
    load_balancer.finalize_model()
    layer.host_tensor_sharer.finalize_host_tensor_sharing.assert_called_once_with(
        layer._add_host_weight_from_tensor
    )


@pytest.mark.parametrize("fault_tolerance", [True, False], ids=["fault_tolerance", "off"])
def test_the_cpp_load_balancer_keeps_the_only_references_to_the_host_weights(
    monkeypatch, fault_tolerance
):
    """Its weight updater loads slots from them, for online EPLB and for restoring lost experts alike: the Python
    views go once it has them."""
    monkeypatch.setattr(_utils, "_wide_ep_fault_tolerance", fault_tolerance)
    monkeypatch.delenv(FAULT_TOLERANCE_RANK_ENV, raising=False)
    sharer_class = DeploymentHostCopies if fault_tolerance else HostMoeTensorSharer
    sharer = sharer_class(0, 2, MPI.COMM_SELF)
    sharer.set_shared_memory_base_name(f"test_spare_slots_{os.getpid()}")
    for expert_id in range(2):
        sharer.share_host_tensor_with_shape(expert_id, "w", torch.full((4, 64), float(expert_id)))
    sharer.finalize_layer_weights()
    try:
        host_weights = []
        sharer.finalize_host_tensor_sharing(lambda expert_id, name, t: host_weights.append(t))
        assert [int(t[0, 0]) for t in host_weights] == [0, 1]
        host_weights.clear()  # views of the shared memory, which must be released to close it
        assert not sharer.host_weights
        sharer.pre_shutdown_cleanup()
    finally:
        sharer.post_shutdown_cleanup()


def test_shutdown_waits_for_weight_updates_before_releasing_host_weights(monkeypatch):
    """Online EPLB's weight updates read the host weights until the C++ shutdown returns."""
    load_balancer = _load_balancer(monkeypatch, largest_failure=2, eplb_slots=1)
    (layer,) = load_balancer.single_layer_load_balancers
    events = []
    load_balancer.load_balancer_impl.shutdown.side_effect = lambda: events.append("shutdown")
    layer.host_tensor_sharer.pre_shutdown_cleanup.side_effect = lambda: events.append("release")
    assert load_balancer.shutdown()
    assert events == ["shutdown", "release"]


@pytest.mark.parametrize("fault_tolerance", [True, False], ids=["fault_tolerance", "off"])
def test_shutdown_waits_for_the_node_only_without_fault_tolerance(monkeypatch, fault_tolerance):
    """With fault tolerance a rank of the node may be gone: each rank releases its own mappings without a node
    collective, and an iteration in progress keeps only its own rank from releasing."""
    load_balancer = _load_balancer(monkeypatch, largest_failure=2, eplb_slots=1)
    monkeypatch.setattr(f"{_MLB}.fault_tolerance_mode", lambda: fault_tolerance)
    node = Mock(allgather=Mock(side_effect=lambda safe: [safe, True]))
    load_balancer.shared_mpi_comm = node
    load_balancer.in_iter = True
    assert not load_balancer.shutdown()
    load_balancer.load_balancer_impl.shutdown.assert_not_called()
    load_balancer.in_iter = False
    assert load_balancer.shutdown()
    load_balancer.load_balancer_impl.shutdown.assert_called_once()
    if fault_tolerance:
        node.allgather.assert_not_called()
        node.barrier.assert_not_called()
    else:
        assert node.allgather.call_count == 2
        assert node.barrier.call_count == 2


@pytest.mark.parametrize(
    "largest_failure, eplb_slots, tuned_slots",
    [
        # One spare slot per rank: the MoE autotuner profiles EPLB's slot alone.
        (2, 1, 1),
        # No spare slot: the MoE autotuner profiles every local slot, as without fault tolerance.
        (0, 1, None),
    ],
)
def test_moe_autotuner_profiles_only_the_active_slots(
    monkeypatch, largest_failure, eplb_slots, tuned_slots
):
    """The MoE ops' tuner layout is settled when the layer registers its weights with the load balancer."""
    load_balancer = _load_balancer(
        monkeypatch, largest_failure=largest_failure, eplb_slots=eplb_slots
    )
    moe = MoEEplbWeightLayoutMixin()
    assert moe.tuner_layout_kwargs == {}
    (moe.layer_load_balancer,) = load_balancer.single_layer_load_balancers
    moe.num_experts = EXPERTS
    moe.expert_size_per_partition = moe.layer_load_balancer.slot_count_per_rank
    moe.initial_local_expert_ids = [0]
    moe.register_all_parameter_slot_and_to_fix_weight_fns({"w": torch.zeros(EXPERTS, 1)})
    expected = {} if tuned_slots is None else {"tuner_num_local_experts": tuned_slots}
    assert moe.tuner_layout_kwargs == expected


@pytest.mark.parametrize("spare_slot_count", [2, 0], ids=["spare_slots", "online_eplb"])
def test_slot_tensors_go_to_the_storage_of_their_layer(spare_slot_count):
    """A layer with spare slots takes its contiguous slot tensors as spare slot storage; online EPLB without them
    keeps its host-accessible storage. Non-contiguous slots are refused either way."""
    moe = MoEEplbWeightLayoutMixin()
    moe.layer_load_balancer = Mock(spare_slot_count=spare_slot_count)
    moe.layer_load_balancer.is_dynamic_routing.return_value = True
    moe.weights = torch.zeros(3, 4, 2)
    moe.register_to_fix_weight_fn("weights")
    if spare_slot_count:
        moe.layer_load_balancer.use_spare_slot_storage.assert_called_once()
        moe.layer_load_balancer.make_tensor_host_accessible.assert_not_called()
    else:
        moe.layer_load_balancer.make_tensor_host_accessible.assert_called_once()
        moe.layer_load_balancer.use_spare_slot_storage.assert_not_called()
    moe.scales = torch.zeros(3, 4, 2).transpose(1, 2)
    with pytest.raises(AssertionError, match="should be contiguous"):
        moe.register_to_fix_weight_fn("scales")


def test_deepseek_r1_ep32_static_256_slots_get_two_spare_slots_per_rank(monkeypatch):
    """DeepSeek-R1 at EP32 on eight nodes of four GPUs: the 28 live ranks after a node failure need
    ceil(256 / 28) = 10 slots each, two more than EPLB's 8."""
    monkeypatch.setattr(f"{_MLB}.fault_tolerance_mode", lambda: True)
    with (
        patch("tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer"),
        patch(f"{_ER}.DeploymentHostCopies"),
    ):
        load_balancer = MoeLoadBalancer(
            0, 32, 0, mapping=Mapping(world_size=32, tp_size=32, gpus_per_node=4)
        )
        layer = load_balancer.add_layer(256, 8, 8)
    load_balancer.load_balancer_impl.add_layer.assert_called_once_with(256, 8, 10)
    assert (layer.slot_count_per_rank, layer.spare_slot_count) == (10, 2)


def _placement(*rows):
    return np.array(rows, dtype=np.int32)


def test_a_lost_rank_s_experts_go_to_spare_slots_and_return_when_it_rejoins():
    """Static routing at four ranks of two experts and two spare slots each."""
    initial = _placement([0, 1, -1, -1], [2, 3, -1, -1], [4, 5, -1, -1], [6, 7, -1, -1])
    # Rank 3 fails: its experts go to the spare slots of the ranks with the fewest filled.
    without_3 = plan_layer_placement(initial, initial, np.array([1, 1, 1, 0], bool), 2, 8)
    np.testing.assert_array_equal(
        without_3, _placement([0, 1, 6, -1], [2, 3, 7, -1], [4, 5, -1, -1], [-1, -1, -1, -1])
    )
    # Rank 2 fails too: the experts restored before stay where they are.
    without_2_3 = plan_layer_placement(without_3, initial, np.array([1, 1, 0, 0], bool), 2, 8)
    np.testing.assert_array_equal(
        without_2_3, _placement([0, 1, 6, 4], [2, 3, 7, 5], [-1] * 4, [-1] * 4)
    )
    # Rank 2 rejoins: the spare slots holding its experts are emptied.
    np.testing.assert_array_equal(
        plan_layer_placement(without_2_3, initial, np.array([1, 1, 1, 0], bool), 2, 8), without_3
    )
    # Every rank is back: the placement is the initial one.
    np.testing.assert_array_equal(
        plan_layer_placement(without_3, initial, np.ones(4, bool), 2, 8), initial
    )


def test_a_lost_expert_never_replaces_an_extra_replica():
    """Only empty spare slots take a lost expert while a rank is lost: the live ranks' active slots keep theirs."""
    initial = _placement([0, 1], [2, 3], [0, 1])
    with pytest.raises(ValueError, match=r"live ranks \[0, 2\] cannot hold expert 2"):
        plan_layer_placement(initial, initial, np.array([1, 0, 1], bool), 2, 4)


def test_a_plan_fails_when_the_live_ranks_cannot_hold_every_expert():
    initial = _placement([0, 1], [2, 3])
    with pytest.raises(ValueError, match=r"live ranks \[0\] cannot hold expert 2"):
        plan_layer_placement(initial, initial, np.array([1, 0], bool), 2, 4)


def test_a_survivor_lost_during_an_admission_leaves_the_spare_slot_the_joining_rank_frees():
    """Ranks 2 and 3 were lost and their experts restored into the spare slots of ranks 0 and 1. Rank 2's
    replacement joins, and rank 1 fails before the admission installs. On a join the active slots take their experts
    of the initial placement: the joining rank holds expert 2 again, so rank 0's spare slot that held it takes a lost
    expert, and the live ranks keep every expert, although only one of their spare slots was empty. The plan is the
    one rule for this (the membership authority leaves it to the live ranks)."""
    initial = _placement([0, -1], [1, -1], [2, -1], [3, -1])
    agreed = _placement([0, 2], [1, 3], [2, -1], [-1, -1])  # rank 2's row: its own slots
    live = np.array([True, False, True, False])
    placement = plan_layer_placement(agreed, initial, live, 1, 4)
    np.testing.assert_array_equal(placement, _placement([0, 1], [-1, -1], [2, 3], [-1, -1]))


def _online_balancer(monkeypatch, current, ep_rank=0):
    """Online EPLB at four ranks of two EPLB slots and two spare slots, whose placement is ``current``, with
    weight "w" whose unbacked slots take 4 MiB each to back."""
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


def test_every_live_rank_plans_the_same_restore_from_the_agreed_placement(monkeypatch):
    live = [True, False, True, True]
    # An interrupted online update left rank 2 with another view of the placement than rank 0's.
    views = {
        0: _ONLINE,
        2: _placement([1, 0, -1, -1], [3, 2, -1, -1], [2, 0, -1, -1], [3, 2, -1, -1]),
    }
    plans = [
        _online_balancer(monkeypatch, view, ep_rank)[0].expert_restore.plan_expert_restore(
            live, [_ONLINE]
        )
        for ep_rank, view in views.items()
    ]
    for plan in plans:
        # Online EPLB keeps the experts the live ranks' active slots hold; expert 3 goes to a spare slot.
        np.testing.assert_array_equal(plan.placements[0], _ONLINE_WITHOUT_1)
    assert plans[0].digest == plans[1].digest
    # Rank 0 backs one 4 MiB slot, rank 2 none.
    assert [plan.device_bytes for plan in plans] == [4 << 20, 0]
    other = _online_balancer(monkeypatch, _ONLINE)[0].expert_restore.plan_expert_restore(
        [True, True, False, True], [_ONLINE]
    )
    assert other.digest != plans[0].digest


@pytest.mark.parametrize("ep_rank, restore_bytes", [(0, 4 << 20), (1, 0), (2, 0)])
def test_the_largest_failure_restore_is_the_most_a_rank_backs_for_a_failure_it_survives(
    monkeypatch, ep_rank, restore_bytes
):
    """Nodes [0, 1] and [2, 3]: a failure takes node 1 or rank 1. Losing node 1 loses no expert the other ranks lack;
    losing rank 1 loses expert 3, which goes to rank 0's spare slot (4 MiB). Rank 1 survives only node 1's loss."""
    load_balancer, _ = _online_balancer(monkeypatch, _ONLINE, ep_rank)
    assert load_balancer.expert_restore.largest_failure_restore_bytes() == restore_bytes


def test_a_joining_rank_plans_the_same_restore_as_the_live_ranks(monkeypatch):
    """The joining rank's own view of the deployment is only its startup, and its slots may no longer hold its
    initial experts: the record's joining ranks, not each rank's history, decide that the active slots return to the
    initial placement."""
    # Rank 1's replacement reports its slots holding [3, 2]; its initial experts are [2, 3].
    agreed = _placement([1, 0, 3, -1], [3, 2, -1, -1], [2, 0, -1, -1], [1, 2, -1, -1])
    live, rejoining = [True] * 4, [False, True, False, False]
    survivor, _ = _online_balancer(monkeypatch, _ONLINE_WITHOUT_1)
    survivor.expert_restore.live_ranks = (
        True,
        False,
        True,
        True,
    )  # restored without rank 1 at the failure
    joining, _ = _online_balancer(monkeypatch, agreed, ep_rank=1)
    plans = [
        balancer.expert_restore.plan_expert_restore(live, [agreed], rejoining=rejoining)
        for balancer in (survivor, joining)
    ]
    assert plans[0].digest == plans[1].digest
    np.testing.assert_array_equal(plans[0].placements[0], _INITIAL)


def test_a_rejoin_under_online_eplb_empties_every_spare_slot(monkeypatch):
    """While rank 1 was lost, online EPLB moved expert 1 into rank 0's restored spare slot, so no active slot holds
    it. When rank 1 rejoins, every active slot takes its expert of the initial placement: the active slots hold every
    expert again and every spare slot is empty, the same plan on every rank."""
    degraded = _placement([3, 2, 1, -1], [-1] * 4, [3, 2, -1, -1], [0, 2, -1, -1])
    # The lowest survivor's view, with rank 1's slots as its replacement reports them: its initial experts.
    agreed = degraded.copy()
    agreed[1] = [2, 3, -1, -1]
    live, rejoining = [True] * 4, [False, True, False, False]
    plans = []
    for ep_rank in range(WORLD_SIZE):
        view = agreed if ep_rank == 1 else degraded
        load_balancer, _ = _online_balancer(monkeypatch, view, ep_rank)
        plans.append(
            load_balancer.expert_restore.plan_expert_restore(live, [agreed], rejoining=rejoining)
        )
    assert len({plan.digest for plan in plans}) == 1
    assert [plan.device_bytes for plan in plans] == [0] * WORLD_SIZE
    for plan in plans:
        np.testing.assert_array_equal(plan.placements[0], _INITIAL)


@pytest.mark.parametrize("layer_updates_per_iter", [1, 0], ids=["online", "static"])
def test_active_slots_keep_their_experts_until_a_rank_rejoins(monkeypatch, layer_updates_per_iter):
    """One rule in both routing modes: a failure leaves the live ranks' active slots as they are (static routing
    never moves them from the initial placement), a join returns them to the initial placement."""
    load_balancer, _ = _online_balancer(monkeypatch, _ONLINE)
    monkeypatch.setattr(load_balancer, "layer_updates_per_iter", layer_updates_per_iter)
    failover = load_balancer.expert_restore.plan_expert_restore(
        [True, False, True, True], [_ONLINE]
    )
    np.testing.assert_array_equal(failover.placements[0], _ONLINE_WITHOUT_1)
    rejoin = load_balancer.expert_restore.plan_expert_restore(
        [True] * 4, [_ONLINE_WITHOUT_1], rejoining=[False, True, False, False]
    )
    np.testing.assert_array_equal(rejoin.placements[0], _INITIAL)
    # A rejoining rank that is not live yet (it failed again) changes nothing.
    np.testing.assert_array_equal(
        load_balancer.expert_restore.plan_expert_restore(
            [True, False, True, True], [_ONLINE], rejoining=[False, True, False, False]
        ).placements[0],
        _ONLINE_WITHOUT_1,
    )


def test_restore_backs_then_restores_and_the_release_empties_spare_slots_later(monkeypatch):
    """The order of the restore of a failure and of the rejoin after it: a spare slot is backed before the C++ load
    balancer loads an expert into it and routing can reach it; the spare slots a placement empties keep their memory
    until release_empty_spare_slots."""
    load_balancer, layer = _online_balancer(monkeypatch, _ONLINE)
    impl = layer.single_layer_load_balancer_impl
    events = []
    load_balancer.load_balancer_impl.restore_placement.side_effect = lambda dead, placements: (
        events.append(("restore", dead, placements))
    )
    backed = set()

    def set_backed(tensors, slots, slot_counts, back):
        """As the C++ op: only the slots whose state changes map or release memory."""
        assert tensors == [layer.unbacked_slot_tensors["w"]] and slot_counts == [len(slots)]
        events.append(("backed", slots, back))
        changed = set(slots) - backed if back else set(slots) & backed
        backed.symmetric_difference_update(changed)
        return len(changed) * (4 << 20)

    monkeypatch.setattr(
        torch.ops.trtllm, "moe_unbacked_slots_set_backed", set_backed, raising=False
    )
    plan = load_balancer.expert_restore.plan_expert_restore([True, False, True, True], [_ONLINE])
    times = {}
    assert load_balancer.expert_restore.restore_lost_experts(plan, times) == 4 << 20
    assert events == [("backed", [2], True), ("restore", [1], [_ONLINE_WITHOUT_1.tolist()])]
    assert list(times) == ["back", "load and install"]
    assert backed == {2}
    assert load_balancer.expert_restore.live_ranks == (True, False, True, True)

    events.clear()
    impl.get_old_rank_expert_ids.return_value = _ONLINE_WITHOUT_1.tolist()
    # The empty spare slot was never backed.
    assert load_balancer.expert_restore.release_empty_spare_slots() == 0
    assert events == [("backed", [3], False)]

    events.clear()
    rejoin = load_balancer.expert_restore.plan_expert_restore(
        [True] * 4, [_ONLINE_WITHOUT_1], rejoining=[False, True, False, False]
    )
    assert rejoin.device_bytes == 0
    # Every rank's active slots take their experts of the initial placement; the spare slot keeps its memory.
    assert load_balancer.expert_restore.restore_lost_experts(rejoin) == 0
    assert events == [("restore", [], [_INITIAL.tolist()])]
    assert backed == {2}

    # Routing no longer reads it: the release maps it to the scratch again.
    events.clear()
    impl.get_old_rank_expert_ids.return_value = _INITIAL.tolist()
    assert load_balancer.expert_restore.release_empty_spare_slots() == 4 << 20
    assert events == [("backed", [2, 3], False)]
    assert backed == set()


def test_a_plan_the_load_balancer_rejects_keeps_the_live_ranks(monkeypatch):
    """The C++ load balancer validates every layer's placement before it loads any slot."""
    load_balancer, _ = _online_balancer(monkeypatch, _ONLINE)
    load_balancer.load_balancer_impl.restore_placement.side_effect = RuntimeError(
        "expert 3 has no replica"
    )
    monkeypatch.setattr(
        torch.ops.trtllm, "moe_unbacked_slots_set_backed", Mock(return_value=0), raising=False
    )
    plan = load_balancer.expert_restore.plan_expert_restore([True, False, True, True], [_ONLINE])
    with pytest.raises(RuntimeError, match="expert 3 has no replica"):
        load_balancer.expert_restore.restore_lost_experts(plan)
    assert load_balancer.expert_restore.live_ranks == (True,) * 4


def test_restore_without_a_change_does_nothing(monkeypatch):
    load_balancer, _ = _online_balancer(monkeypatch, _ONLINE)
    plan = load_balancer.expert_restore.plan_expert_restore([True] * 4, [_ONLINE])
    assert plan.device_bytes == 0
    assert load_balancer.expert_restore.restore_lost_experts(plan) == 0
    load_balancer.load_balancer_impl.restore_placement.assert_not_called()


def test_spare_slots_take_the_experts_whose_replicas_were_all_lost(monkeypatch):
    """Static routing with two replicas of each expert: a failure of two ranks that held both replicas of experts 2
    and 3 still leaves every expert served, from the spare slots."""
    load_balancer = _load_balancer(
        monkeypatch, largest_failure=2, eplb_slots=2, layer_updates_per_iter=0
    )
    (layer,) = load_balancer.single_layer_load_balancers
    assert layer.need_load_shared_weights()
    initial = _placement([0, 1, -1, -1], [2, 3, -1, -1], [0, 1, -1, -1], [2, 3, -1, -1])
    layer.set_initial_weight_assignments(initial.ravel().tolist())
    layer.single_layer_load_balancer_impl.get_old_rank_expert_ids.return_value = initial.tolist()
    monkeypatch.setattr(
        torch.ops.trtllm,
        "moe_unbacked_slots_bytes_to_back",
        lambda tensor, slots: len(slots) * (4 << 20),
        raising=False,
    )
    placement = load_balancer.expert_restore.plan_expert_restore(
        [True, False, True, False], [initial]
    ).placements[0]
    assert sorted(set(placement[[0, 2]].ravel()) - {-1}) == [0, 1, 2, 3]


# A replacement rank (rejoin): its MPI world is only itself.


def _replacement(monkeypatch, rank):
    monkeypatch.setattr(_utils, "_wide_ep_fault_tolerance", True)
    monkeypatch.setenv(FAULT_TOLERANCE_RANK_ENV, f"{rank},8,4,1")


def test_a_replacement_keeps_its_node_layout_without_mpi(monkeypatch):
    """Its startup group (mpi_comm()) splits by the mapping's gpus_per_node, with only itself live."""
    from tensorrt_llm._torch.pyexecutor.rank_failure_recovery import StartupGroup

    _replacement(monkeypatch, 5)
    mapping = Mapping(world_size=8, rank=5, tp_size=8, gpus_per_node=4)
    monkeypatch.setattr(rank_failure_recovery, "_mapping", mapping)
    world = StartupGroup(_utils.wide_ep_fault_tolerance_rank(), range(8))
    monkeypatch.setattr("tensorrt_llm.mpi_comm", lambda: world)
    monkeypatch.setattr("tensorrt_llm.local_mpi_size", lambda: 4)
    load_balancer = MoeLoadBalancer.__new__(MoeLoadBalancer)
    load_balancer.ep_rank, load_balancer.ep_size, load_balancer.mapping = 5, 8, mapping
    load_balancer._setup_mpi_comm()
    node = load_balancer.shared_mpi_comm
    assert (node.Get_rank(), node.Get_size()) == (1, 4)  # its share of the node's host copies
    node.barrier()  # the node's host copies already exist: nobody to wait for
    assert node.allgather(True) == [None, True, None, None]


@pytest.mark.parametrize("replacement", [False, True], ids=["original", "replacement"])
def test_a_replacement_tunes_with_the_tuner_layout_of_the_ranks_it_rejoins(
    monkeypatch, replacement
):
    """Rank 5 of two nodes of four: an original rank and its replacement derive the same spare slots, so
    their MoE ops get the same tuner layout and tactics tuned for one serve the other."""
    if replacement:
        _replacement(monkeypatch, 5)
    else:
        monkeypatch.setattr(_utils, "_wide_ep_fault_tolerance", True)
        monkeypatch.delenv(FAULT_TOLERANCE_RANK_ENV, raising=False)
    with (
        patch("tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer"),
        patch(_USES_GDRCOPY, return_value=True),
        patch(f"{_ER}.DeploymentHostCopies"),
        patch("torch.cuda.Event"),
    ):
        load_balancer = MoeLoadBalancer(
            5, 8, 1, mapping=Mapping(world_size=8, tp_size=8, gpus_per_node=4)
        )
        layer = load_balancer.add_layer(16, 2, 2, aux_stream=Mock())
    moe = MoEEplbWeightLayoutMixin()
    moe.layer_load_balancer, moe.num_experts, moe.initial_local_expert_ids = layer, 16, [0, 1]
    moe.expert_size_per_partition = layer.slot_count_per_rank
    moe.register_all_parameter_slot_and_to_fix_weight_fns({})
    # Losing node 1 leaves its 8 slots' experts to the 4 ranks of node 0: 2 spare slots each.
    assert (layer.slot_count_per_rank, layer.spare_slot_count) == (4, 2)
    assert moe.tuner_layout_kwargs == {"tuner_num_local_experts": 2}


# The host copies of a node with fault tolerance: they outlive their processes while the deployment serves.


@pytest.fixture
def tracked(monkeypatch):
    """The segments Python's resource tracker would unlink when this process ends, by the names they are created
    under (the tracker gets their POSIX names, which start with '/'); segments the test leaves are unlinked."""
    names = set()
    monkeypatch.setattr(
        resource_tracker, "register", lambda name, rtype: names.add(name.removeprefix("/"))
    )
    monkeypatch.setattr(
        resource_tracker, "unregister", lambda name, rtype: names.discard(name.removeprefix("/"))
    )
    yield names
    for name in os.listdir("/dev/shm"):
        if name.startswith(f"test_host_copies_{os.getpid()}"):
            os.unlink(f"/dev/shm/{name}")


def _sharer(
    rank,
    value,
    node_size=1,
    dtype=torch.float32,
    shape=(4,),
    base_name=None,
    sharer_class=DeploymentHostCopies,
):
    """The host copies of local ``rank``'s experts (two per rank) of a layer, each filled with ``value`` plus its
    expert id, under ``base_name`` (default: the test's), by a ``sharer_class``."""
    sharer = sharer_class(
        0,
        2 * node_size,
        Mock(Get_rank=Mock(return_value=rank), Get_size=Mock(return_value=node_size)),
    )
    sharer.set_shared_memory_base_name(base_name or f"test_host_copies_{os.getpid()}")
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
        if name.startswith(f"test_host_copies_{os.getpid()}")
    )


@pytest.mark.parametrize("fault_tolerance", [True, False], ids=["fault_tolerance", "off"])
def test_host_copies_are_named_after_their_deployment(monkeypatch, fault_tolerance):
    """With fault tolerance they outlive their processes: another deployment on the node, also one started with the
    same join secret, must not find them, nor its node supervisor unlink them."""
    monkeypatch.setattr(f"{_MLB}.fault_tolerance_mode", lambda: fault_tolerance)
    monkeypatch.delenv("TRTLLM_EPLB_SHM_NAME", raising=False)
    monkeypatch.setattr(EplbExpertRestore, "_rank_nodes", lambda self: [0, 0, 1, 1])
    names = []
    for _ in range(2):  # two deployments, one after the other: rank 0 draws the nonce
        monkeypatch.setattr(rank_failure_recovery, "_deployment_nonce", None)
        with patch("tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer"):
            names.append(MoeLoadBalancer(0, WORLD_SIZE, 0).shared_memory_base_name)
    if fault_tolerance:
        assert all(name.startswith("moe_shared_") for name in names)
        assert names[0] != names[1]
    else:
        assert names == ["moe_shared", "moe_shared"]


@pytest.mark.parametrize("fault_tolerance", [True, False], ids=["fault_tolerance", "off"])
def test_a_rank_s_host_copies_end_with_it_until_the_deployment_serves(
    monkeypatch, tracked, fault_tolerance
):
    """Until then a failure ends the deployment, as without fault tolerance; from then on a replacement rank
    reattaches them, and the node supervisor unlinks them once the deployment ended."""
    monkeypatch.setattr(_utils, "_wide_ep_fault_tolerance", fault_tolerance)
    monkeypatch.delenv(FAULT_TOLERANCE_RANK_ENV, raising=False)
    sharer_class = DeploymentHostCopies if fault_tolerance else HostMoeTensorSharer
    sibling = _sharer(1, 10.0, node_size=2, sharer_class=sharer_class)
    sibling.finalize_layer_weights()
    tracked.clear()  # by the sibling's process
    sharer = _sharer(0, 0.0, node_size=2, sharer_class=sharer_class)
    sharer.finalize_layer_weights()
    sharer.finalize_host_tensor_sharing()
    assert {sharer.get_shared_memory_name(rank) for rank in (0, 1)} <= tracked
    if fault_tolerance:
        sharer.keep_for_the_deployment()
        assert sharer.get_shared_memory_name(0) not in tracked
        assert sharer.get_shared_memory_name(1) not in tracked
    sharer.pre_shutdown_cleanup()
    sharer.post_shutdown_cleanup()
    sibling.pre_shutdown_cleanup()
    assert (sharer.get_shared_memory_name(0) in _segments()) == fault_tolerance


@pytest.fixture
def replacement_of_rank_0(monkeypatch):
    def become():
        monkeypatch.setattr(_utils, "_wide_ep_fault_tolerance", True)
        monkeypatch.setenv(FAULT_TOLERANCE_RANK_ENV, "0,8,4,1")

    monkeypatch.setattr(_utils, "_wide_ep_fault_tolerance", True)
    monkeypatch.delenv(FAULT_TOLERANCE_RANK_ENV, raising=False)
    return become


def test_a_replacement_takes_over_the_host_copies_of_the_rank_it_replaces(
    tracked, replacement_of_rank_0
):
    """Its rank's host copies outlived it, and the replacement reattaches them instead of filling new ones."""
    original = _sharer(0, 1.0)
    original.finalize_layer_weights()
    original.keep_for_the_deployment()
    replacement_of_rank_0()
    replacement = _sharer(0, -1.0)  # what it loaded must not overwrite the host copies
    replacement.finalize_layer_weights()
    assert replacement.own_shm.name == original.own_shm.name
    assert replacement.host_weights[(1, "w")].tolist() == [2.0] * 4
    assert original.host_weights[(1, "w")].tolist() == [2.0] * 4
    assert not tracked  # the deployment's, also if the replacement fails
    original.pre_shutdown_cleanup()
    replacement.pre_shutdown_cleanup()
    replacement.post_shutdown_cleanup()
    assert _segments() == [replacement.get_shared_memory_name()]


@pytest.mark.parametrize(
    "found", ["missing", "incomplete", "other host weights", "open to other users"]
)
def test_a_replacement_builds_its_rank_s_host_copies_again_unless_they_are_intact(
    tracked, replacement_of_rank_0, found
):
    """A segment its rank's process died before filling, of the same size from another deployment, or that other
    users could have written is not reattached: the replacement fills it again from what it loaded."""
    if found == "incomplete":
        original = _sharer(0, 1.0)
        original.finalize_layer_weights()
        original.own_shm.buf[:32] = bytes(32)  # its process died before the fingerprint
        original.pre_shutdown_cleanup()
    elif found == "open to other users":
        original = _sharer(0, 1.0)
        original.finalize_layer_weights()
        os.fchmod(original.own_shm._fd, 0o666)
        assert not original.holds_host_copies(0)
        original.pre_shutdown_cleanup()
    elif found == "other host weights":
        original = _sharer(0, 1.0, dtype=torch.float64, shape=(2,))  # the same size
        original.finalize_layer_weights()
        original.pre_shutdown_cleanup()
    replacement_of_rank_0()
    replacement = _sharer(0, -1.0)
    replacement.finalize_layer_weights()
    assert replacement.host_weights[(1, "w")].tolist() == [0.0] * 4
    assert replacement.holds_host_copies(0)
    # Its own until it joins the deployment: if it fails before, the next replacement builds them again.
    assert tracked == {replacement.get_shared_memory_name()}
    replacement.pre_shutdown_cleanup()
    replacement.post_shutdown_cleanup()
    assert _segments() == []


class _Store:
    """The deployment's store as the load balancer uses it (values by key; signing aside), holding ``record``."""

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
    """Rank 3's replacement on the node of ranks 2 and 3, with ``record`` in the deployment's store: the host
    copies of rank 3 it built, its load balancer over them, the store, and the host weights the load balancer gets
    (by expert)."""
    from tensorrt_llm._torch.distributed import membership

    monkeypatch.setattr(_utils, "_wide_ep_fault_tolerance", True)
    monkeypatch.setenv(FAULT_TOLERANCE_RANK_ENV, "3,4,2,1")
    store = _Store(record)
    monkeypatch.setattr(membership, "rendezvous_store", lambda: store)
    monkeypatch.setattr(f"{_ER}._HOST_COPIES_CHECK_S", 0.01)
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


def test_a_replacement_waits_for_the_host_copies_its_node_s_other_replacement_builds(
    monkeypatch, tracked
):
    """Every process of the node was lost with its host copies: rank 2's replacement builds rank 2's again and
    announces them in the store, which rank 3's replacement waits for; rank 3's announces its own."""
    record = MembershipRecord(
        generation=1, state=MembershipState.COMMITTED, live_mask=0b0011, incarnations=(0, 0, 1, 1)
    )
    replacement, load_balancer, store, host_weights = _replacement_of_rank_3(monkeypatch, record)
    other = _sharer(0, 10.0, node_size=2)

    def build():
        other.finalize_layer_weights()
        store.set(host_copies_key(2), b"")

    built = threading.Timer(0.2, build)
    built.start()
    try:
        load_balancer.finalize_model()
    finally:
        built.join()
    assert host_copies_key(3) in store.values
    assert host_weights[0] == [10.0] * 4
    replacement.pre_shutdown_cleanup()
    other.pre_shutdown_cleanup()


@pytest.mark.parametrize(
    "record, announced, error, message",
    [
        # Rank 2 is live: nobody builds its host copies again.
        (
            MembershipRecord(
                generation=1,
                state=MembershipState.COMMITTED,
                live_mask=0b0111,
                incarnations=(0, 0, 0, 1),
            ),
            False,
            ReplacementCannotStart,
            r"live ranks \[2\]",
        ),
        # Rank 2's replacement announced host copies the node does not hold.
        (
            MembershipRecord(
                generation=1,
                state=MembershipState.COMMITTED,
                live_mask=0b0011,
                incarnations=(0, 0, 1, 1),
            ),
            True,
            ReplacementCannotStart,
            r"replacements of ranks \[2\] .* built are missing",
        ),
        (
            MembershipRecord(generation=1, state=MembershipState.ABORT, live_mask=0b0011),
            False,
            RuntimeError,
            "deployment ended",
        ),
    ],
    ids=["live", "announced", "aborted"],
)
def test_a_replacement_without_the_host_copies_of_its_node_does_not_start(
    monkeypatch, tracked, record, announced, error, message
):
    replacement, load_balancer, store, _ = _replacement_of_rank_3(monkeypatch, record)
    if announced:
        store.set(host_copies_key(2), b"")
    with pytest.raises(error, match=message) as raised:
        load_balancer.finalize_model()
    if error is ReplacementCannotStart:
        assert raised.value.code == REPLACEMENT_CANNOT_START
    replacement.pre_shutdown_cleanup()
    replacement.post_shutdown_cleanup()


def test_host_copies_round_trip_under_the_names_of_their_deployment(
    monkeypatch, tracked, replacement_of_rank_0
):
    """An original rank creates its host copies, a replacement reattaches them, and the node supervisor unlinks
    them, all by the names of the load balancer's one base name (``host_copies_prefix``)."""
    monkeypatch.setenv("TRTLLM_EPLB_SHM_NAME", f"test_host_copies_{os.getpid()}")
    monkeypatch.setattr(EplbExpertRestore, "_rank_nodes", lambda self: [0, 0, 1, 1])
    with patch("tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer"):
        load_balancer = MoeLoadBalancer(0, WORLD_SIZE, 0)
    base_name = load_balancer.shared_memory_base_name
    original = _sharer(0, 1.0, base_name=base_name)
    original.finalize_layer_weights()
    load_balancer.single_layer_load_balancers = [
        SimpleNamespace(need_load_shared_weights=lambda: True, host_tensor_sharer=original)
    ]
    prefix = load_balancer.expert_restore.keep_host_copies_for_the_deployment()
    assert _segments() == [original.get_shared_memory_name()]
    assert original.get_shared_memory_name().startswith(prefix) and prefix.startswith(base_name)
    replacement_of_rank_0()
    replacement = _sharer(0, -1.0, base_name=base_name)
    replacement.finalize_layer_weights()
    assert replacement.own_shm.name == original.own_shm.name
    assert replacement.host_weights[(1, "w")].tolist() == [2.0] * 4
    original.pre_shutdown_cleanup()
    replacement.pre_shutdown_cleanup()

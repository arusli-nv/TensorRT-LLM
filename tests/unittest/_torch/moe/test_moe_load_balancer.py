# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import gc
import os
import unittest
from collections.abc import Iterator
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
import torch
from mpi4py import MPI
from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.moe.fused_moe.eplb_expert_restore import (
    EplbExpertRestore, FaultTolerantMoeLoadBalancer)
from tensorrt_llm._torch.moe.fused_moe.impl_blocks import \
    MoEEplbWeightLayoutMixin
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import (
    MoeLoadBalancer, MoeLoadBalancerIterContext, SingleLayerMoeLoadBalancer,
    get_moe_load_balancer, maybe_create_moe_load_balancer,
    moe_load_balancer_add_single_layer)
from tensorrt_llm.llmapi.llm_args import MoeLoadBalancerConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


class TestMoeLoadBalancer(unittest.TestCase):
    """
    Test cases for the MoeLoadBalancer class.
    """

    def setUp(self):
        os.environ["TLLM_HOST_ACCESSIBLE_ALLOW_MANAGED_FALLBACK"] = "1"

    @patch('tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer')
    def test_moe_load_balancer_init(self, mock_load_balancer_impl):
        """Test initialization of MoeLoadBalancer."""

        torch.cuda.set_device(0)

        # Setup
        ep_rank = 0
        ep_size = 4
        layer_updates_per_iter = 2

        # Exercise
        balancer = MoeLoadBalancer(ep_rank, ep_size, layer_updates_per_iter)

        # Verify
        mock_load_balancer_impl.assert_called_once_with(
            ep_rank,
            ep_size,
            layer_updates_per_iter,
            rank_failure_recovery=False)
        self.assertEqual(balancer.ep_rank, ep_rank)
        self.assertEqual(balancer.ep_size, ep_size)
        self.assertEqual(balancer.layer_updates_per_iter,
                         layer_updates_per_iter)
        self.assertEqual(balancer.load_balancer_impl,
                         mock_load_balancer_impl.return_value)
        self.assertIsNone(balancer._previous_balancer)

    @patch('tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer')
    def test_moe_load_balancer_add_layer(self, mock_load_balancer_impl):
        """Test adding a layer to MoeLoadBalancer."""

        torch.cuda.set_device(0)

        # Setup
        ep_rank = 0
        ep_size = 4
        layer_updates_per_iter = 2
        expert_count = 8
        top_k = 2
        slot_count_per_rank = 4

        mock_single_layer = MagicMock()
        mock_load_balancer_impl.return_value.add_layer.return_value = mock_single_layer

        # Exercise
        balancer = MoeLoadBalancer(ep_rank, ep_size, layer_updates_per_iter)
        result = balancer.add_layer(expert_count, top_k, slot_count_per_rank)

        # Verify
        mock_load_balancer_impl.return_value.add_layer.assert_called_once_with(
            expert_count, top_k, slot_count_per_rank)
        self.assertIsInstance(result, SingleLayerMoeLoadBalancer)
        self.assertEqual(result.single_layer_load_balancer_impl,
                         mock_single_layer)

    @patch('tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer')
    def test_nested_context_managers(self, mock_load_balancer_impl):
        """Test nested MoeLoadBalancer context managers."""

        torch.cuda.set_device(0)

        # Setup
        outer_balancer = MoeLoadBalancer(0, 4, 2)
        inner_balancer = MoeLoadBalancer(1, 4, 2)

        # Exercise & Verify
        self.assertIsNone(get_moe_load_balancer())
        with outer_balancer as active_balancer:
            self.assertIs(active_balancer, outer_balancer)
            self.assertEqual(get_moe_load_balancer(), outer_balancer)

            with inner_balancer:
                self.assertEqual(get_moe_load_balancer(), inner_balancer)

            # After exiting inner context
            self.assertEqual(get_moe_load_balancer(), outer_balancer)

        # After exiting outer context
        self.assertIsNone(get_moe_load_balancer())

    @patch('tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer')
    def test_moe_load_balancer_add_single_layer_with_context(
            self, mock_load_balancer_impl):
        """Test moe_load_balancer_add_single_layer with active context."""

        torch.cuda.set_device(0)

        # Setup
        ep_rank = 0
        ep_size = 4
        layer_updates_per_iter = 2
        expert_count = 8
        top_k = 2
        slot_count_per_rank = 4

        mock_single_layer = MagicMock()
        mock_load_balancer_impl.return_value.add_layer.return_value = mock_single_layer

        # Exercise
        with MoeLoadBalancer(ep_rank, ep_size, layer_updates_per_iter):
            result = moe_load_balancer_add_single_layer(expert_count, top_k,
                                                        slot_count_per_rank)

        # Verify
        mock_load_balancer_impl.return_value.add_layer.assert_called_once_with(
            expert_count, top_k, slot_count_per_rank)
        self.assertIsInstance(result, SingleLayerMoeLoadBalancer)
        self.assertEqual(result.single_layer_load_balancer_impl,
                         mock_single_layer)

    def test_moe_load_balancer_add_single_layer_without_context(self):
        """Test moe_load_balancer_add_single_layer without active context."""

        torch.cuda.set_device(0)

        # Exercise
        result = moe_load_balancer_add_single_layer(8, 2, 4)

        # Verify
        self.assertIsNone(result)

    @patch('tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer')
    def test_kimi_k25_creates_moe_load_balancer(self, mock_load_balancer_impl):
        """Kimi-K2.5 wraps a DeepSeek MoE text model and must setup EPLB."""

        mapping = Mapping(world_size=4, rank=0, tp_size=4, moe_ep_size=4)
        pretrained_config = PretrainedConfig()
        pretrained_config.architectures = ['KimiK25ForConditionalGeneration']
        model_config = ModelConfig(pretrained_config=pretrained_config,
                                   mapping=mapping,
                                   moe_load_balancer=MoeLoadBalancerConfig(
                                       num_slots=416, layer_updates_per_iter=2))

        with maybe_create_moe_load_balancer(model_config, mapping) as balancer:
            self.assertIsInstance(balancer, MoeLoadBalancer)
            self.assertEqual(model_config.moe_load_balancer.num_local_slots,
                             104)
            self.assertEqual(get_moe_load_balancer(), balancer)

        mock_load_balancer_impl.assert_called_once_with(
            0, 4, 2, rank_failure_recovery=False)

    @patch('tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer')
    def test_exception_in_context(self, mock_load_balancer_impl):
        """Test handling of exceptions inside the context manager."""

        torch.cuda.set_device(0)

        # Setup
        class TestException(Exception):
            pass

        # Exercise & Verify
        try:
            with MoeLoadBalancer(0, 4, 2) as balancer:
                self.assertEqual(get_moe_load_balancer(), balancer)
                raise TestException("Test exception")
        except TestException:
            # Exception should be propagated
            pass
        else:
            self.fail("Expected TestException to be raised")

        # Verify the global state is cleaned up
        self.assertIsNone(get_moe_load_balancer())

    @patch('tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer')
    def test_single_layer_moe_load_balancer_methods(self,
                                                    mock_load_balancer_impl):
        """Test methods of SingleLayerMoeLoadBalancer."""

        torch.cuda.set_device(0)

        # Setup
        mock_single_layer_impl = MagicMock()
        layer = SingleLayerMoeLoadBalancer(mock_single_layer_impl,
                                           MPI.COMM_WORLD,
                                           expert_count=4,
                                           slot_count_per_rank=2)

        # Mock out torch.ops.trtllm functions
        with patch('torch.ops.trtllm.moe_load_balance_wait_gpu_stage') as mock_wait, \
             patch('torch.ops.trtllm.moe_load_balance_set_cpu_stage') as mock_set_cpu, \
             patch('torch.ops.trtllm.moe_load_balance_statistic') as mock_statistic, \
             patch('torch.ops.trtllm.moe_load_balance_routing') as mock_route:

            # Exercise - test each method
            # add_weight_slot
            mock_weight = MagicMock()
            layer._add_weight_slot(1, "weight1", mock_weight)
            mock_single_layer_impl.add_single_weight_slot.assert_called_once_with(
                1, "weight1", mock_weight, spare=False)

            # add_host_weight
            mock_host_weight = MagicMock()
            layer._add_host_weight(2, "weight2", mock_host_weight)
            mock_single_layer_impl.add_single_host_weight.assert_called_once_with(
                2, "weight2", mock_host_weight)

            # set_initial_weight_assignments
            initial_assignments = [0, 1, 2, 3]
            layer.set_initial_weight_assignments(initial_assignments)
            mock_single_layer_impl.set_initial_weight_assignments.assert_called_once_with(
                initial_assignments)

            # wait_for_gpu_stage
            mock_wait.return_value = torch.tensor([1])
            layer.start_wait_gpu_stage()
            layer.done_wait_gpu_stage()
            result = layer.statistic_flag_tensor
            mock_wait.assert_called_once_with(
                mock_single_layer_impl.get_pointer())
            self.assertEqual(result, mock_wait.return_value)

            # statistic
            mock_expert_ids = torch.tensor([[0, 1], [2, 3]])
            mock_enabled = torch.tensor([1])
            layer.statistic_flag_tensor = mock_enabled
            layer.update_statistic_with_global_ids(mock_expert_ids, True, False)
            mock_statistic.assert_called_once_with(
                mock_expert_ids, mock_enabled,
                mock_single_layer_impl.get_pointer(), True, False)

            # route
            mock_selected_experts = torch.tensor([[0, 1], [2, 3]])
            mock_route.return_value = torch.tensor([[0, 1], [2, 3]])
            result = layer.route(mock_selected_experts)
            assert torch.equal(result, mock_route.return_value)

            # set_cpu_stage
            layer.start_set_cpu_stage()
            layer.done_set_cpu_stage()
            mock_set_cpu.assert_called_once_with(
                mock_single_layer_impl.get_pointer())

    @patch('tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer')
    def test_moe_load_balancer_lifecycle_methods(self, mock_load_balancer_impl):
        """Test lifecycle methods of MoeLoadBalancer."""

        torch.cuda.set_device(0)

        # Setup
        balancer = MoeLoadBalancer(0, 4, 2)

        # Exercise - test each method
        # finalize_model
        balancer.finalize_model()
        mock_load_balancer_impl.return_value.finalize_model.assert_called_once()

        # set_warm_up_iter_count
        balancer.set_warm_up_iter_count(10)
        mock_load_balancer_impl.return_value.set_warm_up_iter_count.assert_called_once_with(
            10)

        # reconfigure_mask_only
        balancer.reconfigure_mask_only([2])
        mock_load_balancer_impl.return_value.reconfigure_mask_only.assert_called_once_with(
            [2])

        balancer.set_iter_info(True, True)

        with MoeLoadBalancerIterContext(balancer):
            mock_load_balancer_impl.return_value.start_iter.assert_called_once_with(
                0, True, True)

        mock_load_balancer_impl.return_value.end_iter.assert_called_once_with(0)

        # shutdown
        balancer.shutdown()
        mock_load_balancer_impl.return_value.shutdown.assert_called_once()

    @patch('tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer')
    def test_reconfigure_mask_only_rejects_active_iteration(
            self, mock_load_balancer_impl):
        """Test mask-only reconfigure is rejected during an active iteration."""

        torch.cuda.set_device(0)

        balancer = MoeLoadBalancer(0, 4, 2)
        balancer.in_iter = True

        with self.assertRaisesRegex(RuntimeError, "iteration is active"):
            balancer.reconfigure_mask_only([2])

        mock_load_balancer_impl.return_value.reconfigure_mask_only.assert_not_called(
        )

    def test_real_statistic_kernel(self):
        """Test the real statistic kernel functionality."""

        torch.cuda.set_device(0)

        # Setup parameters
        ep_rank = 0
        ep_size = 2  # Small number of ranks for testing
        expert_count = 4
        top_k = 2
        slots_per_rank = 2  # Each rank has 2 slots, total 4 slots for 4 experts

        # Create a real MoeLoadBalancer
        balancer = MoeLoadBalancer(ep_rank, ep_size, 1)

        balancer.set_use_gpu_memcpy(True)

        # Add a layer with initial weight assignments
        # Each slot is assigned to exactly one expert initially
        layer = balancer.add_layer(expert_count, top_k, slots_per_rank)
        initial_assignments = [0, 1, 2, 3]  # Expert i is assigned to slot i
        layer.set_initial_weight_assignments(initial_assignments)

        # Finalize the model
        balancer.finalize_model()

        # enable statistic, disable weight update
        balancer.set_iter_info(True, False)

        # Create sample token data - each token selects 2 experts
        # 4 tokens, each selecting 2 experts
        gathered_raw_expert_ids = torch.tensor(
            [
                [0, 1],  # Token 0 selects experts 0 and 1
                [1, 2],  # Token 1 selects experts 1 and 2
                [2, 3],  # Token 2 selects experts 2 and 3
                [0, 3]  # Token 3 selects experts 0 and 3
            ],
            dtype=torch.int32,
            device="cuda")

        try:
            with MoeLoadBalancerIterContext(balancer):
                # Wait for GPU stage and get enabled flag
                layer.start_wait_gpu_stage()
                layer.done_wait_gpu_stage()

                # Run statistic - just test it runs without error
                layer.update_statistic_with_global_ids(gathered_raw_expert_ids,
                                                       True, True)

                # Set CPU stage to signal completion
                layer.start_set_cpu_stage()
                layer.done_set_cpu_stage()

            # Test passed if we got here without exceptions
            self.assertTrue(True, "Statistic kernel ran successfully")

        except Exception as e:
            self.fail(f"Statistic kernel test failed with exception: {e}")
        finally:
            # Clean up
            balancer.shutdown()

    def test_real_routing_kernel(self):
        """Test the real routing kernel functionality and verify results."""

        torch.cuda.set_device(0)

        # Setup parameters
        ep_rank = 0
        ep_size = 2
        expert_count = 4
        top_k = 2
        slots_per_rank = 2  # Total 4 slots (2 slots per rank)

        # Create a real MoeLoadBalancer
        balancer = MoeLoadBalancer(ep_rank, ep_size, 1)

        balancer.set_use_gpu_memcpy(True)

        # Add a layer with known initial weight assignments
        layer = balancer.add_layer(expert_count, top_k, slots_per_rank)

        # Set initial assignments: expert i is on slot i
        initial_assignments = [0, 1, 2, 3]
        layer.set_initial_weight_assignments(initial_assignments)

        # Finalize the model
        balancer.finalize_model()

        # enable statistic, disable weight update
        balancer.set_iter_info(True, False)

        # Create sample token data - tokens selecting different experts
        token_selected_experts = torch.tensor(
            [
                [0, 1],  # Token 0 selects experts 0 and 1
                [1, 2],  # Token 1 selects experts 1 and 2
                [2, 3],  # Token 2 selects experts 2 and 3
                [0, 3]  # Token 3 selects experts 0 and 3
            ],
            dtype=torch.int32,
            device="cuda")

        try:
            with MoeLoadBalancerIterContext(balancer):
                # Wait for GPU stage
                layer.start_wait_gpu_stage()
                layer.done_wait_gpu_stage()

                # Run routing
                routed_slots = layer.route(token_selected_experts)

                # Set CPU stage
                layer.start_set_cpu_stage()
                layer.done_set_cpu_stage()

            # Verify results - with our initial assignment, expert i should map to slot i
            expected_slots = torch.tensor(
                [
                    [0, 1],  # Experts 0,1 -> Slots 0,1
                    [1, 2],  # Experts 1,2 -> Slots 1,2
                    [2, 3],  # Experts 2,3 -> Slots 2,3
                    [0, 3]  # Experts 0,3 -> Slots 0,3
                ],
                dtype=torch.int32,
                device="cuda")

            self.assertTrue(
                torch.all(torch.eq(routed_slots, expected_slots)),
                f"Routing results don't match expected slots.\nExpected: {expected_slots}\nGot: {routed_slots}"
            )

        except Exception as e:
            self.fail(f"Routing kernel test failed with exception: {e}")
        finally:
            # Clean up
            balancer.shutdown()


@pytest.fixture
def mock_runtime() -> Iterator[MagicMock]:
    shared_comm = MagicMock()
    shared_comm.Get_rank.return_value = 1
    shared_comm.Get_size.return_value = 4
    with (
            patch("tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer") as
            runtime,
            patch.object(
                MoeLoadBalancer,
                "_setup_mpi_comm",
                lambda self: setattr(self, "shared_mpi_comm", shared_comm),
            ),
    ):
        yield runtime


@pytest.mark.cpu_only
@pytest.mark.parametrize("updates", [0, 2])
def test_nemotron_vl_decoder_inherits_eplb_and_registers_moe(
        updates: int, mock_runtime: MagicMock) -> None:
    from tensorrt_llm._torch.models.modeling_nemotron_h_multimodal import \
        NemotronHMultimodalModel
    from tensorrt_llm._torch.moe.fused_moe import (RenormalizeMoeRoutingMethod,
                                                   SimpleActivation)
    from tensorrt_llm._torch.moe.fused_moe.configurable_moe import \
        ConfigurableMoE
    from tensorrt_llm._torch.moe.fused_moe.cutlass import \
        TrtllmCutlassW4a16Nvfp4Impl
    from tensorrt_llm._torch.utils import ActivationType

    text_config = PretrainedConfig(architectures=["NemotronHForCausalLM"],
                                   vocab_size=32)
    model_config = ModelConfig(
        pretrained_config=PretrainedConfig(
            architectures=["NemotronH_Omni_Reasoning_V3"],
            llm_config=text_config,
            img_context_token_id=1,
            video_context_token_id=2,
            sound_config=None,
        ),
        mapping=Mapping(world_size=4,
                        rank=1,
                        tp_size=4,
                        moe_ep_size=4,
                        enable_attention_dp=True),
        quant_config=QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4),
        moe_backend="CUTLASS",
        moe_load_balancer=MoeLoadBalancerConfig(num_slots=16,
                                                layer_updates_per_iter=updates),
        skip_create_weights_in_init=True,
    )

    def create_decoder(decoder_config: ModelConfig) -> torch.nn.Module:
        assert decoder_config.pretrained_config.architectures == [
            "NemotronHForCausalLM"
        ]
        assert decoder_config.moe_load_balancer is not model_config.moe_load_balancer
        assert decoder_config.moe_load_balancer.num_local_slots == 4
        assert decoder_config.moe_load_balancer.slot_start == 4
        assert decoder_config.moe_load_balancer.slot_end == 8
        assert get_moe_load_balancer() is balancer
        decoder = torch.nn.Module()
        decoder.config = decoder_config.pretrained_config
        decoder.moe = ConfigurableMoE(
            routing_method=RenormalizeMoeRoutingMethod(top_k=2),
            num_experts=8,
            hidden_size=128,
            intermediate_size=128,
            dtype=torch.bfloat16,
            model_config=decoder_config,
            moe_cls=TrtllmCutlassW4a16Nvfp4Impl,
            activation=SimpleActivation(ActivationType.Relu2),
            layer_idx=0,
        )
        return decoder

    with (
            patch("torch.cuda.Stream"),
            patch("torch.cuda.Event"),
            patch("tensorrt_llm._torch.distributed.AllReduce",
                  return_value=torch.nn.Identity()),
            patch.object(ConfigurableMoE,
                         "_create_comm_strategy_auto",
                         return_value=None),
            patch(
                "tensorrt_llm._torch.models.modeling_nemotron_h_multimodal."
                "AutoModelForCausalLM.from_config",
                side_effect=create_decoder,
            ) as decoder_factory,
            maybe_create_moe_load_balancer(model_config, model_config.mapping)
            as balancer,
    ):
        try:
            model = NemotronHMultimodalModel(model_config)
            decoder_factory.assert_called_once()
            mock_runtime.return_value.add_layer.assert_called_once_with(8, 2, 4)
            assert len(balancer.single_layer_load_balancers) == 1
            layer_balancer = balancer.single_layer_load_balancers[0]
            assert model.llm.moe.layer_load_balancer is layer_balancer
            assert model.llm.moe.backend.layer_load_balancer is layer_balancer
            assert model.llm.moe.backend.expert_size_per_partition == 4
            assert model.llm.moe.backend.initial_local_expert_ids == [
                2, 3, 4, 5
            ]
        finally:
            balancer.shutdown()
    assert get_moe_load_balancer() is None


@pytest.mark.cpu_only
def test_cpp_shutdown_before_host_weights_release(
        mock_runtime: MagicMock) -> None:
    """Shut the C++ load balancer down before releasing the host weights its updates read."""
    balancer = MoeLoadBalancer(0, 4, 2)
    balancer.shared_mpi_comm.allgather.return_value = [True]
    layer = MagicMock(spec=SingleLayerMoeLoadBalancer)
    balancer.single_layer_load_balancers.append(layer)
    order = []
    mock_runtime.return_value.shutdown.side_effect = lambda: order.append(
        "C++ shutdown")
    layer.py_pre_shutdown_cleanup.side_effect = lambda: order.append(
        "host weights released")

    assert balancer.shutdown()
    assert order == ["C++ shutdown", "host weights released"]


class _WeightLayout(MoEEplbWeightLayoutMixin):
    pass


def _skip_without_gdrcopy_slots():
    """Online EPLB's spare slots are host accessible through GDRCopy only (fault tolerance rejects others)."""
    from tensorrt_llm.bindings.internal import runtime

    # A load balancer initializes host accessible device memory.
    balancer = runtime.MoeLoadBalancer(0, 1, 0)
    uses_gdrcopy = runtime.host_accessible_device_memory_uses_gdrcopy()
    balancer.shutdown()
    del balancer
    if not uses_gdrcopy:
        pytest.skip("host accessible EPLB slots are not GDRCopy mappings here")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("layer_updates_per_iter", [0, 1],
                         ids=["static", "online"])
def test_a_failed_rank_s_expert_is_restored_into_a_spare_slot_until_it_rejoins(
        monkeypatch, layer_updates_per_iter):
    """Restore rank 1's expert 3 into rank 0's first spare slot, then empty and release it when rank 1 rejoins.

    Rank 0 of two runs in one process. Under online EPLB, an update moves expert 2 into the spare slot in between.
    """
    from tensorrt_llm._torch.moe.fused_moe import moe_load_balancer as mlb

    if layer_updates_per_iter:
        _skip_without_gdrcopy_slots()
    monkeypatch.setenv("TLLM_HOST_ACCESSIBLE_ALLOW_MANAGED_FALLBACK", "1")
    monkeypatch.setattr(mlb, "wide_ep_fault_tolerance_enabled", lambda: True)
    # Two ranks on two nodes: rank 0 gets three spare slots after EPLB's three (rank 1's slots may hold three
    # experts rank 0 does not).
    monkeypatch.setattr(EplbExpertRestore, "_rank_nodes", lambda self: [0, 1])
    torch.cuda.set_device(0)
    expert_count = 4
    # Slots of 8 MiB (whole VMM chunks), of 3 MiB + 4 bytes (chunks shared with the neighbour slots) and of 4 bytes
    # (a per-slot scale, whose spare slot fills no whole chunk: it keeps the storage it has without spare slots).
    slot_elements = {"w": 1 << 21, "s": (3 << 18) + 1, "alpha": 1}
    mib = 1 << 20
    # The one VMM chunk all unbacked pieces of the process map: released once no piece maps it.
    scratch = 2 * mib
    balancer = FaultTolerantMoeLoadBalancer(
        0,
        2,
        layer_updates_per_iter,
        shared_memory_base_name=f"test_restore_{os.getpid()}",
        mapping=None)
    try:
        # Top-1: route() selects one expert per token.
        layer = balancer.add_layer(expert_count, 1, 3)
        assert (layer.slot_count_per_rank, layer.spare_slot_count) == (6, 3)
        slot_count = layer.slot_count_per_rank
        assignments = layer.with_spare_slots([0, 1, 2, 3, 0, 1])
        local_experts = assignments[:slot_count]
        host = {
            name: [torch.randn(elements) for _ in range(expert_count)]
            for name, elements in slot_elements.items()
        }
        weights = {}
        active = layer.active_slot_count_per_rank
        for name, elements in slot_elements.items():
            # As loaded: the active slots hold their experts, the spare slots nothing.
            loaded = torch.stack(
                [host[name][e] for e in local_experts[:active]] +
                [torch.zeros(elements)] * (slot_count - active))
            storage = layer.create_spare_slot_storage(name, loaded)
            weights[name] = loaded.cuda() if storage is None else storage
        for name in slot_elements:
            for expert_id in range(expert_count):
                layer.host_tensor_sharer.share_host_tensor_with_shape(
                    expert_id, name, host[name][expert_id])
            layer.add_to_migrate_weight_fn(layer.use_spare_slot_storage,
                                           (weights[name], name))
            for slot in range(layer.slot_count_per_rank):
                if layer_updates_per_iter:
                    layer.add_register_weight_fn(
                        lambda slot=slot, name=name: layer.register_weight_slot(
                            slot, name, weights[name][slot]), ())
        layer.host_tensor_sharer.finalize_layer_weights()
        layer.set_initial_weight_assignments(assignments)

        def weight_owner():
            # A MoE layer's weight owner: registering its weights settles the
            # tuner layout of its MoE ops.
            owner = _WeightLayout()
            owner.layer_load_balancer, owner.num_experts = layer, expert_count
            owner.expert_size_per_partition = layer.slot_count_per_rank
            owner.initial_local_expert_ids = local_experts
            owner.register_all_parameter_slot_and_to_fix_weight_fns({})
            return owner

        # The tactics are tuned on EPLB's three slots.
        owner = weight_owner()
        tuner_layout = {"tuner_num_local_experts": 3}
        assert owner.tuner_layout_kwargs == tuner_layout
        alpha_storage = weights["alpha"].data_ptr()
        balancer.register_weight_slots_after_to_cuda()
        balancer.finalize_model()
        balancer.set_warm_up_iter_count(0)
        assert {
            name: id(tensor)
            for name, tensor in layer.unbacked_slot_tensors.items()
        } == {
            "w": id(weights["w"]),
            "s": id(weights["s"])
        }
        if not layer_updates_per_iter:
            assert weights["alpha"].data_ptr() == alpha_storage

        def route(experts):
            selected = torch.tensor(experts, dtype=torch.int32,
                                    device="cuda")[:, None]
            with MoeLoadBalancerIterContext(balancer, False, False):
                layer.start_wait_gpu_stage()
                layer.done_wait_gpu_stage()
                slots = layer.route(selected)
                layer.start_set_cpu_stage()
                layer.done_set_cpu_stage()
            return slots.flatten().tolist()

        # Expert 3 is in rank 1's first slot.
        assert set(route([3] * 8)) == {slot_count}

        balancer.abandon_iterations()
        plan = balancer.expert_restore.plan_expert_restore(
            [True, False], [layer.get_old_rank_expert_ids()])
        # Backing spare slot 3: its four whole "w" chunks, and the two "s" chunks holding its bytes (one shared with
        # spare slot 4); spare slots 4 and 5 stay unbacked.
        assert plan.device_bytes == 8 * mib + 4 * mib
        free = torch.cuda.mem_get_info()[0]
        assert balancer.expert_restore.restore_lost_experts(
            plan) == plan.device_bytes
        assert free - torch.cuda.mem_get_info(
        )[0] >= plan.device_bytes - scratch
        # Only spare slot 3 is backed: backing it again maps nothing, backing slot 4 or 5 would.
        assert layer.spare_slot_bytes_to_back([3]) == 0
        assert all(
            layer.spare_slot_bytes_to_back([slot]) > 0 for slot in (4, 5))
        assert layer.get_old_rank_expert_ids()[0] == [0, 1, 2, 3, -1, -1]
        for name in slot_elements:
            for slot in range(4):
                torch.testing.assert_close(weights[name][slot].cpu(),
                                           host[name][slot],
                                           rtol=0,
                                           atol=0)
        assert set(route([3] * 8)) == {3}
        assert route([0, 1, 2]) == [0, 1, 2]
        # A restore moves experts, never the tuner layout the tactics were tuned with.
        assert owner.tuner_layout_kwargs == tuner_layout
        assert weight_owner().tuner_layout_kwargs == tuner_layout

        if layer_updates_per_iter:
            # Online EPLB balances over the restored spare slot as over the others: by load, experts 3, 0, 1 and 2
            # go to slots 0 to 3, so only the spare slot holds expert 2.
            selected = torch.tensor([3] * 4 + [0] * 3 + [1] * 2,
                                    dtype=torch.int32,
                                    device="cuda")[:, None]
            with MoeLoadBalancerIterContext(balancer, True, True):
                layer.start_wait_gpu_stage()
                layer.done_wait_gpu_stage()
                layer.update_statistic_with_global_ids(selected, True, True)
                layer.start_set_cpu_stage()
                layer.done_set_cpu_stage()
        balancer.abandon_iterations()
        if layer_updates_per_iter:
            assert layer.get_old_rank_expert_ids()[0] == [3, 0, 1, 2, -1, -1]
        rejoin = balancer.expert_restore.plan_expert_restore(
            [True, True], [layer.get_old_rank_expert_ids()],
            rejoining=[False, True])
        assert rejoin.device_bytes == 0
        assert balancer.expert_restore.restore_lost_experts(rejoin) == 0
        free = torch.cuda.mem_get_info()[0]
        assert balancer.expert_restore.release_empty_spare_slots(
        ) == plan.device_bytes
        assert torch.cuda.mem_get_info(
        )[0] - free >= plan.device_bytes - scratch
        # No spare slot is backed.
        assert all(
            layer.spare_slot_bytes_to_back([slot]) > 0
            for slot in layer.spare_slots)
        # The initial placement again: rank 0's active slots are loaded with their experts of it.
        expected = [[0, 1, 2, -1, -1, -1], [3, 0, 1, -1, -1, -1]]
        assert layer.get_old_rank_expert_ids() == expected
        for name in slot_elements:
            for slot in range(3):
                torch.testing.assert_close(weights[name][slot].cpu(),
                                           host[name][slot],
                                           rtol=0,
                                           atol=0)
        # The expert in rank 1's first slot has no other replica.
        assert set(route([expected[1][0]] * 8)) == {slot_count}
    finally:
        balancer.shutdown()


_EXPERTS, _HIDDEN, _INTERMEDIATE = 4, 2048, 2048


def _vmm_granularity() -> int:
    """The VMM granularity of the unbacked slot memory on the current device."""
    from cuda.bindings import driver

    prop = driver.CUmemAllocationProp()
    prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = torch.cuda.current_device()
    error, granularity = driver.cuMemGetAllocationGranularity(
        prop, driver.CUmemAllocationGranularity_flags.
        CU_MEM_ALLOC_GRANULARITY_MINIMUM)
    assert error == driver.CUresult.CUDA_SUCCESS
    return granularity


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_unbacked_slots_back_and_unback():
    """Back a spare slot in place and unback it, taking device memory only while it is backed."""
    from tensorrt_llm.bindings import MemoryCounters

    torch.cuda.set_device(0)
    chunk = _vmm_granularity()
    counters = MemoryCounters.instance()
    before = counters.gpu
    # Four slots of one VMM chunk each, of which slots 2 and 3 are spare.
    slots = torch.ops.trtllm.moe_unbacked_slot_tensor([4, chunk], torch.uint8,
                                                      [2, 3])
    address = slots.data_ptr()
    # The two backed slots and at most the scratch chunk the spare slots share.
    assert counters.gpu - before <= 3 * chunk
    assert torch.ops.trtllm.moe_unbacked_slots_bytes_to_back(slots,
                                                             [2]) == chunk
    assert torch.ops.trtllm.moe_unbacked_slots_set_backed([slots], [2], [1],
                                                          True) == chunk
    assert slots.data_ptr() == address
    slots[2].fill_(7)
    slots[3].fill_(1)
    assert bool(slots[2].eq(7).all())
    backed = counters.gpu
    assert torch.ops.trtllm.moe_unbacked_slots_set_backed([slots], [2], [1],
                                                          False) == chunk
    assert backed - counters.gpu == chunk
    del slots
    assert counters.gpu == before


def _load_moe_layer(monkeypatch, fault_tolerance: bool,
                    meta_init: bool) -> dict:
    """Creates, loads and post-processes one Cutlass BF16 MoE layer with static EPLB as rank 0 of two, the way the
    model loader does, and measures the device memory it takes: in the caching allocator and in unbacked slot
    memory (VMM, which the caching allocator does not see)."""
    from tensorrt_llm._torch.models.modeling_utils import MetaInitMode
    from tensorrt_llm._torch.moe.fused_moe import (RenormalizeMoeRoutingMethod,
                                                   create_moe)
    from tensorrt_llm._torch.moe.fused_moe import moe_load_balancer as mlb
    from tensorrt_llm.bindings import MemoryCounters

    monkeypatch.setattr(mlb, "wide_ep_fault_tolerance_enabled",
                        lambda: fault_tolerance)
    # Two ranks on two nodes: with fault tolerance, rank 0 alone must hold the four experts, two slots more than
    # EPLB's two.
    monkeypatch.setattr(EplbExpertRestore, "_rank_nodes", lambda self: [0, 1])
    eplb_config = MoeLoadBalancerConfig(num_slots=_EXPERTS,
                                        layer_updates_per_iter=0)
    eplb_config.setup(ep_rank=0, ep_size=2)
    pretrained_config = PretrainedConfig()
    pretrained_config.num_experts = _EXPERTS
    pretrained_config.hidden_size = _HIDDEN
    pretrained_config.intermediate_size = _INTERMEDIATE
    pretrained_config.torch_dtype = torch.bfloat16
    model_config = ModelConfig(pretrained_config=pretrained_config,
                               mapping=Mapping(world_size=2,
                                               rank=0,
                                               tp_size=2,
                                               moe_ep_size=2,
                                               enable_attention_dp=True),
                               moe_backend="CUTLASS",
                               moe_load_balancer=eplb_config)
    shapes = {
        "w1": (_INTERMEDIATE, _HIDDEN),
        "w3": (_INTERMEDIATE, _HIDDEN),
        "w2": (_HIDDEN, _INTERMEDIATE)
    }
    weights = {
        f"{expert_id}.{name}.weight":
        torch.full(shape, expert_id + 1.0, dtype=torch.bfloat16)
        for expert_id in range(_EXPERTS)
        for name, shape in shapes.items()
    }
    counters = MemoryCounters.instance()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    allocated, reserved = torch.cuda.memory_allocated(
    ), torch.cuda.memory_reserved()
    unbacked_before = counters.gpu
    balancer_cls = MoeLoadBalancer
    if fault_tolerance:
        balancer_cls = functools.partial(FaultTolerantMoeLoadBalancer,
                                         mapping=None)
    balancer = balancer_cls(
        0, 2, 0, shared_memory_base_name=f"test_load_memory_{os.getpid()}")
    try:
        with balancer, MetaInitMode() if meta_init else nullcontext():
            moe = create_moe(
                routing_method=RenormalizeMoeRoutingMethod(top_k=2),
                model_config=model_config,
                layer_idx=0,
                communication_method="ALLGATHER")
        # The model loader puts the model on the device before it loads the weights.
        moe._apply(lambda t: torch.empty_like(t, device="cuda")
                   if t.device.type == "meta" else t)
        moe.to("cuda")
        moe.load_weights([weights])
        moe.post_load_weights()
        balancer.register_weight_slots_after_to_cuda()
        balancer.finalize_model()
        torch.cuda.synchronize()
        unbacked = counters.gpu - unbacked_before
        peak = torch.cuda.max_memory_allocated() - allocated + unbacked
        torch.cuda.empty_cache()
        (layer, ) = balancer.single_layer_load_balancers
        return {
            "peak": peak,
            "after": torch.cuda.memory_reserved() - reserved + unbacked,
            "unbacked": sorted(layer.unbacked_slot_tensors),
            "loaded": list(moe.backend.initial_local_expert_ids),
            "w2": moe.backend.w2_weight[:2].float().mean(dim=(1, 2)).tolist(),
        }
    finally:
        balancer.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("meta_init", [True, False],
                         ids=["meta_init", "regular_init"])
def test_spare_slots_take_no_device_memory_while_the_weights_load(
        monkeypatch, meta_init):
    """With Wide-EP fault tolerance the slot tensors are created with their spare slots unbacked and the loaders
    skip those slots, so loading a layer takes the device memory it takes without fault tolerance, at its peak
    and after it: within one VMM chunk per unbacked slot tensor and the scratch chunk of the process."""
    torch.cuda.set_device(0)
    # The first load also makes what a process allocates once.
    _load_moe_layer(monkeypatch, fault_tolerance=False, meta_init=meta_init)
    gc.collect()
    with_spares = _load_moe_layer(monkeypatch,
                                  fault_tolerance=True,
                                  meta_init=meta_init)
    gc.collect()
    without = _load_moe_layer(monkeypatch,
                              fault_tolerance=False,
                              meta_init=meta_init)
    # Rank 0 loads its two experts; its two spare slots stay empty.
    assert with_spares["loaded"] == without["loaded"] == [0, 1]
    assert with_spares["w2"] == without["w2"] == [1.0, 2.0]
    assert with_spares["unbacked"] == ["w2_weight", "w3_w1_weight"]
    assert without["unbacked"] == []
    tolerance = (len(with_spares["unbacked"]) + 1) * _vmm_granularity()
    # Two spare slots of w3_w1 and w2 in bfloat16, which loading took before.
    assert 2 * 3 * _HIDDEN * _INTERMEDIATE * 2 > 4 * tolerance
    for key in ("peak", "after"):
        assert abs(with_spares[key] -
                   without[key]) <= tolerance, (key, with_spares[key],
                                                without[key])


if __name__ == '__main__':
    unittest.main()

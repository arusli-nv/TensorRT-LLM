# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import json
import os
import subprocess
import sys
import unittest
from collections.abc import Iterator
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
import torch
from mpi4py import MPI
from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
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
                                           expert_count=4)

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
                1, "weight1", mock_weight)

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

    def test_one_slot_per_expert_routing_reads_the_current_placement(self):
        """Under fault tolerance a static placement of one slot per expert, with empty (-1) spare slots, routes
        each expert to its slot, also after restore_placement moves it; an invalid expert id goes to no slot."""
        torch.cuda.set_device(0)
        ep_size, expert_count, top_k, slots_per_rank = 2, 4, 2, 3
        num_slots = ep_size * slots_per_rank
        from tensorrt_llm.bindings.internal import runtime as _tbr
        native_balancer = _tbr.MoeLoadBalancer

        # Only the native balancer recovers: this single process is not a WideEP job.
        def native(*args, **kwargs):
            return native_balancer(*args, rank_failure_recovery=True)

        with patch('tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer',
                   native):
            balancer = MoeLoadBalancer(0, ep_size, 0)
        layer = balancer.add_layer(expert_count, top_k, slots_per_rank)
        layer.set_initial_weight_assignments([2, 0, -1, 3, 1, -1])
        balancer.finalize_model()
        experts = torch.tensor([[0, 1], [2, 3], [3, -1]],
                               dtype=torch.int32,
                               device="cuda")

        def expect_routes(expert_slots):
            # Per expert its slot; an invalid expert goes to slot num_slots.
            slots = torch.tensor(expert_slots + [num_slots],
                                 dtype=torch.int32,
                                 device="cuda")
            torch.testing.assert_close(layer.route(experts),
                                       slots[torch.where(
                                           experts >= 0, experts,
                                           expert_count)],
                                       rtol=0,
                                       atol=0)

        try:
            expect_routes([1, 4, 0, 3])
            # Expert 0 moves into rank 0's spare slot.
            balancer.load_balancer_impl.restore_placement(
                [], [[[2, -1, 0], [3, 1, -1]]])
            expect_routes([2, 4, 0, 3])
        finally:
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


class _WeightLayout(MoEEplbWeightLayoutMixin):
    pass


@pytest.mark.parametrize("spare_slot_count", [0, 1])
def test_static_routing_with_spare_slots_moves_their_storage_and_shares_host_weights(
        spare_slot_count):
    """Static routing registers no slot functions: with spare slots it registers their storage, whose migration
    (``use_spare_slot_storage``) registers the slots, and shares the host weights a lost rank's experts are restored
    from; without them nothing."""
    module = _WeightLayout()
    module.num_experts = 2
    module.layer_load_balancer = MagicMock(spare_slot_count=spare_slot_count)
    module.layer_load_balancer.is_dynamic_routing.return_value = False
    module.layer_load_balancer.get_load_expert_ids.return_value = [1]
    weights = {name: torch.zeros(1, 3) for name in ("w1", "w2")}
    module.register_all_parameter_slot_and_to_fix_weight_fns(weights)
    balancer = module.layer_load_balancer
    sharer = balancer.host_tensor_sharer
    assert balancer.add_to_migrate_weight_fn.call_count == 2 * spare_slot_count
    balancer.add_register_weight_fn.assert_not_called()
    if spare_slot_count:
        assert [
            c.args[:2]
            for c in sharer.share_host_tensor_with_shape.call_args_list
        ] == [(1, "w1"), (1, "w2")]
        assert [
            c.args[:2]
            for c in sharer.pre_register_host_tensor_with_shape.call_args_list
        ] == [(0, "w1"), (0, "w2")]
    else:
        balancer.get_load_expert_ids.assert_not_called()
        sharer.share_host_tensor_with_shape.assert_not_called()


@pytest.mark.parametrize("shares_host_weights", [False, True])
def test_raw_weights_are_paged_out_wherever_host_weights_are_shared(
        shares_host_weights):
    module = _WeightLayout()
    module.layer_load_balancer = MagicMock()
    module.layer_load_balancer.is_dynamic_routing.return_value = False
    module.layer_load_balancer.need_load_shared_weights.return_value = shares_host_weights
    module._add_raw_shared_weights_for_unmap(["raw"])
    assert module.layer_load_balancer._add_raw_host_weight_for_unmap.call_count == int(
        shares_host_weights)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_unbacked_slot_tensor_takes_device_memory_for_its_backed_slots_only():
    """A slot tensor made with unbacked slots takes device memory for its other slots and the one scratch chunk
    of the process, which its unbacked slots read and write."""
    import tensorrt_llm  # noqa: F401  (registers the trtllm ops)
    from tensorrt_llm.bindings import MemoryCounters

    torch.cuda.set_device(0)
    mib = 1 << 20
    slot_elements = 2 * mib  # 8 MiB slots of float32: whole VMM chunks
    assert torch.ops.trtllm.moe_unbacked_slots_save_memory(
        4 * slot_elements, 4, [2, 3])
    # A [slots] scale has no chunk within its unbacked slots.
    assert not torch.ops.trtllm.moe_unbacked_slots_save_memory(4, 4, [2, 3])
    counters = MemoryCounters.instance()
    before = counters.gpu
    slots = torch.ops.trtllm.moe_unbacked_slot_tensor([4, slot_elements],
                                                      torch.float32, [2, 3])
    assert slots.is_cuda and slots.is_contiguous()
    assert slots.shape == (4, slot_elements) and slots.dtype == torch.float32
    # The two backed slots, and the scratch unless another unbacked range of the process maps it already.
    assert 16 * mib <= counters.gpu - before <= 18 * mib
    slots[:2] = torch.arange(2, dtype=torch.float32, device="cuda")[:, None]
    slots[2] = 7.0
    assert slots[0].eq(0).all() and slots[1].eq(1).all()
    assert slots[3].eq(7.0).all()  # the scratch
    del slots
    torch.cuda.synchronize()
    assert counters.gpu == before


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
def test_online_eplb_loads_slots_whose_spare_slot_maps_no_memory(monkeypatch):
    """With WideEP fault tolerance, online EPLB moves experts between the active slots with CPU copies from the
    host weights, as without spare slots; the spare slots map no memory and stay empty while every rank is live."""
    from tensorrt_llm._torch.moe.fused_moe import moe_load_balancer as mlb

    _skip_without_gdrcopy_slots()
    monkeypatch.setenv("TLLM_HOST_ACCESSIBLE_ALLOW_MANAGED_FALLBACK", "1")
    monkeypatch.setattr(mlb, "fault_tolerance_mode", lambda: True)
    # Two ranks on two nodes: if rank 1 fails, its three slots may hold three experts rank 0 does not, so rank 0
    # gets three spare slots after EPLB's three.
    monkeypatch.setattr(MoeLoadBalancer, "_rank_nodes", lambda self: [0, 1])
    torch.cuda.set_device(0)
    expert_count, slot_elements = 4, 1 << 21  # 8 MiB slots: whole VMM chunks
    balancer = MoeLoadBalancer(
        0, 2, 1, shared_memory_base_name=f"test_online_unbacked_{os.getpid()}")
    try:
        layer = balancer.add_layer(expert_count, 2, 3)
        assert (layer.slot_count_per_rank, layer.spare_slot_count) == (6, 3)
        assignments = layer.with_spare_slots([0, 1, 2, 3, 0, 1])
        assert assignments[:6] == [0, 1, 2, -1, -1, -1]
        weights = layer.create_spare_slot_storage(
            "w",
            torch.tensor(assignments[:6],
                         dtype=torch.float32)[:, None].add(1).expand(
                             -1, slot_elements).contiguous())
        # What the spare slots read unless a copy writes them.
        weights[3:] = -1.0
        for expert_id in range(expert_count):
            layer.host_tensor_sharer.share_host_tensor_with_shape(
                expert_id, "w", torch.full((slot_elements, ), expert_id + 1.0))
        layer.host_tensor_sharer.finalize_layer_weights()
        layer.add_to_migrate_weight_fn(layer.use_spare_slot_storage,
                                       (weights, "w"))
        for slot in range(layer.slot_count_per_rank):
            layer.add_register_weight_fn(
                lambda slot=slot: layer.register_weight_slot(
                    slot, "w", weights[slot]), ())
        layer.set_initial_weight_assignments(assignments)
        balancer.register_weight_slots_after_to_cuda()
        balancer.finalize_model()
        assert list(layer.spare_slots) == [3, 4, 5]

        # Every token picks experts 2 and 3: EPLB replicates them over the active slots.
        balancer.set_warm_up_iter_count(0)
        expert_ids = torch.tensor([[2, 3]] * 8,
                                  dtype=torch.int32,
                                  device="cuda")
        with MoeLoadBalancerIterContext(balancer, True, True):
            layer.start_wait_gpu_stage()
            layer.done_wait_gpu_stage()
            layer.update_statistic_with_global_ids(expert_ids, True, True)
            layer.start_set_cpu_stage()
            layer.done_set_cpu_stage()
        # Waits for the weight update in flight before releasing the host weights it reads.
        assert balancer.shutdown()
        torch.cuda.synchronize()

        local = layer.get_old_rank_expert_ids()[0]
        assert local[3:] == [-1] * 3 and sorted(local[:3]) != [0, 1, 2]
        for slot, expert_id in enumerate(local[:3]):
            assert weights[slot].eq(expert_id + 1).all()
        assert weights[3:].eq(-1.0).all()
        # A spare slot has no memory of its own until a failure backs it, before an expert is loaded into it.
        backed = torch.ops.trtllm.moe_unbacked_slots_set_backed([weights], [3],
                                                                [1], True)
        assert backed > 0
        assert torch.ops.trtllm.moe_unbacked_slots_set_backed([weights], [3],
                                                              [1],
                                                              False) == backed
    finally:
        balancer.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("layer_updates_per_iter", [0, 1],
                         ids=["static", "online"])
def test_a_lost_rank_s_expert_is_restored_into_a_spare_slot_until_it_rejoins(
        monkeypatch, layer_updates_per_iter):
    """Rank 0 of two in one process. Rank 1 fails: its expert 3 moves into rank 0's first spare slot, which gets
    device memory of its own and the expert's exact host weights, and routing reaches it there; the other spare slots
    stay unbacked. Under online EPLB, an update then moves expert 2 into the spare slot instead, with CPU copies that
    write each allocation the slot's bytes lie in. Rank 1 rejoins: the active slots take their experts of the initial
    placement again, loaded from the host weights, routing returns to rank 1, and the spare slot is emptied and its
    memory released."""
    from tensorrt_llm._torch.moe.fused_moe import moe_load_balancer as mlb

    if layer_updates_per_iter:
        _skip_without_gdrcopy_slots()
    monkeypatch.setenv("TLLM_HOST_ACCESSIBLE_ALLOW_MANAGED_FALLBACK", "1")
    monkeypatch.setattr(mlb, "fault_tolerance_mode", lambda: True)
    # Two ranks on two nodes: rank 0 gets three spare slots after EPLB's three (rank 1's slots may hold three
    # experts rank 0 does not).
    monkeypatch.setattr(MoeLoadBalancer, "_rank_nodes", lambda self: [0, 1])
    torch.cuda.set_device(0)
    expert_count = 4
    # Slots of 8 MiB (whole VMM chunks), of 3 MiB + 4 bytes (chunks shared with the neighbour slots) and of 4 bytes
    # (a per-slot scale, whose spare slot fills no whole chunk: it keeps the storage it has without spare slots).
    slot_elements = {"w": 1 << 21, "s": (3 << 18) + 1, "alpha": 1}
    mib = 1 << 20
    # The one VMM chunk all unbacked pieces of the process map: released once no piece maps it.
    scratch = 2 * mib
    balancer = MoeLoadBalancer(
        0,
        2,
        layer_updates_per_iter,
        shared_memory_base_name=f"test_restore_{os.getpid()}")
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
        plan = balancer.plan_expert_restore([True, False],
                                            [layer.get_old_rank_expert_ids()])
        # Backing spare slot 3: its four whole "w" chunks, and the two "s" chunks holding its bytes (one shared with
        # spare slot 4); spare slots 4 and 5 stay unbacked.
        assert plan.device_bytes == 8 * mib + 4 * mib
        free = torch.cuda.mem_get_info()[0]
        assert balancer.restore_lost_experts(plan) == plan.device_bytes
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
        rejoin = balancer.plan_expert_restore([True, True],
                                              [layer.get_old_rank_expert_ids()],
                                              rejoining=[False, True])
        assert rejoin.device_bytes == 0
        assert balancer.restore_lost_experts(rejoin) == 0
        free = torch.cuda.mem_get_info()[0]
        assert balancer.release_empty_spare_slots() == plan.device_bytes
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
        assert owner.tuner_layout_kwargs == tuner_layout
        assert weight_owner().tuner_layout_kwargs == tuner_layout
    finally:
        balancer.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_online_eplb_update_does_not_stall_behind_a_driver_call_waiting_for_the_gpu(
        tmp_path):
    """A driver call that waits for the GPU while it holds the driver lock (in KV cache manager V2: a copy kernel's
    first launch, a cache removal) must not stall an online EPLB weight update in flight, because the GPU waits in
    EPLB's GPU stage wait for that update. Online EPLB's updates call no driver function (CPU copies), so the update
    ends, the wait ends and the call returns. The driver call here is the KV cache copy kernel's first launch under
    lazy loading, which synchronizes the context: in a fresh process, where that kernel is not loaded yet."""
    _skip_without_gdrcopy_slots()
    script = tmp_path / "eplb_update_during_kv_copy_kernel_load.py"
    script.write_text(_EPLB_UPDATE_DURING_KV_COPY_KERNEL_LOAD)
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("OMPI_", "PMIX_", "PMI_"))
    }
    env.update(CUDA_MODULE_LOADING="LAZY",
               TLLM_HOST_ACCESSIBLE_ALLOW_MANAGED_FALLBACK="1")
    try:
        child = subprocess.run([sys.executable, str(script)],
                               env=env,
                               capture_output=True,
                               text=True,
                               timeout=300)
    except subprocess.TimeoutExpired as error:
        pytest.fail(
            "online EPLB stalled behind the copy kernel's load; the process said: "
            f"{error.stdout}")
    assert child.returncode == 0, child.stderr[-4000:]
    result = {}
    for line in child.stdout.splitlines():
        if line.startswith("{"):
            result.update(json.loads(line))
    if not result.get("load_waited_for_gpu"):
        pytest.skip(
            "the copy kernel's lazy load did not wait for the GPU here: the stall cannot form"
        )
    assert result.get("copied") and result.get("forwards_done"), result


# Run in a fresh process by the test above: the copy kernel must not be loaded yet.
_EPLB_UPDATE_DURING_KV_COPY_KERNEL_LOAD = r'''
import json
import os
import threading

import torch

from tensorrt_llm._torch.moe.fused_moe import moe_load_balancer as mlb
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import MoeLoadBalancer, MoeLoadBalancerIterContext
from tensorrt_llm.bindings.internal.batch_manager import kv_cache_manager_v2_utils as copy_helpers

mlb.fault_tolerance_mode = lambda: True
MoeLoadBalancer._rank_nodes = lambda self: [0, 1]  # rank 0 of two nodes: spare slots after EPLB's three
torch.cuda.set_device(0)
expert_count, slot_elements = 4, 1 << 21  # 8 MiB slots: whole VMM chunks
balancer = MoeLoadBalancer(0, 2, 1, shared_memory_base_name=f"stall_{os.getpid()}")
layer = balancer.add_layer(expert_count, 2, 3)
assignments = layer.with_spare_slots([0, 1, 2, 3, 0, 1])
weights = layer.create_spare_slot_storage(
    "w", torch.tensor(assignments[:6], dtype=torch.float32)[:, None].add(1).expand(-1, slot_elements).contiguous())
for expert_id in range(expert_count):
    layer.host_tensor_sharer.share_host_tensor_with_shape(expert_id, "w", torch.full((slot_elements,), expert_id + 1.0))
layer.host_tensor_sharer.finalize_layer_weights()
layer.add_to_migrate_weight_fn(layer.use_spare_slot_storage, (weights, "w"))
for slot in range(layer.slot_count_per_rank):
    layer.add_register_weight_fn(lambda slot=slot: layer.register_weight_slot(slot, "w", weights[slot]), ())
layer.set_initial_weight_assignments(assignments)
balancer.register_weight_slots_after_to_cuda()
balancer.finalize_model()
balancer.set_warm_up_iter_count(0)
expert_ids = torch.tensor([[2, 3]] * 8, dtype=torch.int32, device="cuda")


def forward():
    layer.start_wait_gpu_stage()
    layer.done_wait_gpu_stage()
    layer.update_statistic_with_global_ids(expert_ids, True, True)
    layer.start_set_cpu_stage()
    layer.done_set_cpu_stage()


# The GPU is busy for seconds; behind that, iteration 0 hands the layer back, which starts its weight update, and
# iteration 1 waits in the GPU stage for that update.
torch.cuda._sleep(int(6e9))
with MoeLoadBalancerIterContext(balancer, True, True):
    forward()
with MoeLoadBalancerIterContext(balancer, True, True):
    forward()
forwards_done = torch.cuda.Event()
forwards_done.record()

# Meanwhile a host-tier resume launches the KV cache copy kernel for the first time.
pages, page_bytes = 20, 1 << 16
host = torch.ones(pages, page_bytes, dtype=torch.uint8).pin_memory()
device = torch.zeros(pages, page_bytes, dtype=torch.uint8, device="cuda")
stream = torch.cuda.Stream()
tasks = [copy_helpers.MemToMemTask(dst=device[i].data_ptr(), src=host[i].data_ptr()) for i in range(pages)]
copied = threading.Event()


def copy():
    assert copy_helpers.copy_host_to_device(tasks, page_bytes, stream.cuda_stream) == 0
    copied.set()


threading.Thread(target=copy, daemon=True).start()
# While the GPU sleeps, a lazy load that synchronizes the context still waits.
print(json.dumps({"load_waited_for_gpu": not copied.wait(1.0)}), flush=True)
print(json.dumps({"copied": copied.wait(120)}), flush=True)
forwards_done.synchronize()
stream.synchronize()
print(json.dumps({"forwards_done": True}), flush=True)
assert balancer.shutdown()
'''

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

    monkeypatch.setattr(mlb, "fault_tolerance_mode", lambda: fault_tolerance)
    # Two ranks on two nodes: with fault tolerance, rank 0 alone must hold the four experts, two slots more than
    # EPLB's two.
    monkeypatch.setattr(MoeLoadBalancer, "_rank_nodes", lambda self: [0, 1])
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
    balancer = MoeLoadBalancer(
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
    """With WideEP fault tolerance the slot tensors are created with their spare slots unbacked and the loaders
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


@pytest.mark.cpu_only
def test_shutdown_unmaps_the_host_weights_after_the_weight_updates_end(
) -> None:
    """Online EPLB's weight updates read the host weights until the C++
    shutdown, which waits for them, returns."""
    node = MagicMock()
    node.Get_rank.return_value = 0
    node.Get_size.return_value = 1
    node.allgather.side_effect = lambda value: [value]
    with (
            patch("tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer") as
            runtime,
            patch.object(
                MoeLoadBalancer,
                "_setup_mpi_comm",
                lambda self: setattr(self, "shared_mpi_comm", node),
            ),
            patch("torch.cuda.Event"),
    ):
        runtime.return_value.add_layer.return_value.get_layer_id.return_value = 0
        balancer = MoeLoadBalancer(
            0, 1, 1, shared_memory_base_name=f"test_shutdown_{os.getpid()}")
        layer = balancer.add_layer(2, 1, 2, aux_stream=MagicMock())
        sharer = layer.host_tensor_sharer
        for expert_id in range(2):
            sharer.share_host_tensor_with_shape(
                expert_id, "w", torch.full((4, 64), float(expert_id)))
        sharer.finalize_layer_weights()
        balancer.finalize_model()
        mapped = []
        runtime.return_value.shutdown.side_effect = lambda: mapped.append(
            sharer.own_shm.buf is not None)
        assert balancer.shutdown()
    assert mapped == [True]


if __name__ == '__main__':
    unittest.main()

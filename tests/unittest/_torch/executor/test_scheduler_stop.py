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

import gc
import threading
import time
import weakref
from concurrent.futures import Future
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

pytestmark = pytest.mark.cpu_only


def test_native_kv_bridge_follows_bounded_gpu_join() -> None:
    owner, events = _stop_owner()
    with pytest.raises(RuntimeError, match="terminal disposition"):
        owner._hold_scheduler_stop(SimpleNamespace(control_id="loss-1"),
                                   deadline_monotonic=time.monotonic() + 5)
    assert events == [
        "sampler", "gpu_join", "kv_bridge", "final_gpu_join", "publish"
    ]
    assert owner._scheduler_stop_receipt.local_issuer_join_complete


def test_incomplete_gpu_join_does_not_enter_native_bridge() -> None:
    owner, events = _stop_owner()
    owner.model_engine.join_execution.side_effect = TimeoutError(
        "pending GPU event")
    with pytest.raises(TimeoutError, match="pending GPU event"):
        owner._hold_scheduler_stop(SimpleNamespace(control_id="loss-1"),
                                   deadline_monotonic=time.monotonic() + 5)
    assert events == ["sampler"]
    owner.model_engine.execution_state.retain_failed_stop.assert_called_once()
    owner._serving_task_control.publish_stopped.assert_not_called()


def _stop_owner() -> tuple[PyExecutor, list[str]]:
    owner = object.__new__(PyExecutor)
    events: list[str] = []
    identity = SimpleNamespace(worker=SimpleNamespace(rank=0), generation=0)
    command = SimpleNamespace(identity=identity, transaction_id="loss-1")
    owner._serving_task_control = SimpleNamespace(
        wait_for_scheduler_stop=Mock(return_value=command),
        publish_stopped=Mock(
            side_effect=lambda receipt: events.append("publish")),
    )
    owner.worker_thread = SimpleNamespace(ident=threading.get_ident())
    owner.dist = SimpleNamespace(rank=0)
    owner._scheduler_stop_receipt = None
    owner._scheduler_stop_terminal = SimpleNamespace(wait=lambda timeout: True)
    owner._require_idle_request_owners = Mock()
    owner.previous_batch = None
    owner.max_num_active_requests = 8
    owner.sampler = object()
    owner.execution_stream = owner.encoder_stream = owner.sample_stream = object(
    )
    owner.iter_counter = 1
    owner.is_shutdown = False
    owner.waiting_queue = []
    owner.active_requests = []
    native = SimpleNamespace(refresh_blocks=lambda: events.append("kv_bridge"))
    snapshot = SimpleNamespace(fingerprint=(1, ),
                               native_owner=native,
                               stream=object(),
                               retained_owners=(native, ))
    owner._snapshot_idle_v1_kv = Mock(return_value=snapshot)
    owner._join_idle_sampler = Mock(
        side_effect=lambda deadline: events.append("sampler"))
    joined = SimpleNamespace(
        stopped=True,
        ticket_retired=True,
        watchdog_retired=True,
        generation=0,
        execution_sequence=1,
        events=(object(), ),
    )
    state = SimpleNamespace(generation=0,
                            _graphs={1: object()},
                            validate_join=Mock(),
                            retain_failed_stop=Mock())

    def join(**kwargs: object) -> SimpleNamespace:
        events.append("gpu_join")
        joined.stopped = kwargs["stop"]
        joined.watchdog_retired = kwargs["stop"]
        return joined

    def finish(receipt: object, **kwargs: object) -> object:
        events.append("final_gpu_join")
        return receipt

    owner.model_engine = SimpleNamespace(
        execution_state=state,
        captured_execution_identity=Mock(return_value=SimpleNamespace(
            graph_digest="a" * 64, address_digest="b" * 64)),
        retain_scheduler_join=Mock(),
        join_execution=Mock(side_effect=join),
        finish_execution_stop=Mock(side_effect=finish),
    )
    return owner, events


def test_pending_native_bridge_event_withholds_scheduler_receipt() -> None:
    owner, events = _stop_owner()
    owner.model_engine.finish_execution_stop.side_effect = TimeoutError(
        "pending bridge event")
    with pytest.raises(TimeoutError, match="pending bridge event"):
        owner._hold_scheduler_stop(SimpleNamespace(control_id="loss-1"),
                                   deadline_monotonic=time.monotonic() + 5)
    assert events == ["sampler", "gpu_join", "kv_bridge"]
    assert owner._scheduler_stop_receipt is None
    owner.model_engine.execution_state.retain_failed_stop.assert_called_once()
    owner._serving_task_control.publish_stopped.assert_not_called()


def _idle_owner() -> tuple[PyExecutor, list[str]]:
    owner, events = _stop_owner()
    command = owner._serving_task_control.wait_for_scheduler_stop.return_value
    command.frontend_request_epoch = 2
    owner._serving_task_control.wait_for_idle_boundary = Mock(
        return_value=command)
    owner._serving_task_control.publish_idle = Mock(
        side_effect=lambda receipt: events.append("publish_idle"))
    owner.executor_request_queue = SimpleNamespace(idle_boundary_epoch=1)
    owner.model_engine.summarize_resident_admission = Mock(
        return_value=object())
    owner.model_engine.hold_execution_idle = Mock(
        side_effect=lambda receipt: events.append("seal"))

    def host_progress(*args: object) -> list:
        events.append("host")
        owner.is_shutdown = True
        return []

    owner._fetch_new_requests = Mock(side_effect=host_progress)
    owner._handle_control_request = Mock()
    return owner, events


def test_idle_boundary_joins_seals_and_preserves_host_progress() -> None:
    owner, events = _idle_owner()
    with pytest.raises(RuntimeError, match="terminal shutdown"):
        owner._hold_scheduler_idle(SimpleNamespace(control_id="loss-1"),
                                   deadline_monotonic=time.monotonic() + 5)
    assert events == [
        "sampler",
        "gpu_join",
        "kv_bridge",
        "gpu_join",
        "seal",
        "publish_idle",
        "host",
    ]
    receipt = owner._serving_task_control.publish_idle.call_args.args[0]
    assert receipt.execution_sequence == 1
    assert receipt.queue_ingress_epoch == 1
    assert receipt.frontend_request_epoch == 2
    owner._serving_task_control.publish_stopped.assert_not_called()


def test_idle_boundary_pending_bridge_withholds_seal_and_receipt() -> None:
    owner, events = _idle_owner()
    initial = owner.model_engine.join_execution(stop=False)
    events.clear()
    owner.model_engine.join_execution.side_effect = [
        initial, TimeoutError("pending idle bridge")
    ]
    with pytest.raises(TimeoutError, match="pending idle bridge"):
        owner._hold_scheduler_idle(SimpleNamespace(control_id="loss-1"),
                                   deadline_monotonic=time.monotonic() + 5)
    assert events == ["sampler", "kv_bridge"]
    owner.model_engine.hold_execution_idle.assert_not_called()
    owner._serving_task_control.publish_idle.assert_not_called()
    owner.model_engine.execution_state.retain_failed_stop.assert_called_once()


def test_pending_sampler_future_is_retained_before_model_join() -> None:
    from tensorrt_llm._torch.moe.fused_moe.communication import execution_state
    from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine

    owner, events = _stop_owner()
    state = object.__new__(execution_state.ExecutionState)
    state._lock = threading.RLock()
    state._phase = "ready"
    state._local_join_owned = False
    state.ep_group_health = SimpleNamespace(
        snapshot=lambda: SimpleNamespace(generation=0))
    model = object.__new__(PyTorchModelEngine)
    model.execution_state = state
    state._model_owner = model
    owner.model_engine = model
    future = Future()
    future_ref = weakref.ref(future)
    owner.sampler = SimpleNamespace(_async_worker_futures=(future, ))
    owner._join_idle_sampler = MethodType(PyExecutor._join_idle_sampler, owner)
    state_id = id(state)
    try:
        with pytest.raises(TimeoutError):
            owner._hold_scheduler_stop(SimpleNamespace(control_id="loss-1"),
                                       deadline_monotonic=time.monotonic() +
                                       0.03)
        assert state._phase == "blocked"
        assert execution_state._UNCERTAIN_STATES[state_id] is state
        assert model._scheduler_join_owner is owner._scheduler_join_owners
        assert owner._scheduler_join_owners.sampler_owners[-1] == (future, )
        assert events == []
        del owner, model, state, future
        gc.collect()
        assert future_ref() is not None
    finally:
        execution_state._UNCERTAIN_STATES.pop(state_id, None)


def test_recovery_router_requires_exact_default_type() -> None:
    from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import \
        DefaultADPRouter

    owner = object.__new__(PyExecutor)
    owner.adp_router = object.__new__(DefaultADPRouter)
    owner._qualify_recovery_router()
    owner.adp_router = SimpleNamespace()
    with pytest.raises(RuntimeError, match="DefaultADPRouter"):
        owner._qualify_recovery_router()


def _resident_placement_owner(
        monkeypatch: pytest.MonkeyPatch) -> tuple[object, ...]:
    from tensorrt_llm._torch.moe import fused_moe
    from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine

    class MoE(SimpleNamespace):
        pass

    monkeypatch.setattr(fused_moe, "ConfigurableMoE", MoE)
    rows = [[0, 1], [0, 1], [0, 1]]
    layer = SimpleNamespace(
        is_static_routing=lambda: True,
        get_old_rank_expert_ids=lambda: rows,
        get_layer_idx=lambda: 3,
    )
    weights = SimpleNamespace(shape=(2, 16),
                              device=SimpleNamespace(type="cuda"))
    backend = SimpleNamespace(
        layer_load_balancer=layer,
        initial_local_expert_ids=[0, 1],
        initial_global_assignments=[0, 1] * 3,
        expert_size_per_partition=2,
        w3_w1_weight=weights,
        w2_weight=weights,
    )
    balancer = SimpleNamespace(in_iter=False,
                               single_layer_load_balancers=[layer])
    module = MoE(
        model_config=SimpleNamespace(moe_load_balancer=balancer),
        layer_load_balancer=layer,
        backend=backend,
        expert_size_per_partition=2,
        ep_size=3,
        ep_rank=0,
        num_slots=6,
        num_experts=2,
        initial_global_assignments=[0, 1] * 3,
        initial_local_expert_ids=[0, 1],
        comm=SimpleNamespace(execution_geometry=lambda: (3, 0, 128)),
    )
    owner = SimpleNamespace(
        scheduler_thread_id=threading.get_ident(),
        request_capacity=8,
        request=SimpleNamespace(control_id="loss-1"),
        execution=SimpleNamespace(stopped=True, generation=0),
        native_after=SimpleNamespace(kv_manager=SimpleNamespace(
            get_num_available_tokens=lambda limit: 64,
            get_num_kv_blocks=lambda length: length // 8,
            tokens_per_block=8)),
    )
    model = object.__new__(PyTorchModelEngine)
    model.mapping = SimpleNamespace(world_size=3, rank=0)
    model.execution_state = SimpleNamespace(generation=0, validate_join=Mock())
    model._scheduler_join_owner = owner
    model._qualify_execution_owners = Mock()
    model.model = SimpleNamespace(modules=lambda: [module])
    model.max_num_tokens = 128
    model.batch_size = 8
    model.max_seq_len = 8
    proposal = SimpleNamespace(
        digest="a" * 64,
        base_generation=0,
        survivors=(SimpleNamespace(rank=0), SimpleNamespace(rank=2)),
        failed_worker=SimpleNamespace(rank=1),
    )
    stopped = SimpleNamespace(
        scheduler_thread_id=threading.get_ident(),
        transaction_id="loss-1",
        model_generation=0,
    )
    return model, proposal, stopped, owner, module, rows


def test_resident_placement_inspection_reports_host_provenance(
        monkeypatch: pytest.MonkeyPatch) -> None:
    model, proposal, stopped, owner, module, _ = _resident_placement_owner(
        monkeypatch)
    result = model.inspect_resident_placement(owner)
    assert result.layers == ((3, 2, ((0, 1), (0, 1), (0, 1))), )
    assert result.admitted_ranks == (1, 2)
    assert result.fixed_slots == ((3, 2, 2), )
    assert result.max_tokens_per_logical_rank == 128
    assert result.available_kv_tokens == 64
    assert result.max_active_requests == 8
    assert result.request_sequence_length == 8
    assert any(retained is module for retained in result.owners)
    model.execution_state.validate_join.assert_called_with(owner.execution)


def test_resident_placement_inspection_excludes_uncovered_rank(
        monkeypatch: pytest.MonkeyPatch) -> None:
    model, proposal, stopped, owner, module, rows = _resident_placement_owner(
        monkeypatch)
    rows[:] = [[0, 0], [1, 1], [0, 0]]
    module.initial_global_assignments = [
        expert for row in rows for expert in row
    ]
    module.backend.initial_global_assignments = module.initial_global_assignments
    module.initial_local_expert_ids = [0, 0]
    module.backend.initial_local_expert_ids = [0, 0]
    result = model.inspect_resident_placement(owner)
    assert result.admitted_ranks == (2, )


def test_resident_placement_inspection_rejects_changed_loader_assignment(
        monkeypatch: pytest.MonkeyPatch) -> None:
    model, proposal, stopped, owner, module, _ = _resident_placement_owner(
        monkeypatch)
    module.backend.initial_local_expert_ids = [1, 0]
    with pytest.raises(RuntimeError, match="resident loader slots"):
        model.inspect_resident_placement(owner)


@pytest.mark.parametrize("load_format,sleep", [("GMS", None),
                                               ("AUTO", object())])
def test_recovery_rejects_unqualified_allocator_ownership(
        load_format: str, sleep: object) -> None:
    from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine

    model = object.__new__(PyTorchModelEngine)
    model.llm_args = SimpleNamespace(load_format=load_format,
                                     sleep_config=sleep)
    model.mapping = SimpleNamespace(enable_lm_head_tp_in_adp=False,
                                    attn_tp_size=1,
                                    moe_tp_size=1,
                                    pp_size=1,
                                    attn_cp_size=1)
    with pytest.raises(RuntimeError, match="allocator ownership"):
        model._qualify_execution_owners(())


def test_resident_placement_inspection_rejects_insufficient_native_capacity(
        monkeypatch: pytest.MonkeyPatch) -> None:
    model, _, _, owner, _, _ = _resident_placement_owner(monkeypatch)
    owner.native_after.kv_manager.get_num_available_tokens = lambda limit: 63
    result = model.inspect_resident_placement(owner)
    assert result.admitted_ranks == ()


def test_capture_identity_preserves_original_graph_across_generation(
        monkeypatch: pytest.MonkeyPatch) -> None:
    model, _, _, owner, _, _ = _resident_placement_owner(monkeypatch)
    graph = SimpleNamespace(raw_cuda_graph=lambda: 11,
                            raw_cuda_graph_exec=lambda: 12)
    footprint = SimpleNamespace(
        graph_key=(1, 0),
        graph_owner_id=id(graph),
        raw_graph=11,
        executable=12,
        capture_generation=0,
        workspace_geometry=(3, 0, 128),
        mapping_token=object(),
        descriptor_addresses=(1, 2, 3, 4),
        spans=(),
    )
    state = model.execution_state
    state._graphs = {id(graph): (graph, )}
    state.captured_footprints = (footprint, )
    state.abort, state.descriptor, state.published, state.slot = tuple(
        SimpleNamespace(data_ptr=lambda pointer=pointer: pointer)
        for pointer in (1, 2, 3, 4))
    owner.native_after.fingerprint = (100, ((200, 8, 2), ))
    original = model.captured_execution_identity(owner)
    assert model.captured_execution_identity(owner) == original
    state.generation = 1
    assert model.captured_execution_identity(owner) == original


def test_recovery_inspection_requires_accepted_original_stop() -> None:
    owner, _ = _stop_owner()
    proposal = SimpleNamespace(placement_digest="a" * 64,
                               base_generation=0,
                               next_generation=1,
                               failed_worker=SimpleNamespace(rank=1))
    owner._serving_task_control.validate_preparation = Mock()
    owner._scheduler_join_owners = object()
    prepared = SimpleNamespace(placement_digest="a" * 64,
                               base_generation=0,
                               next_generation=1)
    owner.model_engine.prepare_resident_placement = Mock(return_value=prepared)
    with pytest.raises(RuntimeError, match="original scheduler STOP"):
        owner._inspect_recovery_placement(proposal)
    owner.model_engine.prepare_resident_placement.assert_not_called()
    stopped = object()
    owner._scheduler_stop_receipt = stopped
    assert owner._inspect_recovery_placement(proposal) is prepared
    owner._serving_task_control.validate_preparation.assert_called_once_with(
        proposal, stopped)
    proposal.placement_digest = "b" * 64
    with pytest.raises(RuntimeError,
                       match="differs from prepared resident placement"):
        owner._inspect_recovery_placement(proposal)


def _placement_stop_receipt() -> object:
    from tensorrt_llm.executor.recovery import (SchedulerStopReceipt,
                                                WorkerTaskIdentity)
    from tensorrt_llm.executor.worker_process_monitor import \
        capture_worker_process_identity

    return SchedulerStopReceipt(
        identity=WorkerTaskIdentity(1, 0, 1,
                                    capture_worker_process_identity(0)),
        transaction_id="loss-1",
        scheduler_thread_id=threading.get_ident(),
        iteration=2,
        owner_counts=(),
        model_generation=0,
        registered_stream_events_completed=1,
        ticket_retired=True,
        local_issuer_join_complete=True,
        native_kv_inventory_complete=True,
        owner_reference="original_scheduler",
    )


def test_prepared_placement_preserves_survivor_slots_and_binds_next_digest(
        monkeypatch: pytest.MonkeyPatch) -> None:
    model, _, _, owner, module, rows = _resident_placement_owner(monkeypatch)
    source = model.inspect_resident_placement(owner)
    model._resident_placement_inspection = source
    owner.execution.owner = model.execution_state
    stopped = _placement_stop_receipt()
    prepared = model.prepare_resident_placement(1, stopped, owner)
    assert prepared.layers == ((3, 2, ((0, 1), (-1, -1), (0, 1))), )
    assert prepared.layers[0][2][0] is prepared.source.layers[0][2][0]
    assert prepared.layers[0][2][2] is prepared.source.layers[0][2][2]
    assert prepared.placement_digest == dict(source.masked_placement_digests)[1]
    assert prepared.placement_digest != source.placement_digest
    assert (prepared.base_generation, prepared.next_generation) == (0, 1)
    assert prepared.stopped is stopped
    assert any(retained is module.backend.w2_weight
               for retained in prepared.source.owners)
    assert rows == [[0, 1], [0, 1], [0, 1]]
    assert model.execution_state.generation == 0


@pytest.mark.parametrize("changed", ["join", "weight", "capacity", "rank"])
def test_prepared_placement_rejects_changed_or_unadmitted_owners(
        monkeypatch: pytest.MonkeyPatch, changed: str) -> None:
    model, _, _, owner, module, _ = _resident_placement_owner(monkeypatch)
    model._resident_placement_inspection = model.inspect_resident_placement(
        owner)
    owner.execution.owner = model.execution_state
    failed_rank = 1
    if changed == "join":
        model._scheduler_join_owner = object()
    elif changed == "weight":
        module.backend.w2_weight = SimpleNamespace(
            shape=(2, 16), device=SimpleNamespace(type="cuda"))
    elif changed == "capacity":
        owner.native_after.kv_manager.get_num_available_tokens = lambda limit: 63
    else:
        failed_rank = 0
    with pytest.raises(RuntimeError):
        model.prepare_resident_placement(failed_rank, _placement_stop_receipt(),
                                         owner)


@pytest.mark.parametrize(
    "field", ["attn_tp_size", "moe_tp_size", "pp_size", "attn_cp_size"])
def test_recovery_rejects_required_shard_loss(field: str) -> None:
    from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine

    model = object.__new__(PyTorchModelEngine)
    model.mapping = SimpleNamespace(enable_lm_head_tp_in_adp=False,
                                    attn_tp_size=1,
                                    moe_tp_size=1,
                                    pp_size=1,
                                    attn_cp_size=1)
    setattr(model.mapping, field, 2)
    with pytest.raises(RuntimeError, match="shard loss"):
        model._qualify_execution_owners(())


def test_recovery_rejects_captured_dummy_allreduce() -> None:
    from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine

    model = object.__new__(PyTorchModelEngine)
    model.mapping = SimpleNamespace(enable_lm_head_tp_in_adp=False,
                                    attn_tp_size=1,
                                    moe_tp_size=1,
                                    pp_size=1,
                                    attn_cp_size=1)
    module = SimpleNamespace(enable_dummy_allreduce=True)
    with pytest.raises(RuntimeError, match="dummy world allreduce"):
        model._qualify_execution_owners((module, ))


def _kv_layout_owner(monkeypatch: pytest.MonkeyPatch) -> tuple[object, ...]:
    import math

    import torch

    import tensorrt_llm._utils as utils

    def tensor(pointer: int, shape: tuple[int, ...],
               dtype: torch.dtype) -> SimpleNamespace:
        return SimpleNamespace(
            device=torch.device("cuda", 0),
            shape=shape,
            dtype=dtype,
            data_ptr=lambda: pointer,
            numel=lambda: math.prod(shape),
            element_size=lambda: dtype.itemsize,
            storage_offset=lambda: 0,
            is_contiguous=lambda: True,
            stride=lambda: tuple(
                math.prod(shape[index + 1:]) for index in range(len(shape))),
        )

    monkeypatch.setattr(utils, "binding_to_torch_dtype", lambda dtype: dtype)
    pool = tensor(100000, (4, 2, 1, 16), torch.float16)
    manager = SimpleNamespace(
        num_pools=1,
        impl=SimpleNamespace(num_pools=1),
        blocks_in_secondary_pool=0,
        enable_indexer_k_cache=False,
        is_linear_attention=False,
        is_vswa=False,
        num_kv_heads_per_layer=[1, 1],
        num_local_layers=2,
        blocks_in_primary_pool=4,
        kv_factor=1,
        tokens_per_block=8,
        head_dim=2,
        dtype=torch.float16,
        _stream=SimpleNamespace(device=pool.device),
        max_blocks_per_seq=2,
        kv_cache_pool_pointers=torch.tensor([[pool.data_ptr(), 0]],
                                            dtype=torch.int64),
        kv_cache_pool_mapping=torch.tensor([[0, 0], [0, 1]], dtype=torch.int32),
    )
    offsets = tensor(200000, (1, 2, 2, 2), torch.int32)
    metadata = SimpleNamespace(kv_cache_manager=manager,
                               max_num_sequences=2,
                               kv_cache_block_offsets=offsets)
    runner = SimpleNamespace(graphs={2: object()},
                             graph_metadata={2: {
                                 "attn_metadata": metadata
                             }})
    owner = object.__new__(PyExecutor)
    client = SimpleNamespace(
        execution_geometry=lambda: (3, 0, 128, 2, None, 0, 1000, 512, 256))
    owner.model_engine = SimpleNamespace(execution_state=SimpleNamespace(
        _clients=(client, )))
    return owner, manager, pool, runner, metadata


def test_idle_kv_layout_records_actual_host_values_and_captured_spans(
        monkeypatch: pytest.MonkeyPatch) -> None:
    owner, manager, pool, runner, metadata = _kv_layout_owner(monkeypatch)
    layout, retained = owner._inspect_idle_v1_kv_layout(manager, pool, runner)
    assert layout[:2] == (((100000, 0), ), ((0, 0), (0, 1)))
    assert layout[2][:2] == (100000, 256)
    assert layout[3][0][1][:2] == (200000, 32)
    assert retained == (metadata, metadata.kv_cache_block_offsets)


@pytest.mark.parametrize("table", ["pointer", "mapping"])
def test_idle_kv_layout_rejects_stale_host_table_values(
        monkeypatch: pytest.MonkeyPatch, table: str) -> None:
    owner, manager, pool, runner, _ = _kv_layout_owner(monkeypatch)
    if table == "pointer":
        manager.kv_cache_pool_pointers[0, 0] += 8
    else:
        manager.kv_cache_pool_mapping[1, 1] = 0
    with pytest.raises(RuntimeError, match="original full pool"):
        owner._inspect_idle_v1_kv_layout(manager, pool, runner)


def test_idle_kv_layout_rejects_changed_primary_pool_layout(
        monkeypatch: pytest.MonkeyPatch) -> None:
    owner, manager, pool, runner, _ = _kv_layout_owner(monkeypatch)
    pool.is_contiguous = lambda: False
    with pytest.raises(RuntimeError, match="primary pool ownership/layout"):
        owner._inspect_idle_v1_kv_layout(manager, pool, runner)


def test_idle_kv_layout_rejects_captured_offsets_aliasing_peer_workspace(
        monkeypatch: pytest.MonkeyPatch) -> None:
    owner, manager, pool, runner, metadata = _kv_layout_owner(monkeypatch)
    metadata.kv_cache_block_offsets.data_ptr = lambda: 1512
    with pytest.raises(RuntimeError, match="remote-writable MoE storage"):
        owner._inspect_idle_v1_kv_layout(manager, pool, runner)

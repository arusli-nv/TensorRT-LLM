# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import gc
import hashlib
import os
import socket
import threading
import time
from contextlib import AbstractContextManager, nullcontext
from multiprocessing import resource_tracker, shared_memory
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from mpi4py import MPI

import tensorrt_llm
import tensorrt_llm.bindings.internal.runtime as _tbr
from tensorrt_llm._torch.distributed import AllReduce
from tensorrt_llm._torch.mmap_utils import advise_tensor_pageout
from tensorrt_llm._torch.modules.multi_stream_utils import do_multi_stream
from tensorrt_llm._torch.utils import EventType
from tensorrt_llm._utils import is_mpi_peer_failure, peer_loss_recovery_enabled
from tensorrt_llm.executor.recovery.config import fault_tolerance_config
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

# madvise(2) advice that builds this process's page-table entries for resident pages (Linux 5.14+).
_MADV_POPULATE_READ = 22


def _populate_in_background(segments: List[shared_memory.SharedMemory]) -> None:
    """Map the pages of host copies that other local ranks wrote, off the startup path.

    Without it, the first copy from a segment page-faults on every page: a
    restore's GPU copies of 1.4 GB took ~400 ms instead of ~35 ms. Best effort:
    older kernels reject the advice and the faults come back.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    spans = [(np.frombuffer(segment.buf,
                            dtype=np.uint8).ctypes.data, segment.size)
             for segment in segments]

    def populate() -> None:
        for address, size in spans:
            if libc.madvise(ctypes.c_void_p(address), ctypes.c_size_t(size),
                            _MADV_POPULATE_READ):
                logger.debug("Prefaulting EPLB host copies failed: "
                             f"{os.strerror(ctypes.get_errno())}")
                return

    threading.Thread(target=populate,
                     name="eplb_host_copy_populate",
                     daemon=True).start()


def _without_replicas(placement: np.ndarray) -> np.ndarray:
    """``placement`` ([ep_size, slots_per_rank] expert ids) with -1 for every replica beyond an expert's first.

    The kept replica is the one with the lowest local slot (an expert's home
    slot), ties to the lowest rank.
    """
    placement = placement.copy()
    seen = set()
    for slot, rank in np.ndindex(placement.shape[::-1]):
        expert = int(placement[rank, slot])
        if expert in seen:
            placement[rank, slot] = -1
        elif expert >= 0:
            seen.add(expert)
    return placement


def _tensor_to_weight(t: torch.Tensor) -> _tbr.MoeWeight:
    """
    Convert a tensor to a MoeWeight object.

    Args:
        t: The tensor to convert
    """
    assert t.dim() <= 2, "t.dim() should be less than or equal to 2"
    shape = [1, 1]
    pitch = 1
    elt_size = torch.tensor([], dtype=t.dtype).element_size()
    if t.dim() == 2:
        shape[0] = t.size(0)
        shape[1] = t.size(1)
        pitch = t.stride(0)
    elif t.dim() == 1:
        shape[1] = t.size(0)
        pitch = t.size(0)
    else:
        pass
    mw = _tbr.MoeWeight()
    mw.height = shape[0]
    mw.width = shape[1] * elt_size
    mw.pitch = pitch * elt_size
    mw.weight_ptr = t.data_ptr()
    return mw


class HostMoeTensorSharer:
    """
    A class representing a host tensor sharer.
    """

    def __init__(self, layer_id: int, expert_count: int,
                 shared_mpi_comm: MPI.Comm):
        """
        Initialize a HostMoeTensorSharer instance.

        Args:
            shared_mpi_comm: The MPI communicator for shared memory
        """
        self.shared_mpi_comm = shared_mpi_comm
        self.layer_id = layer_id
        self.expert_count = expert_count
        self.shared_memory_base_name = None

        self.local_rank = self.shared_mpi_comm.Get_rank()
        self.local_size = self.shared_mpi_comm.Get_size()

        self.keep_experts(range(expert_count))

        self.name_info = {}  # key is weight name, value is (dtype, shape)
        self.host_weights = {}

        self.own_shm = None
        self.imported_shms = []

        self.shared_tensors = {}
        self.names = []

        self.loaded_shared_weights = []

    def keep_experts(self, expert_ids: Iterable[int]) -> None:
        """Keep host copies of ``expert_ids`` only (before any is shared), split over the node's ranks in runs."""
        self.expert_ids = sorted(expert_ids)
        # The experts this rank loads and writes to its shared memory.
        self.own_expert_ids = self.rank_expert_ids(self.local_rank)

    def rank_expert_ids(self, local_rank: int) -> List[int]:
        """The experts local rank ``local_rank`` of the node loads and shares."""
        count = len(self.expert_ids)
        return self.expert_ids[local_rank * count //
                               self.local_size:(local_rank + 1) * count //
                               self.local_size]

    def set_shared_memory_base_name(self, shared_memory_base_name):
        """
        Set the shared memory base name for the layer.

        Args:
            shared_memory_base_name: The base name for the shared memory
        """
        self.shared_memory_base_name = shared_memory_base_name

    def get_shared_memory_name(self, rank: Optional[int] = None):
        """
        Get the shared memory name for the layer.

        Args:
            rank: The rank who created the shared memory. Current rank if None
        """
        if rank is None:
            rank = self.local_rank
        assert 0 <= rank < self.local_size
        assert isinstance(self.shared_memory_base_name,
                          str), "self.shared_memory_base_name must be a string"
        shared_memory_name = f"{self.shared_memory_base_name}_l{self.layer_id}_lr{rank}_all"
        return shared_memory_name

    def pre_register_host_tensor_with_shape(self, expert_id: int, name: str,
                                            dtype, tensor_shape):
        """
        Pre-register a host tensor with shape.
        This function is invoked by the processes that don't load the weights.
        It just records the shape of the weight tensor, and the process that
        loads the weights will call share_host_tensor_with_shape to share the
        actual weight tensor. After the weight tensor is shared, e.g. after the
        barrier in finalize_model, the processes that pre-registers the weight
        tensor will add the weight tensor to the layer.

        Args:
            expert_id: The ID of the expert
            name: The name of the weight
        """
        assert len(tensor_shape
                   ) <= 2, "tensor_shape dim must be less than or equal to 2"
        assert 0 <= expert_id < self.expert_count
        assert expert_id not in self.own_expert_ids
        if name not in self.name_info:
            self.name_info[name] = (dtype, tensor_shape)
        else:
            assert dtype == self.name_info[name][0] and tensor_shape == self.name_info[name][1], \
                f'weights name={name}, dtype={dtype}, shape={tensor_shape}, but already registered with dtype={self.name_info[name][0]}, shape={self.name_info[name][1]}'

    def share_host_tensor_with_shape(self, expert_id: int, name: str,
                                     t: torch.Tensor):
        """
        Share a host tensor with shape.
        This function is invoked by the processes that load the weights.
        It creates a shared memory for the weight tensor, and then shares the
        weight tensor with the processes that pre-register the weight tensor.

        Args:
            expert_id: The ID of the expert
            name: The name of the weight
            t: The weight tensor
        """
        assert len(
            t.shape) <= 2, "tensor_shape dim must be less than or equal to 2"
        assert t.is_contiguous() == True, "t.is_contiguous() must be True"
        assert (expert_id, name) not in self.shared_tensors.keys()
        assert expert_id in self.own_expert_ids
        self.shared_tensors[(expert_id, name)] = t
        dtype = t.dtype
        tensor_shape = t.shape
        if name not in self.name_info:
            self.name_info[name] = (dtype, tensor_shape)
        else:
            assert dtype == self.name_info[name][0] and tensor_shape == self.name_info[name][1], \
                f'weights name={name}, dtype={dtype}, shape={tensor_shape}, but already registered with dtype={self.name_info[name][0]}, shape={self.name_info[name][1]}'

    @staticmethod
    def align_size(size: int):
        return (size + 256 - 1) // 256 * 256

    def add_raw_host_weight_for_unmap(self,
                                      raw_weight_tensors: List[torch.Tensor]):
        """
        Add a raw weight (mmapped Tensor) to HostMoeTensorSharer for later `madvise` to save host memory.

        Args:
            raw_weight_tensors: A list of raw weight tensors
        """
        self.loaded_shared_weights.extend(raw_weight_tensors)

    def finalize_layer_weights(self):
        self.names = list(sorted(self.name_info.keys()))
        assert len(self.shared_tensors.keys()) == len(
            self.own_expert_ids) * len(self.names)

        total_size = 0
        for name in self.names:
            dtype, shape = self.name_info[name]
            for expert_id in self.own_expert_ids:
                t = self.shared_tensors[(expert_id, name)]
                assert dtype == t.dtype and shape == t.shape
                data_size = t.numel() * t.element_size()
                aligned_size = self.align_size(data_size)
                total_size += aligned_size

        # A rank of a node that holds fewer experts than it has ranks may share none.
        if total_size:
            shm_name = self.get_shared_memory_name()
            try:
                shm = shared_memory.SharedMemory(name=shm_name,
                                                 create=True,
                                                 size=total_size)
            except FileExistsError:
                tensorrt_llm.logger.warning(
                    f'Found exist EPLB shared memory name: {shm_name}, unlinking...'
                )
                existing_shm = shared_memory.SharedMemory(name=shm_name)
                existing_shm.close()
                existing_shm.unlink()
                shm = shared_memory.SharedMemory(name=shm_name,
                                                 create=True,
                                                 size=total_size)
            self.own_shm = shm

        offset = 0
        for name in self.names:
            for expert_id in self.own_expert_ids:
                t = self.shared_tensors[(expert_id, name)].contiguous().cpu()
                data_size = t.numel() * t.element_size()
                aligned_size = self.align_size(data_size)
                shm.buf[offset:offset + data_size] = t.flatten().view(
                    torch.int8).numpy().tobytes()
                dtype = t.dtype
                tensor_shape = t.shape
                elt_count = t.numel()
                st = torch.frombuffer(shm.buf,
                                      dtype=dtype,
                                      offset=offset,
                                      count=elt_count).view(tensor_shape)
                key = (expert_id, name)
                assert key not in self.host_weights.keys(
                ), f"key={key} already exists"
                self.host_weights[key] = st
                offset += aligned_size
        self.shared_tensors = {}

        for raw_weight in self.loaded_shared_weights:
            advise_tensor_pageout(raw_weight)

        self.loaded_shared_weights = []

        gc.collect()

    def finalize_host_tensor_sharing(self, add_host_weight_fn: Callable = None):
        """
        Finalize the host tensor sharing.
        """
        for rank in range(self.local_size):
            rank_expert_ids = self.rank_expert_ids(rank)
            if rank == self.local_rank or not rank_expert_ids:
                continue

            shm_name = self.get_shared_memory_name(rank)
            shm = shared_memory.SharedMemory(name=shm_name)
            self.imported_shms.append(shm)

            offset = 0
            for name in self.names:
                dtype, shape = self.name_info[name]
                elt_count = int(np.prod(shape))
                data_size = torch.tensor([],
                                         dtype=dtype).element_size() * elt_count
                aligned_size = self.align_size(data_size)
                for expert_id in rank_expert_ids:
                    t = torch.frombuffer(shm.buf,
                                         dtype=dtype,
                                         offset=offset,
                                         count=elt_count).view(shape)
                    key = (expert_id, name)
                    assert key not in self.host_weights.keys(
                    ), f"key={key} already exists"
                    self.host_weights[key] = t
                    offset += aligned_size

        if add_host_weight_fn is not None:
            for key, t in self.host_weights.items():
                add_host_weight_fn(key[0], key[1], t)

        self.host_weights.clear()
        _populate_in_background(self.imported_shms)

    def pre_shutdown_cleanup(self):
        """
        Clean up the resources before C++ shutdown and barrier
        """
        for shm in self.imported_shms:
            shm.close()
            resource_tracker.unregister(shm._name, "shared_memory")
        self.imported_shms = None
        if self.own_shm:
            self.own_shm.close()

    def post_shutdown_cleanup(self):
        """
        Clean up the resources before C++ shutdown and barrier
        """
        if self.own_shm:
            self.own_shm.unlink()
            self.own_shm = None


class SingleLayerMoeLoadBalancer:
    """
    A class representing a single layer of the Mixture of Experts (MoE) load balancer.
    This class wraps the C++ implementation of SingleLayerMoeLoadBalancer.
    """

    def __init__(
            self,
            single_layer_load_balancer_impl: _tbr.SingleLayerMoeLoadBalancer,
            shared_mpi_comm: MPI.Comm,
            expert_count: int,
            updates_enabled: bool = True,
            repeated_count=1,
            aux_stream: Optional[torch.cuda.Stream] = None,
            host_copies: Optional[bool] = None,
            slot_count_per_rank: Optional[int] = None,
            ghost_slot_count: int = 0,
            host_copy_ranks: Optional[List[int]] = None):
        """
        Initialize a SingleLayerMoeLoadBalancer instance.

        Args:
            single_layer_load_balancer_impl: The C++ implementation of SingleLayerMoeLoadBalancer
            shared_mpi_comm: The MPI communicator for shared memory
            expert_count: total number of experts
            updates_enabled: whether to enable weight updates
            repeated_count: the repeated count of current layer, used when forward is repeated more than once like MTP.
            host_copies: keep every expert's weights in host memory and the slots loadable from them (default:
                with weight updates); with static routing, for loading slots outside iterations (WideEP peer-loss
                recovery restores a lost rank's experts this way).
            slot_count_per_rank: the number of slots per rank, ``ghost_slot_count`` of them last.
            ghost_slot_count: slots per rank appended after online EPLB's own for WideEP peer-loss recovery; they
                start empty and EPLB's updates fill one only once a restore loaded an expert into it.
            host_copy_ranks: with ring host copies, the EP ranks whose home experts' host copies this node keeps.
        """
        self.single_layer_load_balancer_impl = single_layer_load_balancer_impl
        self.single_layer_load_balancer_ptr = single_layer_load_balancer_impl.get_pointer(
        )
        self.expert_count = expert_count
        self.updates_enabled = updates_enabled
        self.host_copies = updates_enabled if host_copies is None else host_copies
        self.repeated_count = repeated_count
        self.slot_count_per_rank = slot_count_per_rank
        self.ghost_slot_count = ghost_slot_count
        layer_id = self.single_layer_load_balancer_impl.get_layer_id()
        self.host_tensor_sharer = HostMoeTensorSharer(
            layer_id, expert_count,
            shared_mpi_comm) if self.host_copies else None
        self.host_copy_ranks = host_copy_ranks
        # Ring host copies: per expert, its home EP rank, and this node's copies by (expert, weight name).
        self.home_ranks: Optional[np.ndarray] = None
        self.ring_host_weights: Dict[Tuple[int, str], torch.Tensor] = {}
        # (local slot, weight name) -> the slot's weight tensor.
        self.weight_slots: Dict[Tuple[int, str], torch.Tensor] = {}
        self.register_weight_fns = []
        self.to_migrate_weight_fns = []
        # Local slots whose storage maps no memory until a restore (ghost extra slots), or None.
        self.ghost_slots: Optional[List[int]] = None
        self.initial_weight_assignments: Optional[List[int]] = None

        if self.updates_enabled:
            self.aux_stream = aux_stream if aux_stream is not None else torch.cuda.Stream(
            )
            self.event_dict = {
                key: torch.cuda.Event()
                for key in [EventType.Main, EventType.MoeBalancer]
            }
        else:
            self.aux_stream = None
            self.event_dict = None

        self.statistic_flag_tensor = None
        self.local_statistic_tensor = None
        self.func_called_count = {
            name: 0
            for name in [
                "start_wait_gpu_stage",
                "done_wait_gpu_stage",
                "start_set_cpu_stage",
                "done_set_cpu_stage",
                "update_local_statistic",
                "get_local_statistic_tensor",
                "update_statistic_with_gathered_statistic",
                "update_statistic_with_local_ids",
                "update_statistic_with_global_ids",
                "route",
            ]
        }

    def get_layer_idx(self):
        return self.single_layer_load_balancer_impl.get_layer_id()

    def get_old_rank_expert_ids(self):
        return self.single_layer_load_balancer_impl.get_old_rank_expert_ids()

    def get_load_expert_ids(self):
        assert self.host_copies, "should not call get_load_expert_ids without host copies"
        return self.host_tensor_sharer.own_expert_ids

    def get_repeat_count(self):
        return self.repeated_count

    def is_static_routing(self):
        return not self.updates_enabled

    def is_dynamic_routing(self):
        return self.updates_enabled

    def need_load_shared_weights(self):
        return self.host_copies

    def set_shared_memory_base_name(self, shared_memory_base_name):
        """
        Set the shared memory base name for the layer.

        Args:
            shared_memory_base_name: The base name for the shared memory
        """
        if self.host_copies:
            self.host_tensor_sharer.set_shared_memory_base_name(
                shared_memory_base_name)

    def _add_weight_slot(self, slot_id: int, name: str,
                         weight_slot: _tbr.MoeWeight):
        """
        Add a weight slot to the layer.

        Args:
            slot_id: The ID of the slot
            name: The name of the weight
            weight_slot: The weight object
        """
        self.single_layer_load_balancer_impl.add_single_weight_slot(
            slot_id, name, weight_slot)

    def register_weight_slot(self, local_slot_id: int, name: str,
                             t: torch.Tensor):
        """
        Register a weight slot to the layer.

        Args:
            local_slot_id: The ID of the slot at local rank
            name: The name of the weight
            t: The weight tensor
        """
        moe_weight = _tensor_to_weight(t)
        self._add_weight_slot(local_slot_id, name, moe_weight)
        self.weight_slots[(local_slot_id, name)] = t

    def _add_raw_host_weight_for_unmap(self,
                                       raw_weight_tensors: List[torch.Tensor]):
        """
        Add a raw weight (mmapped Tensor) to LoadBalancer for later `madvise` to save host memory.

        Args:
            raw_weight_tensors: A list of raw weight tensors
        """
        self.host_tensor_sharer.add_raw_host_weight_for_unmap(
            raw_weight_tensors)

    def _add_host_weight(self, expert_id: int, name: str,
                         host_weight: _tbr.MoeWeight):
        """
        Add a host weight to the layer.

        Args:
            expert_id: The ID of the expert
            name: The name of the weight
            host_weight: The host weight object
        """
        self.single_layer_load_balancer_impl.add_single_host_weight(
            expert_id, name, host_weight)

    def _add_host_weight_from_tensor(self, expert_id: int, name: str,
                                     t: torch.Tensor):
        """
        Add a host weight to the layer from a tensor.
        """
        host_weight = _tensor_to_weight(t)
        self._add_host_weight(expert_id, name, host_weight)

    def with_ghost_slots(self, eplb_assignments: List[int]) -> List[int]:
        """``eplb_assignments`` (the expert of each EPLB slot) with each rank's ghost slots appended.

        A ghost slot loads its rank's first expert, which only fills the
        weight tensor: routing starts with the slot empty.
        """
        if not self.ghost_slot_count:
            return eplb_assignments
        per_rank = np.asarray(eplb_assignments).reshape(
            -1, self.slot_count_per_rank - self.ghost_slot_count)
        ghosts = np.repeat(per_rank[:, :1], self.ghost_slot_count, axis=1)
        return np.concatenate([per_rank, ghosts], axis=1).ravel().tolist()

    def set_initial_weight_assignments(self,
                                       initial_weight_assignments: List[int]):
        """
        Set the initial weight assignments for the layer.

        Args:
            initial_weight_assignments: The expert of every slot; routing starts with the ghost slots empty (-1).
        """
        if self.ghost_slot_count:
            placement = np.array(initial_weight_assignments,
                                 dtype=np.int32).reshape(
                                     -1, self.slot_count_per_rank)
            placement[:, -self.ghost_slot_count:] = -1
            initial_weight_assignments = placement.ravel().tolist()
        self.single_layer_load_balancer_impl.set_initial_weight_assignments(
            initial_weight_assignments)
        self.initial_weight_assignments = list(initial_weight_assignments)
        if self.host_copy_ranks is not None and self.home_ranks is None:
            # Ring host copies: keep the experts whose home rank is in host_copy_ranks.
            placement = _without_replicas(
                np.asarray(initial_weight_assignments,
                           dtype=np.int32).reshape(-1,
                                                   self.slot_count_per_rank))
            ranks, _ = np.nonzero(placement >= 0)
            self.home_ranks = np.empty(self.expert_count, dtype=np.int64)
            self.home_ranks[placement[placement >= 0]] = ranks
            self.host_tensor_sharer.keep_experts(
                np.flatnonzero(np.isin(self.home_ranks,
                                       self.host_copy_ranks)).tolist())

    def add_to_migrate_weight_fn(self,
                                 fn: Callable,
                                 args: Tuple,
                                 kwargs: Dict = {}):
        self.to_migrate_weight_fns.append((fn, args, kwargs))

    def add_register_weight_fn(self,
                               fn: Callable,
                               args: Tuple,
                               kwargs: Dict = {}):
        """
        Add weight register function, this function doesn't run fn directly but run all functions after model.to("cuda")
        so this function can be called when model is not on GPU yet.
        """
        self.register_weight_fns.append((fn, args, kwargs))

    def make_tensor_host_accessible(self, wt: torch.Tensor):
        if self.ghost_slots is None:
            torch.ops.trtllm.migrate_to_host_accessible(wt)
        else:
            # Restores load these slots with GPU copies, so they need not be host accessible.
            torch.ops.trtllm.moe_ghost_slot_storage(wt, self.ghost_slots)
        torch.cuda.empty_cache()

    def register_weight_slots_after_to_cuda(self):
        """
        Register weights after model has been moved to cuda, should be invoked after model.to("cuda") and before finalize_model.
        """
        for fn, args, kwargs in self.to_migrate_weight_fns:
            fn(*args, **kwargs)

        self.to_migrate_weight_fns = []

        for fn, args, kwargs in self.register_weight_fns:
            fn(*args, **kwargs)

        self.register_weight_fns = []

    def py_finalize_model(self):
        """
        Finalize the model after all layers have been added.
        This must be called before starting any iterations.
        """
        if self.host_copies:
            self.host_tensor_sharer.finalize_host_tensor_sharing(
                self._add_host_weight_from_tensor if self.host_copy_ranks is
                None else lambda expert_id, name, t: self.ring_host_weights.
                update({(expert_id, name): t}))

    def start_wait_gpu_stage(self):
        """
        Start to wait for the GPU stage to complete.
        """
        assert self.func_called_count["start_wait_gpu_stage"] == 0
        self.func_called_count["start_wait_gpu_stage"] += 1
        if self.updates_enabled:
            if do_multi_stream():
                self.event_dict[EventType.Main].record()
                with torch.cuda.stream(self.aux_stream):
                    self.event_dict[EventType.Main].wait()
                    self.statistic_flag_tensor = torch.ops.trtllm.moe_load_balance_wait_gpu_stage(
                        self.single_layer_load_balancer_ptr)
                    self.event_dict[EventType.MoeBalancer].record()
            else:
                self.statistic_flag_tensor = torch.ops.trtllm.moe_load_balance_wait_gpu_stage(
                    self.single_layer_load_balancer_ptr)

    def done_wait_gpu_stage(self):
        """
        Done waiting for the GPU stage to complete.
        """
        assert self.func_called_count["start_wait_gpu_stage"] == 1
        assert self.func_called_count["done_wait_gpu_stage"] == 0
        self.func_called_count["done_wait_gpu_stage"] += 1
        if self.updates_enabled:
            if do_multi_stream():
                self.event_dict[EventType.MoeBalancer].wait()

    def start_set_cpu_stage(self):
        """
        Start to set the CPU stage.
        """
        assert self.func_called_count["done_wait_gpu_stage"] == 1
        assert self.func_called_count["start_set_cpu_stage"] == 0
        self.func_called_count["start_set_cpu_stage"] += 1
        if self.updates_enabled:
            if do_multi_stream():
                self.event_dict[EventType.Main].record()
                with torch.cuda.stream(self.aux_stream):
                    self.event_dict[EventType.Main].wait()
                    torch.ops.trtllm.moe_load_balance_set_cpu_stage(
                        self.single_layer_load_balancer_ptr)
                    self.event_dict[EventType.MoeBalancer].record()
            else:
                torch.ops.trtllm.moe_load_balance_set_cpu_stage(
                    self.single_layer_load_balancer_ptr)

    def done_set_cpu_stage(self):
        """
        Done setting the CPU stage.
        """
        assert self.func_called_count["start_set_cpu_stage"] == 1
        for name in self.func_called_count:
            self.func_called_count[name] = 0
        self.statistic_flag_tensor = None
        if self.updates_enabled:
            if do_multi_stream():
                self.event_dict[EventType.MoeBalancer].wait()

    def update_local_statistic(self, local_raw_expert_ids: torch.Tensor,
                               is_first_stage: bool, is_last_stage: bool):
        """
        Update local statistics of the expert IDs.

        Args:
            local_raw_expert_ids: The local raw expert IDs
            is_first_stage: Whether this is the first stage
            is_last_stage: Whether this is the last stage
        """
        assert self.func_called_count["done_wait_gpu_stage"] == 1
        assert self.func_called_count["update_statistic_with_global_ids"] == 0
        self.func_called_count["update_local_statistic"] += 1
        if self.updates_enabled:
            if self.local_statistic_tensor is None:
                self.local_statistic_tensor = torch.empty(
                    (self.expert_count, ),
                    dtype=torch.int32,
                    device=torch.device('cuda'))
            if do_multi_stream():
                self.event_dict[EventType.Main].record()
                with torch.cuda.stream(self.aux_stream):
                    self.event_dict[EventType.Main].wait()
                    torch.ops.trtllm.moe_hierarchical_statistic_local_device(
                        local_raw_expert_ids, self.local_statistic_tensor,
                        self.statistic_flag_tensor,
                        self.single_layer_load_balancer_ptr, is_first_stage,
                        is_last_stage)
            else:
                torch.ops.trtllm.moe_hierarchical_statistic_local_device(
                    local_raw_expert_ids, self.local_statistic_tensor,
                    self.statistic_flag_tensor,
                    self.single_layer_load_balancer_ptr, is_first_stage,
                    is_last_stage)

    def get_local_statistic_tensor(self) -> Optional[torch.Tensor]:
        """
        Get the local statistic tensor.
        Returns:
            The local statistic tensor if using statistic else None
        """
        assert self.func_called_count["update_local_statistic"] > 0
        self.func_called_count["get_local_statistic_tensor"] += 1
        if self.updates_enabled:
            if do_multi_stream():
                with torch.cuda.stream(self.aux_stream):
                    self.event_dict[EventType.MoeBalancer].record()
                self.event_dict[EventType.MoeBalancer].wait()
            return self.local_statistic_tensor
        return None

    def update_statistic_with_gathered_statistic(
            self, gathered_local_statistic_tensor: torch.Tensor):
        """
        Update statistics of the expert IDs, using gathered local statistic tensors.

        Args:
            gathered_local_statistic_tensor: gathered local statistics info, should have shape (world_size, self.expert_count)
        """
        assert self.func_called_count["get_local_statistic_tensor"] > 0
        assert self.func_called_count["update_statistic_with_local_ids"] == 0
        assert self.func_called_count["update_statistic_with_global_ids"] == 0
        self.func_called_count["update_statistic_with_gathered_statistic"] += 1

        def _update_statistic():
            global_statistic_info = torch.sum(gathered_local_statistic_tensor,
                                              dim=0,
                                              dtype=torch.int32)
            torch.ops.trtllm.moe_hierarchical_statistic_update(
                global_statistic_info, self.statistic_flag_tensor,
                self.single_layer_load_balancer_ptr)

        if self.updates_enabled:
            if do_multi_stream():
                self.event_dict[EventType.Main].record()
                with torch.cuda.stream(self.aux_stream):
                    self.event_dict[EventType.Main].wait()
                    _update_statistic()
            else:
                _update_statistic()

    def update_statistic_with_local_ids(self,
                                        local_raw_expert_ids: torch.Tensor,
                                        is_first_stage: bool,
                                        is_last_stage: bool,
                                        allreduce: Optional[AllReduce] = None):
        """
        Update statistics of the expert IDs, using local raw expert IDs.

        Args:
            local_raw_expert_ids: The local raw expert IDs
            is_first_stage: Whether this is the first stage
            is_last_stage: Whether this is the last stage
            allreduce: The allreduce object
        """
        assert self.func_called_count["done_wait_gpu_stage"] == 1
        assert self.func_called_count[
            "update_statistic_with_gathered_statistic"] == 0
        assert self.func_called_count["update_statistic_with_global_ids"] == 0
        self.func_called_count["update_statistic_with_local_ids"] += 1

        def _update_statistic():
            if is_last_stage:
                global_statistic_info = allreduce(self.local_statistic_tensor)
                torch.ops.trtllm.moe_hierarchical_statistic_update(
                    global_statistic_info, self.statistic_flag_tensor,
                    self.single_layer_load_balancer_ptr)

        if self.updates_enabled:
            self.update_local_statistic(local_raw_expert_ids, is_first_stage,
                                        is_last_stage)
            if do_multi_stream():
                with torch.cuda.stream(self.aux_stream):
                    _update_statistic()
            else:
                _update_statistic()

    def update_statistic_with_global_ids(self,
                                         gathered_raw_expert_ids: torch.Tensor,
                                         is_first_stage: bool,
                                         is_last_stage: bool):
        """
        Update statistics of the expert IDs, using gathered raw expert IDs from all ranks.

        Args:
            gathered_raw_expert_ids: The gathered raw expert IDs from all ranks
            is_first_stage: Whether this is the first stage
            is_last_stage: Whether this is the last stage
        """
        assert self.func_called_count["done_wait_gpu_stage"] == 1
        assert self.func_called_count[
            "update_statistic_with_gathered_statistic"] == 0
        assert self.func_called_count["update_statistic_with_local_ids"] == 0
        self.func_called_count["update_statistic_with_global_ids"] += 1
        if self.updates_enabled:
            if do_multi_stream():
                self.event_dict[EventType.Main].record()
                with torch.cuda.stream(self.aux_stream):
                    self.event_dict[EventType.Main].wait()
                    torch.ops.trtllm.moe_load_balance_statistic(
                        gathered_raw_expert_ids, self.statistic_flag_tensor,
                        self.single_layer_load_balancer_ptr, is_first_stage,
                        is_last_stage)
            else:
                torch.ops.trtllm.moe_load_balance_statistic(
                    gathered_raw_expert_ids, self.statistic_flag_tensor,
                    self.single_layer_load_balancer_ptr, is_first_stage,
                    is_last_stage)

    def route(self,
              token_selected_experts: torch.Tensor,
              offset_by_ep_rank: bool = False) -> torch.Tensor:
        """
        Route the tokens to experts.

        Args:
            token_selected_experts: The experts selected by each token
            offset_by_ep_rank: Whether to offset the round robin position by ep_rank

        Returns:
            A tensor of routed slot IDs
        """
        if self.is_dynamic_routing():
            assert self.func_called_count["done_wait_gpu_stage"] == 1
            self.func_called_count["route"] += 1
        return torch.ops.trtllm.moe_load_balance_routing(
            token_selected_experts, offset_by_ep_rank,
            self.single_layer_load_balancer_ptr)

    def one_slot_table(self) -> Optional[torch.Tensor]:
        """The device expert -> slot table [num_experts] while every expert has exactly one slot, else None.

        ``route`` then reduces to this lookup. Its address is fixed and placement updates rewrite it in place, so
        captured CUDA graphs read the current placement. Like ``route``, a call that returns a table commits the
        layer to one slot per expert for every later placement.
        """
        table = torch.ops.trtllm.moe_load_balance_one_slot_table(
            self.single_layer_load_balancer_ptr)
        return table if table.numel() > 0 else None

    def py_pre_shutdown_cleanup(self):
        """
        Clean up the resources before C++ shutdown and barrier
        """
        if self.host_copies:
            # Views of the shared memory, which closes only without them.
            self.ring_host_weights.clear()
            self.host_tensor_sharer.pre_shutdown_cleanup()

    def py_post_shutdown_cleanup(self):
        """
         Clean up the resources after C++ shutdown and barrier
         """
        if self.host_copies:
            self.host_tensor_sharer.post_shutdown_cleanup()


# Global variable to store the current active MoeLoadBalancer instance
_current_moe_load_balancer = threading.local()


def restore_lost_experts(placement: np.ndarray, lost: List[int],
                         dead_ranks: Iterable[int]) -> List[int]:
    """Place each ``lost`` expert into a slot of a live rank in ``placement`` (``[ep_size, slots_per_rank]``
    expert ids, -1 = no expert), in place.

    Empty (ghost) slots of live ranks are taken first; otherwise the slot of a
    replica of the expert with the most replicas (so no expert loses
    its last copy). Ties go to the rank that has taken the fewest restored
    experts so far, then the lowest rank and slot: a minimal change that
    spreads the loading across ranks and is the same on every rank.

    Returns:
        The lost experts that found no slot.
    """
    usable = np.ones(placement.shape, dtype=bool)
    usable[list(dead_ranks)] = False
    counts = np.bincount(placement[placement >= 0],
                         minlength=max(lost) + 1 if lost else 0)
    restored_per_rank = np.zeros(placement.shape[0], dtype=np.int64)
    unplaced = []
    for expert in lost:
        ranks, slots = np.nonzero(usable)
        held = placement[ranks, slots]
        replicas = np.where(held >= 0, counts[np.maximum(held, 0)], 0)
        free = (held < 0) | (replicas >= 2)
        if not free.any():
            unplaced.append(expert)
            continue
        ranks, slots, held, replicas = (a[free]
                                        for a in (ranks, slots, held, replicas))
        best = np.lexsort(
            (slots, ranks, restored_per_rank[ranks], -replicas, held >= 0))[0]
        rank, slot = ranks[best], slots[best]
        if held[best] >= 0:
            counts[held[best]] -= 1
        placement[rank, slot] = expert
        counts[expert] = 1
        restored_per_rank[rank] += 1
    return unplaced


def ring_copy_nodes(hosts: List[str]) -> List[Tuple[int, ...]]:
    """Per rank (``hosts``: its host), the ranks of the node keeping its home experts' copies: the node before
    its own in first-rank order, the last node for the first (a single node keeps its own)."""
    nodes = list(dict.fromkeys(hosts))
    copy_nodes = {
        host: tuple(rank for rank, h in enumerate(hosts) if h == previous)
        for host, previous in zip(nodes, nodes[-1:] + nodes[:-1])
    }
    return [copy_nodes[host] for host in hosts]


class MoeLoadBalancer:
    """
    A class representing a Mixture of Experts (MoE) load balancer.
    This class can be used as a context manager to manage the lifecycle of a MoeLoadBalancer.
    """

    def __init__(self,
                 ep_rank: int,
                 ep_size: int,
                 layer_updates_per_iter: int,
                 shared_memory_base_name: Optional[str] = None):
        """
        Initialize a MoeLoadBalancer instance.

        Args:
            ep_rank: The rank of the current process in expert parallelism
            ep_size: The total number of processes in expert parallelism
            layer_updates_per_iter: The number of layers to update per iteration
            shared_memory_base_name: Shared memory base name, will use 'moe_shared' if None
        """
        self.is_shutdown = True
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.layer_updates_per_iter = layer_updates_per_iter
        self.load_balancer_impl = _tbr.MoeLoadBalancer(ep_rank, ep_size,
                                                       layer_updates_per_iter)
        self._previous_balancer = None
        self.single_layer_load_balancers = []
        self.shared_memory_base_name = shared_memory_base_name or os.getenv(
            'TRTLLM_EPLB_SHM_NAME', 'moe_shared')
        self._setup_mpi_comm()
        self.is_shutdown = False

        self.iter_id = 0
        self.in_iter = False

        self.enable_statistic = False
        self.enable_update_weights = False

        self.next_layer_repeated_count = None
        # With online EPLB and WideEP peer-loss recovery, the most ranks one admitted failure takes: the
        # survivors' slots, ghost slots included, must hold every expert.
        self._largest_loss = self._gather_largest_loss() if (
            self.is_dynamic_routing()
            and peer_loss_recovery_enabled()) else None
        # Ring host copies, with a static placement: per EP rank, ring_copy_nodes. Online EPLB keeps every
        # expert's host copy on every node.
        self.copy_nodes = self._gather_copy_nodes() if (
            self.is_static_routing() and peer_loss_recovery_enabled()) else None

    def __del__(self):
        if not self.is_shutdown:
            self.shutdown()

    def is_static_routing(self):
        # if we don't update, then it is statistic routing.
        return self.layer_updates_per_iter == 0

    def is_dynamic_routing(self):
        return not self.is_static_routing()

    def _setup_mpi_comm(self):
        global_mpi_comm = tensorrt_llm.mpi_comm()
        shared_mpi_comm = global_mpi_comm.Split_type(
            split_type=MPI.COMM_TYPE_SHARED)
        shared_size = shared_mpi_comm.Get_size()
        local_size = tensorrt_llm.local_mpi_size()
        assert shared_size == local_size, \
            f"Interesting, shared size {shared_size} is not same as local size {local_size}"
        self.shared_mpi_comm = shared_mpi_comm

    def _gather_largest_loss(self) -> int:
        """The most ranks one failure domain (FaultToleranceConfig.failure_domain) holds; collective."""
        from tensorrt_llm._torch.pyexecutor.peer_loss_recovery import \
            failure_domains
        hosts = tensorrt_llm.mpi_comm().allgather(socket.gethostname())
        return max(map(
            len, failure_domains(hosts,
                                 fault_tolerance_config().failure_domain)),
                   default=0)

    def _gather_copy_nodes(self) -> List[Tuple[int, ...]]:
        return ring_copy_nodes(tensorrt_llm.mpi_comm().allgather(
            socket.gethostname()))

    def set_use_gpu_memcpy(self, use_gpu_memcpy: bool):
        self.load_balancer_impl.set_use_gpu_memcpy(use_gpu_memcpy)

    def set_repeated_for_next_layer(self, repeated_count: int):
        """
        Set repeat count for next layer.

        Args:
            repeated_count: The repeat count for next layer
        """
        assert repeated_count > 0, "repeat count must be greater than 0"
        self.next_layer_repeated_count = repeated_count

    def add_layer(
        self,
        expert_count: int,
        top_k: int,
        slot_count_per_rank: int,
        aux_stream: Optional[torch.cuda.Stream] = None
    ) -> SingleLayerMoeLoadBalancer:
        """
        Add a new layer to the load balancer.

        Args:
            expert_count: The number of experts in the layer
            top_k: The number of experts each token selects
            slot_count_per_rank: The number of EPLB slots per rank. With online EPLB, WideEP peer-loss recovery
                appends the ghost slots the survivors of a failure need to hold every expert.
            aux_stream: The auxiliary stream for overlapping

        Returns:
            A SingleLayerMoeLoadBalancer instance for the new layer
        """
        ghost_slot_count = 0
        if self._largest_loss is not None:
            survivors = self.ep_size - self._largest_loss
            ghost_slot_count = max(
                0, -(-expert_count // survivors) - slot_count_per_rank)
        single_layer_load_balancer_impl = self.load_balancer_impl.add_layer(
            expert_count, top_k, slot_count_per_rank + ghost_slot_count)
        updates_enabled = not self.is_static_routing()
        repeat_count = 1
        if self.next_layer_repeated_count is not None:
            repeat_count = self.next_layer_repeated_count
            self.next_layer_repeated_count = None
        single_layer_load_balancer = SingleLayerMoeLoadBalancer(
            single_layer_load_balancer_impl,
            self.shared_mpi_comm,
            expert_count,
            updates_enabled=updates_enabled,
            repeated_count=repeat_count,
            aux_stream=aux_stream,
            host_copies=updates_enabled or peer_loss_recovery_enabled(),
            slot_count_per_rank=slot_count_per_rank + ghost_slot_count,
            ghost_slot_count=ghost_slot_count,
            host_copy_ranks=None if self.copy_nodes is None else [
                rank for rank, node in enumerate(self.copy_nodes)
                if self.ep_rank in node
            ])
        single_layer_load_balancer.set_shared_memory_base_name(
            self.shared_memory_base_name)
        self.single_layer_load_balancers.append(single_layer_load_balancer)
        return single_layer_load_balancer

    def register_weight_slots_after_to_cuda(self):
        """
        Register weights after model has been moved to cuda, should be invoked after model.to("cuda") and before finalize_model.
        """
        mode = self.extra_slots_mode()
        if mode in ("ghost", "replicas+ghost"):
            # Ghost slots are not host accessible: restores and online EPLB's updates copy on the GPU.
            self.set_use_gpu_memcpy(True)
            for layer in self.single_layer_load_balancers:
                placement = np.asarray(layer.initial_weight_assignments,
                                       dtype=np.int32).reshape(
                                           self.ep_size, -1)
                if mode == "ghost":
                    placement = _without_replicas(placement)
                layer.ghost_slots = np.flatnonzero(
                    placement[self.ep_rank] < 0).tolist()
        for layer in self.single_layer_load_balancers:
            layer.register_weight_slots_after_to_cuda()

    def extra_slots_mode(self) -> Optional[str]:
        """What WideEP peer-loss recovery keeps in the EPLB slots beyond one per expert; None without it.

        With a static placement, the fault-tolerance config's ``extra_slots``:
        ``empty`` (not routed to while healthy; a restore fills them) or
        ``ghost`` (also without device memory until a restore maps it). Online
        EPLB keeps ``replicas`` in them (a loss reroutes to them or restores
        into them); where its slots cannot hold every expert after a failure,
        ``replicas+empty`` or ``replicas+ghost``: the ghost slots appended to
        each rank are ``extra_slots`` slots too.
        """
        if not peer_loss_recovery_enabled():
            return None
        config = fault_tolerance_config()
        if self.is_static_routing():
            return config.extra_slots
        if any(layer.ghost_slot_count
               for layer in self.single_layer_load_balancers):
            return f"replicas+{config.extra_slots}"
        return "replicas"

    def finalize_model(self):
        """
        Finalize the model after all layers have been added.
        This must be called before starting any iterations.
        """

        # All shared memory create before this barrier.
        self.shared_mpi_comm.barrier()
        # All shared memory mapping after this barrier.
        for single_layer_load_balancer in self.single_layer_load_balancers:
            single_layer_load_balancer.py_finalize_model()
        self.load_balancer_impl.finalize_model()
        mode = self.extra_slots_mode()
        if mode in ("empty", "ghost"):
            # Before warmup, so autotuning and CUDA graph capture see the
            # healthy routing.
            emptied = self.empty_replica_slots()
            if self.ep_rank == 0:
                logger.info(
                    f"WideEP peer-loss recovery keeps {emptied} EPLB slots empty for restores"
                )
        elif mode not in (None, "replicas") and self.ep_rank == 0:
            layer = self.single_layer_load_balancers[0]
            logger.info(
                f"WideEP peer-loss recovery appends {layer.ghost_slot_count} {mode.split('+')[1]} slots per "
                f"rank to online EPLB's {layer.slot_count_per_rank - layer.ghost_slot_count}"
            )
        torch.cuda.empty_cache()

    def set_warm_up_iter_count(self, iter_count: int):
        """
        Set the number of warm-up iterations.

        Args:
            iter_count: The number of warm-up iterations
        """
        self.load_balancer_impl.set_warm_up_iter_count(iter_count)

    def set_iter_info(self, enable_statistic: Optional[bool],
                      enable_update_weights: Optional[bool]):
        if enable_statistic is not None:
            self.enable_statistic = enable_statistic
        if enable_update_weights is not None:
            self.enable_update_weights = enable_update_weights

    def reconfigure_mask_only(self, dead_ranks: list[int]) -> None:
        """
        Reconfigure EPLB routing so slots on dead EP ranks are unreachable.

        This validates that every expert still has at least one surviving
        replica. The caller must separately gate degraded-mode capacity/HBM
        headroom before invoking this method.
        """
        if self.in_iter:
            raise RuntimeError(
                "Cannot reconfigure EPLB mask while an iteration is active")
        self.load_balancer_impl.reconfigure_mask_only(list(dead_ranks))

    def covers_loss_of(self, dead_ranks: Iterable[int]) -> bool:
        """Whether serving can continue without all of ``dead_ranks`` (see ``recovery_placements``)."""
        try:
            self.recovery_placements(dead_ranks)
        except ValueError:
            return False
        return True

    def quiesce(self, timeout_s: float) -> bool:
        """Wait until started iterations and EPLB weight updates are done, so the placement stops changing.

        Returns False if that takes longer than ``timeout_s``. After an interrupted iteration, call
        only once its device work is done.
        """
        if self.in_iter:
            raise RuntimeError("Cannot quiesce while an iteration is active")
        return self.load_balancer_impl.quiesce(int(timeout_s * 1000))

    def reset_update_schedule(self) -> None:
        """Restart the EPLB layer update rotation and clear the load factors (only while quiesced)."""
        self.load_balancer_impl.reset_update_schedule()

    def current_placements(self) -> List[np.ndarray]:
        """Per layer, the ``[ep_size, slots_per_rank]`` expert ids routing uses now."""
        return [
            np.asarray(layer.get_old_rank_expert_ids(), dtype=np.int32)
            for layer in self.single_layer_load_balancers
        ]

    def recovery_placements(
            self,
            dead_ranks: Iterable[int],
            base: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        """Per layer, the ``[ep_size, slots_per_rank]`` placement to serve with after losing ``dead_ranks``.

        The current placement (or ``base``, per layer, the placement the
        survivors agreed on) with the slots of ``dead_ranks`` cleared. In a
        layer with host weights, each expert left without a replica takes over
        a slot of a replicated expert (``restore_lost_experts``). Every rank
        computes the same placements from the same base.

        Raises:
            ValueError: A rank is out of range, or an expert would be left
                without a replica.
        """
        dead = sorted(set(dead_ranks))
        if any(rank < 0 or rank >= self.ep_size for rank in dead):
            raise ValueError(
                f"Dead ranks {dead} out of range for EP size {self.ep_size}")
        if base is not None and len(base) != len(
                self.single_layer_load_balancers):
            raise ValueError(f"Base placement has {len(base)} layers, expected "
                             f"{len(self.single_layer_load_balancers)}")
        placements, missing = [], {}
        for i, layer in enumerate(self.single_layer_load_balancers):
            if base is None:
                placement = self._masked_placement(layer, dead)
            else:
                placement = np.array(base[i], dtype=np.int32)
                placement[dead] = -1
            lost = sorted(
                set(range(layer.expert_count)) -
                set(placement[placement >= 0].tolist()))
            if lost and layer.host_copies:
                # Ring host copies: an expert whose copy node is lost too has no copy left.
                uncopied = [
                    expert for expert in lost if self.copy_nodes is not None
                    and not self._copy_ranks(layer, expert, dead)
                ]
                lost = uncopied + restore_lost_experts(
                    placement, [e for e in lost if e not in uncopied], dead)
            if lost:
                missing[layer.get_layer_idx()] = lost
            placements.append(placement)
        if missing:
            raise ValueError(
                f"No surviving replica for experts (by MoE layer): {missing}")
        return placements

    def _copy_ranks(self, layer: "SingleLayerMoeLoadBalancer", expert: int,
                    dead: Iterable[int]) -> List[int]:
        """The live ranks of the node that keeps ``expert``'s ring host copy."""
        return [
            rank for rank in self.copy_nodes[layer.home_ranks[expert]]
            if rank not in dead
        ]

    @staticmethod
    def placement_digest(dead_ranks: Iterable[int],
                         placements: List[np.ndarray],
                         copy_plan: Optional[np.ndarray] = None) -> str:
        """A digest of ``placements``, ``dead_ranks`` and ``copy_plan``: ranks that agree on it route and restore alike."""
        digest = hashlib.sha256(
            np.asarray(sorted(set(dead_ranks)), dtype=np.int64).tobytes())
        for placement in placements:
            digest.update(np.asarray(placement.shape, dtype=np.int64).tobytes())
            digest.update(placement.tobytes())
        if copy_plan is not None:
            digest.update(copy_plan.tobytes())
        return digest.hexdigest()

    def restore_copy_plan(self, dead_ranks: Iterable[int],
                          placements: List[np.ndarray],
                          workspace_bytes: int) -> np.ndarray:
        """Ring host copies: rows (chunk, layer, target, slot, expert, weight by sorted name, source, workspace
        offset, bytes) that load the slots ``placements`` changes, in chunk order; the same on every rank.

        Sources take turns over the live ranks of the expert's copy node. Per target, a chunk fills at most half
        the workspace (``workspace_bytes`` per rank), alternating halves.
        ValueError if a weight does not fit or ``placements`` loads a slot that still routes an expert.
        """
        dead = sorted(set(dead_ranks))
        half = workspace_bytes // 2
        chunk = np.zeros(self.ep_size, dtype=np.int64)
        used = np.zeros(self.ep_size, dtype=np.int64)
        turns: Dict[Tuple[int, ...], int] = {}
        rows = []
        for layer_idx, (layer, placement) in enumerate(
                zip(self.single_layer_load_balancers, placements)):
            sizes = [
                t.nbytes for (slot, _), t in sorted(layer.weight_slots.items())
                if slot == 0
            ]
            if max(sizes) > half:
                raise ValueError(
                    f"A {max(sizes)}-byte expert weight of MoE layer {layer.get_layer_idx()} does not fit in "
                    f"half the {workspace_bytes}-byte NVLinkOneSided workspace that ring host copies stage through"
                )
            masked = self._masked_placement(layer, dead)
            loaded = (placement != masked) & (placement >= 0)
            if (masked[loaded] >= 0).any():
                # A re-plan reverts only ghost backing, so a routed slot staged over would keep stale weights.
                raise ValueError(
                    f"Ring host copies only restore into empty slots; MoE layer {layer.get_layer_idx()} "
                    "would overwrite a routed one")
            for target, slot in zip(*np.nonzero(loaded)):
                expert = int(placement[target, slot])
                live = self._copy_ranks(layer, expert, dead)
                node = self.copy_nodes[layer.home_ranks[expert]]
                turns[node] = turns.get(node, -1) + 1
                for weight, size in enumerate(sizes):
                    if used[target] + size > half:
                        chunk[target] += 1
                        used[target] = 0
                    rows.append((chunk[target], layer_idx, target, slot, expert,
                                 weight, live[turns[node] % len(live)],
                                 chunk[target] % 2 * half + used[target], size))
                    used[target] += HostMoeTensorSharer.align_size(size)
        plan = np.array(rows, dtype=np.int64).reshape(-1, 9)
        return plan[np.argsort(plan[:, 0], kind="stable")]

    def stage_restored_slots(self, copy_plan: np.ndarray,
                             workspace: torch.Tensor,
                             barrier: Callable[[], None]) -> None:
        """Copy the experts of ``copy_plan`` into their slots through ``workspace`` ([ep_size, bytes] uint8, idle).

        Per chunk: write the weights this rank sources into the targets' workspaces, ``barrier`` (all live
        ranks), copy those it receives into its slots. Alternating halves keep writes off the last chunk's reads.
        """
        layers = self.single_layer_load_balancers
        names = [
            sorted({name
                    for _, name in layer.weight_slots}) for layer in layers
        ]
        stream = torch.cuda.current_stream()
        for chunk in range(int(copy_plan[:, 0].max(initial=-1)) + 1):
            rows = copy_plan[copy_plan[:, 0] == chunk].tolist()
            for _, i, target, _, expert, weight, source, offset, size in rows:
                if source == self.ep_rank:
                    host = layers[i].ring_host_weights[(expert,
                                                        names[i][weight])]
                    workspace[target, offset:offset + size].copy_(
                        host.reshape(-1).view(torch.uint8))
            stream.synchronize()
            barrier()
            for _, i, target, slot, _, weight, _, offset, size in rows:
                if target == self.ep_rank:
                    dst = layers[i].weight_slots[(slot, names[i][weight])]
                    dst.reshape(-1).view(torch.uint8).copy_(
                        workspace[target, offset:offset + size])
        stream.synchronize()

    def restore_backing_bytes(self, placements: List[np.ndarray]) -> int:
        """Device memory that installing ``placements`` maps behind ghost slots (0 without them)."""
        return self.load_balancer_impl.restore_backing_bytes(
            [placement.tolist() for placement in placements])

    def back_restored_slots(self, placements: List[np.ndarray]) -> None:
        """Map device memory behind the ghost slots that installing ``placements`` loads.

        All or nothing: raises if the memory cannot be allocated. Only while no
        iteration runs.
        """
        self.load_balancer_impl.back_restored_slots(
            [placement.tolist() for placement in placements])

    def release_uninstalled_backing(self) -> int:
        """Unmap what ``back_restored_slots`` mapped for placements never installed; the bytes freed."""
        return self.load_balancer_impl.release_uninstalled_backing()

    def install_recovery_placement(self, dead_ranks: Iterable[int],
                                   placements: List[np.ndarray]) -> None:
        """Serve with ``placements`` (from ``recovery_placements``) without ``dead_ranks``.

        Slots that change are loaded from the host weights before routing
        switches to them. Only while no iteration runs.
        """
        if self.in_iter:
            raise RuntimeError(
                "Cannot install a recovery placement while an iteration is active"
            )
        dead = sorted(set(dead_ranks))
        restored = sum(
            int((placement != self._masked_placement(layer, dead)).sum())
            for layer, placement in zip(self.single_layer_load_balancers,
                                        placements))
        if not restored:
            self.load_balancer_impl.reconfigure_mask_only(dead)
            return
        start = time.monotonic()
        # Ring host copies: stage_restored_slots loaded the slots.
        self.load_balancer_impl.restore_placement(
            dead, [placement.tolist() for placement in placements],
            slots_loaded=self.copy_nodes is not None)
        logger.info(f"Restored {restored} expert slots without ranks {dead} in "
                    f"{(time.monotonic() - start) * 1e3:.1f} ms")

    def empty_replica_slots(self) -> int:
        """Stop routing to every replica beyond an expert's first, leaving those slots empty for restores.

        Routing then uses one slot per expert, as a placement without spare
        slots does; the slot buffers keep their addresses (CUDA graphs
        captured them; ghost slots map no memory). Which replica stays:
        ``_without_replicas``. Needs host copies, to fill the slots later.
        Returns the number of slots emptied.
        """
        placements = [
            _without_replicas(
                np.asarray(layer.get_old_rank_expert_ids(), dtype=np.int32))
            for layer in self.single_layer_load_balancers
        ]
        emptied = sum(int((p < 0).sum()) for p in placements)
        if emptied:
            # Emptying slots loads none.
            self.load_balancer_impl.restore_placement(
                [], [placement.tolist() for placement in placements],
                slots_loaded=True)
        return emptied

    @staticmethod
    def _masked_placement(layer: "SingleLayerMoeLoadBalancer",
                          dead_ranks: List[int]) -> np.ndarray:
        """``[ep_size, slots_per_rank]`` expert ids, -1 on ``dead_ranks``."""
        placement = np.asarray(layer.get_old_rank_expert_ids(), dtype=np.int32)
        placement[dead_ranks] = -1
        return placement

    def start_iter(self):
        """
        Start a new iteration.
        """
        assert self.in_iter == False, "already in forward"
        self.in_iter = True
        self.load_balancer_impl.start_iter(self.iter_id, self.enable_statistic,
                                           self.enable_update_weights)

    def end_iter(self, interrupted: bool = False):
        """
        End the current iteration.

        Args:
            interrupted: The forward raised. The layers it did not reach never set their CPU stage;
                ``quiesce`` releases them once the iteration's device work is done.
        """
        assert self.in_iter, "not in forward, cannot end_iter"
        self.load_balancer_impl.end_iter(self.iter_id, interrupted=interrupted)
        if interrupted:
            # The layer the forward raised in stopped between its stage calls.
            for layer in self.single_layer_load_balancers:
                layer.func_called_count = dict.fromkeys(layer.func_called_count,
                                                        0)
                layer.statistic_flag_tensor = None
        self.in_iter = False
        self.iter_id += 1

    def shutdown(self):
        """
        Shutdown the load balancer and release resources.
        """
        for single_layer_load_balancer in self.single_layer_load_balancers:
            single_layer_load_balancer.py_pre_shutdown_cleanup()
        self.load_balancer_impl.shutdown()
        # use this sync to make sure all the shm resources can be cleaned up
        self._shared_barrier()
        for single_layer_load_balancer in self.single_layer_load_balancers:
            single_layer_load_balancer.py_post_shutdown_cleanup()
        self._shared_barrier()
        self.is_shutdown = True

    def _shared_barrier(self) -> None:
        """Wait for the node's other ranks.

        With peer-loss recovery, a rank that died no longer uses the shared
        memory, so its absence is tolerated.
        """
        try:
            self.shared_mpi_comm.barrier()
        except MPI.Exception as error:
            if not (peer_loss_recovery_enabled()
                    and is_mpi_peer_failure(error)):
                raise

    def __repr__(self):
        """
        Return a string representation of the load balancer.

        Returns:
            A string representation of the load balancer
        """
        return f"MoeLoadBalancer(ep_rank={self.ep_rank}, ep_size={self.ep_size}, layer_updates_per_iter={self.layer_updates_per_iter})"

    def __enter__(self):
        """
        Enter the context manager.

        Returns:
            The MoeLoadBalancer instance
        """
        # Save the previous load balancer (if any)
        try:
            self._previous_balancer = getattr(_current_moe_load_balancer,
                                              'instance', None)
        except AttributeError:
            self._previous_balancer = None

        # Set the current instance as the active load balancer
        _current_moe_load_balancer.instance = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager.

        Args:
            exc_type: The exception type
            exc_val: The exception value
            exc_tb: The exception traceback

        Returns:
            False to not suppress exceptions
        """
        # Restore the previous load balancer
        if self._previous_balancer is not None:
            _current_moe_load_balancer.instance = self._previous_balancer
        else:
            delattr(_current_moe_load_balancer, 'instance')

        # Do not suppress exceptions
        return False


moe_model_arch_list = [
    'DeepseekV3ForCausalLM',
    'DeepseekV32ForCausalLM',
    'DeepseekV4ForCausalLM',
    'GlmMoeDsaForCausalLM',
    'GptOssForCausalLM',
    'KimiK25ForConditionalGeneration',
    'MixtralForCausalLM',
    'Llama4ForConditionalGeneration',
    'NemotronHForCausalLM',
    'NemotronH_Omni_Reasoning_V3',
    'Qwen2MoeForCausalLM',
    'Qwen3MoeForCausalLM',
    'Qwen3_5MoeForCausalLM',
    'Qwen4ExpForCausalLM',
    'Qwen4ExpForConditionalGeneration',
    # Composite MoE VLMs are routed through their top-level
    # ConditionalGeneration arch, so EPLB setup must recognize those here (the
    # text-only arches above never match for the VLM wrapper). The inner MoE LM
    # (Qwen3MoeForCausalLM / Qwen3_5MoeForCausalLM) inherits the setup() state
    # via the wrapper's deepcopy of model_config. Same pattern as
    # Llama4ForConditionalGeneration.
    'Qwen3VLMoeForConditionalGeneration',
    'Qwen3_5MoeForConditionalGeneration',
]


def maybe_create_moe_load_balancer(
    model_config, mapping: Optional[Mapping]
) -> AbstractContextManager[MoeLoadBalancer | None]:
    ep_rank = model_config.mapping.moe_ep_rank
    ep_size = model_config.mapping.moe_ep_size
    model_arch = model_config.pretrained_config.architectures[0]
    using_ep = mapping and mapping.moe_ep_size > 1
    in_supported_model_arch = model_arch in moe_model_arch_list
    using_smart_router = mapping and mapping.moe_cluster_size > 1
    moe_load_balancer: AbstractContextManager[MoeLoadBalancer
                                              | None] = nullcontext()
    if in_supported_model_arch and using_ep and not using_smart_router and model_config.moe_load_balancer is not None:
        model_config.moe_load_balancer.setup(ep_rank=ep_rank, ep_size=ep_size)
        if model_config.moe_load_balancer.layer_updates_per_iter > 0:
            # TODO: remove this when supported.
            # cpu_arch = platform.machine().lower()
            # assert cpu_arch == 'aarch64', "online load balancer only support aarch64, e.g. GB200 now, x86 coming soon."
            pass

        moe_load_balancer = MoeLoadBalancer(
            ep_rank=ep_rank,
            ep_size=ep_size,
            layer_updates_per_iter=model_config.moe_load_balancer.
            layer_updates_per_iter)
        logger.info(
            f"Created MoE LoadBalancer, layer_updates_per_iter={model_config.moe_load_balancer.layer_updates_per_iter}..."
        )
    return moe_load_balancer


class MoeLoadBalancerIterContext:

    def __init__(self,
                 moe_load_balancer: Optional[MoeLoadBalancer],
                 enable_statistic: Optional[bool] = None,
                 enable_updates: Optional[bool] = None):
        self.moe_load_balancer = moe_load_balancer
        self.enable_statistic = enable_statistic
        self.enable_updates = enable_updates

    def __enter__(self):
        """
        Enter the context manager.

        Returns:
            The MoeLoadBalancerIterContext instance
        """
        if self.moe_load_balancer is not None and not self.moe_load_balancer.is_static_routing(
        ):
            self.moe_load_balancer.set_iter_info(self.enable_statistic,
                                                 self.enable_updates)
            self.moe_load_balancer.start_iter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager.

        Args:
            exc_type: The exception type
            exc_val: The exception value
            exc_tb: The exception traceback

        Returns:
            False to not suppress exceptions
        """
        if self.moe_load_balancer is not None and not self.moe_load_balancer.is_static_routing(
        ):
            self.moe_load_balancer.end_iter(interrupted=exc_type is not None)
        return False


def get_moe_load_balancer() -> Optional[MoeLoadBalancer]:
    """
    Get the current active MoeLoadBalancer instance.

    Returns:
        The current active MoeLoadBalancer instance, or None if not in a MoeLoadBalancer context
    """
    try:
        return getattr(_current_moe_load_balancer, 'instance', None)
    except AttributeError:
        return None


def moe_load_balancer_set_repeated_for_next_layer(repeat_count: int):
    """
    Set repeated count for next Single Layer created.

    Args:
        repeat_count: repeated count
    """
    load_balancer = get_moe_load_balancer()
    if load_balancer is not None:
        load_balancer.set_repeated_for_next_layer(repeat_count)


def moe_load_balancer_add_single_layer(
    expert_count: int,
    top_k: int,
    slot_count_per_rank: int,
    aux_stream: Optional[torch.cuda.Stream] = None
) -> Optional[SingleLayerMoeLoadBalancer]:
    """
    Add a new layer to the current active MoeLoadBalancer.

    Args:
        expert_count: The number of experts in the layer
        top_k: The number of experts each token selects
        slot_count_per_rank: The number of slots per rank
        aux_stream: The auxiliary stream for overlapping

    Returns:
        A SingleLayerMoeLoadBalancer instance for the new layer, or None if not in a MoeLoadBalancer context
    """
    load_balancer = get_moe_load_balancer()
    if load_balancer is not None:
        return load_balancer.add_layer(expert_count, top_k, slot_count_per_rank,
                                       aux_stream)
    return None

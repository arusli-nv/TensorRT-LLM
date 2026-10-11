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

# ruff: noqa: E501


"""
NVLINK One-Sided AllToAll Communication Strategy

This module implements the NVLINK one-sided comm AllToAll throughput communication method for MoE.

NVLINK One-Sided supports post-quant dispatch.
"""

import os
import re
import sys
from contextlib import nullcontext
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import pynvml
import torch

from tensorrt_llm._mnnvl_utils import CftMnnvlMemory, MnnvlCheckpointCommunicator, MnnvlMemory
from tensorrt_llm._torch.alltoall_watchdog import (
    DEFAULT_ALLTOALL_WATCHDOG_POLL_INTERVAL_S,
    DEFAULT_ALLTOALL_WATCHDOG_TIMEOUT_S,
    ActiveRankMaskSnapshot,
    AlltoAllWatchdog,
    AlltoAllWatchdogCoordinator,
    AlltoAllWatchdogTimeout,
    EPGroupHealthLike,
    reject_rank_mask_cuda_graph_capture,
)
from tensorrt_llm._torch.mnnvl_alltoall_workspace import _MnnvlAlltoAllWorkspaceLifecycle
from tensorrt_llm.bindings import internal as _tllm_internal
from tensorrt_llm.logger import logger as tllm_logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.math_utils import pad_up

from .base import Communication

if TYPE_CHECKING:
    from .execution_state import CapturedTensorSpan, ExecutionState

_CFT_DEFAULT_MAX_BATCH_FOR_DISPATCH = 128
_CFT_MAX_BATCH_FOR_DISPATCH_ENV = "TRTLLM_MOE_A2A_CFT_MAX_BATCH_FOR_DISPATCH"
# CFT combine wins at small/medium batch and ties/regresses at large batch, so it is gated by the
# same per-call token-count threshold as dispatch.
_CFT_DEFAULT_MAX_BATCH_FOR_COMBINE = 128
_CFT_MAX_BATCH_FOR_COMBINE_ENV = "TRTLLM_MOE_A2A_CFT_MAX_BATCH_FOR_COMBINE"
FORCE_CFT_ENV = "TRTLLM_MOE_A2A_FORCE_CFT"
_CFT_ALIGNMENT_BYTES = 16
_CFT_MIN_DRIVER_BRANCH = 615


def get_force_cft() -> bool | None:
    value = os.environ.get(FORCE_CFT_ENV)
    if value == "0":
        return False
    if value == "1":
        return True
    return None


def _get_nvidia_driver_version() -> str | None:
    try:
        try:
            pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError_Uninitialized:
            pynvml.nvmlInit()
        value = pynvml.nvmlSystemGetDriverVersion()
    except pynvml.NVMLError as error:
        tllm_logger.warning_once(
            "CFT counted writes disabled: failed to query the NVIDIA driver "
            f"version via NVML ({error}). Falling back to fence-based dispatch.",
            key="moe_a2a_cft_driver_query_failed",
        )
        return None
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def cft_driver_is_supported(driver_version: str | bytes | None) -> bool:
    if isinstance(driver_version, bytes):
        driver_version = driver_version.decode(errors="replace")
    if not driver_version:
        return False
    match = re.match(r"^(\d+)(?:\.|$)", driver_version.strip())
    return bool(match and int(match.group(1)) >= _CFT_MIN_DRIVER_BRANCH)


def resolve_can_use_cft(can_use_cft_counted_writes: bool) -> bool:
    """Apply the TRTLLM_MOE_A2A_FORCE_CFT override, then the driver requirement.

    Workspace sizing and workspace layout both depend on this, so they must
    resolve it identically: a caller that sizes without the override and then
    constructs with it would lay out the CFT region in an undersized buffer.

    The driver check applies even when CFT is forced. Logical endpoints need
    kernel-driver support; under CUDA forward compatibility a newer user-mode
    driver exports the API but endpoint creation fails on the older kernel driver.
    """
    force_cft = get_force_cft()
    requested = can_use_cft_counted_writes if force_cft is None else force_cft
    if not requested:
        return False
    driver_version = _get_nvidia_driver_version()
    if not cft_driver_is_supported(driver_version):
        if driver_version is not None:
            tllm_logger.warning_once(
                "CFT counted writes disabled: NVIDIA driver "
                f"{driver_version} is below required {_CFT_MIN_DRIVER_BRANCH}.00. "
                "Falling back to fence-based dispatch.",
                key=f"moe_a2a_cft_driver_unsupported_{driver_version}",
            )
        return False
    return True


def should_use_cft(
    can_use_cft: bool,
    force_cft: bool | None,
    max_batch: int | None,
    runtime_max_tokens_per_rank: int,
) -> bool:
    if not can_use_cft:
        return False
    if force_cft is not None:
        return force_cft
    if max_batch is None:
        return True
    return runtime_max_tokens_per_rank <= max_batch


def _use_cft_for_dispatch_payloads(use_cft: bool, payloads: list[torch.Tensor]) -> bool:
    if not use_cft:
        return False
    for payload_index, payload in enumerate(payloads):
        bytes_per_token = payload.shape[1] * payload.element_size()
        if bytes_per_token % _CFT_ALIGNMENT_BYTES != 0:
            tllm_logger.warning_once(
                "CFT counted writes disabled: dispatch payload "
                f"{payload_index} has {bytes_per_token} bytes per token, which is not "
                f"{_CFT_ALIGNMENT_BYTES}-byte aligned. Falling back to fence-based dispatch.",
                key=f"moe_a2a_cft_dispatch_alignment_{payload_index}_{bytes_per_token}",
            )
            return False
    return True


def _use_cft_for_combine_payload(
    use_cft: bool, payload: torch.Tensor, use_low_precision: bool
) -> bool:
    if not use_cft:
        return False
    wire_element_size = 1 if use_low_precision else payload.element_size()
    bytes_per_token = payload.shape[-1] * wire_element_size
    if bytes_per_token % _CFT_ALIGNMENT_BYTES != 0:
        tllm_logger.warning_once(
            "CFT counted writes disabled: combine payload has "
            f"{bytes_per_token} bytes per token, which is not "
            f"{_CFT_ALIGNMENT_BYTES}-byte aligned. Falling back to fence-based combine.",
            key=f"moe_a2a_cft_combine_alignment_{bytes_per_token}",
        )
        return False
    return True


def _get_cft_max_batch(env_name: str, default: int) -> int:
    env_value = os.environ.get(env_name)
    if env_value is None:
        return default
    try:
        threshold = int(env_value)
    except ValueError as e:
        raise ValueError(f"{env_name} must be an integer") from e
    if threshold < 0:
        raise ValueError(f"{env_name} must be non-negative")
    return threshold


def _get_cft_max_batch_for_dispatch() -> int | None:
    return _get_cft_max_batch(_CFT_MAX_BATCH_FOR_DISPATCH_ENV, _CFT_DEFAULT_MAX_BATCH_FOR_DISPATCH)


def _get_cft_max_batch_for_combine() -> int | None:
    return _get_cft_max_batch(_CFT_MAX_BATCH_FOR_COMBINE_ENV, _CFT_DEFAULT_MAX_BATCH_FOR_COMBINE)


class NVLinkOneSided(Communication):
    """
    NVLINK one-sided comm AllToAll strategy for throughput scenarios.

    This implementation utilizes symmetric memory to enable peer-to-peer access between GPUs over NVLink.
    The kernels only take the role as one side of the communication: the dispatch kernel puts the data
    into peer ranks' symmetric memory from local buffer, while the combine kernel gets the data from peer
    ranks' symmetric memory and reduces the data into local buffer. It is the most efficient implementation
    by now, but requires symmetric memory size proportional to `max_num_tokens * n_ranks`, which may not
    scale well for very large-scale parallelization.
    """

    # Constants from C++ (must match moeAlltoAllKernels.h)
    MAX_RANKS = 256
    MAX_TOP_K = 8
    MAX_PAYLOADS = 8

    # Shared workspaces/memory across the process, keyed by payload layout and CFT mode.
    _WORKSPACES: Dict[Tuple[object, ...], dict] = {}
    _WORKSPACE_REFCOUNTS: Dict[Tuple[object, ...], int] = {}
    _WORKSPACE: dict | None = None

    # MetaInfo indices - initialized from C++ constants
    FLAG_VAL_OFFSET_INDEX = None
    LOCAL_TOKEN_COUNTER_OFFSET_INDEX = None
    SEND_COUNTERS_OFFSET_INDEX = None
    RECV_COUNTERS_OFFSET_INDEX = None
    DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX = None
    COMBINE_COMPLETION_FLAGS_OFFSET_INDEX = None
    EPLB_GATHERED_STATS_OFFSET_INDEX = None
    PAYLOAD_DATA_OFFSET_INDEX = None

    @staticmethod
    def get_aux_data_size(
        ep_size: int,
        max_num_tokens: int,
        eplb_stats_num_experts: Optional[int] = None,
        can_use_cft_counted_writes: bool = False,
    ) -> int:
        return torch.ops.trtllm.moe_a2a_get_aux_data_size(
            ep_size, max_num_tokens, eplb_stats_num_experts, can_use_cft_counted_writes
        )

    @staticmethod
    def calculate_required_workspace_size(
        ep_size: int,
        top_k: int,
        max_num_tokens: int,
        hidden_size: int,
        dtype: torch.dtype,
        eplb_stats_num_experts: Optional[int] = None,
        extra_payload_bytes_per_token: int = 0,
        can_use_cft_counted_writes: bool = False,
    ) -> int:
        can_use_cft_counted_writes = resolve_can_use_cft(can_use_cft_counted_writes)
        element_size = dtype.itemsize

        # Auxiliary data size
        workspace_size = NVLinkOneSided.get_aux_data_size(
            ep_size, max_num_tokens, eplb_stats_num_experts, can_use_cft_counted_writes
        )

        # Dispatch needs workspace for [ep_size, max_tokens] tokens,
        # but due to the variety of quantization recipes, we cannot know the exact size, so we conservatively estimate assuming no quantization.
        # Meanwhile, we consider the alignment requirement as in moeA2ADispatchOp and moeA2ACombineOp.
        # (Unquantized) token hidden states
        workspace_size += ep_size * max_num_tokens * hidden_size * element_size
        workspace_size = pad_up(workspace_size, 128)
        # token_selected_experts
        workspace_size += ep_size * max_num_tokens * top_k * 4
        workspace_size = pad_up(workspace_size, 128)
        # token_final_scales
        workspace_size += ep_size * max_num_tokens * top_k * 4
        workspace_size = pad_up(workspace_size, 128)
        # Required workspace for combine [ep_size, max_tokens] tokens
        workspace_size += ep_size * max_num_tokens * hidden_size * element_size
        workspace_size = pad_up(workspace_size, 128)
        # CFT combine: dedicated combine RECEIVE region C (peer pushes land here;
        # prepareCombine never touches it -> no proxy aliasing). Same size as the combine region.
        if can_use_cft_counted_writes:
            workspace_size += ep_size * max_num_tokens * hidden_size * element_size
            workspace_size = pad_up(workspace_size, 128)
        # extra payload bytes per token
        workspace_size += ep_size * max_num_tokens * extra_payload_bytes_per_token
        workspace_size = pad_up(workspace_size, 128)

        return workspace_size

    @classmethod
    def _init_constants(cls):
        """Initialize constants from C++ if not already done."""
        if cls.FLAG_VAL_OFFSET_INDEX is None:
            thop = _tllm_internal.thop
            cls.FLAG_VAL_OFFSET_INDEX = int(thop.MOE_A2A_FLAG_VAL_OFFSET_INDEX)
            cls.LOCAL_TOKEN_COUNTER_OFFSET_INDEX = int(
                thop.MOE_A2A_LOCAL_TOKEN_COUNTER_OFFSET_INDEX
            )
            cls.SEND_COUNTERS_OFFSET_INDEX = int(thop.MOE_A2A_SEND_COUNTERS_OFFSET_INDEX)
            cls.RECV_COUNTERS_OFFSET_INDEX = int(thop.MOE_A2A_RECV_COUNTERS_OFFSET_INDEX)
            cls.DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX = int(
                thop.MOE_A2A_DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX
            )
            cls.COMBINE_COMPLETION_FLAGS_OFFSET_INDEX = int(
                thop.MOE_A2A_COMBINE_COMPLETION_FLAGS_OFFSET_INDEX
            )
            cls.EPLB_GATHERED_STATS_OFFSET_INDEX = int(
                thop.MOE_A2A_EPLB_GATHERED_STATS_OFFSET_INDEX
            )
            cls.PAYLOAD_DATA_OFFSET_INDEX = int(thop.MOE_A2A_PAYLOAD_DATA_OFFSET_INDEX)

    def __init__(
        self,
        mapping: Mapping,
        num_slots: int,
        top_k: int,
        max_num_tokens_per_rank: int,
        payload_in_workspace: bool = False,
        hidden_size: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        num_experts: Optional[int] = None,
        use_low_precision_combine: bool = False,
        can_use_cft_counted_writes: bool = False,
        ep_group_health: EPGroupHealthLike | None = None,
        alltoall_watchdog_timeout_s: Optional[float] = None,
        alltoall_watchdog_poll_interval_s: float = DEFAULT_ALLTOALL_WATCHDOG_POLL_INTERVAL_S,
        alltoall_watchdog_on_timeout: Optional[Callable[[AlltoAllWatchdogTimeout], None]] = None,
        require_execution_state: bool = False,
    ) -> None:
        """
        Initialize NVLinkOneSided with workspace allocation.

        Args:
            mapping: TensorRT-LLM Mapping object containing rank information
            num_slots: Number of routing slots (token_selected_experts values are in [0, num_slots)).
                Note: The terminology is mapped to `num_experts` in this class and the kernels.
            top_k: Number of experts per token
            max_num_tokens_per_rank: Maximum number of tokens per rank (for workspace allocation)
            payload_in_workspace: If True, final_hidden_states is already in workspace
            hidden_size: Hidden dimension size (optional, for auto workspace calculation)
            dtype: Data type (optional, for auto workspace calculation)
            num_experts: (Optional) Number of experts for EPLB stats (must be <= num_slots). DO NOT provide this parameter if EPLB is not enabled.
                Note: The terminology is mapped to `eplb_stats_num_experts` in this class and the kernels.
            use_low_precision_combine: If True, quantize the combine payload to FP8 for NVLink
                transfer (halves NVLink bandwidth usage, output precision is preserved).
                Corresponds to model_config.use_low_precision_moe_combine.
            can_use_cft_counted_writes: If True, allow CFT handle-based counted
                writes (fabric.try_put.counted via Logical Endpoints) for dispatch.
                Requires sm_100+ (Blackwell or later), a build against CUDA 13.4+, an
                NVLink fabric, and a driver exporting the CUDA logical endpoint API.
                Defaults to False: CFT is opt-in, so the fence-based path remains the
                default on every architecture. Callers that have verified the CFT
                prerequisites may pass True, or set TRTLLM_MOE_A2A_FORCE_CFT=1 to force
                CFT for supported workloads (0 forces the fence path).
            ep_group_health: Optional read-only committed EP membership. When present, rank-mask handling is
                enabled in the CUDA kernels, and its mask defines the peers expected by the watchdog. Timeout
                detection never mutates it. CUDA graphs are rejected until membership-scoped recapture lands.
            alltoall_watchdog_timeout_s: Optional timeout for the host-side AlltoAll watchdog. If None, the
                watchdog is disabled.
            alltoall_watchdog_poll_interval_s: Poll interval for the watchdog thread.
            alltoall_watchdog_on_timeout: Optional callback invoked when the watchdog reports suspects.
        """
        super().__init__(mapping)
        self.requires_execution_state = require_execution_state
        self._execution_state = None
        if mapping.has_cp_helix():
            raise ValueError(
                "NVLinkOneSided does not support Helix context parallelism because "
                "its MNNVL communicator covers only the tensor-parallel group"
            )

        if self.mapping.world_size != self.ep_size:
            raise RuntimeError("Currently NVLinkOneSided only supports pure EP for MoE.")
        # Check the kernel rank cap before allocating the (large) symmetric workspace.
        if self.ep_size > self.MAX_RANKS:
            raise RuntimeError(
                f"NVLinkOneSided supports at most {self.MAX_RANKS} EP ranks, got ep_size={self.ep_size}."
            )

        # Store needed parameters
        self.num_experts = num_slots
        self.top_k = top_k
        self.max_num_tokens_per_rank = max_num_tokens_per_rank
        self.payload_in_workspace = payload_in_workspace
        self.use_low_precision_combine = use_low_precision_combine
        if num_experts is not None:
            assert num_experts > 0 and num_experts <= num_slots, (
                "num_experts must be in (0, num_slots]"
            )
            tllm_logger.info(
                "NVLinkOneSided AlltoAll: EPLB is enabled, with num_slots="
                f"{num_slots} and num_experts={num_experts}"
            )
        self.enable_eplb = num_experts is not None
        self.eplb_stats_num_experts = num_experts
        self._force_cft = get_force_cft()
        # Opt-in only: no caller passes can_use_cft_counted_writes=True, so
        # without the override the CFT path cannot be reached at all. Leaving
        # the variable unset keeps CFT disabled, as before.
        can_use_cft_counted_writes = resolve_can_use_cft(can_use_cft_counted_writes)
        self.can_use_cft_counted_writes = can_use_cft_counted_writes
        if self._force_cft is None:
            self.cft_max_batch_for_dispatch = _get_cft_max_batch_for_dispatch()
            self.cft_max_batch_for_combine = _get_cft_max_batch_for_combine()
        else:
            self.cft_max_batch_for_dispatch = None
            self.cft_max_batch_for_combine = None
        if can_use_cft_counted_writes:
            tllm_logger.info(
                "NVLinkOneSided AlltoAll: CFT handle-based counted writes enabled for dispatch"
            )
            if self._force_cft is True:
                tllm_logger.info("NVLinkOneSided AlltoAll: CFT forced for supported workloads")
            elif self.cft_max_batch_for_dispatch is not None:
                tllm_logger.info(
                    "NVLinkOneSided AlltoAll: CFT dispatch disabled above "
                    f"runtime_max_tokens_per_rank={self.cft_max_batch_for_dispatch}"
                )
        else:
            tllm_logger.info(
                "NVLinkOneSided AlltoAll: CFT handle-based counted writes disabled for dispatch"
            )

        # Initialize constants from C++
        self._init_constants()

        # Get workspace size
        auto_workspace_size = None
        if hidden_size is not None and dtype is not None:
            auto_workspace_size = self.calculate_required_workspace_size(
                self.ep_size,
                self.top_k,
                max_num_tokens_per_rank,
                hidden_size,
                dtype,
                eplb_stats_num_experts=self.eplb_stats_num_experts,
                can_use_cft_counted_writes=self.can_use_cft_counted_writes,
            )
        workspace_mb_env = os.environ.get("TRTLLM_MOE_A2A_WORKSPACE_MB")
        if workspace_mb_env:
            self.workspace_size_per_rank = int(workspace_mb_env) * 1024 * 1024
            msg = f"NVLinkOneSided: Forcing workspace size to {self.workspace_size_per_rank} bytes (TRTLLM_MOE_A2A_WORKSPACE_MB={workspace_mb_env})."
            if auto_workspace_size is not None:
                msg += f"Automatically calculated workspace size is {auto_workspace_size} bytes."
                msg += "Auto calculation is conservative, so only consider overriding it if you have a specific reason."
            tllm_logger.warning(msg)
        elif auto_workspace_size is not None:
            self.workspace_size_per_rank = auto_workspace_size
        else:
            tllm_logger.warning(
                "NVLinkOneSided: hidden_size and dtype are not provided (which are required for calculating workspace size)."
                "Using default workspace size 2048MB."
            )
            self.workspace_size_per_rank = 2048 * 1024 * 1024

        # Initialize or reuse workspace.  The C++ op computes payload offsets
        # from the current tensors at dispatch time, while the Python singleton
        # owns the symmetric memory backing those offsets.  Keep separate
        # workspaces for different payload layouts so one test/layer cannot
        # reuse stale one-sided state from another shape.
        MnnvlMemory.initialize()
        self._workspace_key = (
            self.workspace_size_per_rank,
            self.max_num_tokens_per_rank,
            self.ep_rank,
            self.ep_size,
            self.eplb_stats_num_experts,
            self.num_experts,
            self.top_k,
            hidden_size,
            dtype,
            self.use_low_precision_combine,
            self.can_use_cft_counted_writes,
        )

        workspace_state = NVLinkOneSided._WORKSPACES.get(self._workspace_key)
        memory_cls = CftMnnvlMemory if self.can_use_cft_counted_writes else MnnvlMemory
        workspace_created = workspace_state is None
        if workspace_state is None:
            tllm_logger.info(
                f"NVLinkOneSided: Allocating workspace with size {self.workspace_size_per_rank} bytes."
                f"ep_rank: {self.ep_rank}, ep_size: {self.ep_size}, top_k: {self.top_k}, max_num_tokens_per_rank: {self.max_num_tokens_per_rank}"
            )
            mnnvl_mem = memory_cls(mapping, self.workspace_size_per_rank)
            workspace = mnnvl_mem.as_torch_strided_tensor(torch.uint8)
            metainfo = torch.ops.trtllm.moe_a2a_initialize(
                workspace,
                self.ep_rank,
                self.ep_size,
                self.max_num_tokens_per_rank,
                self.eplb_stats_num_experts,
                self.can_use_cft_counted_writes,
            )
            workspace_state = {
                "workspace_size_per_rank": self.workspace_size_per_rank,
                "max_num_tokens_per_rank": self.max_num_tokens_per_rank,
                "ep_rank": self.ep_rank,
                "ep_size": self.ep_size,
                "eplb_stats_num_experts": self.eplb_stats_num_experts,
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "hidden_size": hidden_size,
                "dtype": dtype,
                "use_low_precision_combine": self.use_low_precision_combine,
                "can_use_cft_counted_writes": self.can_use_cft_counted_writes,
                "mnnvl_mem": mnnvl_mem,
                "workspace": workspace,
                "metainfo": metainfo,
                "cft_initialized": False,
            }
        else:
            expected_workspace_state = {
                "workspace_size_per_rank": self.workspace_size_per_rank,
                "max_num_tokens_per_rank": self.max_num_tokens_per_rank,
                "ep_rank": self.ep_rank,
                "ep_size": self.ep_size,
                "eplb_stats_num_experts": self.eplb_stats_num_experts,
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "hidden_size": hidden_size,
                "dtype": dtype,
                "use_low_precision_combine": self.use_low_precision_combine,
                "can_use_cft_counted_writes": self.can_use_cft_counted_writes,
            }
            for key, expected_value in expected_workspace_state.items():
                assert workspace_state[key] == expected_value, (
                    f"reuse workspace with different {key}"
                )

        self._destroyed = False
        self._workspace_registered = False
        self._workspace_state = workspace_state
        self.mnnvl_mem = workspace_state["mnnvl_mem"]
        self.workspace = workspace_state["workspace"]
        self.max_num_tokens_per_rank = workspace_state["max_num_tokens_per_rank"]
        self.ep_group_health = ep_group_health
        # Keep the kernel specialization stable for this communicator's lifetime.
        self._rank_mask_enabled = ep_group_health is not None
        if alltoall_watchdog_timeout_s is None and self.ep_group_health is not None:
            alltoall_watchdog_timeout_s = DEFAULT_ALLTOALL_WATCHDOG_TIMEOUT_S
        flag_val_offset_index = self.FLAG_VAL_OFFSET_INDEX
        dispatch_flags_offset_index = self.DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX
        combine_flags_offset_index = self.COMBINE_COMPLETION_FLAGS_OFFSET_INDEX
        if (
            flag_val_offset_index is None
            or dispatch_flags_offset_index is None
            or combine_flags_offset_index is None
        ):
            raise RuntimeError("MoE All-to-All metadata indices are not initialized")
        self._workspace_lifecycle = _MnnvlAlltoAllWorkspaceLifecycle.get_or_create(
            workspace_state=workspace_state,
            memory=self.mnnvl_mem,
            workspace=self.workspace,
            metainfo=workspace_state["metainfo"],
            metainfo_index={
                "FLAG_VAL_OFFSET_INDEX": flag_val_offset_index,
                "DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX": dispatch_flags_offset_index,
                "COMBINE_COMPLETION_FLAGS_OFFSET_INDEX": combine_flags_offset_index,
            },
            ep_rank=self.ep_rank,
            ep_size=self.ep_size,
            health=self.ep_group_health,
        )
        # Keep combine storage at a fixed offset across changing dispatch
        # payloads, so a later dispatch cannot overwrite a peer's live combine.
        if hidden_size is not None and dtype is not None:
            self._reserve_combine_region(hidden_size, dtype)

        self._workspace_lifecycle.register(
            self,
            watchdog_timeout_s=alltoall_watchdog_timeout_s,
            watchdog_poll_interval_s=alltoall_watchdog_poll_interval_s,
            watchdog_on_timeout=alltoall_watchdog_on_timeout,
            defer_watchdog_start=require_execution_state,
        )
        if workspace_created:
            NVLinkOneSided._WORKSPACES[self._workspace_key] = workspace_state
        NVLinkOneSided._WORKSPACE = workspace_state
        NVLinkOneSided._WORKSPACE_REFCOUNTS[self._workspace_key] = (
            NVLinkOneSided._WORKSPACE_REFCOUNTS.get(self._workspace_key, 0) + 1
        )
        self._workspace_registered = True

        # Initialize CFT Logical Endpoints by binding the LE to the workspace.
        # The LE IS the workspace — no separate allocation or payload layout needed.
        if self.can_use_cft_counted_writes and not workspace_state.get("cft_initialized", False):
            torch.ops.trtllm.moe_a2a_cft_initialize(
                self.workspace,
                self.mnnvl_mem.local_mem_handle,
                int(self.workspace.size(1)),
                self.ep_rank,
                self.ep_size,
            )
            workspace_state["cft_initialized"] = True

        # Initialize dispatch state
        self._dispatch_state = {"phase": "idle"}

        # Invalid token expert ID (default to -1), the kernels in TRTLLM-gen is hard-code to support -1 only.
        self.invalid_token_expert_id: int = -1

    def execution_watchdog(self) -> AlltoAllWatchdog | None:
        return self._alltoall_watchdog

    def execution_geometry(self) -> tuple[int, int, int, int, torch.dtype, int, int, int, int]:
        self._require_mapped()
        if self.can_use_cft_counted_writes:
            raise RuntimeError("Descriptor execution requires fence communication; CFT is enabled")
        if self.use_low_precision_combine:
            raise RuntimeError(
                "Descriptor execution requires BF16 combine; low precision is enabled"
            )
        memory = self.execution_lifecycle().memory
        state = self._workspace_state
        hidden_size, dtype = state["hidden_size"], state["dtype"]
        offset = state["combine_storage_offset"]
        if (
            dtype is not torch.bfloat16
            or self.workspace.size(0) != self.ep_size
            or self.workspace.stride(0) != memory.rank_stride
            or not 0 < self.workspace.size(1) <= memory.segment_size <= memory.rank_stride
            or offset + self.ep_size * self.max_num_tokens_per_rank * hidden_size * dtype.itemsize
            > self.workspace.size(1)
        ):
            raise RuntimeError("Invalid immutable workspace or canonical combine tail")
        return (
            self.ep_size,
            self.ep_rank,
            self.max_num_tokens_per_rank,
            hidden_size,
            dtype,
            offset,
            self.workspace.data_ptr(),
            self.workspace.stride(0),
            self.workspace.size(1),
        )

    def execution_dispatch_kwargs(
        self, slot: torch.Tensor, payloads: list[torch.Tensor]
    ) -> dict[str, torch.Tensor | int]:
        geometry = self.execution_geometry()
        limit = geometry[5]
        payload_start = int(self.moe_a2a_metainfo[self.PAYLOAD_DATA_OFFSET_INDEX])
        source_bytes = ((limit - payload_start) // self.ep_size) // 128 * 128
        required = 0
        for payload in payloads:
            if payload.dim() != 2:
                raise RuntimeError("Unqualified dispatch payload schema")
            size = self.max_num_tokens_per_rank * payload.size(1) * payload.element_size()
            required += (size + 127) // 128 * 128
        if not 0 < required <= source_bytes:
            raise RuntimeError("Dispatch exceeds fixed logical source stride")
        return dict(execution_descriptor=slot, dispatch_payload_limit=limit)

    def execution_combine_kwargs(
        self, slot: torch.Tensor, output: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if not self.payload_in_workspace:
            raise RuntimeError(
                "Descriptor combine requires the current scheduler plan's workspace output"
            )
        geometry = self.execution_geometry()
        expected = self.workspace.data_ptr() + self.ep_rank * self.workspace.stride(0) + geometry[5]
        if (
            output.data_ptr() != expected
            or output.dtype is not torch.bfloat16
            or not output.is_contiguous()
            or output.size(-1) != geometry[3]
            or output.numel() * output.element_size() > self.workspace.size(1) - geometry[5]
        ):
            raise RuntimeError("FC2 must write the canonical mapped tail")
        return dict(execution_descriptor=slot)

    def execution_lifecycle(self) -> _MnnvlAlltoAllWorkspaceLifecycle:
        return self._require_workspace_lifecycle()

    def bind_execution_state(self, state: "ExecutionState") -> None:
        if not self.requires_execution_state:
            raise RuntimeError("Frontend did not request descriptor binding")
        if self._execution_state is not None and self._execution_state is not state:
            raise RuntimeError("Frontend already belongs to another execution state")
        if (
            state.lifecycle is not self.execution_lifecycle()
            or state.ep_group_health is not self.ep_group_health
        ):
            raise RuntimeError("Execution state health or workspace differs from frontend")
        state.validate_frontend(self)
        self._execution_state = state
        self._capture_allocation_spans = {}

    def _record_capture_spans(
        self, phase: str, tensors: tuple[tuple[str, torch.Tensor, str], ...]
    ) -> None:
        if not torch.cuda.is_current_stream_capturing():
            return
        from .execution_state import CapturedTensorSpan

        state = self._execution_state
        state.validate_frontend_epoch(self)
        geometry = self.execution_geometry()
        spans = []
        for role, tensor, origin in tensors:
            if not tensor.is_contiguous() or tensor.device != self.workspace.device:
                raise RuntimeError("Captured native allocation layout or device differs")
            pointer, size = int(tensor.data_ptr()), tensor.numel() * tensor.element_size()
            if origin == "native_torch_empty":
                for peer in range(geometry[0]):
                    begin = geometry[6] + peer * geometry[7]
                    if max(pointer, begin) < min(pointer + size, begin + geometry[8]):
                        raise RuntimeError("Captured private allocation aliases a peer workspace")
            elif origin == "canonical_workspace_tail":
                if pointer != geometry[6] + geometry[1] * geometry[7] + geometry[5]:
                    raise RuntimeError("Captured expert output is not the canonical local tail")
                if size > geometry[8] - geometry[5]:
                    raise RuntimeError("Captured expert output exceeds the canonical local tail")
            else:
                raise RuntimeError("Captured allocation origin is unqualified")
            spans.append(
                CapturedTensorSpan(
                    role, pointer, size, tuple(tensor.shape), str(tensor.dtype), origin
                )
            )
        self._capture_allocation_spans[phase] = (state.execution_sequence, tuple(spans))

    def captured_allocation_spans(self, phase: str) -> tuple[int, tuple["CapturedTensorSpan", ...]]:
        if phase not in self._capture_allocation_spans:
            raise RuntimeError("Native capture produced no allocation spans for the phase")
        return self._capture_allocation_spans[phase]

    def _require_execution_binding(self, kwargs: dict[str, object]) -> "ExecutionState | None":
        state = self._execution_state
        if self.requires_execution_state and state is None:
            raise RuntimeError("Model execution state must bind before warmup or capture")
        if state is not None:
            if any(
                name in kwargs
                for name in ("active_rank_mask", "execution_descriptor", "dispatch_payload_limit")
            ):
                raise ValueError("Bound execution forbids mask or descriptor overrides")
            state.validate_frontend_epoch(self)
        else:
            reject_rank_mask_cuda_graph_capture(self._rank_mask_enabled)
        return state

    @property
    def moe_a2a_metainfo(self) -> torch.Tensor:
        return self._require_workspace_lifecycle().metainfo

    @property
    def _watchdog_coordinator(self) -> AlltoAllWatchdogCoordinator:
        return self._require_workspace_lifecycle().coordinator

    @property
    def _alltoall_watchdog(self) -> AlltoAllWatchdog | None:
        return self._require_workspace_lifecycle().watchdog_for(self)

    def _require_workspace_lifecycle(self) -> _MnnvlAlltoAllWorkspaceLifecycle:
        lifecycle = self._workspace_lifecycle
        if lifecycle is None:
            raise RuntimeError("NVLinkOneSided workspace has been destroyed")
        return lifecycle

    @staticmethod
    def is_platform_supported() -> bool:
        """
        Check if NVLINK one-sided comm is supported on current hardware.
        """
        return MnnvlMemory.supports_mnnvl()

    def supports_post_quant_dispatch(self) -> bool:
        """
        NVLINK one-sided comm supports post-quant dispatch.
        """
        return True

    def destroy(self):
        lifecycle = getattr(self, "_workspace_lifecycle", None)
        with lifecycle.mutable_clients() if lifecycle is not None else nullcontext():
            self._destroy_impl()

    def _destroy_impl(self) -> None:
        """Release shared state during explicit, rank-coordinated teardown."""
        if getattr(self, "_destroyed", False):
            return

        state = getattr(self, "_execution_state", None)
        if state is not None:
            state.before_frontend_destroy(self)
        self._destroyed = True
        lifecycle = getattr(self, "_workspace_lifecycle", None)
        if lifecycle is not None and getattr(self, "_workspace_registered", False):
            lifecycle.unregister(self)
        workspace_key = getattr(self, "_workspace_key", None)
        if workspace_key is None or not getattr(self, "_workspace_registered", False):
            self._workspace_lifecycle = None
            return
        self._workspace_registered = False

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        refcount = NVLinkOneSided._WORKSPACE_REFCOUNTS.get(workspace_key, 0) - 1
        if refcount > 0:
            NVLinkOneSided._WORKSPACE_REFCOUNTS[workspace_key] = refcount
        else:
            NVLinkOneSided._WORKSPACE_REFCOUNTS.pop(workspace_key, None)
            workspace_state = NVLinkOneSided._WORKSPACES.pop(workspace_key, None)
            if NVLinkOneSided._WORKSPACE is workspace_state:
                NVLinkOneSided._WORKSPACE = None
            if workspace_state is not None:
                workspace_state.clear()

        self.mnnvl_mem = None
        self.workspace = None
        self._workspace_state = None
        self._workspace_lifecycle = None
        self._dispatch_state = {"phase": "destroyed"}

    def __del__(self) -> None:
        if sys.is_finalizing():
            return
        lifecycle = getattr(self, "_workspace_lifecycle", None)
        try:
            with lifecycle.mutable_clients() if lifecycle is not None else nullcontext():
                self._finalize_impl()
        except RuntimeError:
            return

    def _finalize_impl(self) -> None:
        if sys.is_finalizing():
            return
        state = getattr(self, "_execution_state", None)
        if state is not None:
            try:
                state.before_frontend_destroy(self)
            except RuntimeError:
                return
        # Finalizers cannot safely perform collective cache eviction because
        # their order is nondeterministic across ranks.
        lifecycle = getattr(self, "_workspace_lifecycle", None)
        if lifecycle is not None and getattr(self, "_workspace_registered", False):
            lifecycle.unregister(self)
        workspace_key = getattr(self, "_workspace_key", None)
        if workspace_key is not None and getattr(self, "_workspace_registered", False):
            refcount = NVLinkOneSided._WORKSPACE_REFCOUNTS.get(workspace_key, 0) - 1
            if refcount > 0:
                NVLinkOneSided._WORKSPACE_REFCOUNTS[workspace_key] = refcount
            else:
                NVLinkOneSided._WORKSPACE_REFCOUNTS.pop(workspace_key, None)
        self._workspace_registered = False
        self._workspace_lifecycle = None

    def is_workload_feasible(self, all_rank_num_tokens: List[int], num_chunks: int) -> bool:
        """
        Check if NVLINK one-sided comm is feasible for the given workload at runtime.

        This method performs runtime checks based on workload characteristics such as
        token counts, number of chunks, and other runtime parameters.
        """
        return True

    def use_cft_for_dispatch(self, runtime_max_tokens_per_rank: int) -> bool:
        return should_use_cft(
            self.can_use_cft_counted_writes,
            self._force_cft,
            self.cft_max_batch_for_dispatch,
            runtime_max_tokens_per_rank,
        )

    def use_cft_for_combine(self, runtime_max_tokens_per_rank: int) -> bool:
        return should_use_cft(
            self.can_use_cft_counted_writes,
            self._force_cft,
            self.cft_max_batch_for_combine,
            runtime_max_tokens_per_rank,
        )

    def _require_mapped(self) -> None:
        if not self.mnnvl_mem.mapped:
            raise RuntimeError("Native MoE All-to-All workspace handles are unmapped")

    def checkpoint_resource_key(self) -> int:
        """Identify wrappers sharing the same MNNVL workspace lifecycle."""
        return id(self._require_workspace_lifecycle())

    def checkpoint_prepare(self) -> None:
        """Collectively detach handles after every shared owner is idle."""
        if self.can_use_cft_counted_writes:
            raise RuntimeError(
                "Checkpointing a CFT-backed MoE All-to-All workspace is not supported"
            )
        self._require_workspace_lifecycle().checkpoint_prepare()

    def checkpoint_restore(
        self,
        comm: MnnvlCheckpointCommunicator | None = None,
    ) -> None:
        """Collectively restore handles and all shared frontend state.

        Args:
            comm: An mpi4py-like communicator exposing ``Get_rank()``,
                ``Get_size()``, ``allgather()``, and ``barrier()``. Its local
                rank and size must match the communicator used for the
                original allocation. Every rank must call this method
                symmetrically.
        """
        if comm is None:
            comm = self.mnnvl_mem.comm
        if comm is None:
            raise RuntimeError("MNNVL workspace communicator is not initialized")
        self._require_workspace_lifecycle().checkpoint_restore(
            comm,
            lambda: torch.ops.trtllm.moe_a2a_initialize(
                self.workspace,
                self.ep_rank,
                self.ep_size,
                self.max_num_tokens_per_rank,
                self.eplb_stats_num_experts,
                self.can_use_cft_counted_writes,
            ),
        )

    def _mnnvl_checkpoint_is_idle(self) -> bool:
        return self._dispatch_state.get("phase") == "idle"

    def _mnnvl_checkpoint_reset(self) -> None:
        self._dispatch_state = {"phase": "idle"}

    def _reserve_combine_region(self, hidden_size: int, dtype: torch.dtype) -> int:
        """Keep dispatch and combine storage disjoint across workspace reuse."""
        layout = (hidden_size, dtype.itemsize)
        old_layout = self._workspace_state.get("combine_storage_layout")
        if old_layout is not None:
            if old_layout != layout:
                raise ValueError(
                    "shared A2A workspace requires a stable combine shape and dtype size"
                )
            return self._workspace_state["combine_storage_offset"]

        # Native combine stores a dense [EP, runtime_max_tokens, hidden] tensor.
        # Match the existing allocation: reserve a second full-size region only
        # for CFT-capable workspaces. Low-precision combine fits in this bound.
        region_bytes = pad_up(
            self.ep_size * self.max_num_tokens_per_rank * hidden_size * dtype.itemsize, 128
        )
        num_regions = 2 if self.can_use_cft_counted_writes else 1
        offset = (self.workspace_size_per_rank - num_regions * region_bytes) // 128 * 128
        aux_bytes = int(self.moe_a2a_metainfo[self.PAYLOAD_DATA_OFFSET_INDEX])
        if offset < aux_bytes:
            raise ValueError("A2A workspace is too small for stable combine regions")
        dispatch_end = getattr(self, "_dispatch_state", {}).get("dispatch_payload_end", aux_bytes)
        if dispatch_end > offset:
            raise ValueError("A2A dispatch payload overlaps the reserved combine region")
        self._workspace_state["combine_storage_layout"] = layout
        self._workspace_state["combine_storage_offset"] = offset
        return offset

    def _check_dispatch_region(self, payloads: List[torch.Tensor], max_tokens: int) -> int:
        if not 0 < max_tokens <= self.max_num_tokens_per_rank:
            raise ValueError("runtime token count exceeds the configured A2A capacity")
        end = int(self.moe_a2a_metainfo[self.PAYLOAD_DATA_OFFSET_INDEX])
        for payload in payloads:
            end = pad_up(
                end + self.ep_size * max_tokens * payload.shape[1] * payload.element_size(), 128
            )
        limit = self._workspace_state.get("combine_storage_offset", self.workspace_size_per_rank)
        # Check before the native dispatch can write into any peer's memory.
        if end > limit:
            raise ValueError("A2A dispatch payload overlaps the reserved combine region")
        return end

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        hidden_states_sf: Optional[torch.Tensor],
        token_selected_slots: torch.Tensor,
        token_final_scales: Optional[torch.Tensor],
        all_rank_num_tokens: List[int],
        use_dp_padding: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """
        Dispatch phase: scatter/send data to different ranks

        Args:
            hidden_states: Input tensor [local_num_tokens, hidden_size]
            hidden_states_sf: Input scaling factor [local_num_tokens, sf_size]
            token_selected_slots: Selected expert slots [local_num_tokens, top_k]
            token_final_scales: Router weights [local_num_tokens, top_k]
            all_rank_num_tokens: Token counts per rank [ep_size]
            use_dp_padding: Whether to use DP padding (optional)
            **kwargs: Strategy-specific arguments. In rank-mask mode, ``active_rank_mask`` may override the
                committed membership for dispatch. Without an override, the committed mask and generation are
                captured together; combine reuses that mask and fails closed if the generation changes first.
                The masked kernel rejects inactive routes before remote access; that sentinel is an internal abort
                artifact, not valid model output.

        Returns:
            Tuple of (hidden_states, hidden_states_sf, token_selected_slots, token_final_scales)
            Each tensor has shape [ep_size, max_tokens_per_rank, ...]
        """
        self._require_mapped()
        if self._dispatch_state.get("phase") == "dispatched":
            raise RuntimeError("dispatch called twice without an intervening combine")
        state = self._require_execution_binding(kwargs)

        # Calculate runtime_max_tokens_per_rank from all_rank_num_tokens
        runtime_max_tokens_per_rank = max(all_rank_num_tokens)
        can_use_cft_for_dispatch = self.use_cft_for_dispatch(runtime_max_tokens_per_rank)
        if state is not None and can_use_cft_for_dispatch:
            raise RuntimeError("Bound descriptor execution requires fence communication")

        # Build payloads list - match TRTLLMGen baseline order for optimal performance
        # Order: [hidden_states, hidden_states_sf (optional), token_selected_slots, token_final_scales (optional)]

        payloads = []
        payloads.append(hidden_states)
        if hidden_states_sf is not None:
            payloads.append(hidden_states_sf)

        payloads.append(token_selected_slots)
        if token_final_scales is not None:
            payloads.append(token_final_scales)
        dispatch_payload_end = self._check_dispatch_region(payloads, runtime_max_tokens_per_rank)
        can_use_cft_for_dispatch = _use_cft_for_dispatch_payloads(
            can_use_cft_for_dispatch, payloads
        )
        expert_id_payload_index = len(payloads) - (2 if token_final_scales is not None else 1)
        can_fuse_sanitize = can_use_cft_for_dispatch

        eplb_local_stats = kwargs.get("eplb_local_stats")
        if eplb_local_stats is not None:
            assert self.enable_eplb, "eplb_local_stats provided but enable_eplb is False"
            assert eplb_local_stats.dim() == 1, "eplb_local_stats must be a 1D tensor"
            assert eplb_local_stats.size(0) == self.eplb_stats_num_experts, (
                "eplb_local_stats size must match eplb_stats_num_experts"
            )
        requested_active_rank_mask = kwargs.get("active_rank_mask")
        if not self._rank_mask_enabled and requested_active_rank_mask is not None:
            raise ValueError("active_rank_mask requires committed EP group health")
        active_rank_mask_snapshot = self._watchdog_coordinator.capture_active_rank_mask(
            requested_active_rank_mask
        )
        active_rank_mask = active_rank_mask_snapshot.active_rank_mask

        recv_buffers, combine_payload_offset, eplb_gathered_stats = (
            torch.ops.trtllm.moe_a2a_dispatch(
                token_selected_slots,
                payloads,
                self.workspace,
                self.moe_a2a_metainfo,
                runtime_max_tokens_per_rank,
                self.ep_rank,
                self.ep_size,
                self.top_k,
                self.num_experts,
                eplb_local_stats,
                can_use_cft_for_dispatch,
                expert_id_payload_index if can_fuse_sanitize else None,
                int(self.invalid_token_expert_id) if can_fuse_sanitize else None,
                self._rank_mask_enabled,
                active_rank_mask,
                **(state.dispatch_kwargs(self, payloads) if state is not None else {}),
            )
        )
        if state is not None:
            if torch.cuda.is_current_stream_capturing():
                self._record_capture_spans(
                    "dispatch",
                    tuple(
                        (f"gather_payload_{index}", tensor, "native_torch_empty")
                        for index, tensor in enumerate(recv_buffers)
                    )
                    + (("gather_stats", eplb_gathered_stats, "native_torch_empty"),),
                )
            state.watch_collective(self, "dispatch", active_rank_mask)
        else:
            self._watchdog_coordinator.watch_collective(
                self._alltoall_watchdog, "dispatch", active_rank_mask
            )
        if eplb_gathered_stats.numel() == 0:
            eplb_gathered_stats = None
        self._dispatch_state["eplb_gathered_stats"] = eplb_gathered_stats
        expected_end = (
            self._workspace_state["combine_storage_offset"]
            if state is not None
            else dispatch_payload_end
        )
        if int(combine_payload_offset) != expected_end:
            raise RuntimeError(
                f"native A2A dispatch layout disagrees with the reserved layout: "
                f"returned={combine_payload_offset}, expected={expected_end}, "
                f"descriptor_mode={state is not None}"
            )
        self._dispatch_state["dispatch_payload_end"] = dispatch_payload_end
        self._dispatch_state["combine_payload_offset"] = self._workspace_state.get(
            "combine_storage_offset", dispatch_payload_end
        )
        self._dispatch_state["local_num_tokens"] = token_selected_slots.size(0)
        self._dispatch_state["runtime_max_tokens_per_rank"] = runtime_max_tokens_per_rank
        self._dispatch_state["active_rank_mask_snapshot"] = active_rank_mask_snapshot
        self._dispatch_state["phase"] = "dispatched"

        # Extract results from recv_buffers
        # Payload order matches input:
        # [hidden_states, hidden_states_sf (optional), token_selected_slots, token_final_scales (optional)]
        hidden_states_recv = recv_buffers[0]
        if hidden_states_sf is not None:
            hidden_states_sf_recv = recv_buffers[1]
            token_selected_slots_recv = recv_buffers[2]
            token_final_scales_recv = recv_buffers[3] if token_final_scales is not None else None
        else:
            hidden_states_sf_recv = None
            token_selected_slots_recv = recv_buffers[1]
            token_final_scales_recv = recv_buffers[2] if token_final_scales is not None else None

        if not can_fuse_sanitize:
            torch.ops.trtllm.moe_a2a_sanitize_expert_ids(
                token_selected_slots_recv,
                self.workspace,
                self.moe_a2a_metainfo,
                self.ep_rank,
                int(self.invalid_token_expert_id),
                **(dict(execution_descriptor=state.slot) if state is not None else {}),
            )

        # Flatten 3D tensors to 2D for compatibility with MoE backends
        # recv_buffers have shape [ep_size, max_tokens_per_rank, ...], flatten to [ep_size * max_tokens_per_rank, ...]
        hidden_states_recv = hidden_states_recv.view(-1, hidden_states_recv.shape[-1])
        if hidden_states_sf_recv is not None:
            hidden_states_sf_recv = hidden_states_sf_recv.view(-1, hidden_states_sf_recv.shape[-1])
        token_selected_slots_recv = token_selected_slots_recv.view(
            -1, token_selected_slots_recv.shape[-1]
        )
        if token_final_scales_recv is not None:
            token_final_scales_recv = token_final_scales_recv.view(
                -1, token_final_scales_recv.shape[-1]
            )

        return (
            hidden_states_recv,
            hidden_states_sf_recv,
            token_selected_slots_recv,
            token_final_scales_recv,
        )

    def get_eplb_gathered_statistics(self) -> Optional[torch.Tensor]:
        """
        Return gathered EPLB statistics from the last dispatch, if available.
        """
        assert self.enable_eplb, "EPLB is not enabled"
        return self._dispatch_state.get("eplb_gathered_stats")

    def combine(
        self,
        final_hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Combine phase: gather/receive data from different ranks

        Args:
            final_hidden_states: Output from MoE computation
                Shape: [ep_size, max_tokens_per_rank, hidden_size] or
                       [ep_size * max_tokens_per_rank, hidden_size] (will be reshaped)
            **kwargs: Strategy-specific arguments. In rank-mask mode, ``active_rank_mask`` must match the mask
                captured by dispatch for this collective when supplied. A committed-generation change since
                dispatch aborts the collective epoch.

        Returns:
            Combined output tensor [local_num_tokens, hidden_size]

        """
        self._require_mapped()
        if self._dispatch_state.get("phase") != "dispatched":
            raise RuntimeError("combine called before a successful dispatch")
        state = self._require_execution_binding(kwargs)

        local_num_tokens = self._dispatch_state.get("local_num_tokens")
        combine_payload_offset = self._dispatch_state.get("combine_payload_offset")
        runtime_max_tokens_per_rank = self._dispatch_state.get("runtime_max_tokens_per_rank")

        if (
            local_num_tokens is None
            or combine_payload_offset is None
            or runtime_max_tokens_per_rank is None
        ):
            raise RuntimeError("combine called but dispatch state is missing")

        # Reshape if needed (handle case where input is flattened)
        if final_hidden_states.dim() == 2:
            # Flattened: [ep_size * max_tokens_per_rank, hidden_size]
            # Reshape to: [ep_size, max_tokens_per_rank, hidden_size]
            hidden_size = final_hidden_states.shape[-1]
            final_hidden_states = final_hidden_states.view(
                self.ep_size, runtime_max_tokens_per_rank, hidden_size
            )
        elif final_hidden_states.dim() == 3:
            # Already shaped: [ep_size, max_tokens_per_rank, hidden_size]
            pass
        else:
            raise ValueError(
                f"final_hidden_states must be 2D or 3D, got {final_hidden_states.dim()}D"
            )
        active_rank_mask_snapshot = self._dispatch_state.get("active_rank_mask_snapshot")
        if not isinstance(active_rank_mask_snapshot, ActiveRankMaskSnapshot):
            raise RuntimeError("combine called but dispatch rank-mask snapshot is missing")
        requested_active_rank_mask = kwargs.get("active_rank_mask")
        if not self._rank_mask_enabled and requested_active_rank_mask is not None:
            raise ValueError("active_rank_mask requires committed EP group health")
        active_rank_mask = self._watchdog_coordinator.active_rank_mask_for_combine(
            active_rank_mask_snapshot,
            requested_active_rank_mask,
        )
        use_cft_for_combine = _use_cft_for_combine_payload(
            self.use_cft_for_combine(runtime_max_tokens_per_rank),
            final_hidden_states,
            self.use_low_precision_combine,
        )
        if state is not None and use_cft_for_combine:
            raise RuntimeError("Bound descriptor execution requires fence communication")
        combine_payload_offset = self._reserve_combine_region(
            final_hidden_states.shape[-1], final_hidden_states.dtype
        )
        output = torch.ops.trtllm.moe_a2a_combine(
            final_hidden_states,
            int(local_num_tokens),
            self.workspace,
            self.moe_a2a_metainfo,
            int(runtime_max_tokens_per_rank),
            self.ep_rank,
            self.ep_size,
            self.top_k,
            int(combine_payload_offset),
            bool(self.payload_in_workspace),
            bool(self.use_low_precision_combine),
            use_cft_for_combine,
            self._rank_mask_enabled,
            active_rank_mask,
            **(state.combine_kwargs(self, final_hidden_states) if state is not None else {}),
        )
        if state is not None:
            if torch.cuda.is_current_stream_capturing():
                self._record_capture_spans(
                    "combine",
                    (
                        ("expert_output", final_hidden_states, "canonical_workspace_tail"),
                        ("combined_output", output, "native_torch_empty"),
                    ),
                )
            state.watch_collective(self, "combine", active_rank_mask)
        else:
            self._watchdog_coordinator.watch_collective(
                self._alltoall_watchdog, "combine", active_rank_mask
            )

        # Reset state for next round
        self.reset_state()

        return output

    def reset_state(self) -> None:
        """Reset the dispatch/combine state machine to ``idle``.

        Safe to call between forward passes (or from an error handler) to
        recover from a forward that called ``dispatch`` but did not reach
        ``combine`` — e.g. because an OOM aborted the forward. Without this,
        the next ``dispatch`` would raise ``dispatch called twice``.
        """
        self._dispatch_state = {"phase": "idle"}

    def get_combine_payload_tensor_in_workspace(
        self, runtime_max_tokens_per_rank: int, hidden_size: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Return the combine payload tensor in the workspace, which could be used
        as the output of MoE kernel to avoid extra copy.
        Passing the returned tensor to combine lets the C++ op detect workspace ownership.

        Args:
            runtime_max_tokens_per_rank: Runtime max tokens per rank
            hidden_size: Hidden dimension size
            dtype: Data type

        Returns:
            Tensor view into workspace [ep_size, max_tokens_per_rank, hidden_size]
        """
        self._require_mapped()
        if self._dispatch_state.get("phase") != "dispatched":
            raise RuntimeError(
                "get_combine_payload_tensor_in_workspace called before a successful dispatch"
            )

        combine_payload_offset = self._dispatch_state.get("combine_payload_offset")
        if combine_payload_offset is None:
            raise RuntimeError("combine_payload_offset not found in dispatch state")

        combine_payload_offset = self._reserve_combine_region(hidden_size, dtype)
        self._dispatch_state["combine_payload_offset"] = combine_payload_offset
        result = torch.ops.trtllm.moe_a2a_get_combine_payload_tensor(
            self.workspace,
            int(self.ep_rank),
            int(self.ep_size),
            int(runtime_max_tokens_per_rank),
            int(combine_payload_offset),
            dtype,
            int(hidden_size),
        )

        return result

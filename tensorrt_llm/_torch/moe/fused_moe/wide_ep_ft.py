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
"""Shared WideEP fault-tolerance options for MoE communication paths."""

from __future__ import annotations

import os
from typing import Any, Optional

from tensorrt_llm._torch.alltoall_watchdog import (
    DEFAULT_ALLTOALL_WATCHDOG_POLL_INTERVAL_S,
    DEFAULT_ALLTOALL_WATCHDOG_TIMEOUT_S,
)
from tensorrt_llm._utils import fault_tolerance_mode, fault_tolerance_replacement

from .ep_group_health import EPGroupHealth

_ENABLE_ENV = "TLLM_FAULT_TOLERANCE_MODE"
_TIMEOUT_ENV = "TRTLLM_ALLTOALL_WATCHDOG_TIMEOUT_S"
_POLL_INTERVAL_ENV = "TRTLLM_ALLTOALL_WATCHDOG_POLL_INTERVAL_S"

# This object contains membership committed by higher-layer recovery
# coordination.  Detection threads must treat it as read-only.
_HEALTH_KEY = "wide_ep_ft_ep_group_health"
_TIMEOUT_KEY = "alltoall_watchdog_timeout_s"
_POLL_INTERVAL_KEY = "alltoall_watchdog_poll_interval_s"


def _env_enabled() -> bool:
    return os.environ.get(_ENABLE_ENV) == "1"


def _float_option(extra_attrs: dict, key: str, env_name: str, default: float) -> float:
    if key in extra_attrs:
        return float(extra_attrs[key])
    if env_name in os.environ:
        return float(os.environ[env_name])
    return default


def get_wide_ep_ft_options(
    model_config: Any,
) -> tuple[Optional[EPGroupHealth], Optional[float], float]:
    """Return committed EP membership and watchdog timing for a model.

    Callers can inject ``wide_ep_ft_ep_group_health`` through
    ``ModelConfig.extra_attrs``, set ``TLLM_FAULT_TOLERANCE_MODE=1`` or turn on
    ``enable_wide_ep_fault_tolerance`` to create one process-local membership
    object shared by all MoE communication layers.
    The AlltoAll watchdog reads this object to determine expected peers and
    reports suspects through its ``on_timeout`` seam; it never mutates the
    committed membership directly. With ``enable_wide_ep_fault_tolerance`` the
    timeout is None: no watchdog runs, as the membership authority reads the
    kernels' wait evidence. The kernels read the mask at execution time, so
    rank-mask mode works with CUDA graphs once each communicator is bound to an
    execution state, which ``enable_wide_ep_fault_tolerance`` does; without it,
    a model that uses CUDA graphs is refused here.
    """

    extra_attrs = getattr(model_config, "extra_attrs", {})
    health = extra_attrs.get(_HEALTH_KEY) or extra_attrs.get("ep_group_health")
    rank_mask_enabled = health is not None or _env_enabled() or fault_tolerance_mode()
    if (
        rank_mask_enabled
        and getattr(model_config, "use_cuda_graph", False)
        and not fault_tolerance_mode()
    ):
        # Only enable_wide_ep_fault_tolerance binds the communicators to an execution state.
        raise ValueError(
            "WideEP rank-mask mode does not support CUDA graphs without "
            "enable_wide_ep_fault_tolerance; disable CUDA graphs or turn it on"
        )
    if health is None and rank_mask_enabled:
        health = EPGroupHealth(model_config.mapping.moe_ep_size)
        if fault_tolerance_replacement() is not None:
            # Isolated startup: the all-to-all reaches no peer until the admission installs the membership.
            for rank in range(health.moe_world_size):
                if rank != model_config.mapping.moe_ep_rank:
                    health.mark_failed(rank)
        extra_attrs[_HEALTH_KEY] = health

    poll_interval_s = _float_option(
        extra_attrs,
        _POLL_INTERVAL_KEY,
        _POLL_INTERVAL_ENV,
        DEFAULT_ALLTOALL_WATCHDOG_POLL_INTERVAL_S,
    )
    if health is None or fault_tolerance_mode():
        # With enable_wide_ep_fault_tolerance the membership authority reads the kernels' wait evidence:
        # no watchdog.
        return health, None, poll_interval_s

    timeout_s = _float_option(
        extra_attrs,
        _TIMEOUT_KEY,
        _TIMEOUT_ENV,
        DEFAULT_ALLTOALL_WATCHDOG_TIMEOUT_S,
    )
    return health, timeout_s, poll_interval_s

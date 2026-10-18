# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WideEP fault-tolerance settings, read and validated once per process.

``TLLM_FAULT_TOLERANCE_MODE``, ``TLLM_FAULT_TOLERANCE_AUTHORITY`` and ``TLLM_FAULT_TOLERANCE_AUTHORITY_KEY`` turn
fault tolerance on and reach the recovery authority. Every other setting is a field of
:class:`FaultToleranceConfig`, loaded from the YAML or JSON file that ``TLLM_FAULT_TOLERANCE_CONFIG`` names
(defaults without it). The workers and the authority read the same file; the authority refuses a worker whose
settings differ (:meth:`FaultToleranceConfig.digest`).
"""

import functools
import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import ConfigDict, Field, PositiveFloat, model_validator

from tensorrt_llm.llmapi.utils import StrictBaseModel

__all__ = ["CONFIG_ENV", "FaultToleranceConfig", "fault_tolerance_config"]

CONFIG_ENV = "TLLM_FAULT_TOLERANCE_CONFIG"
# How long the workers' own fencing (the hung rank's evidence thread, then a node-mate) gets before the
# authority's fence command runs.
COMMAND_FENCE_GRACE_S = 5.0


class FaultToleranceConfig(StrictBaseModel):
    """WideEP fault-tolerance settings."""

    model_config = ConfigDict(frozen=True)

    extra_slots: Literal["empty", "borrowed"] = Field(
        default="empty",
        description="EPLB slots beyond one per expert, with a static placement: ``empty`` (routing never uses "
        "them; a restore fills them) or ``borrowed`` (also without device memory until a restore, the KV cache can "
        "use it). Online EPLB keeps replicas in them; setting this with online EPLB is an error.",
    )
    recovery_timeout_s: PositiveFloat = Field(
        default=30.0,
        description="How long a worker's recovery may take, including finishing a long step first.",
    )
    authority_timeout_s: PositiveFloat = Field(
        default=20.0,
        description="Seconds from the first ``Stopped`` to ``Commit`` at the authority; below "
        "``recovery_timeout_s`` so that workers observe its ``Abort`` rather than their own timeout.",
    )
    hang_detection: bool = Field(
        default=False,
        description="Workers report heartbeats and long all-to-all waits; the authority confirms a hung rank and "
        "fences it.",
    )
    hang_suspect_s: PositiveFloat = Field(
        default=2.0, description="A worker reports a device wait after this long."
    )
    hang_confirm_s: PositiveFloat = Field(
        default=30.0,
        description="How long the evidence of a hang must hold before the authority confirms it; above the "
        "slowest legitimate step plus host stalls.",
    )
    heartbeat_s: PositiveFloat = Field(
        default=1.0,
        description="Worker heartbeat interval.",
    )
    heartbeat_timeout_s: PositiveFloat = Field(
        default=5.0,
        description="The authority counts a worker without a heartbeat for this long as missing.",
    )

    @model_validator(mode="after")
    def _check_order(self) -> "FaultToleranceConfig":
        if self.authority_timeout_s >= self.recovery_timeout_s:
            raise ValueError("authority_timeout_s must be below recovery_timeout_s")
        if self.heartbeat_s >= self.heartbeat_timeout_s:
            raise ValueError("heartbeat_s must be below heartbeat_timeout_s")
        return self

    def hang_resolution_s(self) -> float:
        """The longest a confirmed hang can take until a fence ends the hung rank (before recovery starts)."""
        return self.hang_suspect_s + self.hang_confirm_s + COMMAND_FENCE_GRACE_S

    def digest(self) -> str:
        """A short hash of these settings; the workers and the authority must hold the same one."""
        settings = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(settings.encode()).hexdigest()[:16]


@functools.cache
def fault_tolerance_config() -> FaultToleranceConfig:
    """The settings of this process (from ``TLLM_FAULT_TOLERANCE_CONFIG``, else the defaults).

    Raises:
        ValueError: The file does not hold a valid configuration.
    """
    path = os.environ.get(CONFIG_ENV)
    if not path:
        return FaultToleranceConfig()
    fields = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(fields, dict):
        raise ValueError(f"{CONFIG_ENV}={path} must hold a mapping of settings")
    return FaultToleranceConfig(**fields)

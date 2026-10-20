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
from typing import Literal, Optional

import yaml
from pydantic import ConfigDict, Field, PositiveFloat, model_validator

from tensorrt_llm.llmapi.utils import StrictBaseModel

__all__ = ["CONFIG_ENV", "STALLED_EXIT_CODE", "FaultToleranceConfig", "fault_tolerance_config"]

CONFIG_ENV = "TLLM_FAULT_TOLERANCE_CONFIG"
# A worker asks the MPI runtime to kill a hung rank this long after it received the authority's Fence, unless the
# hung rank's own evidence thread ended its process first.
RUNTIME_FENCE_GRACE_S = 1.0
# How long the MPI runtime may take to kill a rank before the authority's fence command takes over.
RUNTIME_FENCE_CALL_S = 2.0
# How long the workers' own fencing gets before the authority's fence command runs: a heartbeat interval until
# the workers receive the Fence, then the runtime fence's grace and call.
COMMAND_FENCE_GRACE_S = 5.0
# The exit code of a worker the authority told that the deployment hangs and no fence can recover it (EX_TEMPFAIL):
# the deployment must restart.
STALLED_EXIT_CODE = 75


class FaultToleranceConfig(StrictBaseModel):
    """WideEP fault-tolerance settings."""

    model_config = ConfigDict(frozen=True)

    extra_slots: Optional[Literal["empty", "ghost"]] = Field(
        default=None,
        description="EPLB slots beyond one per expert: ``ghost`` (no device memory until a restore maps it, so the "
        "KV cache keeps that memory while healthy) or ``empty`` (memory held, routing never uses them; a restore "
        "fills them). Default: ``ghost`` with a static placement, ``empty`` with online EPLB, whose weight updates "
        "load ghost slot memory with GPU copies. Online EPLB keeps replicas in its slots; where they cannot hold "
        "every expert after the loss of a failure domain, each rank gets as many slots of this kind after them as "
        "that takes. A restore fills them only where replicas cannot, and EPLB's updates use a slot once a restore "
        "filled it.",
    )
    failure_domain: Literal["rank", "node"] = Field(
        default="node",
        description="The largest loss to recover from: every rank of one host (rank 0's excluded), or one rank. "
        "Startup refuses an EPLB placement whose slots cannot serve every expert without such a loss.",
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
        "fences it, or reports a stall of every rank as terminal.",
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
        description="Worker heartbeat interval, at most 2 s: a worker receives the authority's ``Fence`` within "
        "one, and its runtime fence must end before the authority's fence command starts.",
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
        if self.heartbeat_s + RUNTIME_FENCE_GRACE_S + RUNTIME_FENCE_CALL_S > COMMAND_FENCE_GRACE_S:
            limit = COMMAND_FENCE_GRACE_S - RUNTIME_FENCE_GRACE_S - RUNTIME_FENCE_CALL_S
            raise ValueError(
                f"heartbeat_s must be at most {limit}s, so that the workers' runtime fence ends before the "
                "authority's fence command starts"
            )
        return self

    def hang_resolution_s(self) -> float:
        """The longest a hang or a stall of every rank can take until it is acted on (before recovery starts).

        Peers wait on a hung rank once their device wait is reported or its heartbeat is missing; the evidence
        must then hold for ``hang_confirm_s``, and the fence command runs last.
        """
        return (
            max(self.hang_suspect_s, self.heartbeat_timeout_s)
            + self.hang_confirm_s
            + COMMAND_FENCE_GRACE_S
        )

    def shrink_timeout_s(self) -> float:
        """The longest a survivor may wait from detecting a loss until the survivors agree on the failed ranks.

        The agreement needs every survivor. One can detect a hung peer before the others (Open MPI's in-band
        failure detector declares a stopped process failed within seconds), while the others stay blocked on
        that peer until the authority confirms the hang and fences it.
        """
        return (self.hang_resolution_s() if self.hang_detection else 0.0) + self.recovery_timeout_s

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

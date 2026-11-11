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
"""Whether WideEP fault tolerance is on in this process, and the rank identity its launcher gives it."""

import os
from functools import lru_cache
from typing import NamedTuple, Optional, Sequence, Tuple

# "<rank>,<world size>,<GPUs per node>,<replacement>", set for every rank under WideEP fault tolerance.
FAULT_TOLERANCE_RANK_ENV = "TLLM_FAULT_TOLERANCE_RANK"

_enabled = False


def set_wide_ep_fault_tolerance_enabled(enabled: bool) -> None:
    """Turn WideEP fault tolerance on or off in this process."""
    global _enabled
    _enabled = enabled


def wide_ep_fault_tolerance_enabled() -> bool:
    """Return whether WideEP fault tolerance is on."""
    return _enabled


def check_fault_tolerance_requirements(checks: Sequence[Tuple[bool, str]]) -> None:
    """Raise ValueError naming every unmet ``(met, requirement)`` check."""
    missing = [requirement for met, requirement in checks if not met]
    if missing:
        raise ValueError(
            "WideEP fault tolerance requires "
            + "; ".join(missing)
            + "; turn it off (enable_wide_ep_fault_tolerance=False) or change the configuration"
        )


class RankIdentity(NamedTuple):
    """A rank under WideEP fault tolerance, where each rank is its own MPI world."""

    rank: int
    world_size: int
    gpus_per_node: int
    replacement: bool


def wide_ep_fault_tolerance_rank() -> Optional[RankIdentity]:
    """Return the rank identity the launcher gave this process, or None."""
    return _parse_rank_identity(os.environ.get(FAULT_TOLERANCE_RANK_ENV))


def fault_tolerance_replacement() -> Optional[RankIdentity]:
    """Return the rank this process replaces, or None for an original rank."""
    identity = wide_ep_fault_tolerance_rank() if _enabled else None
    return identity if identity is not None and identity.replacement else None


@lru_cache(maxsize=1)
def _parse_rank_identity(value: Optional[str]) -> Optional[RankIdentity]:
    if value is None:
        return None
    rank, world_size, gpus_per_node, replacement = (int(field) for field in value.split(","))
    return RankIdentity(rank, world_size, gpus_per_node, bool(replacement))

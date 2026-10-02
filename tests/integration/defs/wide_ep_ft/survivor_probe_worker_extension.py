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
"""Test-only Ray worker extension for process identity and survivor RPC probes."""

from __future__ import annotations

import os

from fault_injector import capture_process_identity
from survivor_probe import probe_digest, validate_probe_proposal


class SurvivorProbeWorkerExtension:
    def probe_identity(self) -> dict:
        import ray
        import torch.distributed as distributed

        rank = distributed.get_rank()
        return {
            **capture_process_identity(rank, os.getpid()),
            "ray_node_id": ray.get_runtime_context().get_node_id(),
        }

    def prepare_survivor_probe(self, proposal: dict) -> dict:
        import torch.distributed as distributed

        rank = distributed.get_rank()
        validate_probe_proposal(proposal, rank)
        digest = probe_digest(proposal)
        previous_digest = getattr(self, "_survivor_probe_digest", None)
        if previous_digest is not None and previous_digest != digest:
            raise ValueError("conflicting diagnostic proposal")
        self._survivor_probe_digest = digest
        return {
            "rank": rank,
            "pid": os.getpid(),
            "digest": digest,
            "base_generation": proposal["base_generation"],
            "next_generation": proposal["next_generation"],
        }

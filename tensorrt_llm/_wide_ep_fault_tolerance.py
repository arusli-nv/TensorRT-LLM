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
"""Whether Wide-EP fault tolerance is on in this process, and the rank identity its launcher gives it."""

import os
from functools import lru_cache
from typing import TYPE_CHECKING, NamedTuple, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from .llmapi.llm_args import TorchLlmArgs

# Set for every rank under Wide-EP fault tolerance to "<rank>,<world size>,<GPUs per node>,<replacement>".
FAULT_TOLERANCE_RANK_ENV = "TLLM_FAULT_TOLERANCE_RANK"

_enabled = False


def set_wide_ep_fault_tolerance_enabled(enabled: bool) -> None:
    """Turn Wide-EP fault tolerance on or off in this process."""
    global _enabled
    _enabled = enabled


def wide_ep_fault_tolerance_enabled() -> bool:
    """Return whether Wide-EP fault tolerance is on."""
    return _enabled


def check_fault_tolerance_requirements(checks: Sequence[Tuple[bool, str]]) -> None:
    """Raise ValueError naming every unmet ``(met, requirement)`` check."""
    missing = [requirement for met, requirement in checks if not met]
    if missing:
        raise ValueError(
            "Wide-EP fault tolerance requires "
            + "; ".join(missing)
            + "; turn it off (enable_wide_ep_fault_tolerance=False) or change the configuration"
        )


class RankIdentity(NamedTuple):
    """A rank under Wide-EP fault tolerance, where each rank is its own MPI world."""

    rank: int
    world_size: int
    gpus_per_node: int
    replacement: bool

    def env_value(self) -> str:
        """Return the value of ``FAULT_TOLERANCE_RANK_ENV`` that gives a process this identity."""
        return ",".join(str(int(field)) for field in self)


def wide_ep_fault_tolerance_rank() -> Optional[RankIdentity]:
    """Return the rank identity the launcher gave this process, or None without fault tolerance."""
    return _parse_rank_identity(os.environ.get(FAULT_TOLERANCE_RANK_ENV)) if _enabled else None


def fault_tolerance_replacement() -> Optional[RankIdentity]:
    """Return the rank this process replaces, or None for an original rank."""
    identity = wide_ep_fault_tolerance_rank()
    return identity if identity is not None and identity.replacement else None


@lru_cache(maxsize=1)
def _parse_rank_identity(value: Optional[str]) -> Optional[RankIdentity]:
    if value is None:
        return None
    rank, world_size, gpus_per_node, replacement = (int(field) for field in value.split(","))
    return RankIdentity(rank, world_size, gpus_per_node, bool(replacement))


def check_wide_ep_fault_tolerance_args(args: "TorchLlmArgs") -> None:
    """Raise ValueError naming every setting of ``args`` that Wide-EP fault tolerance does not support."""
    adp_config = args.attention_dp_config
    spec_mode = None if args.speculative_config is None else args.speculative_config.spec_dec_mode
    transceiver = args.cache_transceiver_config
    load_balancer = args.moe_config.load_balancer
    checks = [
        (
            args.enable_attention_dp,
            "attention DP (enable_attention_dp=True), so that a failed rank takes only its own "
            "requests with it",
        ),
        (
            args.pipeline_parallel_size == 1,
            "no pipeline parallelism (pipeline_parallel_size=1), because a failed pipeline stage "
            "leaves its layers on no rank",
        ),
        (
            args.max_beam_width in (None, 1),
            "no beam search (max_beam_width=1), which is not supported yet: continuing an "
            "interrupted request by recompute would need to restore its beams",
        ),
        (
            not args.disable_overlap_scheduler,
            "the overlap scheduler (disable_overlap_scheduler=False), because Wide-EP fault "
            "tolerance accepts or discards each step when the next one starts, which only the "
            "overlap scheduler's executor loop does yet",
        ),
        (
            load_balancer is not None,
            "EPLB (moe_config.load_balancer with num_slots), whose spare slots take a failed "
            "rank's experts",
        ),
        (
            load_balancer is None or load_balancer.mode == "standard",
            "standard EPLB (moe_config.load_balancer.mode='standard'), because per-layer dynamic "
            "EPLB refills its helper slots in every forward pass and has no spare slots for a "
            "failed rank's experts",
        ),
        (
            adp_config is None
            or not (
                adp_config.enable_kv_balanced_admission
                or adp_config.enable_kv_cache_aware_routing
                or adp_config.kv_cache_routing_conversation_affinity
            ),
            "the default attention-DP routing and admission (no enable_kv_balanced_admission, "
            "enable_kv_cache_aware_routing or kv_cache_routing_conversation_affinity in "
            "attention_dp_config), because the others keep per-rank state that a replacement "
            "would not share",
        ),
        (
            args.kv_cache_config.event_buffer_max_size == 0,
            "no KV cache events (kv_cache_config.event_buffer_max_size=0), which are not "
            "supported yet: a failed rank's KV cache blocks would need removal events",
        ),
        (
            transceiver is None or transceiver.backend is None,
            "no disaggregated serving (no cache_transceiver_config.backend), because the KV cache "
            "transfers between serving instances do not follow a membership change",
        ),
        (
            args.kv_connector_config is None,
            "no KV cache connector (kv_connector_config=None), because a connector saves the KV "
            "cache of a step before the step is accepted",
        ),
        (
            spec_mode is None or not (spec_mode.has_spec_drafter() or spec_mode.is_dspark()),
            "no NGram or user-provided drafter and no DSpark (speculative_config: none, or another "
            "decoding_type such as MTP), which are not supported yet: their drafting state outside "
            "the model forward is not rolled back when a rank failure discards a step",
        ),
        (
            not args.enable_lm_head_tp_in_adp,
            "no LM head TP in attention DP (enable_lm_head_tp_in_adp=False), because a failed "
            "rank would take its share of the vocabulary with it",
        ),
        (
            args.torch_compile_config is None or not args.torch_compile_config.enable_userbuffers,
            "no userbuffers (torch_compile_config.enable_userbuffers=False), because their setup "
            "exchanges memory handles over an MPI world of every rank, and each rank is its own "
            "MPI world",
        ),
        (
            args.sleep_config is None,
            "no sleep mode (sleep_config=None), because sleep releases and restores the GPU "
            "memory of the weights, whose spare slots fault tolerance backs and unbacks itself",
        ),
        (
            args.num_serve_frontends == 1,
            "one serving frontend (num_serve_frontends=1), which is not supported yet with more: "
            "the frontends that attach to the executor do not follow the membership, as /health "
            "and the re-run of a failed rank's requests need",
        ),
        (
            args.orchestrator_type is None,
            "the MPI executor (orchestrator_type unset, not 'rpc' or 'ray'), because the "
            "membership, the rank supervisors and the executor proxy that follows them come with "
            "the MPI executor only",
        ),
    ]
    check_fault_tolerance_requirements(checks)

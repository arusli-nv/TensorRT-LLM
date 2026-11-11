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
"""Tests for the configurations that enable_wide_ep_fault_tolerance rejects when the LLM arguments are built."""

import re

import pytest

from tensorrt_llm.llmapi.llm_args import (
    AttentionDpConfig,
    CacheTransceiverConfig,
    DSparkDecodingConfig,
    KvCacheConfig,
    KvCacheConnectorConfig,
    MTPDecodingConfig,
    NGramDecodingConfig,
    SleepConfig,
    TorchCompileConfig,
    TorchLlmArgs,
)

pytestmark = pytest.mark.cpu_only


def _args(**fields) -> TorchLlmArgs:
    """Return a supported configuration updated with ``fields``."""
    return TorchLlmArgs(
        **{
            "model": "dummy",
            "enable_wide_ep_fault_tolerance": True,
            "enable_attention_dp": True,
            "moe_config": {"load_balancer": {"num_slots": 8}},
            **fields,
        }
    )


@pytest.mark.parametrize(
    "fields",
    [
        {},
        # The balance counters follow per-step gathers over the serving ranks only.
        {
            "attention_dp_config": AttentionDpConfig(
                enable_balance=True, batching_wait_iters=0, timeout_iters=60
            )
        },
        {"speculative_config": MTPDecodingConfig(num_nextn_predict_layers=1)},
        # Under fault tolerance no modeling_v2 target is resolved, so "require" fails later, when the model loads.
        # The default "auto" is covered by the empty case above.
        {"modeling_v2": "require"},
        {"torch_compile_config": TorchCompileConfig(enable_userbuffers=False)},
        {"prefill_cuda_graph_backend": "piecewise"},
    ],
)
def test_a_supported_configuration_is_accepted(fields):
    assert _args(**fields).enable_wide_ep_fault_tolerance


@pytest.mark.parametrize(
    "fields, fix",
    [
        ({"enable_attention_dp": False}, "enable_attention_dp=True"),
        ({"pipeline_parallel_size": 2}, "pipeline_parallel_size=1"),
        ({"max_beam_width": 2}, "max_beam_width=1"),
        ({"disable_overlap_scheduler": True}, "disable_overlap_scheduler=False"),
        ({"moe_config": {}}, "moe_config.load_balancer with num_slots"),
        # Per-layer dynamic EPLB is EPLB too, but without spare slots.
        (
            {"moe_config": {"load_balancer": {"mode": "per_layer_dynamic", "num_slots": 16}}},
            "moe_config.load_balancer.mode='standard'",
        ),
        *(
            (
                {"attention_dp_config": AttentionDpConfig(**{option: True})},
                "no enable_kv_balanced_admission, enable_kv_cache_aware_routing or "
                "kv_cache_routing_conversation_affinity in attention_dp_config",
            )
            for option in (
                "enable_kv_balanced_admission",
                "enable_kv_cache_aware_routing",
                "kv_cache_routing_conversation_affinity",
            )
        ),
        (
            {"kv_cache_config": KvCacheConfig(event_buffer_max_size=1024)},
            "kv_cache_config.event_buffer_max_size=0",
        ),
        (
            {"cache_transceiver_config": CacheTransceiverConfig(backend="DEFAULT")},
            "no disaggregated serving (no cache_transceiver_config.backend)",
        ),
        (
            {
                "kv_connector_config": KvCacheConnectorConfig(
                    connector_module="connector",
                    connector_scheduler_class="Scheduler",
                    connector_worker_class="Worker",
                )
            },
            "kv_connector_config=None",
        ),
        (
            {"speculative_config": NGramDecodingConfig(max_draft_len=1)},
            "speculative_config: none, or another decoding_type such as MTP",
        ),
        ({"enable_lm_head_tp_in_adp": True}, "enable_lm_head_tp_in_adp=False"),
        # torch.compile enables userbuffers by default.
        (
            {"torch_compile_config": TorchCompileConfig()},
            "torch_compile_config.enable_userbuffers=False",
        ),
        ({"sleep_config": SleepConfig()}, "sleep_config=None"),
        ({"num_serve_frontends": 2}, "num_serve_frontends=1"),
        ({"orchestrator_type": "rpc"}, "orchestrator_type unset"),
        ({"orchestrator_type": "ray"}, "orchestrator_type unset"),
    ],
)
def test_unsupported_configuration_names_field(fields, fix):
    with pytest.raises(ValueError, match=re.escape(fix)) as error:
        _args(**fields)
    assert "enable_wide_ep_fault_tolerance=False" in str(error.value)


def test_dspark_is_rejected(tmp_path):
    (tmp_path / "config.json").write_text('{"dspark_block_size": 2}')
    with pytest.raises(
        ValueError, match="no NGram or user-provided drafter and no DSpark"
    ) as error:
        _args(
            speculative_config=DSparkDecodingConfig(
                max_draft_len=2, speculative_model=str(tmp_path)
            )
        )
    assert "decoding_type such as MTP" in str(error.value)
    assert "enable_wide_ep_fault_tolerance=False" in str(error.value)


def test_fault_tolerance_off_skips_checks():
    assert not _args(
        enable_wide_ep_fault_tolerance=False,
        disable_overlap_scheduler=True,
        num_serve_frontends=2,
        sleep_config=SleepConfig(),
    ).enable_wide_ep_fault_tolerance

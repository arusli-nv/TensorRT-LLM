# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Reject invalid deployment configuration, provenance and readiness evidence."""

import copy
from pathlib import Path

import pytest
import yaml
from deployment_profile import (
    PROMPTS,
    expert_coverage_report,
    file_sha256,
    fixed_assignments,
    read_placements,
    single_rank_coverage,
    validate_execution_log,
    validate_hardware,
    validate_healthy,
    validate_profile,
    validate_ray_version,
    validate_resolved_profile,
    validate_runtime,
    verify_placements,
)


@pytest.fixture
def profile() -> tuple[dict, dict]:
    config = yaml.safe_load(Path(__file__).with_name("deepseek_r1_ep32.yaml").read_text())
    config["moe_config"]["load_balancer"].update(
        num_slots=512,
        layer_updates_per_iter=0,
        initial_global_assignments={
            layer: fixed_assignments(256, 32, 512) for layer in range(3, 61)
        },
    )
    config["cuda_graph_config"] = {"batch_sizes": [1, 2, 4, 8], "enable_padding": True}
    model = {
        "model_type": "deepseek_v3",
        "n_routed_experts": 256,
        "num_hidden_layers": 61,
        "first_k_dense_replace": 3,
    }
    return config, model


@pytest.mark.parametrize(
    "fault",
    [
        "empty",
        "missing_layer",
        "wrong_slots",
        "missing_expert",
        "graphs_off",
        "wrong_topology",
        "wrong_backend",
        "prefill_graphs",
    ],
)
def test_unsupported_deployment_is_rejected(profile: tuple[dict, dict], fault: str) -> None:
    """Can unsupported placement, topology or execution settings qualify a deployment?"""
    config, model = copy.deepcopy(profile)
    load_balancer = config["moe_config"]["load_balancer"]
    if fault == "empty":
        load_balancer["initial_global_assignments"] = {}
    elif fault == "missing_layer":
        del load_balancer["initial_global_assignments"][20]
    elif fault == "wrong_slots":
        load_balancer["num_slots"] = 288
    elif fault == "missing_expert":
        load_balancer["initial_global_assignments"][20] = [0] * 512
    elif fault == "graphs_off":
        config["cuda_graph_config"] = None
    elif fault == "wrong_topology":
        config["moe_expert_parallel_size"] = 16
    elif fault == "wrong_backend":
        config["moe_config"]["backend"] = "TRTLLM"
    else:
        config["cuda_graph_config"]["mode"] = "prefill"
    with pytest.raises(ValueError):
        validate_profile(config, model)


@pytest.mark.parametrize(
    "fault",
    ["wrong_uuid", "premature", "missing_result", "wrong_prompt", "empty_tokens", "wrong_answer"],
)
def test_invalid_readiness_cannot_authorize_injection(fault: str) -> None:
    """Can stale, incomplete or incorrect healthy output authorize fault injection?"""
    record = {
        "run_id": "current",
        "state": "paused_between_requests",
        "results": [
            {"prompt": prompt, "text": answer, "token_ids": [1]}
            for prompt, answer in zip(PROMPTS, ("Paris", "Rome"))
        ],
    }
    if fault == "wrong_uuid":
        record["run_id"] = "old"
    elif fault == "premature":
        record["state"] = "loading"
    elif fault == "missing_result":
        record["results"].pop()
    elif fault == "wrong_prompt":
        record["results"][0]["prompt"] = "unrelated"
    elif fault == "empty_tokens":
        record["results"][0]["token_ids"] = []
    else:
        record["results"][0]["text"] = "wrong"
    with pytest.raises(ValueError):
        validate_healthy(record, "current")


@pytest.mark.parametrize("fault", ["image_unbound", "runtime_changed", "payload_unverified"])
def test_runtime_provenance_mismatch_is_rejected(fault: str) -> None:
    """Can an unbound image, changed runtime or unverified payload pass provenance checks?"""
    manifest = {
        "tensorrt_llm_version": "1.4.0rc0",
        "torch_version": "qualified",
        "torch_cuda": "13.4",
        "nccl_runtime_version": [2, 30, 7],
        "architecture": "aarch64",
        "source_commit": "a" * 40,
        "runtime_image_sha256": "b" * 64,
        "installed_payload_verified_file_count": 13124,
    }
    runtime = dict(manifest)
    if fault == "image_unbound":
        del manifest["runtime_image_sha256"]
    elif fault == "runtime_changed":
        runtime["torch_cuda"] = "13.0"
    else:
        manifest["installed_payload_verified_file_count"] = 0
    with pytest.raises(ValueError):
        validate_runtime(runtime, manifest)


def test_image_hash_changes_with_payload(tmp_path: Path) -> None:
    """Does changing the image payload change its provenance hash?"""
    image = tmp_path / "image.sqsh"
    image.write_bytes(b"verified payload")
    expected = file_sha256(image)
    image.write_bytes(b"different payload")
    assert file_sha256(image) != expected


@pytest.mark.parametrize("fault", ["topology", "graph_override", "nested_override"])
def test_additional_unqualified_options_are_rejected(
    profile: tuple[dict, dict], fault: str
) -> None:
    """Can extra topology, graph or nested options bypass the qualified profile?"""
    config, model = profile
    if fault == "topology":
        config["pipeline_parallel_size"] = 2
    elif fault == "graph_override":
        config["cuda_graph_config"]["max_batch_size"] = 1
    else:
        config["moe_config"]["unsupported"] = True
    with pytest.raises(ValueError, match="Unqualified"):
        validate_profile(config, model)


@pytest.mark.parametrize(
    "fault", ["stale", "graphs_off", "graph_capacity", "pp_loss", "cp_loss", "placement_changed"]
)
def test_resolved_execution_must_match_admission(profile: tuple[dict, dict], fault: str) -> None:
    """Can normalized runtime arguments drift from the admitted deployment?"""
    config, model = profile
    resolved = copy.deepcopy(config)
    resolved["cuda_graph_config"].update(mode="decode", max_batch_size=8)
    resolved["pipeline_parallel_size"] = 1
    resolved["context_parallel_size"] = 1
    record = {"run_id": "current", "configuration": resolved}
    if fault == "stale":
        record["run_id"] = "old"
    elif fault == "graphs_off":
        resolved["cuda_graph_config"] = None
    elif fault == "graph_capacity":
        resolved["cuda_graph_config"]["max_batch_size"] = 1
    elif fault == "pp_loss":
        resolved["pipeline_parallel_size"] = 2
    elif fault == "cp_loss":
        resolved["context_parallel_size"] = 2
    else:
        resolved["moe_config"]["load_balancer"]["layer_updates_per_iter"] = 1
    with pytest.raises(ValueError):
        validate_resolved_profile(record, "current", config, model)


@pytest.mark.parametrize("fault", ["duplicate_host", "duplicate_gpu", "wrong_driver"])
def test_hardware_coverage_rejects_duplicate_resources(fault: str) -> None:
    """Can duplicate hosts or GPUs, or a different driver, pass hardware qualification?"""
    hosts = {f"node-{node}" for node in range(8)}
    reports = [
        {
            "hostname": f"node-{node}",
            "gpu_memory_csv": "\n".join(
                f"NVIDIA GB200, GPU-{node}-{gpu}, 580.126.20, 189471, 57000, 132471"
                for gpu in range(4)
            ),
        }
        for node in range(8)
    ]
    if fault == "duplicate_host":
        reports[1]["hostname"] = reports[0]["hostname"]
    elif fault == "duplicate_gpu":
        reports[1]["gpu_memory_csv"] = reports[0]["gpu_memory_csv"]
    else:
        reports[1]["gpu_memory_csv"] = reports[1]["gpu_memory_csv"].replace(
            "580.126.20", "615.71.09"
        )
    with pytest.raises(ValueError):
        validate_hardware(
            reports,
            hosts,
            {
                "tensor_parallel_size": 32,
                "moe_expert_parallel_size": 32,
                "moe_tensor_parallel_size": 1,
                "gpus_per_node": 4,
            },
            {"driver_report": "NVIDIA GB200, 580.126.20"},
        )


def test_healthy_288_slots_with_partial_failure_coverage_are_valid(
    profile: tuple[dict, dict],
) -> None:
    """Can healthy 288-slot placement qualify despite incomplete rank-loss redundancy?"""
    config, model = profile
    config["moe_config"]["load_balancer"]["num_slots"] = 288
    config["moe_config"]["load_balancer"]["initial_global_assignments"] = {
        layer: list(range(256)) + list(range(32)) for layer in range(3, 61)
    }
    validate_profile(config, model)


def test_custom_per_layer_layout_does_not_require_all_rank_redundancy(
    profile: tuple[dict, dict],
) -> None:
    """Can a fully covered custom layer qualify without redundancy across every rank?"""
    config, model = profile
    config["moe_config"]["load_balancer"]["initial_global_assignments"][20] = [
        expert for expert in range(256) for _ in range(2)
    ]
    validate_profile(config, model)


@pytest.mark.parametrize("version", [None, "2.54.0", "2.55.2"])
def test_unqualified_ray_version_is_rejected(version: str | None) -> None:
    """Can evidence from a different private actor-state API pass qualification?"""
    with pytest.raises(ValueError, match="2.55.1"):
        validate_ray_version(version)
    validate_ray_version("2.55.1")


@pytest.mark.parametrize("ep_size", [4, 8, 72])
def test_topology_is_derived_from_requested_configuration(profile, ep_size):
    """Can another EP size use the same structural validators without claiming qualification?"""
    config, model = profile
    config["tensor_parallel_size"] = config["moe_expert_parallel_size"] = ep_size
    slots = ((288 + ep_size - 1) // ep_size) * ep_size
    config["moe_config"]["load_balancer"].update(
        num_slots=slots,
        initial_global_assignments={
            layer: fixed_assignments(256, ep_size, slots) for layer in range(3, 61)
        },
    )
    validate_profile(config, model)


@pytest.mark.parametrize("sparse_step", [1, 2])
def test_qwen_checkpoint_layout(profile, sparse_step) -> None:
    """Can Qwen3 use the same profile checker without silently dropping sparse layers?"""
    config, _ = copy.deepcopy(profile)
    model = {
        "model_type": "qwen3_moe",
        "num_experts": 128,
        "num_hidden_layers": 94,
        "decoder_sparse_step": sparse_step,
    }
    config["moe_config"]["load_balancer"].update(
        num_slots=144,
        initial_global_assignments={layer: fixed_assignments(128, 16, 144) for layer in range(94)},
    )
    config.update(tensor_parallel_size=16, moe_expert_parallel_size=16)
    if sparse_step == 1:
        validate_profile(config, model)
    else:
        with pytest.raises(ValueError, match="Unsupported checkpoint"):
            validate_profile(config, model)


def test_nominal_copies_on_same_rank_do_not_prove_coverage() -> None:
    """Does losing a rank remove experts whose copies all reside on that rank?"""
    coverage = single_rank_coverage([0, 0, 1, 1, 2, 2, 3, 3], 4, 4)
    assert coverage == {0: [0], 1: [1], 2: [2], 3: [3]}


def test_missing_expert_is_rejected_before_failure() -> None:
    """Can an already incomplete placement enter rank-loss coverage analysis?"""
    with pytest.raises(ValueError, match="missing logical"):
        single_rank_coverage([0, 1, 0, 1], 3, 2)


@pytest.mark.parametrize("assignment", [[0, 1, -1, 2], [0, 1, 2, 3], [0, 1, True, 2]])
def test_invalid_expert_identity_is_rejected(assignment: list[int]) -> None:
    """Can out-of-range or boolean expert IDs enter coverage analysis?"""
    with pytest.raises(ValueError, match="invalid logical"):
        single_rank_coverage(assignment, 3, 2)


def test_middle_rank_loss_preserves_every_expert() -> None:
    """Does cyclic 512-slot placement preserve every expert after any single rank loss?"""
    assignment = fixed_assignments(256, 32, 512)
    assert len(assignment) == 512
    assert single_rank_coverage(assignment, 256, 32) == {rank: [] for rank in range(32)}
    assert assignment[7 * 16 : 8 * 16] == assignment[23 * 16 : 24 * 16]


def test_288_slots_compute_partial_coverage_and_512_cover_all_nonzero_ranks() -> None:
    """Do 288 and 512 slots report their distinct rank-loss coverage without admitting recovery?"""
    partial = fixed_assignments(256, 32, 288)
    assert partial == list(range(256)) + list(range(32))
    report = expert_coverage_report({3: partial, 60: partial}, 256, 32)
    assert report["eligible_ranks"] == [1, 2, 29, 30, 31]
    assert report["ineligible_ranks"] == list(range(3, 29))
    assert report["missing_experts_by_layer_and_removed_rank"][3][3] == [32, 33, 34, 35]
    full = fixed_assignments(256, 32, 512)
    full_report = expert_coverage_report({3: full, 60: full}, 256, 32)
    assert full_report["eligible_ranks"] == list(range(1, 32))
    assert full_report["ineligible_ranks"] == []
    assert "not full-model" in full_report["scope"]


def test_eligibility_intersects_layers_instead_of_union_or_first_layer() -> None:
    """Must an eligible rank preserve expert coverage in every MoE layer?"""
    partial = expert_coverage_report({3: [0, 1, 0, 2], 4: [0, 1, 2, 1]}, 3, 4)
    assert partial["eligible_ranks"] == []
    assert partial["ineligible_ranks"] == [1, 2, 3]
    assert partial["missing_experts_by_layer_and_removed_rank"][3][2] == []
    assert partial["missing_experts_by_layer_and_removed_rank"][4][1] == []


def test_colocated_copies_never_make_a_rank_eligible() -> None:
    """Can colocated expert copies make a rank eligible for removal?"""
    report = expert_coverage_report({3: [0, 0, 1, 1, 2, 2, 3, 3]}, 4, 4)
    assert report["eligible_ranks"] == [] and report["ineligible_ranks"] == [1, 2, 3]


def test_empty_layer_evidence_is_rejected() -> None:
    """Can absent MoE-layer evidence establish expert coverage?"""
    with pytest.raises(ValueError, match="at least one"):
        expert_coverage_report({}, 256, 32)


def test_fixed_slot_partition_is_explicit() -> None:
    """Can a slot count that does not divide evenly across ranks define placement?"""
    with pytest.raises(ValueError, match="uniform slot"):
        fixed_assignments(256, 32, 289)


def test_interleaved_slurm_chunks_are_reassembled() -> None:
    """Can interleaved Slurm log chunks reconstruct each worker placement?"""
    log = (
        "0: [RANK 0] initial_global_assignments (layer 3) = [0,\n"
        "1: [RANK 1] initial_global_assignments (layer 3) = [0, 1, 0, 1]\n"
        "0:  1, 0, 1]\n"
    )
    records = read_placements(log)
    verify_placements(records, {3: [0, 1, 0, 1]}, 2)


@pytest.mark.parametrize(
    "log",
    [
        "0: [RANK 0] initial_global_assignments (layer 3) = [0,",
        "0: [RANK 1] initial_global_assignments (layer 3) = [0, 1]",
        "0: [RANK 0] initial_global_assignments (layer 3) = [0, 1]\n"
        "0: [RANK 0] initial_global_assignments (layer 3) = [0, 1]",
    ],
)
def test_incomplete_mislabelled_or_duplicate_records_fail(log: str) -> None:
    """Can partial, mislabelled or duplicate placement records pass validation?"""
    with pytest.raises(ValueError):
        read_placements(log)


def test_missing_worker_is_not_inferred_from_other_workers() -> None:
    """Can another worker supply the placement evidence of a missing worker?"""
    with pytest.raises(ValueError, match="missing"):
        verify_placements({(0, 3): [0, 1, 0, 1]}, {3: [0, 1, 0, 1]}, 2)


def test_conflicting_worker_placement_fails() -> None:
    """Can a worker placement that differs from the plan pass validation?"""
    with pytest.raises(ValueError, match="differs"):
        verify_placements({(0, 3): [0, 1, 0, 1], (1, 3): [1, 0, 1, 0]}, {3: [0, 1, 0, 1]}, 2)


@pytest.fixture
def execution_case(profile) -> tuple[str, dict, dict]:
    config, model = copy.deepcopy(profile)
    config["cuda_graph_config"]["mode"] = "decode"
    assignments = config["moe_config"]["load_balancer"]["initial_global_assignments"][3]
    lines = []
    for rank in range(32):
        for layer in range(3, 61):
            lines.extend(
                [
                    f"{rank}: [RANK {rank}] initial_global_assignments (layer {layer}) = {assignments}",
                    f"{rank}: [RANK {rank}] Selected communication strategy: NVLinkOneSided (forced)",
                    f"{rank}: [RANK {rank}] CFT handle-based counted writes disabled for dispatch",
                ]
            )
        lines.append(
            f"{rank}: [RANK {rank}] [rank={rank}][purpose=final_executor] cuda_graph_capture: done in 2.0s"
        )
    return "\n".join(lines), config, model


@pytest.mark.parametrize(
    "marker", ["Selected communication strategy:", "CFT handle-based counted writes"]
)
def test_duplicate_rank_cannot_compensate_for_missing_execution_evidence(
    execution_case: tuple[str, dict, dict], marker: str
) -> None:
    """Can duplicate rank evidence conceal a missing communication or CFT record?"""
    log, config, model = execution_case
    lines = log.splitlines()
    replacement = next(line for line in lines if line.startswith("0:") and marker in line)
    lines = [replacement if line.startswith("1:") and marker in line else line for line in lines]
    with pytest.raises(ValueError, match="per-rank"):
        validate_execution_log("\n".join(lines), config, model)


def test_conflicting_execution_rank_labels_are_rejected(
    execution_case: tuple[str, dict, dict],
) -> None:
    """Can conflicting Slurm and native rank labels pass execution validation?"""
    log, config, model = execution_case
    log = log.replace("1: [RANK 1] Selected", "1: [RANK 0] Selected", 1)
    with pytest.raises(ValueError, match="rank labels"):
        validate_execution_log(log, config, model)


def test_missing_final_capture_is_rejected(execution_case: tuple[str, dict, dict]) -> None:
    """Can execution qualify without final graph capture on every worker?"""
    log, config, model = execution_case
    log = "\n".join(
        line
        for line in log.splitlines()
        if not (line.startswith("31:") and "cuda_graph_capture" in line)
    )
    with pytest.raises(ValueError, match="graph-capture"):
        validate_execution_log(log, config, model)


def test_worker_evidence_can_be_checked_before_shutdown(
    execution_case: tuple[str, dict, dict],
) -> None:
    """Can complete worker evidence establish deployment readiness before shutdown?"""
    log, config, model = execution_case
    evidence = validate_execution_log(log, config, model)
    assert evidence["rank_layer_records"] == 1856
    assert evidence["workers_with_final_graph_capture"] == list(range(32))


def test_graph_off_profile_is_rejected(
    execution_case: tuple[str, dict, dict],
) -> None:
    """Can a graph-off deployment qualify for the decode-graph profile?"""
    log, config, model = execution_case
    config["cuda_graph_config"] = None
    log = "\n".join(line for line in log.splitlines() if "cuda_graph_capture" not in line)
    with pytest.raises(ValueError, match="graph buckets"):
        validate_execution_log(log, config, model)


@pytest.mark.parametrize("value", ["false", "1.0", '"0"'])
def test_worker_placement_rejects_noninteger_expert_ids(value: str) -> None:
    """Can boolean, floating-point or string values identify logical experts?"""
    log = f"0: [RANK 0] initial_global_assignments (layer 3) = [{value}]"
    with pytest.raises(ValueError):
        read_placements(log)


def test_requested_partial_coverage_is_reported_without_recovery_admission() -> None:
    """Does matched placement retain a report of experts missing after each rank loss?"""
    assignments = [0, 1, 2, 0]
    records = {(rank, 3): assignments for rank in range(2)}
    verify_placements(records, {3: assignments}, 2)
    report = expert_coverage_report({3: assignments}, 3, 2)
    assert report["missing_experts_by_layer_and_removed_rank"] == {3: {0: [1], 1: [2]}}


def test_execution_evidence_reports_all_layer_coverage_without_claiming_recovery(
    execution_case: tuple[str, dict, dict],
) -> None:
    """Does execution evidence retain partial coverage and reject changed worker placement?"""
    log, config, model = execution_case
    original = list(range(256)) * 2
    partial = list(range(256)) + list(range(32))
    load_balancer = config["moe_config"]["load_balancer"]
    load_balancer["num_slots"] = 288
    load_balancer["initial_global_assignments"] = {layer: partial for layer in range(3, 61)}
    log = log.replace(str(original), str(partial))
    evidence = validate_execution_log(log, config, model)
    coverage = evidence["coverage_eligibility"]
    assert coverage["eligible_ranks"]
    assert coverage["ineligible_ranks"]
    assert 0 not in coverage["eligible_ranks"]
    assert len(coverage["missing_experts_by_layer_and_removed_rank"]) == 58
    assert "no capacity, recovery or CFT qualification" in evidence["scope"]
    changed = list(partial)
    changed[0], changed[1] = changed[1], changed[0]
    log = log.replace(
        f"0: [RANK 0] initial_global_assignments (layer 3) = {partial}",
        f"0: [RANK 0] initial_global_assignments (layer 3) = {changed}",
    )
    with pytest.raises(ValueError, match="differs"):
        validate_execution_log(log, config, model)

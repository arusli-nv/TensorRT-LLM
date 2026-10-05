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
"""Static placement and evidence checks for Ray WideEP test profiles."""

import collections
import hashlib
import json
import re
from pathlib import Path

PROMPTS = ("The capital of France is", "The capital of Italy is")


def file_sha256(path: Path) -> str:
    """Hash a file incrementally without loading a container image into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deployment_shape(config: dict) -> tuple[int, int, int]:
    """Derive model ranks, node count and GPUs per node for attention-DP / expert-TP1."""
    ranks, gpus = config["moe_expert_parallel_size"], config["gpus_per_node"]
    if (
        any(type(value) is not int or value <= 0 for value in (ranks, gpus))
        or ranks < 2
        or ranks % gpus
    ):
        raise ValueError("Deployment requires uniform whole-node EP placement")
    if config["tensor_parallel_size"] != ranks or config["moe_tensor_parallel_size"] != 1:
        raise ValueError("Deployment requires attention DP and unsharded experts")
    return ranks, ranks // gpus, gpus


def moe_layout(model: dict) -> tuple[int, range]:
    """Read routed experts and contiguous MoE layers from supported checkpoint schemas."""
    if model.get("model_type") == "deepseek_v3" and model.get("moe_layer_freq", 1) == 1:
        experts, first = model["n_routed_experts"], model["first_k_dense_replace"]
    elif (
        model.get("model_type") == "qwen3_moe"
        and model.get("decoder_sparse_step", 1) == 1
        and not model.get("mlp_only_layers")
    ):
        experts, first = model["num_experts"], 0
    else:
        raise ValueError("Unsupported checkpoint MoE layout")
    layers = model["num_hidden_layers"]
    if (
        any(type(value) is not int for value in (experts, first, layers))
        or experts <= 0
        or not 0 <= first < layers
    ):
        raise ValueError("Invalid checkpoint expert or layer count")
    return experts, range(first, layers)


def validate_profile(config: dict, model: dict) -> None:
    """Check structural constraints; successful validation is not physical qualification."""
    experts, layers = moe_layout(model)
    ranks, _, _ = deployment_shape(config)
    fixed = {
        "moe_tensor_parallel_size": 1,
        "enable_attention_dp": True,
        "enable_lm_head_tp_in_adp": False,
        "disable_overlap_scheduler": True,
        "enable_autotuner": False,
    }
    variable = {
        "tensor_parallel_size",
        "moe_expert_parallel_size",
        "gpus_per_node",
        "max_batch_size",
        "max_num_tokens",
        "max_seq_len",
    }
    if set(config) != set(fixed) | variable | {
        "moe_config",
        "kv_cache_config",
        "cuda_graph_config",
    }:
        raise ValueError("Unqualified deployment configuration options")
    if any(config[key] != value for key, value in fixed.items()):
        raise ValueError("Unsupported execution profile")
    if any(type(config[key]) is not int or config[key] <= 0 for key in variable):
        raise ValueError("Positive integer capacities are required")
    moe, cache = config["moe_config"], config["kv_cache_config"]
    if set(moe) != {"backend", "load_balancer"} or set(cache) != {"free_gpu_memory_fraction"}:
        raise ValueError("Unqualified nested deployment options")
    if moe["backend"] != "CUTEDSL" or not 0 < cache["free_gpu_memory_fraction"] < 1:
        raise ValueError("Unsupported MoE backend or KV capacity")
    placement = moe["load_balancer"]
    if set(placement) != {"num_slots", "layer_updates_per_iter", "initial_global_assignments"}:
        raise ValueError("Unqualified load-balancer options")
    if placement["layer_updates_per_iter"] != 0:
        raise ValueError("Placement must be static")
    planned = {
        int(layer): value for layer, value in placement["initial_global_assignments"].items()
    }
    if not planned or set(planned) != set(layers):
        raise ValueError("Placement must cover every checkpoint MoE layer")
    for assignment in planned.values():
        if len(assignment) != placement["num_slots"]:
            raise ValueError("Layer placement differs from the slot count")
        single_rank_coverage(assignment, experts, ranks)
    graph = config["cuda_graph_config"]
    if isinstance(graph, dict) and set(graph) - {"batch_sizes", "enable_padding", "mode"}:
        raise ValueError("Unqualified CUDA graph options")
    buckets = graph.get("batch_sizes", []) if isinstance(graph, dict) else []
    if (
        not buckets
        or any(
            type(size) is not int or not 0 < size <= config["max_batch_size"] for size in buckets
        )
        or buckets != sorted(set(buckets))
        or graph.get("enable_padding") is not True
        or graph.get("mode", "decode") != "decode"
    ):
        raise ValueError("Profile requires decode graph buckets")


def validate_resolved_profile(record: dict, run_id: str, config: dict, model: dict) -> None:
    """Validate admitted settings after runtime normalization, allowing unrelated defaults."""
    if record.get("run_id") != run_id:
        raise ValueError("Stale resolved configuration")
    resolved = record.get("configuration", {})
    projection = {key: resolved.get(key) for key in config}
    for section in ("moe_config", "kv_cache_config", "cuda_graph_config"):
        source = resolved.get(section) or {}
        projection[section] = {key: source.get(key) for key in config[section]}
    graph = resolved.get("cuda_graph_config") or {}
    projection["cuda_graph_config"]["mode"] = graph.get("mode")
    if (
        resolved.get("pipeline_parallel_size") != 1
        or resolved.get("context_parallel_size") != 1
        or graph.get("max_batch_size") != config["max_batch_size"]
    ):
        raise ValueError("Resolved topology or graph capacity differs from qualification")
    validate_profile(projection, model)
    requested = json.loads(json.dumps(config))
    requested["cuda_graph_config"].setdefault("mode", "decode")
    if json.loads(json.dumps(projection)) != requested:
        raise ValueError("Resolved deployment differs from requested profile")


def validate_healthy(record: dict, run_id: str) -> None:
    """Require a current-run pause after both correct, nonempty healthy completions."""
    if record.get("run_id") != run_id or record.get("state") != "paused_between_requests":
        raise ValueError("Healthy between-request handshake is missing or stale")
    results = record.get("results", [])
    if len(results) != 2:
        raise ValueError("Healthy readiness requires two completions")
    for prompt, answer, result in zip(PROMPTS, ("paris", "rome"), results):
        if (
            result.get("prompt") != prompt
            or answer not in result.get("text", "").lower()
            or not result.get("token_ids")
        ):
            raise ValueError("Healthy readiness contains incorrect or empty inference")


def validate_runtime(runtime: dict, manifest: dict) -> None:
    """Check observed library versions against the image-bound verified build manifest."""
    required = (
        "tensorrt_llm_version",
        "torch_version",
        "torch_cuda",
        "nccl_runtime_version",
        "architecture",
    )
    if any(key not in manifest or runtime.get(key) != manifest[key] for key in required):
        raise ValueError("Observed runtime differs from the verified build manifest")
    if manifest.get("installed_payload_verified_file_count", 0) <= 0 or not re.fullmatch(
        r"[0-9a-f]{40}", manifest.get("source_commit", "")
    ):
        raise ValueError("Build manifest lacks exact-checkout verification")
    if not re.fullmatch(r"[0-9a-f]{64}", manifest.get("runtime_image_sha256", "")):
        raise ValueError("Build manifest is not bound to a runtime image")


def validate_ray_version(version: str | None) -> None:
    """Reject a missing or unqualified version of the private actor-notification API."""
    if version != "2.55.1":
        raise ValueError("This profile requires qualified Ray 2.55.1")


def validate_hardware(
    reports: list[dict], hostnames: set[str], config: dict, manifest: dict
) -> None:
    """Check host/GPU coverage against the requested shape and recorded build hardware."""
    _, nodes, gpus = deployment_shape(config)
    expected = {tuple(row.strip().split(", ")) for row in manifest["driver_report"].splitlines()}
    if len(expected) != 1 or len(next(iter(expected))) != 2:
        raise ValueError("Manifest requires one qualified GPU/driver combination")
    if (
        len(hostnames) != nodes
        or len(reports) != nodes
        or {row["hostname"] for row in reports} != hostnames
    ):
        raise ValueError("Incomplete or duplicate hardware reports")
    gpu_uuids = set()
    for report in reports:
        rows = [line.split(", ") for line in report["gpu_memory_csv"].splitlines()]
        if len(rows) != gpus or any(
            len(row) != 6 or (row[0], row[2]) not in expected for row in rows
        ):
            raise ValueError("Hardware differs from its qualified manifest")
        for row in rows:
            if not row[1] or row[1] in gpu_uuids:
                raise ValueError("Duplicate or missing GPU identity")
            gpu_uuids.add(row[1])


def read_placements(log: str) -> dict[tuple[int, int], list[int]]:
    """Reassemble Slurm-labelled arrays; reject partial, duplicate, or mislabelled records."""
    records = {}
    pending = {}
    for line in log.splitlines():
        labelled = re.fullmatch(r"\s*(\d+): (.*)", line)
        if not labelled:
            continue
        rank = int(labelled[1])
        message = labelled[2]
        begin = re.search(
            r"\[RANK (\d+)\].*initial_global_assignments \(layer (\d+)\) = (\[.*)", message
        )
        if begin:
            if int(begin[1]) != rank or rank in pending:
                raise ValueError("Mislabelled or interrupted placement record")
            pending[rank] = (int(begin[2]), begin[3])
        elif rank in pending:
            if not re.fullmatch(r"[0-9, \]]+", message):
                raise ValueError("Invalid placement continuation")
            layer, raw = pending[rank]
            pending[rank] = (layer, raw + message)
        else:
            continue
        layer, raw = pending[rank]
        if raw.endswith("]"):
            key = (rank, layer)
            if key in records:
                raise ValueError("Duplicate rank/layer placement")
            assignment = json.loads(raw)
            if not isinstance(assignment, list) or any(
                type(expert) is not int for expert in assignment
            ):
                raise ValueError("Worker placement must contain integer expert IDs")
            records[key] = assignment
            del pending[rank]
    if pending:
        raise ValueError("Incomplete placement arrays")
    return records


def verify_placements(
    records: dict[tuple[int, int], list[int]],
    planned: dict[int, list[int]],
    ep_size: int,
) -> None:
    expected = {(rank, layer) for rank in range(ep_size) for layer in planned}
    if set(records) != expected:
        raise ValueError(
            f"Worker placement evidence mismatch: missing={expected - set(records)}, extra={set(records) - expected}"
        )
    for (rank, layer), assignment in records.items():
        if assignment != planned[layer]:
            raise ValueError(f"Worker placement differs from plan: rank={rank}, layer={layer}")


def validate_execution_log(log: str, config: dict, model: dict) -> dict:
    """Require actual placement and execution evidence from every logical worker.

    Args:
        log: Slurm-labelled model log collected before fault injection.
        config: Native deployment arguments whose placement must match worker logs.
        model: Checkpoint configuration identifying all MoE layers and experts.

    Returns:
        JSON-serializable placement, communication, and graph-capture evidence.
    """
    validate_profile(config, model)
    planned = {
        int(layer): value
        for layer, value in config["moe_config"]["load_balancer"][
            "initial_global_assignments"
        ].items()
    }
    ep_size = config["moe_expert_parallel_size"]
    records = read_placements(log)
    experts, _ = moe_layout(model)
    verify_placements(records, planned, ep_size)
    transport_counts = collections.Counter()
    cft_counts = collections.Counter()
    captured_ranks = set()
    for line in log.splitlines():
        relevant = any(
            marker in line
            for marker in (
                "Selected communication strategy:",
                "CFT handle-based counted writes",
                "cuda_graph_capture: done",
            )
        )
        if not relevant:
            continue
        labelled = re.fullmatch(r"\s*(\d+): (.*)", line)
        producer = re.search(r"\[RANK (\d+)\]", line)
        if not labelled or not producer or int(labelled[1]) != int(producer[1]):
            raise ValueError("Missing or conflicting execution evidence rank labels")
        rank = int(labelled[1])
        message = labelled[2]
        transport = re.search(r"Selected communication strategy: (.+)", message)
        if transport:
            if not transport[1].startswith("NVLinkOneSided "):
                raise ValueError("Unexpected EP communication evidence")
            transport_counts[rank] += 1
        cft = re.search(r"CFT handle-based counted writes (.+)", message)
        if cft:
            if cft[1] != "disabled for dispatch":
                raise ValueError("This profile requires verified fence-based execution")
            cft_counts[rank] += 1
        capture = re.search(
            r"\[rank=(\d+)\]\[purpose=final_executor\].*cuda_graph_capture: done in ([\d.]+)s",
            message,
        )
        if capture:
            if int(capture[1]) != rank:
                raise ValueError("Conflicting final graph-capture rank labels")
            if float(capture[2]) > 0:
                captured_ranks.add(rank)
    expected_counts = {rank: len(planned) for rank in range(ep_size)}
    if dict(transport_counts) != expected_counts:
        raise ValueError("Incomplete or duplicate per-rank EP communication evidence")
    if dict(cft_counts) != expected_counts:
        raise ValueError("Incomplete or duplicate per-rank CFT evidence")
    if captured_ranks != set(range(ep_size)):
        raise ValueError("Missing positive final graph-capture evidence on one or more workers")
    return {
        "scope": "healthy fixed placement and per-rank coverage only; no capacity, recovery or CFT qualification",
        "coverage_eligibility": expert_coverage_report(planned, experts, ep_size),
        "rank_layer_records": len(records),
        "transport_records": sum(transport_counts.values()),
        "transport_records_by_rank": dict(sorted(transport_counts.items())),
        "cft_disabled_records": sum(cft_counts.values()),
        "cft_disabled_records_by_rank": dict(sorted(cft_counts.items())),
        "workers_with_final_graph_capture": sorted(captured_ranks),
    }


def normalize_ray_log(log: str, identities: list[dict], address: str) -> str:
    """Attribute forwarded model logs to the recorded Ray actor process, not arrival order."""
    if len({worker["actor_id"] for worker in identities}) != len(identities):
        raise ValueError("Duplicate Ray actor identities")
    source_ranks = {(worker["pid"], worker["node_ip"]): worker["rank"] for worker in identities}
    if len(source_ranks) != len(identities):
        raise ValueError("Duplicate Ray actor process identities")
    normalized = []
    partial_rank = None
    for line in log.splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        producer = re.fullmatch(
            r"0: \(RayWorkerWrapper(?:\[rank=\d+\])? pid=(\d+)(?:, ip=([0-9.]+))?\) (.*)", line
        )
        if producer is None:
            continuation = re.fullmatch(r"0: ([0-9, \]]+)", line)
            if partial_rank is not None and continuation is not None:
                normalized.append(f"{partial_rank}: {continuation[1]}")
                if continuation[1].endswith("]"):
                    partial_rank = None
            else:
                partial_rank = None
            continue
        rank = source_ranks.get((int(producer[1]), producer[2] or address))
        if rank is None:
            raise ValueError("Unmapped Ray model-log producer")
        message = producer[3]
        relevant = any(
            marker in message
            for marker in (
                "initial_global_assignments",
                "Selected communication strategy:",
                "CFT handle-based counted writes",
                "cuda_graph_capture: done",
            )
        )
        inner = re.search(r"\[RANK (\d+)\]", message)
        if relevant and inner is not None and int(inner[1]) != rank:
            raise ValueError("Conflicting native Ray log rank")
        if relevant and inner is None:
            message = f"[RANK {rank}] {message}"
        normalized.append(f"{rank}: {message}")
        partial_rank = (
            rank if "initial_global_assignments" in message and not message.endswith("]") else None
        )
    return "\n".join(normalized)


def fixed_assignments(expert_count: int, ep_size: int, num_slots: int) -> list[int]:
    """Fill uniform physical slots with cyclic logical expert IDs."""
    if (
        any(type(value) is not int for value in (expert_count, ep_size, num_slots))
        or expert_count <= 0
        or ep_size <= 1
        or num_slots < expert_count
        or num_slots % ep_size
    ):
        raise ValueError("Fixed placement requires full coverage and uniform slot partitions")
    return [slot % expert_count for slot in range(num_slots)]


def single_rank_coverage(
    assignments: list[int], expert_count: int, ep_size: int
) -> dict[int, list[int]]:
    """Return missing logical experts for each removed rank; reject malformed layouts."""
    if expert_count <= 0 or ep_size <= 1 or not assignments or len(assignments) % ep_size:
        raise ValueError("Invalid expert count, EP size, or slot partition")
    if any(type(expert) is not int or not 0 <= expert < expert_count for expert in assignments):
        raise ValueError("Placement contains an invalid logical expert")
    experts = set(range(expert_count))
    if set(assignments) != experts:
        raise ValueError("Healthy placement is missing logical experts")
    local_slots = len(assignments) // ep_size
    return {
        rank: sorted(
            experts
            - set(assignments[: rank * local_slots] + assignments[(rank + 1) * local_slots :])
        )
        for rank in range(ep_size)
    }


def expert_coverage_report(
    layer_assignments: dict[int, list[int]], expert_count: int, ep_size: int
) -> dict:
    """Intersect ranks preserving every expert in every layer; exclude rank zero from pools."""
    if not layer_assignments:
        raise ValueError("Expert coverage requires at least one MoE layer")
    missing = {
        layer: single_rank_coverage(assignments, expert_count, ep_size)
        for layer, assignments in sorted(layer_assignments.items())
    }
    eligible = [
        rank for rank in range(1, ep_size) if all(not row[rank] for row in missing.values())
    ]
    return {
        "scope": "Routed-expert coverage only; not full-model or MVP recovery admission",
        "ep_size": ep_size,
        "missing_experts_by_layer_and_removed_rank": missing,
        "eligible_ranks": eligible,
        "ineligible_ranks": [rank for rank in range(1, ep_size) if rank not in eligible],
    }


def main() -> None:
    """Prepare native arguments or bind a verified image manifest, without GPUs."""
    import argparse

    import yaml
    from process_probe import write_record

    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--config", type=Path)
    operation.add_argument("--image", type=Path)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--build-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num-slots", type=int)
    args = parser.parse_args()
    if args.image:
        if not args.build_manifest or not args.output:
            parser.error("Image binding requires --build-manifest and --output")
        manifest = json.loads(args.build_manifest.read_text())
        manifest.update(
            runtime_image_sha256=file_sha256(args.image), runtime_image_filename=args.image.name
        )
        validate_runtime(manifest, manifest)
        write_record(args.output, manifest)
    else:
        if not args.model_config or not args.output_dir:
            parser.error("Profile preparation requires --model-config and --output-dir")
        config, model = (
            yaml.safe_load(args.config.read_text()),
            json.loads(args.model_config.read_text()),
        )
        ranks, _, _ = deployment_shape(config)
        slots = (
            config["moe_config"]["load_balancer"]["num_slots"]
            if args.num_slots is None
            else args.num_slots
        )
        experts, moe_layers = moe_layout(model)
        assignments = fixed_assignments(experts, ranks, slots)
        layers = {layer: assignments for layer in moe_layers}
        config["moe_config"]["load_balancer"].update(
            num_slots=slots, layer_updates_per_iter=0, initial_global_assignments=layers
        )
        config["cuda_graph_config"] = config.get("cuda_graph_config") or {
            "batch_sizes": [1 << index for index in range(config["max_batch_size"].bit_length())],
            "enable_padding": True,
        }
        validate_profile(config, model)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        write_record(
            args.output_dir / "placement_plan.json",
            expert_coverage_report(layers, experts, ranks),
        )


if __name__ == "__main__":
    main()

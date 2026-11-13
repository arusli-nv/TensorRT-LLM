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
"""Source checks, with ``ast``, of the shared code that Wide-EP fault tolerance relies on.

The tables list the reviewed exceptions. An entry that no longer matches the code also fails.
"""

import ast
import functools
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

REPO = Path(__file__).resolve().parents[4]
TORCH = "tensorrt_llm/_torch"
COMMUNICATOR = f"{TORCH}/distributed/communicator.py"
HOST_EXCHANGE = f"{TORCH}/distributed/host_exchange.py"
PY_EXECUTOR = f"{TORCH}/pyexecutor/py_executor.py"
_WARMUP = f"{TORCH}/pyexecutor/engine/runners/decoder/warmup.py"
# The modules that fault tolerance adds.
FAULT_TOLERANCE_MODULES = (
    HOST_EXCHANGE,
    f"{TORCH}/distributed/membership",
    f"{TORCH}/distributed/membership_mnnvl_memory.py",
    f"{TORCH}/distributed/startup_record.py",
    f"{TORCH}/moe/fused_moe/communication/execution_state.py",
    f"{TORCH}/moe/fused_moe/eplb_expert_restore.py",
    f"{TORCH}/pyexecutor/rank_failure_memory.py",
    f"{TORCH}/pyexecutor/rank_failure_recovery.py",
    "tensorrt_llm/executor/rank_failure_requests.py",
)
# The shared modules that run while a rank is absent: serving, failover and rejoin. The checks below
# cover these and FAULT_TOLERANCE_MODULES only, so they never flag code outside the feature's path.
SHARED_MODULES = (
    COMMUNICATOR,
    f"{TORCH}/distributed/mnnvl_memory.py",
    f"{TORCH}/moe/fused_moe/moe_load_balancer.py",
    f"{TORCH}/pyexecutor/cuda_graph_runner.py",
    _WARMUP,
    f"{TORCH}/pyexecutor/kv_cache/kv_cache_manager_v2.py",
    PY_EXECUTOR,
    f"{TORCH}/pyexecutor/scheduler/adp_router.py",
    "tensorrt_llm/executor/worker.py",
)
# The other modules that the startup of a fault-tolerant deployment runs.
STARTUP_MODULES = (
    f"{TORCH}/pyexecutor/py_executor_creator.py",
    f"{TORCH}/pyexecutor/model_loader.py",
    f"{TORCH}/_experimental/modeling_v2/_router_index.py",
    f"{TORCH}/models/modeling_utils.py",
    f"{TORCH}/models/modeling_deepseekv3.py",
    f"{TORCH}/distributed/initialization.py",
    f"{TORCH}/autotuner.py",
    f"{TORCH}/moe/fused_moe/configurable_moe.py",
    f"{TORCH}/moe/fused_moe/communication/nvlink_one_sided.py",
    f"{TORCH}/pyexecutor/_util.py",
    f"{TORCH}/pyexecutor/resource_manager.py",
)


@functools.lru_cache(maxsize=None)
def _tree(path: str) -> ast.Module:
    return ast.parse((REPO / path).read_text(), filename=path)


def _sources() -> list:
    """Return the repository-relative Python files of FAULT_TOLERANCE_MODULES and SHARED_MODULES."""
    paths = []
    for module in FAULT_TOLERANCE_MODULES + SHARED_MODULES:
        root = REPO / module
        for path in sorted(root.rglob("*.py")) if root.is_dir() else [root]:
            paths.append(path.relative_to(REPO).as_posix())
    return paths


def _functions(tree: ast.AST, prefix: str = ""):
    """Yield every function of ``tree`` with its qualified name, including those in a branch."""
    for node in sorted(_own_nodes(tree), key=lambda node: getattr(node, "lineno", 0)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield prefix + node.name, node
            yield from _functions(node, f"{prefix}{node.name}.")
        elif isinstance(node, ast.ClassDef):
            yield from _functions(node, f"{prefix}{node.name}.")


def _own_nodes(function: ast.AST):
    """Yield the nodes of ``function``'s body, with lambdas but no nested functions or classes."""
    stack = list(ast.iter_child_nodes(function))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stack.extend(ast.iter_child_nodes(node))


def _call_name(call: ast.Call):
    return getattr(call.func, "attr", None) or getattr(call.func, "id", None)


def _assignments(function: ast.AST):
    """Yield the assigned names and the value of every assignment in ``function``'s own body."""
    for node in _own_nodes(function):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)) and node.value:
            targets = [node.target]
        else:
            continue
        names = {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}
        yield names, node.value


def _function_counts(count) -> dict:
    """Return the nonzero ``count`` of every function of the fault-tolerance and shared modules."""
    return {
        f"{path}::{qualname}": n
        for path in _sources()
        for qualname, function in _functions(_tree(path))
        if (n := count(function))
    }


def _check_inventory(found: dict, allowed: dict, table: str, fix: str) -> None:
    """Check that the ``found`` counts match the ``allowed`` (count, reason) entries."""
    problems = [
        f"{site}: {count} found, {allowed.get(site, (0,))[0]} in {table}; {fix}"
        for site, count in sorted(found.items())
        if allowed.get(site, (0,))[0] != count
    ]
    problems += [
        f"{table} lists {site}, which has none: remove it"
        for site in sorted(set(allowed) - set(found))
    ]
    assert not problems, "\n".join(problems)


# --- 1. Every host collective of Distributed runs over MNNVL memory under fault tolerance ---------

# Names whose use in a Distributed method means that it talks to MPI or torch.distributed itself.
DIRECT_TRANSPORT = frozenset(
    {"mpi_comm", "MPI", "tp_comm", "pp_comm", "cp_comm", "dist", "torch"}
    | {"safe_broadcast", "safe_allgather", "safe_gather"}
)


def _public_methods(cls: ast.ClassDef) -> dict:
    methods = {}
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name[0] != "_":
            decorators = {getattr(d, "id", getattr(d, "attr", None)) for d in node.decorator_list}
            if not decorators & {"property", "staticmethod", "classmethod", "setter"}:
                methods[node.name] = node
    return methods


def test_mnnvl_dist_overrides_every_host_collective():
    """Check that ``MnnvlDist`` runs every collective of ``Distributed`` over MNNVL, not MPI."""
    classes = {
        node.name: node
        for path in (COMMUNICATOR, HOST_EXCHANGE)
        for node in _tree(path).body
        if isinstance(node, ast.ClassDef)
    }
    bases = [base.id for base in classes["MnnvlDist"].bases]
    base, mpi, mnnvl = (
        _public_methods(classes[name]) for name in ("Distributed", "MPIDist", "MnnvlDist")
    )
    problems = (
        [] if bases == ["Distributed"] else [f"MnnvlDist derives from {bases}, not Distributed"]
    )
    for name in sorted(set(base) - set(mnnvl)):
        # A method defined only on Distributed must use collectives that MnnvlDist overrides.
        for node in ast.walk(base[name]):
            if isinstance(node, ast.Name) and node.id in DIRECT_TRANSPORT:
                problems.append(f"Distributed.{name} uses {node.id}: override it in MnnvlDist")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and getattr(node.func.value, "id", None) == "self"
                and node.func.attr in set(mpi) - set(base) - set(mnnvl)
            ):
                problems.append(
                    f"Distributed.{name} calls MPIDist.{node.func.attr}, which MnnvlDist lacks"
                )
    assert not problems, "\n".join(problems)


def test_no_caller_bypasses_distributed_collectives():
    """Check that no deployment module bypasses ``MnnvlDist`` through ``tp_comm`` and the like."""
    deployment_modules = set(STARTUP_MODULES) | set(_sources())
    uses = [
        f"{path}:{node.lineno}: .{node.attr}"
        for path in sorted(deployment_modules)
        if path != COMMUNICATOR
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.Attribute) and node.attr in ("tp_comm", "pp_comm", "cp_comm")
    ]
    assert not uses, "Use the collectives of Distributed instead:\n" + "\n".join(uses)


# --- 2. Votes over a TP gather take the rows of the active ranks ----------------------------------

# Gathers with one row or entry per rank. Under fault tolerance a failed rank's entry is None for
# object gathers and zeros for tp_allgather_int64.
GATHERS = frozenset(
    {"allgather", "tp_allgather", "tp_gather", "tp_allgather_int64", "tp_cp_allgather"}
    | {"tp_cp_allgather_int64", "mpi_allgather"}
)
_SKIPS_ABSENT = "skips the None of an absent rank"
_REPLAYED = "a startup gather, which a replacement replays from the startup record"
_REMOTE_TAIL = "remote tail attention DP, rejected with fault tolerance"
_DSPARK = "DSpark, rejected with fault tolerance"
_KV_CONNECTOR = "KV cache connectors, rejected with fault tolerance"
_PIPELINE_PARALLEL = "pipeline parallelism, rejected with fault tolerance"
_DISAGGREGATED = "disaggregated serving, rejected with fault tolerance"
# Functions of FAULT_TOLERANCE_MODULES and SHARED_MODULES that vote over such a gather without
# a gather with active_only=True, with the vote count and the reason.
RAW_GATHER_VOTES = {
    f"{TORCH}/distributed/mnnvl_memory.py::MnnvlMemory.open_mnnvl_memory": (1, _SKIPS_ABSENT),
    f"{TORCH}/moe/fused_moe/moe_load_balancer.py::MoeLoadBalancer.shutdown": (
        1,
        "the node communicator, which FaultTolerantMoeLoadBalancer.shutdown does not use",
    ),
    f"{TORCH}/pyexecutor/cuda_graph_runner.py::DSparkCUDAGraphRunner._agree_graph_shape": (
        3,
        _DSPARK,
    ),
    f"{_WARMUP}::WarmupAgreement.agree_flag": (1, _REPLAYED),
    f"{_WARMUP}::WarmupAgreement.should_run_warmup_batch": (1, _REPLAYED),
    f"{_WARMUP}::WarmupAgreement.assert_all_tp_ranks_have_warmup_batch": (1, _REPLAYED),
    f"{_WARMUP}::DecoderWarmup._run_autotuner_warmup": (1, _REPLAYED),
    f"{PY_EXECUTOR}::PyExecutor._initialize_execution": (
        1,
        "rank 0 (never replaced) at startup; an absent rank only skips the KV cache load snapshot",
    ),
    f"{PY_EXECUTOR}::PyExecutor._dspark_confidence_draft_len": (4, _DSPARK),
    f"{PY_EXECUTOR}::PyExecutor._can_queue": (
        2,
        f"{_REMOTE_TAIL}; and the vote without fault tolerance, in which every rank is active",
    ),
    f"{PY_EXECUTOR}::PyExecutor._kv_connector_sync_status": (1, _KV_CONNECTOR),
    f"{PY_EXECUTOR}::PyExecutor._resume_after_rebalance": (1, _REMOTE_TAIL),
}


def _vote_operand(call: ast.Call):
    """Return the operand of an ``all(x)``, ``np.all(x)`` or ``x.all()`` vote, or None."""
    func = call.func
    if isinstance(func, ast.Name) and func.id == "all" and call.args:
        return call.args[0]
    if isinstance(func, ast.Attribute) and func.attr == "all":
        if getattr(func.value, "id", None) in ("np", "numpy", "torch"):
            return call.args[0] if call.args else None
        return func.value
    return None


def _uses_raw_gather(expr: ast.AST, names: set) -> bool:
    """Return whether ``expr`` reads a gather without ``active_only=True`` or one of ``names``."""
    stack = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Call) and any(
            keyword.arg == "active_only" and getattr(keyword.value, "value", None) is True
            for keyword in node.keywords
        ):
            continue
        if (isinstance(node, ast.Call) and _call_name(node) in GATHERS) or (
            isinstance(node, ast.Name) and node.id in names
        ):
            return True
        stack.extend(ast.iter_child_nodes(node))
    return False


def _raw_gather_votes(function: ast.AST) -> int:
    names, assignments = set(), list(_assignments(function))
    while True:  # the names holding a raw gather or a value computed from one
        grown = names.union(
            *(targets for targets, value in assignments if _uses_raw_gather(value, names))
        )
        if grown == names:
            break
        names = grown
    return sum(
        1
        for node in _own_nodes(function)
        if isinstance(node, ast.Call)
        and (operand := _vote_operand(node)) is not None
        and _uses_raw_gather(operand, names)
    )


def test_gather_votes_use_active_rows():
    """Check that an ``all`` vote over a TP gather takes its rows with ``active_only=True``."""
    _check_inventory(
        _function_counts(_raw_gather_votes),
        RAW_GATHER_VOTES,
        "RAW_GATHER_VOTES",
        "gather the vote's rows with active_only=True, or list the votes with why an absent rank "
        "cannot decide them",
    )


# --- 3. The previous batch is read once it is accepted --------------------------------------------

# Functions that read previous_batch before or without _accept_or_discard_previous_batch, with the
# read count and the reason.
UNACCEPTED_PREVIOUS_BATCH_READS = {
    f"{PY_EXECUTOR}::PyExecutor._accept_or_discard_previous_batch": (1, "the acceptance itself"),
    f"{PY_EXECUTOR}::PyExecutor._discard_previous_batch": (
        1,
        "the discard itself, called by _accept_or_discard_previous_batch",
    ),
    f"{PY_EXECUTOR}::PyExecutor._executor_loop_overlap": (
        4,
        "the next step's inputs take the previous batch's device tensors and a dropped batch "
        "returns its timing events; no result applies before _accept_or_discard_previous_batch",
    ),
    f"{PY_EXECUTOR}::PyExecutor._process_previous_batch": (
        3,
        "every caller calls _accept_or_discard_previous_batch first",
    ),
    f"{PY_EXECUTOR}::PyExecutor._maybe_prefetch_next_iter_mm_encoders": (
        2,
        "reads request ids only",
    ),
    f"{PY_EXECUTOR}::PyExecutor.recompute_active_requests": (
        1,
        "checks that there is no previous batch",
    ),
    f"{PY_EXECUTOR}::PyExecutor._executor_loop_pp": (1, _PIPELINE_PARALLEL),
    f"{PY_EXECUTOR}::PyExecutor._save_kv_to_connector_async": (2, _KV_CONNECTOR),
    f"{PY_EXECUTOR}::PyExecutor._cancel_waiting_disagg_requests": (1, _DISAGGREGATED),
}


def _reads_previous_batch(node: ast.AST) -> bool:
    if isinstance(node, ast.Attribute):
        return node.attr == "previous_batch" and isinstance(node.ctx, ast.Load)
    return (
        isinstance(node, ast.Call)
        and _call_name(node) == "getattr"
        and len(node.args) > 1
        and getattr(node.args[1], "value", None) == "previous_batch"
    )


def _unaccepted_reads(function: ast.AST) -> int:
    nodes = list(_own_nodes(function))
    accepted_at = min(
        (
            node.lineno
            for node in nodes
            if isinstance(node, ast.Call)
            and _call_name(node) == "_accept_or_discard_previous_batch"
        ),
        default=None,
    )
    return sum(
        1
        for node in nodes
        if _reads_previous_batch(node) and (accepted_at is None or node.lineno <= accepted_at)
    )


def test_previous_batch_is_read_after_acceptance():
    """Check that a function reading ``previous_batch`` accepts or discards it first."""
    _check_inventory(
        _function_counts(_unaccepted_reads),
        UNACCEPTED_PREVIOUS_BATCH_READS,
        "UNACCEPTED_PREVIOUS_BATCH_READS",
        "call _accept_or_discard_previous_batch before the reads, or list them with why no result "
        "of a discarded batch applies",
    )


# --- 4. Only the launch path reads the rank identity that the launcher gives ----------------------

# The names under which a module reads the rank identity (TLLM_FAULT_TOLERANCE_RANK).
RANK_IDENTITY_NAMES = frozenset(
    {"TLLM_FAULT_TOLERANCE_RANK", "FAULT_TOLERANCE_RANK_ENV", "wide_ep_fault_tolerance_rank"}
)
# The modules that read the rank identity, each with what it does with it.
RANK_IDENTITY_READERS = {
    "tensorrt_llm/_wide_ep_fault_tolerance.py": "parses it and rejects it without the flag",
    f"{TORCH}/distributed/membership/startup.py": "starts this process as the rank it names",
    "tensorrt_llm/llmapi/_llmapi_node_supervisor.py": "gives it to each rank it starts",
    "tensorrt_llm/llmapi/_node_supervised_mpi_session.py": "gives it to rank 0 of one node",
    "tensorrt_llm/serve/_frontend_processes.py": "drops it for a serving frontend, not a rank",
}


def _names_rank_identity(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id in RANK_IDENTITY_NAMES
    if isinstance(node, ast.Attribute):
        return node.attr in RANK_IDENTITY_NAMES
    if isinstance(node, ast.alias):
        return node.name in RANK_IDENTITY_NAMES
    return isinstance(node, ast.Constant) and node.value in RANK_IDENTITY_NAMES


def test_only_launch_path_reads_rank_identity():
    """Check that only the modules in ``RANK_IDENTITY_READERS`` read the rank identity."""
    readers = set()
    for path in sorted((REPO / "tensorrt_llm").rglob("*.py")):
        if not any(name in path.read_text() for name in RANK_IDENTITY_NAMES):
            continue
        rel = path.relative_to(REPO).as_posix()
        if any(_names_rank_identity(node) for node in ast.walk(_tree(rel))):
            readers.add(rel)
    _check_inventory(
        dict.fromkeys(readers, 1),
        {rel: (1, reason) for rel, reason in RANK_IDENTITY_READERS.items()},
        "RANK_IDENTITY_READERS",
        "take the rank identity from the startup record or mpi_comm() instead",
    )

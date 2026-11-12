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
"""Source checks for the shared code that Wide-EP fault tolerance relies on.

A change to shared code can break ``enable_wide_ep_fault_tolerance`` without failing any test of its own. Examples
are a host collective that ``MnnvlDist`` does not override, a vote over a gather that counts failed ranks, a read of
the previous batch before it is accepted, and a startup collective that a replacement rank cannot join. These tests
parse the sources with ``ast``, without importing TensorRT-LLM or using a GPU, and report how to fix each finding.
The tables here and in the YAML file next to this module list the reviewed exceptions. An entry that no longer
matches the code also fails, so the tables stay current.
"""

import ast
import functools
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.cpu_only

REPO = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
TORCH = "tensorrt_llm/_torch"
COMMUNICATOR = f"{TORCH}/distributed/communicator.py"
HOST_EXCHANGE = f"{TORCH}/distributed/host_exchange.py"
PY_EXECUTOR = f"{TORCH}/pyexecutor/py_executor.py"
# Code that a fault-tolerant deployment never runs. This covers modeling_v2 targets, kernels, image and video
# generation, and the features that fault tolerance rejects at startup.
SCAN_EXCLUDE = (
    "/_experimental/modeling_v2/models/",
    "/kernels/",
    "/cute_dsl_kernels/",
    "/visual_gen/",
    "/disaggregation/",
    "/per_layer_dynamic_eplb/",
)


@functools.lru_cache(maxsize=None)
def _tree(path: str) -> ast.Module:
    return ast.parse((REPO / path).read_text(), filename=path)


def _sources(*roots: str) -> list:
    """Return the repository-relative Python files under ``roots`` that ``SCAN_EXCLUDE`` does not exclude."""
    paths = []
    for root in roots:
        for path in sorted((REPO / root).rglob("*.py")):
            rel = path.relative_to(REPO).as_posix()
            if not any(part in rel for part in SCAN_EXCLUDE):
                paths.append(rel)
    return paths


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate keys, which ``yaml.safe_load`` silently overwrites."""

    def construct_mapping(self, node, deep=False):
        keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            raise yaml.constructor.ConstructorError(
                None, None, f"keys given twice: {duplicates}", node.start_mark
            )
        return super().construct_mapping(node, deep=deep)


def _load_table(path: Path) -> dict:
    return yaml.load(path.read_text(), Loader=_UniqueKeyLoader)  # nosec B506: a SafeLoader


def _functions(tree: ast.AST, prefix: str = ""):
    """Yield every function of ``tree`` with its qualified name, including functions defined in a branch."""
    for node in sorted(_own_nodes(tree), key=lambda node: getattr(node, "lineno", 0)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield prefix + node.name, node
            yield from _functions(node, f"{prefix}{node.name}.")
        elif isinstance(node, ast.ClassDef):
            yield from _functions(node, f"{prefix}{node.name}.")


def _own_nodes(function: ast.AST):
    """Yield the nodes of ``function``'s body, including lambdas but not nested functions or classes."""
    stack = list(ast.iter_child_nodes(function))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stack.extend(ast.iter_child_nodes(node))


def _call_name(call: ast.Call):
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


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


def _check_inventory(found: dict, allowed: dict, table: str, fix: str) -> list:
    """Return the differences between the ``found`` counts and the ``allowed`` (count, reason) entries."""
    problems = [
        f"{site}: {count} found, {allowed.get(site, (0,))[0]} in {table}; {fix}"
        for site, count in sorted(found.items())
        if allowed.get(site, (0,))[0] != count
    ]
    problems += [
        f"{table} lists {site}, which has none: remove it"
        for site in sorted(set(allowed) - set(found))
    ]
    return problems


# --- 1. Every host collective of Distributed runs over MNNVL memory under fault tolerance -------------------------

# MPIDist methods that MnnvlDist leaves on MPI, and why fault tolerance never calls them with a peer.
COLLECTIVES_LEFT_ON_MPI = {
    "cp_allgather": "MnnvlDist requires context_parallel_size=1 (unmet_requirements)",
    "cp_broadcast": "MnnvlDist requires context_parallel_size=1 (unmet_requirements)",
    "cp_allgather_int64": "MnnvlDist requires context_parallel_size=1 (unmet_requirements)",
    "pp_allgather": "MnnvlDist requires pipeline_parallel_size=1 (unmet_requirements)",
    "pp_gather": "MnnvlDist requires pipeline_parallel_size=1 (unmet_requirements)",
    "pp_broadcast": "MnnvlDist requires pipeline_parallel_size=1 (unmet_requirements)",
    "isend": "point to point, used by pipeline parallelism only",
    "send": "point to point, used by pipeline parallelism only",
    "recv": "point to point, used by pipeline parallelism only",
    "isend_object": "point to point, used by pipeline parallelism only",
    "send_object": "point to point, used by pipeline parallelism only",
    "recv_object": "point to point, used by pipeline parallelism only",
}
# Names whose use in a Distributed method means that it talks to MPI or torch.distributed itself.
DIRECT_TRANSPORT = frozenset(
    {
        "mpi_comm",
        "MPI",
        "tp_comm",
        "pp_comm",
        "cp_comm",
        "safe_broadcast",
        "safe_allgather",
        "safe_gather",
    }
    | {"dist", "torch"}
)


def _public_methods(cls: ast.ClassDef) -> dict:
    methods = {}
    for node in cls.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name.startswith(
            "_"
        ):
            continue
        decorators = {getattr(d, "id", getattr(d, "attr", None)) for d in node.decorator_list}
        if not decorators & {"property", "staticmethod", "classmethod", "setter"}:
            methods[node.name] = node
    return methods


def _communicator_classes() -> dict:
    classes = {
        node.name: node
        for path in (COMMUNICATOR, HOST_EXCHANGE)
        for node in _tree(path).body
        if isinstance(node, ast.ClassDef)
    }
    return {
        name: _public_methods(classes[name]) for name in ("Distributed", "MPIDist", "MnnvlDist")
    }


def test_mnnvl_dist_overrides_every_host_collective():
    """Check that ``MnnvlDist`` runs every public collective of ``Distributed`` and ``MPIDist`` over MNNVL memory.

    Under fault tolerance ``Distributed.get`` returns ``MnnvlDist``. A method left on MPI hangs on a failed rank, and
    a replacement rank cannot join it because it is alone in its MPI world.
    """
    classes = _communicator_classes()
    base, mpi, mnnvl = classes["Distributed"], classes["MPIDist"], classes["MnnvlDist"]
    problems = []
    for name in sorted(set(base) | set(mpi)):
        if name in mnnvl or name in COLLECTIVES_LEFT_ON_MPI:
            continue
        if name in mpi:
            problems.append(
                f"MPIDist.{name} runs over MPI under fault tolerance: override it in MnnvlDist, or add it "
                "to COLLECTIVES_LEFT_ON_MPI with why fault tolerance never calls it with a peer"
            )
            continue
        # A method defined only on Distributed must be built from collectives that MnnvlDist overrides.
        for node in ast.walk(base[name]):
            if isinstance(node, ast.Name) and node.id in DIRECT_TRANSPORT:
                problems.append(f"Distributed.{name} uses {node.id}: override it in MnnvlDist")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and getattr(node.func.value, "id", None) == "self"
                and node.func.attr in set(mpi) - set(mnnvl) - set(COLLECTIVES_LEFT_ON_MPI)
            ):
                problems.append(
                    f"Distributed.{name} calls MPIDist.{node.func.attr}, which MnnvlDist does not override"
                )
    problems += [
        f"COLLECTIVES_LEFT_ON_MPI lists {name}, which MnnvlDist does not leave on MPI: remove it"
        for name in COLLECTIVES_LEFT_ON_MPI
        if name not in mpi or name in mnnvl
    ]
    assert not problems, "\n".join(problems)


def test_no_caller_bypasses_distributed_collectives():
    """Check that no caller uses the MPI communicators of ``MPIDist`` (``tp_comm``, ``pp_comm``, ``cp_comm``).

    ``MnnvlDist`` overrides the collectives but not the communicators, so such a caller bypasses fault tolerance.
    """
    uses = [
        f"{path}:{node.lineno}: .{node.attr}"
        for path in _sources(TORCH)
        if path != COMMUNICATOR
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.Attribute) and node.attr in ("tp_comm", "pp_comm", "cp_comm")
    ]
    assert not uses, "Use the collectives of Distributed instead:\n" + "\n".join(uses)


# --- 2. Votes over a TP gather take the rows of the active ranks --------------------------------------------------

# Gathers with one row or entry per rank. Under fault tolerance a failed rank's entry is None for object gathers
# and zeros for tp_allgather_int64.
GATHERS = frozenset(
    {"allgather", "tp_allgather", "tp_gather", "tp_allgather_int64", "tp_cp_allgather"}
    | {"tp_cp_allgather_int64", "mpi_allgather"}
)
_SKIPS_ABSENT = "skips the None of an absent rank"
_REPLAYED = (
    "a startup gather, which a replacement rank replays from the startup record: no rank is absent"
)
_REMOTE_TAIL = (
    "conditional disaggregation (remote tail attention DP), rejected with fault tolerance"
)
_DSPARK = "DSpark, rejected with fault tolerance"
_KV_CONNECTOR = "KV cache connectors, rejected with fault tolerance"
_WARMUP = f"{TORCH}/pyexecutor/engine/runners/decoder/warmup.py"
# Functions that vote over such a gather without Distributed.tp_active_rows, with the vote count and the reason.
RAW_GATHER_VOTES = {
    f"{TORCH}/distributed/ll_exchange.py::_PeerBuffers.create": (
        1,
        "peer buffers of LM-head TP, rejected with fault tolerance; an absent rank only turns them off",
    ),
    f"{TORCH}/distributed/mnnvl_memory.py::MnnvlMemory.open_mnnvl_memory": (1, _SKIPS_ABSENT),
    f"{TORCH}/moe/fused_moe/moe_load_balancer.py::MoeLoadBalancer.shutdown": (
        1,
        "the node communicator, None with fault tolerance",
    ),
    f"{TORCH}/pyexecutor/connectors/kv_cache_connector.py::"
    "KvCacheConnectorManager.configure_prefix_reservations": (2, _KV_CONNECTOR),
    f"{TORCH}/pyexecutor/cuda_graph_runner.py::DSparkCUDAGraphRunner._agree_graph_shape": (
        3,
        _DSPARK,
    ),
    f"{TORCH}/pyexecutor/engine/runners/decoder/inputs.py::"
    "DecoderInputPreparer._get_remote_tail_decode_rank_tokens": (1, _REMOTE_TAIL),
    f"{TORCH}/pyexecutor/engine/runners/decoder/metadata.py::sync_group_all_greedy_sample": (
        1,
        "LM-head TP in attention DP and DSpark confidence scheduling, rejected with fault tolerance",
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
    f"{PY_EXECUTOR}::PyExecutor._can_queue": (1, _REMOTE_TAIL),
    f"{PY_EXECUTOR}::PyExecutor._kv_connector_sync_status": (1, _KV_CONNECTOR),
    f"{PY_EXECUTOR}::PyExecutor._handle_control_request": (1, _SKIPS_ABSENT),
    f"{PY_EXECUTOR}::PyExecutor._resume_after_rebalance": (1, _REMOTE_TAIL),
}


def _vote_operand(call: ast.Call):
    """Return the operand of an all-ranks vote (``all(x)``, ``np.all(x)``, ``x.all()``), or None.

    An ``any`` vote is not checked because a failed rank's None or zero row counts as "no" and leaves it unchanged.
    """
    func = call.func
    if isinstance(func, ast.Name) and func.id == "all" and call.args:
        return call.args[0]
    if isinstance(func, ast.Attribute) and func.attr == "all":
        if getattr(func.value, "id", None) in ("np", "numpy", "torch"):
            return call.args[0] if call.args else None
        return func.value
    return None


def _uses_raw_gather(expr: ast.AST, names: set) -> bool:
    """Return whether ``expr`` reads a gather or one of ``names`` outside a ``tp_active_rows`` call."""
    stack = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Call) and _call_name(node) == "tp_active_rows":
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
    """Check that an ``all`` vote over a TP gather reduces over ``Distributed.tp_active_rows(rows)``.

    Under fault tolerance a failed rank's None or zero row counts as "no" in a vote over the raw gather.
    ``tp_active_rows`` keeps only the active ranks' rows, which is every row without fault tolerance.
    """
    found = {}
    for path in _sources(TORCH, "tensorrt_llm/executor"):
        for qualname, function in _functions(_tree(path)):
            if count := _raw_gather_votes(function):
                found[f"{path}::{qualname}"] = count
    problems = _check_inventory(
        found,
        RAW_GATHER_VOTES,
        "RAW_GATHER_VOTES",
        "vote over dist.tp_active_rows(rows), or list the votes with why an absent rank cannot decide them",
    )
    assert not problems, "\n".join(problems)


# --- 3. The previous batch is read once it is accepted ------------------------------------------------------------

# Functions that read previous_batch before or without _accepted_previous_batch, with the read count and the reason.
UNACCEPTED_PREVIOUS_BATCH_READS = {
    f"{PY_EXECUTOR}::PyExecutor._accepted_previous_batch": (1, "the accessor itself"),
    f"{TORCH}/pyexecutor/step_acceptance.py::accept_previous_batch": (
        2,
        "the acceptance itself, called by _accepted_previous_batch",
    ),
    f"{TORCH}/pyexecutor/rank_failure_recovery.py::RankFailureRecovery.drop_invalid_step": (
        1,
        "the acceptance itself, called by accept_previous_batch",
    ),
    f"{PY_EXECUTOR}::PyExecutor.discard_previous_batch": (
        1,
        "the discard itself, called by drop_invalid_step",
    ),
    f"{PY_EXECUTOR}::PyExecutor._executor_loop_overlap": (
        4,
        "the next step's inputs take the previous batch's device tensors and a dropped batch returns its "
        "timing events; no result applies before _accepted_previous_batch",
    ),
    f"{PY_EXECUTOR}::PyExecutor._process_previous_batch": (
        3,
        "every caller calls _accepted_previous_batch first",
    ),
    f"{PY_EXECUTOR}::PyExecutor._maybe_prefetch_next_iter_mm_encoders": (
        2,
        "reads request ids only",
    ),
    f"{PY_EXECUTOR}::PyExecutor.recompute_active_requests": (
        1,
        "checks that there is no previous batch",
    ),
    f"{PY_EXECUTOR}::PyExecutor._executor_loop_pp": (
        1,
        "pipeline parallelism, rejected with fault tolerance",
    ),
    f"{PY_EXECUTOR}::PyExecutor._save_kv_to_connector_async": (2, _KV_CONNECTOR),
    f"{PY_EXECUTOR}::PyExecutor._cancel_waiting_disagg_requests": (
        1,
        "disaggregated serving, rejected with fault tolerance",
    ),
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
            if isinstance(node, ast.Call) and _call_name(node) == "_accepted_previous_batch"
        ),
        default=None,
    )
    return sum(
        1
        for node in nodes
        if _reads_previous_batch(node) and (accepted_at is None or node.lineno <= accepted_at)
    )


def test_previous_batch_is_read_after_acceptance():
    """Check that a function reading ``previous_batch`` calls ``_accepted_previous_batch`` first.

    With rank failure recovery the previous batch is accepted, or discarded after a membership change, before its
    results apply. A discarded batch must not be consumed.
    """
    found = {}
    for path in _sources(f"{TORCH}/pyexecutor"):
        for qualname, function in _functions(_tree(path)):
            if count := _unaccepted_reads(function):
                found[f"{path}::{qualname}"] = count
    problems = _check_inventory(
        found,
        UNACCEPTED_PREVIOUS_BATCH_READS,
        "UNACCEPTED_PREVIOUS_BATCH_READS",
        "call _accepted_previous_batch before the reads, or list them with why no result of a discarded batch "
        "applies",
    )
    assert not problems, "\n".join(problems)


# --- 4. Startup collectives outside Distributed agree on what the original ranks got ------------------------------

STARTUP_COLLECTIVES = HERE / "wide_ep_ft_startup_collectives.yaml"
ENTRY_KEYS = frozenset({"collectives", "reason", "note"})
# Collectives the startup calls on mpi_comm() (a StoreGroup under fault tolerance) and its subgroups. A replacement
# rank replays those of mpi_comm() from the startup record, not those of its subgroups.
STARTUP_COLLECTIVE_NAMES = frozenset(
    {"allgather", "allreduce", "bcast", "barrier", "Barrier", "Allgather", "Allreduce", "Bcast"}
    | {"Split", "Split_type", "Create_group", "Dup", "subgroup"}
    | {"mpi_allgather", "mpi_broadcast", "mpi_barrier", "local_mpi_barrier"}
)
# The startup record's agreements: a replacement rank takes the original ranks' result of the gather they run.
AGREEMENTS = frozenset({"agree", "agree_rank_capacity"})


def _agreed_nodes(function: ast.AST) -> set:
    """Return the ids of the nodes of ``function`` whose value reaches a ``startup_record.agree*`` call."""
    calls = [
        node
        for node in _own_nodes(function)
        if isinstance(node, ast.Call) and _call_name(node) in AGREEMENTS
    ]
    agreed = {id(n) for call in calls for n in ast.walk(call)}
    names = {n.id for call in calls for n in ast.walk(call) if isinstance(n, ast.Name)}
    assignments = list(_assignments(function))
    while True:  # the names whose value reaches startup_record.agree
        grown = names.union(
            *(
                {n.id for n in ast.walk(value) if isinstance(n, ast.Name)}
                for targets, value in assignments
                if targets & names
            )
        )
        if grown == names:
            break
        names = grown
    return agreed | {
        id(n) for targets, value in assignments if targets & names for n in ast.walk(value)
    }


def test_startup_collectives_are_agreed_or_listed():
    """Check that each startup collective outside ``Distributed`` goes through ``startup_record.agree`` or is listed.

    A replacement rank starts alone, so a collective on a subgroup of ``mpi_comm()`` (a ``StoreGroup``) reaches only
    itself. The check is static and covers only the modules wide_ep_ft_startup_collectives.yaml lists.
    """
    table = _load_table(STARTUP_COLLECTIVES)
    reasons, modules = table["reasons"], table["modules"]
    listed = {
        f"{path}::{qualname}": entry
        for path, functions in modules.items()
        for qualname, entry in (functions or {}).items()
    }
    names = STARTUP_COLLECTIVE_NAMES
    functions = [
        (f"{path}::{qualname}", function)
        for path in modules
        for qualname, function in _functions(_tree(path))
    ]
    agreed = {site: _agreed_nodes(function) for site, function in functions}
    # Functions whose result their caller passes to startup_record.agree, or that it passes as the gather.
    agreed_callees = {
        node.id if isinstance(node, ast.Name) else _call_name(node)
        for site, function in functions
        for node in _own_nodes(function)
        if isinstance(node, (ast.Call, ast.Name)) and id(node) in agreed[site]
    } - names
    found = {}
    for site, function in functions:
        if function.name in agreed_callees:
            continue
        collectives = sorted(
            _call_name(node)
            for node in _own_nodes(function)
            if isinstance(node, ast.Call)
            and _call_name(node) in names
            and id(node) not in agreed[site]
        )
        if collectives:
            found[site] = collectives
    problems = [
        f"{site}: startup collectives {collectives} not agreed on: pass the result that must match the original "
        f"ranks' through startup_record.agree, or list them in {STARTUP_COLLECTIVES.name} with a reason"
        for site, collectives in sorted(found.items())
        if sorted(listed.get(site, {}).get("collectives", ())) != collectives
    ]
    problems += [
        f"{STARTUP_COLLECTIVES.name} lists {site}, which has no such collective: remove it"
        for site in sorted(set(listed) - set(found))
    ]
    problems += [
        f"{site}: {entry} needs 'collectives' and a 'reason' of {sorted(reasons)}, and may have a 'note'"
        for site, entry in sorted(listed.items())
        if entry.get("reason") not in reasons
        or not {"collectives", "reason"} <= set(entry) <= ENTRY_KEYS
    ]
    assert not problems, "\n".join(problems)

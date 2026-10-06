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
"""CPU safety checks for verified faults and same-GPU explicit restart."""

import json
import os
from pathlib import Path

import pytest
from fault_injection import load_config, validate_fault_evidence, validate_restart
from fault_injector import (
    IDENTITY_KEYS,
    process_identity,
    record,
    resources_released,
    validate_injection,
)


def identity(rank: int = 1) -> dict:
    return {
        **process_identity(os.getpid()),
        "rank": rank,
        "actor_id": "actor-a",
        "run_id": "run-a",
        "gpu_uuid": "GPU-a",
    }


@pytest.mark.parametrize("key", IDENTITY_KEYS)
def test_identity_drift_prevents_injection(key: str) -> None:
    """Can a reused PID, changed actor or different rank be signaled?"""
    expected = identity()
    actual = {**expected, key: None}
    with pytest.raises(ValueError, match="identity|target"):
        validate_injection(
            expected,
            actual,
            "worker_sigkill_idle",
            {"run_id": "run-a", "event": "between_requests"},
        )


@pytest.mark.parametrize("rank", [-1, 0])
def test_root_and_invalid_rank_are_rejected(rank: int) -> None:
    """Does this narrow suite accidentally admit root or invalid targets?"""
    with pytest.raises(ValueError, match="target"):
        validate_injection(identity(rank), identity(rank), "worker_sigkill_idle", {})


@pytest.mark.parametrize(
    "trigger",
    [
        {"run_id": "old", "event": "first_nonfinal_output", "token_ids": [1], "finished": False},
        {"run_id": "run-a", "event": "first_nonfinal_output", "token_ids": [], "finished": False},
        {"run_id": "run-a", "event": "first_nonfinal_output", "token_ids": [1], "finished": True},
        {"run_id": "run-a", "event": "between_requests"},
    ],
)
def test_streaming_requires_current_nonfinal_output(trigger: dict) -> None:
    """Can stale, empty or completed output authorize the streaming fault?"""
    with pytest.raises(ValueError):
        validate_injection(identity(), identity(), "worker_sigkill_streaming", trigger)


def test_evidence_is_published_once(tmp_path: Path) -> None:
    """Can a second action overwrite its first injection intent?"""
    record(tmp_path, "injection_intent", {"action": "first"})
    with pytest.raises(FileExistsError):
        record(tmp_path, "injection_intent", {"action": "second"})
    assert json.loads((tmp_path / "injection_intent.json").read_text())["action"] == "first"
    assert len(list(tmp_path.iterdir())) == 1


def snapshot() -> dict:
    return {
        "hostname": "node-a",
        "gpus": [["GPU-a", "100", "580", "GB200"]],
        "compute_apps": [],
        "live_processes": [],
    }


@pytest.mark.parametrize(
    "change",
    [
        {"compute_apps": [["GPU-a", "123", "1024"]]},
        {"live_processes": [{"pid": 123}]},
        {"gpus": [["GPU-a", "165", "580", "GB200"]]},
        {"gpus": [["GPU-other", "100", "580", "GB200"]]},
        {"hostname": "different-node"},
    ],
)
def test_cleanup_requires_independent_resource_release(change: dict) -> None:
    """Can liveness, a remaining context or a changed GPU produce a false clean result?"""
    assert not resources_released([snapshot()], [{**snapshot(), **change}])


def test_empty_or_incomplete_cleanup_cannot_pass() -> None:
    """Can missing observations masquerade as resource cleanup?"""
    assert not resources_released([], [])
    assert not resources_released([snapshot()], [])
    assert resources_released([snapshot()], [snapshot()])


def test_cleanup_backstop_rechecks_process_identity() -> None:
    """Can cleanup signal a recycled PID, and can it terminate its verified child?"""
    import signal
    import subprocess
    import sys

    from fault_injector import NodeProbe

    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        expected = process_identity(process.pid)
        probe = NodeProbe()
        assert not probe.terminate_owned_processes(
            [{**expected, "start_ticks": expected["start_ticks"] + 1}]
        )
        assert process.poll() is None
        assert probe.terminate_owned_processes([expected]) == [process.pid]
        assert process.wait(timeout=5) == -signal.SIGKILL
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_forced_backstop_does_not_hide_failed_native_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Can successful backstop cleanup turn a native resource leak into a pass?"""
    import sys
    from types import SimpleNamespace

    import fault_injection

    dirty = {"compute_apps": [["GPU-a", "123", "1024"]]}

    def terminate(_identities: list[dict]) -> list:
        dirty.clear()
        return []

    probe = SimpleNamespace(
        snapshot=SimpleNamespace(remote=lambda _: {**snapshot(), **dirty}),
        pin_actor_processes=SimpleNamespace(remote=lambda _: []),
        terminate_owned_processes=SimpleNamespace(remote=terminate),
        collect_logs=SimpleNamespace(remote=lambda *_: []),
    )
    ray = SimpleNamespace(
        get=lambda value, **_: value,
        available_resources=lambda: {"GPU": 1},
        _private=SimpleNamespace(
            state=SimpleNamespace(jobs=lambda: [{"JobID": "owned", "IsDead": True}])
        ),
    )
    monkeypatch.setitem(sys.modules, "ray._private.worker", SimpleNamespace(global_worker=None))
    monkeypatch.setattr(fault_injection, "owned_actors", lambda *_: {})
    ticks = iter(range(20))
    monkeypatch.setattr(fault_injection.time, "monotonic", lambda: next(ticks))
    record(tmp_path, "job", {"job_id": "owned"})
    with pytest.raises(TimeoutError, match="Native teardown leaked"):
        fault_injection.cleanup(ray, [probe], [snapshot()], tmp_path, [], 0.1, False)
    assert (tmp_path / "cleanup_failed.json").exists()
    assert json.loads((tmp_path / "cleanup.json").read_text())["forced"]


@pytest.mark.parametrize(
    "change",
    [{"gpu_uuid": "other"}, {"hostname": "other"}, {"actor_id": "actor-a"}, {"pid": os.getpid()}],
)
def test_restart_requires_fresh_workers_on_same_gpus(change: dict) -> None:
    """Can worker reuse or replacement GPU capacity masquerade as the requested restart?"""
    initial = identity()
    restarted = {**initial, "actor_id": "actor-b", "pid": initial["pid"] + 1}
    validate_restart([initial], [restarted])
    with pytest.raises(ValueError):
        validate_restart([initial], [{**restarted, **change}])


def test_generic_error_does_not_prove_fence_fault(tmp_path: Path) -> None:
    """Does a generic terminal error falsely qualify the communication reproducer?"""
    record(tmp_path, "injection_intent", {"scenario": "fence_round_mismatch"})
    record(tmp_path, "injection_result", {"before": 4, "after": 6})
    (tmp_path / "driver.log").write_text("EngineDeadError")
    with pytest.raises(AssertionError, match="fence timeout"):
        validate_fault_evidence(tmp_path, "fence_round_mismatch")
    (tmp_path / "driver.log").write_text("dispatch: Rank 1 timed out waiting for completion flag")
    validate_fault_evidence(tmp_path, "fence_round_mismatch")


@pytest.mark.parametrize("ranks", [2, 7, 16, 32, 72, 128])
def test_static_placement_is_not_fixed_to_ep32(tmp_path: Path, ranks: int) -> None:
    """Can a different native EP capacity use the same static placement preparation?"""
    import yaml

    model = {
        "model_type": "deepseek_v3",
        "n_routed_experts": 256,
        "first_k_dense_replace": 3,
        "num_hidden_layers": 5,
    }
    (tmp_path / "config.json").write_text(json.dumps(model))
    slots = ((256 + ranks - 1) // ranks) * ranks
    config = {
        "moe_expert_parallel_size": ranks,
        "moe_config": {"load_balancer": {"num_slots": slots}},
    }
    path = tmp_path / "args.yaml"
    path.write_text(yaml.safe_dump(config))
    prepared = load_config(path, tmp_path)["moe_config"]["load_balancer"]
    assert prepared["layer_updates_per_iter"] == 0
    assert set(prepared["initial_global_assignments"]) == {3, 4}
    assert all(
        len(row) == slots and set(row) == set(range(256))
        for row in prepared["initial_global_assignments"].values()
    )


@pytest.mark.parametrize(
    "scenario",
    [
        "worker_sigkill_idle",
        "worker_sigkill_streaming",
        "host_collective_abort",
        "fence_round_mismatch",
    ],
)
@pytest.mark.skipif(
    not os.environ.get("WIDEEP_FT_RAY_ADDRESS"), reason="Opt-in dedicated Ray cluster"
)
def test_wideep_fault_and_explicit_restart(scenario: str) -> None:
    """Does this fault report a bounded client error, release resources, and allow a fresh restart?"""
    import subprocess
    import sys

    output = Path(os.environ["WIDEEP_FT_OUTPUT_DIR"]) / scenario
    command = [
        sys.executable,
        str(Path(__file__).with_name("fault_injection.py")),
        "--address",
        os.environ["WIDEEP_FT_RAY_ADDRESS"],
        "--model",
        os.environ["WIDEEP_FT_MODEL"],
        "--config",
        os.environ["WIDEEP_FT_CONFIG"],
        "--scenario",
        scenario,
        "--output-dir",
        str(output),
    ]
    if scenario == "fence_round_mismatch":
        command.extend(["--client-timeout-s", "420"])
    result = subprocess.run(command, check=False)
    assert result.returncode == 0, f"Fault characterization failed; evidence: {output}"


def test_sigkill_publishes_intent_before_terminating_owned_process(tmp_path: Path) -> None:
    """Is one real SIGKILL preceded by durable, matching target evidence?"""
    import signal
    import subprocess
    import sys

    code = """
import os
from pathlib import Path
from fault_injector import WorkerExtension, process_identity, record
class Worker(WorkerExtension):
    def fault_identity(self):
        return {**process_identity(os.getpid()), 'rank': 1, 'actor_id': 'cpu-test'}
worker = Worker()
expected = {**worker.fault_identity(), 'run_id': 'run-a'}
path = Path(__import__('sys').argv[1])
record(path, 'trigger', {'run_id': 'run-a', 'event': 'between_requests'})
worker.inject_fault(expected, str(path), 'worker_sigkill_idle')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == -signal.SIGKILL, result.stderr
    intent = json.loads((tmp_path / "injection_intent.json").read_text())
    assert intent["target"]["rank"] == 1
    assert intent["target"]["actor_id"] == "cpu-test"
    assert intent["trigger"]["run_id"] == "run-a"
    assert not (tmp_path / "injection_result.json").exists()


def test_actor_cleanup_filters_exact_job_and_live_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Can cleanup select actors belonging to another driver?"""
    import sys
    from types import ModuleType, SimpleNamespace

    from fault_injection import owned_actors

    job = object()

    def actors(*, job_id: object) -> dict:
        assert job_id is job
        return {
            "owned": {"JobID": "job-a", "State": "ALIVE"},
            "starting": {"JobID": "job-a", "State": "PENDING_CREATION"},
            "foreign": {"JobID": "job-b", "State": "ALIVE"},
            "dead": {"JobID": "job-a", "State": "DEAD"},
        }

    private = ModuleType("ray._private")
    private.state = SimpleNamespace(actors=actors)
    monkeypatch.setitem(sys.modules, "ray._private", private)
    ray = SimpleNamespace(JobID=SimpleNamespace(from_hex=lambda value: job))
    assert list(owned_actors(ray, "job-a")) == ["owned", "starting"]


def test_group_registration_is_not_communication_failure(tmp_path: Path) -> None:
    """Can benign initialization logs falsely establish the communication fault?"""
    record(tmp_path, "injection_intent", {"scenario": "host_collective_abort"})
    record(tmp_path, "injection_result", {"backend": "gloo", "group_size": 32})
    (tmp_path / "driver.log").write_text("Group is registered; Group valid")
    with pytest.raises(AssertionError, match="collective error"):
        validate_fault_evidence(tmp_path, "host_collective_abort")


def test_partial_startup_pins_only_local_actor_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Can interrupted-startup accounting confuse another node's PID with a local process?"""
    import sys
    from types import SimpleNamespace

    from fault_injector import NodeProbe

    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(get_runtime_context=lambda: SimpleNamespace(get_node_id=lambda: "node-a")),
    )
    actors = {
        "local": {"Address": {"NodeID": "node-a"}, "Pid": os.getpid()},
        "remote": {"Address": {"NodeID": "node-b"}, "Pid": os.getpid()},
        "pending": {"Address": {"NodeID": ""}, "Pid": 0},
    }
    pinned = NodeProbe().pin_actor_processes(actors)
    assert len(pinned) == 1
    assert pinned[0]["actor_id"] == "local"
    assert pinned[0]["start_ticks"] == process_identity(os.getpid())["start_ticks"]

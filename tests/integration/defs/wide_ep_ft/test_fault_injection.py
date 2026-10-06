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
import pickle
import signal
import struct
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import fault_injection
import pytest
from fault_injection import (
    load_config,
    owned_actors,
    validate_fault_evidence,
    validate_restart,
    validate_workers,
)
from fault_injector import (
    IDENTITY_KEYS,
    NodeProbe,
    WorkerExtension,
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
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
    )
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


def test_ray_initialization_preserves_interrupt_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Does SIGTERM still record a failed run after Ray installs its own handler?"""

    def interrupt(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt("test interruption")

    def ray_interrupt(signum: int, _frame: object) -> None:
        raise SystemExit(signum)

    def initialize(**_kwargs: object) -> None:
        signal.signal(signal.SIGTERM, ray_interrupt)

    def create_probe(**_kwargs: object) -> None:
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(init=initialize, remote=create_probe, shutdown=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "ray.util.scheduling_strategies",
        SimpleNamespace(NodeAffinitySchedulingStrategy=object),
    )
    monkeypatch.setattr(fault_injection, "load_config", lambda *_: {"moe_expert_parallel_size": 1})
    args = SimpleNamespace(
        output_dir=tmp_path / "attempt",
        config=tmp_path,
        model=tmp_path,
        run_id="run-a",
        scenario="healthy",
        address="external",
        startup_timeout_s=720,
        client_timeout_s=30,
        shutdown_timeout_s=180,
        cleanup_timeout_s=60,
    )
    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        with pytest.raises(KeyboardInterrupt, match="test interruption"):
            fault_injection.run(args)
        summary = json.loads((args.output_dir / "summary.json").read_text())
        assert summary["state"] == "FAIL" and "KeyboardInterrupt" in summary["error"]
    finally:
        signal.signal(signal.SIGTERM, previous)


@pytest.mark.parametrize("failure", ["exited", "wait_timeout"])
def test_client_termination_failure_still_cleans_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Can an exiting or unresponsive client skip independent resource cleanup?"""
    process = Mock(pid=123, poll=Mock(return_value=None))
    cleanup = Mock()
    monkeypatch.setattr(fault_injection.subprocess, "Popen", Mock(return_value=process))
    monkeypatch.setattr(fault_injection, "cleanup", cleanup)
    kill = Mock(side_effect=ProcessLookupError() if failure == "exited" else None)
    monkeypatch.setattr(fault_injection.os, "killpg", kill)
    if failure == "wait_timeout":
        process.wait.side_effect = subprocess.TimeoutExpired("client", 15)
    args = SimpleNamespace(
        output_dir=tmp_path,
        address="external",
        model=tmp_path,
        config=tmp_path,
        run_id="run-a",
        startup_timeout_s=1e-9,
        client_timeout_s=30,
        shutdown_timeout_s=180,
        cleanup_timeout_s=60,
    )
    expected = TimeoutError if failure == "exited" else subprocess.TimeoutExpired
    with pytest.raises(expected):
        fault_injection.run_phase(args, None, [], [], "initial", "healthy")
    cleanup.assert_called_once()
    assert cleanup.call_args.args[-1] is True
    assert (tmp_path / "initial" / "timing.json").exists()


@pytest.mark.parametrize("failure", ["kill", "shutdown"])
def test_finalization_failure_cannot_publish_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Can a failed probe/client finalization leave a contradictory PASS artifact?"""
    probe = SimpleNamespace(
        read_run_id=SimpleNamespace(remote=lambda _: "run-a"),
        snapshot=SimpleNamespace(remote=lambda _: snapshot()),
    )
    ray = SimpleNamespace(
        init=lambda **_: None,
        remote=lambda **_: lambda _: SimpleNamespace(
            options=lambda **_: SimpleNamespace(remote=lambda: probe)
        ),
        get=lambda value, **_: value,
        nodes=lambda: [{"Alive": True, "Resources": {"GPU": 1}, "NodeID": "node-a"}],
        kill=Mock(side_effect=RuntimeError("kill failed") if failure == "kill" else None),
        shutdown=Mock(
            side_effect=RuntimeError("shutdown failed") if failure == "shutdown" else None
        ),
    )
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(
        sys.modules,
        "ray.util.scheduling_strategies",
        SimpleNamespace(NodeAffinitySchedulingStrategy=lambda *_, **__: None),
    )
    monkeypatch.setattr(fault_injection, "load_config", lambda *_: {"moe_expert_parallel_size": 1})
    first = identity()
    second = {**first, "actor_id": "actor-b", "pid": first["pid"] + 1}
    monkeypatch.setattr(
        fault_injection,
        "run_phase",
        Mock(
            side_effect=[
                {"workers": [row], "received": {"healthy": 1}, "parent_started_s": 0}
                for row in (first, second)
            ]
        ),
    )
    args = SimpleNamespace(
        output_dir=tmp_path / "attempt",
        config=tmp_path,
        model=tmp_path,
        run_id="run-a",
        scenario="healthy",
        address="external",
        startup_timeout_s=720,
        client_timeout_s=30,
        shutdown_timeout_s=180,
        cleanup_timeout_s=60,
    )
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        fault_injection.run(args)
    summary = json.loads((args.output_dir / "summary.json").read_text())
    assert summary["state"] == "FAIL" and f"{failure} failed" in summary["error"]
    ray.shutdown.assert_called_once()


@pytest.mark.parametrize(
    "change", [{}, {"cft_capable": True}, {"cft_capable": None}, {"ep_size": 3}, {"ep_rank": 1}]
)
def test_worker_runtime_matches_non_cft_profile(change: dict) -> None:
    """Can CFT or different EP geometry qualify as the requested fence baseline?"""
    workers = [
        {
            "rank": rank,
            "actor_id": f"actor-{rank}",
            "gpu_uuid": f"GPU-{rank}",
            "graphs_enabled": True,
            "graph_keys": ["1"],
            "communication": {
                "NVLinkOneSided": {
                    "cft_capable": False,
                    "ep_size": 2,
                    "ep_rank": rank,
                }
            },
        }
        for rank in range(2)
    ]
    workers[0]["communication"]["NVLinkOneSided"].update(change)
    if change:
        with pytest.raises(ValueError, match="communication"):
            validate_workers(workers, 2, True)
    else:
        validate_workers(workers, 2, True)


def test_cleanup_preserves_pinned_identity_after_pid_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Can stale actor metadata authorize killing a replacement process?"""
    original = identity()
    identities = [original.copy()]
    calls = []
    snapshots = 0

    def pin(actors: dict, directory: str) -> list[dict]:
        assert directory == str(tmp_path)
        calls.append(actors)
        return [{**original, "start_ticks": original["start_ticks"] + 1}] if actors else []

    def observe(_identities: list[dict]) -> dict:
        nonlocal snapshots
        snapshots += 1
        return {**snapshot(), "compute_apps": [["GPU-a", "123", "1024"]] if snapshots == 1 else []}

    probe = SimpleNamespace(
        snapshot=SimpleNamespace(remote=observe),
        pin_actor_processes=SimpleNamespace(remote=pin),
        terminate_owned_processes=SimpleNamespace(remote=lambda _: []),
        collect_logs=SimpleNamespace(remote=lambda *_: []),
    )
    ray = SimpleNamespace(
        get=lambda value, **_: value,
        ActorID=SimpleNamespace(from_hex=lambda value: value),
        available_resources=lambda: {"GPU": 1},
        _private=SimpleNamespace(
            state=SimpleNamespace(jobs=lambda: [{"JobID": "owned", "IsDead": True}])
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "ray._private.worker",
        SimpleNamespace(
            global_worker=SimpleNamespace(core_worker=SimpleNamespace(kill_actor=Mock()))
        ),
    )
    monkeypatch.setattr(
        fault_injection,
        "owned_actors",
        lambda *_: {original["actor_id"]: {"Pid": original["pid"]}} if snapshots < 2 else {},
    )
    monkeypatch.setattr(fault_injection.time, "sleep", lambda _: None)
    record(tmp_path, "job", {"job_id": "owned", "run_id": "run-a"})
    fault_injection.cleanup(ray, [probe], [snapshot()], tmp_path, identities, 5, True)
    assert identities == [original]
    assert calls == [{}, {}]


def test_forced_backstop_does_not_hide_failed_native_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Can successful backstop cleanup turn a native resource leak into a pass?"""
    dirty = {"compute_apps": [["GPU-a", "123", "1024"]]}

    def terminate(_identities: list[dict]) -> list:
        dirty.clear()
        return []

    probe = SimpleNamespace(
        snapshot=SimpleNamespace(remote=lambda _: {**snapshot(), **dirty}),
        pin_actor_processes=SimpleNamespace(remote=lambda *_: []),
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
    record(tmp_path, "job", {"job_id": "owned", "run_id": "run-a"})
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


def test_fault_targets_request_broadcast_world_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Does injection select WORLD explicitly, regardless of TP-group aliasing?"""
    active, tp_group = Mock(), Mock()
    destroy = Mock()
    group = SimpleNamespace(_get_backend=lambda _: active, size=lambda: 32)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            device=lambda name: name,
            distributed=SimpleNamespace(
                group=SimpleNamespace(WORLD=group), destroy_process_group=destroy
            ),
        ),
    )
    worker = SimpleNamespace(
        fault_identity=identity,
        engine=SimpleNamespace(dist=SimpleNamespace(mapping=SimpleNamespace(tp_group_pg=tp_group))),
    )
    record(tmp_path, "trigger", {"run_id": "run-a", "event": "between_requests"})
    result = WorkerExtension.inject_fault(
        worker, identity(), str(tmp_path), "process_group_destroy"
    )
    destroy.assert_called_once_with(group)
    active.abort.assert_not_called()
    tp_group._get_backend.assert_not_called()
    assert result["group"] == "WORLD"


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
        "process_group_destroy",
        "fence_round_mismatch",
    ],
)
@pytest.mark.skipif(
    not os.environ.get("WIDEEP_FT_RAY_ADDRESS"), reason="Opt-in dedicated Ray cluster"
)
def test_wideep_fault_and_explicit_restart(scenario: str) -> None:
    """Does this fault report a bounded client error, release resources, and allow a fresh restart?"""
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
    """Does cleanup send only serializable local identity data for this driver's actors?"""
    job = object()

    def actors(*, job_id: object) -> dict:
        assert job_id is job
        return {
            "owned": {
                "JobID": "job-a",
                "State": "ALIVE",
                "Pid": 123,
                "Address": {"NodeID": "node-a"},
                "DeathCause": struct.Struct("q"),
            },
            "starting": {
                "JobID": "job-a",
                "State": "PENDING_CREATION",
                "Pid": 0,
                "Address": {"NodeID": ""},
            },
            "foreign": {"JobID": "job-b", "State": "ALIVE"},
            "dead": {"JobID": "job-a", "State": "DEAD"},
        }

    private = ModuleType("ray._private")
    private.state = SimpleNamespace(actors=actors)
    monkeypatch.setitem(sys.modules, "ray._private", private)
    ray = SimpleNamespace(JobID=SimpleNamespace(from_hex=lambda value: job))
    selected = owned_actors(ray, "job-a")
    assert list(selected) == ["owned", "starting"]
    pickle.dumps(selected)


def test_group_registration_is_not_communication_failure(tmp_path: Path) -> None:
    """Can benign initialization logs falsely establish the communication fault?"""
    record(tmp_path, "injection_intent", {"scenario": "process_group_destroy"})
    record(tmp_path, "injection_result", {"group": "WORLD", "backend": "gloo", "group_size": 32})
    (tmp_path / "driver.log").write_text("Group is registered; Group valid")
    with pytest.raises(AssertionError, match="collective error"):
        validate_fault_evidence(tmp_path, "process_group_destroy")
    (tmp_path / "driver.log").write_text("Default process group has not been initialized")
    validate_fault_evidence(tmp_path, "process_group_destroy")


@pytest.mark.parametrize(
    "change",
    [None, {"run_id": "other"}, {"job_id": "other"}, {"start_ticks": 0}, {}],
)
def test_partial_startup_pins_only_verified_local_actor_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: dict | None
) -> None:
    """Can stale Ray metadata pin a process without matching startup identity?"""
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
    current = process_identity(os.getpid())
    record(tmp_path, "job", {"job_id": "owned", "run_id": "run-a"})
    if change is not None:
        record(
            tmp_path / "startup_identities" / current["hostname"],
            f"{current['pid']}-{current['start_ticks']}",
            {**current, "job_id": "owned", "run_id": "run-a", **change},
        )
    pinned = NodeProbe().pin_actor_processes(actors, str(tmp_path))
    assert len(pinned) == (1 if change == {} else 0)
    if pinned:
        assert pinned[0]["actor_id"] == "local"
        assert pinned[0]["run_id"] == "run-a"
        assert pinned[0]["start_ticks"] == current["start_ticks"]

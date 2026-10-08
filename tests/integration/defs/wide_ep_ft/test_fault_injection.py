# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU safety checks and opt-in physical MPI WideEP regressions."""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import fault_injection
import fault_injector
import pytest
from fault_injection import (
    load_config,
    restart_benchmark,
    validate_fault_evidence,
    validate_restart,
    validate_workers,
)
from fault_injector import (
    IDENTITY_KEYS,
    SCENARIOS,
    process_identity,
    record,
    resources_released,
    validate_injection,
)


def identity(rank: int = 1) -> dict:
    return {
        **process_identity(os.getpid()),
        "rank": rank,
        "run_id": "run-a",
        "gpu_uuid": "GPU-a",
    }


def test_client_waits_for_worker_identity_publication(tmp_path: Path, monkeypatch) -> None:
    """Can native readiness precede the last test-hook identity receipt?"""
    args = argparse.Namespace(
        output_dir=tmp_path,
        config=tmp_path / "config.yaml",
        model=tmp_path,
        client_timeout_s=1,
        cleanup_timeout_s=1,
        scenario="healthy",
    )
    args.config.write_text("moe_expert_parallel_size: 2\n")
    llm = Mock()
    llm.generate_async.side_effect = [
        Mock(
            result=Mock(
                return_value=SimpleNamespace(outputs=[SimpleNamespace(text=text, token_ids=[1])])
            )
        )
        for text in ("Paris", "Rome")
    ]

    context = MagicMock()
    context.__enter__.return_value = llm
    module = SimpleNamespace(
        LLM=Mock(return_value=context), SamplingParams=Mock(), __version__="test"
    )
    monkeypatch.setitem(sys.modules, "tensorrt_llm", module)

    def publish(rank: int) -> None:
        record(
            tmp_path / "workers",
            str(rank),
            {
                "rank": rank,
                "gpu_uuid": f"GPU-{rank}",
                "communication": {
                    "NVLinkOneSided": {
                        "cft_capable": False,
                        "ep_size": 2,
                        "ep_rank": rank,
                    }
                },
            },
        )

    publish(0)
    record(tmp_path, "proceed", {})

    def delayed_publication(_seconds: float) -> None:
        assert not (tmp_path / "healthy.json").exists()
        publish(1)

    monkeypatch.setattr(fault_injection.time, "sleep", delayed_publication)
    fault_injection.client(args)
    assert llm.generate_async.call_count == 2
    assert (tmp_path / "shutdown.json").exists()


@pytest.mark.parametrize("key", (*IDENTITY_KEYS, "gpu_uuid"))
def test_identity_drift_prevents_injection(key: str) -> None:
    """Can a reused PID, changed namespace or different rank be signaled?"""
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


def test_worker_gpu_identity_is_independent_of_observer_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Can an injection thread's default device misidentify a worker on another GPU?"""
    comm = argparse.Namespace(
        ep_size=4,
        ep_rank=2,
        can_use_cft_counted_writes=False,
        workspace=argparse.Namespace(device="cuda:2"),
    )
    worker = argparse.Namespace(
        rank=2,
        get_startup_metrics=lambda: {},
        engine=argparse.Namespace(
            model_engine=argparse.Namespace(
                cuda_graph_runner=argparse.Namespace(enabled=True, graphs={1: None})
            )
        ),
    )
    cuda = argparse.Namespace(
        current_device=lambda: 0,
        get_device_properties=lambda device: argparse.Namespace(
            uuid="worker" if device == "cuda:2" else "other"
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", argparse.Namespace(cuda=cuda))
    monkeypatch.setattr(fault_injector, "_nvlink_communication", lambda _: comm)
    monkeypatch.setenv(fault_injector.RUN_ID_ENV, "run-a")
    assert fault_injector.worker_identity(worker)["gpu_uuid"] == "GPU-worker"


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
        {"observation_scope": "worker_identities"},
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


@pytest.mark.parametrize(
    "change", [{}, {"cft_capable": True}, {"cft_capable": None}, {"ep_size": 3}, {"ep_rank": 1}]
)
def test_worker_runtime_matches_non_cft_profile(change: dict) -> None:
    """Can CFT or different EP geometry qualify as the requested fence baseline?"""
    workers = [
        {
            "rank": rank,
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


@pytest.mark.parametrize(
    "change",
    [
        {"gpu_uuid": "other"},
        {"hostname": "other"},
        {"pid": os.getpid()},
    ],
)
def test_restart_requires_fresh_workers_on_same_gpus(change: dict) -> None:
    """Can worker reuse or replacement GPU capacity masquerade as the requested restart?"""
    initial = identity()
    restarted = {**initial, "pid": initial["pid"] + 1}
    validate_restart([initial], [restarted])
    with pytest.raises(ValueError):
        validate_restart([initial], [{**restarted, **change}])


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


@pytest.mark.parametrize("assignments", [None, {0: [0, 1]}])
def test_explicit_placement_remains_static(tmp_path: Path, assignments: dict | None) -> None:
    """Can explicit or null assignments leave dynamic EPLB enabled?"""
    import yaml

    (tmp_path / "config.json").write_text(
        json.dumps(
            dict(
                model_type="deepseek_v3",
                n_routed_experts=2,
                first_k_dense_replace=0,
                num_hidden_layers=1,
            )
        )
    )
    config = dict(
        moe_expert_parallel_size=2,
        moe_config=dict(
            load_balancer=dict(
                num_slots=2,
                layer_updates_per_iter=1,
                initial_global_assignments=assignments,
            )
        ),
    )
    path = tmp_path / "args.yaml"
    path.write_text(yaml.safe_dump(config))
    placement = load_config(path, tmp_path)["moe_config"]["load_balancer"]
    assert placement["layer_updates_per_iter"] == 0
    assert placement["initial_global_assignments"] == {0: [0, 1]}


def test_sigkill_publishes_intent_before_process_death(tmp_path: Path) -> None:
    """Is real process death preceded by a durable, identity-checked injection receipt?"""
    code = """
import os, sys
from pathlib import Path
import fault_injector
from fault_injector import process_identity, record
identity = {**process_identity(os.getpid()), 'rank': 1, 'run_id': 'run-a'}
fault_injector.worker_identity = lambda _: identity
root = Path(sys.argv[1])
record(root, 'trigger', {'run_id': 'run-a', 'event': 'between_requests'})
fault_injector.inject_fault(None, identity, root, 'worker_sigkill_idle')
"""
    process = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert process.returncode == -signal.SIGKILL, process.stderr
    receipt = json.loads((tmp_path / "injection_intent.json").read_text())
    assert receipt["target"]["pid"] > 0
    assert receipt["trigger"]["run_id"] == "run-a"
    assert not (tmp_path / "injection_result.json").exists()


def test_host_probe_rejects_invisible_worker() -> None:
    """Can inaccessible container identities falsely prove host-side cleanup coverage?"""
    worker = identity()
    observed = {"hostname": worker["hostname"], "live_processes": [worker]}
    fault_injection._visible_workers([worker], [observed])
    fault_injection._visible_workers([{**worker, "uid": 0}], [observed])
    with pytest.raises(ValueError, match="observe"):
        fault_injection._visible_workers([worker], [{**observed, "live_processes": []}])
    with pytest.raises(ValueError, match="observe"):
        fault_injection._visible_workers([worker, {**worker, "pid": worker["pid"] + 1}], [observed])


@pytest.mark.parametrize("reused", [False, True])
def test_pid_reuse_prevents_cleanup_signal(monkeypatch: pytest.MonkeyPatch, reused: bool) -> None:
    """Can a stale host-side identity authorize killing a replacement process?"""
    expected = identity()
    send = Mock()
    monkeypatch.setattr(fault_injector, "owned_processes", lambda *_, **__: [expected])
    monkeypatch.setattr(fault_injector.os, "pidfd_open", lambda _: 10)
    monkeypatch.setattr(fault_injector.os, "close", lambda _: None)
    monkeypatch.setattr(fault_injector.signal, "pidfd_send_signal", send)
    monkeypatch.setattr(
        fault_injector,
        "process_identity",
        lambda _: {**expected, "start_ticks": 0} if reused else expected,
    )
    monkeypatch.setattr(Path, "read_bytes", lambda _: b"WIDEEP_FT_RUN_ID=run-a\0")
    monkeypatch.setattr(
        fault_injector.subprocess, "run", lambda *_, **__: argparse.Namespace(stdout="")
    )
    fault_injector.snapshot("run-a", terminate=True)
    if reused:
        send.assert_not_called()
    else:
        send.assert_called_once_with(10, signal.SIGKILL)


@pytest.mark.parametrize("change", [None, "environment", "identity"])
def test_cleanup_ownership_survives_process_discovery_races(
    monkeypatch: pytest.MonkeyPatch, change: str | None
) -> None:
    """Can an old environment marker authorize cleanup of a replacement PID?"""
    expected = {**identity(), "pid": os.getpid() + 1}
    marker = b"WIDEEP_FT_RUN_ID=run-a\0"
    environments = Mock(side_effect=[marker, b"" if change == "environment" else marker])
    identities = Mock(
        side_effect=[expected, {**expected, "start_ticks": 0} if change == "identity" else expected]
    )
    with monkeypatch.context() as patch:
        patch.setattr(Path, "iterdir", lambda _: [Path("/proc") / str(expected["pid"])])
        patch.setattr(Path, "stat", lambda _: argparse.Namespace(st_uid=os.getuid()))
        patch.setattr(Path, "read_bytes", environments)
        patch.setattr(fault_injector, "process_identity", identities)
        found = fault_injector.owned_processes("run-a")
    assert found == ([] if change else [expected])


def test_fence_failure_requires_native_timeout_and_client_error(tmp_path: Path) -> None:
    """Can a generic error or client timeout falsely qualify a fence fault?"""
    worker = identity()
    record(tmp_path, "trigger", {"run_id": "run-a", "event": "between_requests"})
    record(tmp_path, "injection_intent", {"scenario": "fence_round_mismatch", "target": worker})
    record(tmp_path, "injection_result", {"before": 4, "after": 6})
    record(tmp_path, "client_error", {"type": "RequestError"})
    (tmp_path / "launcher.log").write_text("unspecified launch failure")
    with pytest.raises(AssertionError, match="fence timeout"):
        validate_fault_evidence(tmp_path, "fence_round_mismatch", [worker])
    (tmp_path / "launcher.log").write_text("dispatch: Rank 1 timed out waiting for completion flag")
    validate_fault_evidence(tmp_path, "fence_round_mismatch", [worker])


def test_steps_are_filtered_by_owned_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Can cancellation select a sibling job step from the same allocation?"""
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(
        fault_injection.subprocess,
        "run",
        lambda *_, **__: argparse.Namespace(
            stdout=(
                "StepId=123.0 Name=foreign\nStepId=123.1 UserId=42 Name=owned\n"
                "StepId=124.1 Name=owned\nStepId=123.batch Name=batch\n"
            )
        ),
    )
    assert fault_injection._steps("owned") == ["123.1"]


@pytest.mark.parametrize(
    "failed_record", [None, "intervention", "cleanup_control_error", "probe_storage"]
)
def test_forced_cleanup_never_turns_failure_into_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_record: str | None
) -> None:
    """Does cleanup run and preserve failure even when evidence writes fail?"""
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    args = argparse.Namespace(
        output_dir=tmp_path,
        target_rank=1,
        model=tmp_path,
        launcher=["srun"],
        python="python3",
        client_timeout_s=30,
        cleanup_timeout_s=60,
        startup_timeout_s=1,
        shutdown_timeout_s=30,
        ranks=2,
        graphs_requested=False,
    )
    process = Mock()
    process.poll.return_value = 1
    stop = Mock(side_effect=subprocess.CalledProcessError(1, "scontrol") if failed_record else None)
    probe = Mock(
        return_value=[snapshot()],
        side_effect=OSError("probe storage unavailable")
        if failed_record == "probe_storage"
        else None,
    )

    def write(directory: Path, name: str, value: dict) -> None:
        if name == failed_record:
            raise OSError("evidence filesystem unavailable")
        record(directory, name, value)

    monkeypatch.setattr(fault_injection, "record", write)
    monkeypatch.setattr(fault_injection.subprocess, "Popen", lambda *_, **__: process)
    monkeypatch.setattr(fault_injection, "_stop", stop)
    monkeypatch.setattr(fault_injection, "_probe", probe)
    monkeypatch.setattr(fault_injection, "_steps", lambda *_, **__: [])
    with pytest.raises(AssertionError, match="healthy readiness"):
        fault_injection.run_phase(args, [snapshot()], "initial", "worker_sigkill_idle")
    assert any(call.args[3] for call in probe.call_args_list)
    if failed_record == "probe_storage":
        assert stop.call_count == 2
        assert (tmp_path / "initial" / "cleanup_failed.json").exists()
        assert not (tmp_path / "initial" / "forced_cleanup.json").exists()
    else:
        stop.assert_called_once()
        assert (tmp_path / "initial" / "forced_cleanup.json").exists()
    assert not (tmp_path / "summary.json").exists()


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.skipif(
    not os.environ.get("WIDEEP_FT_MPI_LAUNCHER"), reason="Opt-in dedicated MPI allocation"
)
def test_wideep_fault_and_explicit_restart(scenario: str) -> None:
    """Does the qualified MPI fault terminate, release resources, and permit a same-GPU restart?"""
    output = Path(os.environ["WIDEEP_FT_OUTPUT_DIR"]) / scenario
    process = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("fault_injection.py")),
            "--launcher",
            os.environ["WIDEEP_FT_MPI_LAUNCHER"],
            "--launcher-mode",
            os.environ.get("WIDEEP_FT_MPI_LAUNCHER_MODE", "pmix"),
            "--probe-launcher",
            os.environ["WIDEEP_FT_PROBE_LAUNCHER"],
            "--model",
            os.environ["WIDEEP_FT_MODEL"],
            "--config",
            os.environ["WIDEEP_FT_CONFIG"],
            "--scenario",
            scenario,
            "--output-dir",
            str(output),
        ],
        check=False,
    )
    assert process.returncode == 0, f"Fault characterization failed; evidence: {output}"


def test_launcher_exit_race_still_checks_remote_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Does exit between wait timeout and SIGKILL skip remote-step cleanup?"""
    process = Mock()
    process.poll.return_value = None
    process.wait.side_effect = [subprocess.TimeoutExpired("srun", 10), 137]

    def signal_group(*_: object) -> None:
        raise ProcessLookupError()

    steps = Mock(return_value=[])
    monkeypatch.setattr(fault_injection.os, "killpg", signal_group)
    monkeypatch.setattr(fault_injection, "_steps", steps)
    monkeypatch.setattr(fault_injection.time, "monotonic", lambda: 9.0)
    fault_injection._stop(process, "owned", 10.0)
    assert steps.call_count == 2
    assert process.wait.call_count == 2
    assert all(call.kwargs["timeout"] == 1 for call in process.wait.call_args_list)


def test_control_failure_still_reaps_local_launcher(monkeypatch: pytest.MonkeyPatch) -> None:
    """Does scheduler failure prevent local launcher reaping?"""
    process = Mock()
    process.poll.return_value = None
    monkeypatch.setattr(fault_injection.os, "killpg", Mock())
    monkeypatch.setattr(fault_injection, "_steps", Mock(side_effect=RuntimeError("scheduler down")))
    with pytest.raises(RuntimeError, match="scheduler down"):
        fault_injection._stop(process, "owned", fault_injection.time.monotonic() + 60)
    process.wait.assert_called_once()


@pytest.mark.parametrize("daemon", [True, False])
def test_unreadable_model_process_is_not_ignored(
    monkeypatch: pytest.MonkeyPatch, daemon: bool
) -> None:
    """Are only identified Slurm step daemons exempt from environment visibility?"""
    process = Path("/proc/123")
    monkeypatch.setattr(Path, "iterdir", lambda _: [process])
    monkeypatch.setattr(Path, "stat", lambda _: argparse.Namespace(st_uid=os.getuid()))

    def read_bytes(path: Path) -> bytes:
        if path.name == "environ":
            raise PermissionError("not dumpable")
        return b"slurmstepd: [123.batch]\0" if daemon else b"python model_worker.py\0"

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(Path, "read_text", lambda _: "slurmstepd" if daemon else "python")
    assert fault_injector.owned_processes("run-a", allow_unreadable=True) == []
    if daemon:
        assert fault_injector.owned_processes("run-a") == []
    else:
        with pytest.raises(PermissionError):
            fault_injector.owned_processes("run-a")


@pytest.mark.parametrize(
    "observer_job,process_job",
    [("123", "123"), ("123", "1234"), ("123", None), (None, "1234"), ("999", "1234")],
)
def test_unreadable_process_requires_allocation_ownership_proof(
    monkeypatch: pytest.MonkeyPatch, observer_job: str | None, process_job: str | None
) -> None:
    """Can a protected process in another Slurm job block this run, or ours be hidden?"""
    process = Path("/proc") / str(os.getpid() + 1)
    known = {**identity(), "pid": int(process.name)}
    with monkeypatch.context() as patch:
        patch.setenv("SLURM_JOB_ID", "123")
        patch.setattr(Path, "iterdir", lambda _: [process])
        patch.setattr(Path, "stat", lambda _: argparse.Namespace(st_uid=os.getuid()))
        patch.setattr(Path, "read_bytes", Mock(side_effect=PermissionError("not dumpable")))

        def read_text(path: Path) -> str:
            if path.name == "comm":
                return "python"
            job = process_job if path.parent == process else observer_job
            return (
                f"0::/system.slice/slurmstepd.scope/job_{job}/step_0/user/task_0\n"
                if job
                else "0::/\n"
            )

        patch.setattr(Path, "read_text", read_text)
        if observer_job == "123" and process_job == "1234":
            assert fault_injector.owned_processes("run-a") == []
        else:
            with pytest.raises(PermissionError):
                fault_injector.owned_processes("run-a")
        patch.setattr(fault_injector, "process_identity", lambda _: known)
        assert fault_injector.owned_processes("run-a", [known]) == [known]


def test_late_step_exit_cannot_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Can a zero exit arriving after the active deadline qualify as bounded shutdown?"""
    clock = [0.0]
    worker = {
        **identity(0),
        "graphs_enabled": False,
        "communication": {"NVLinkOneSided": {"cft_capable": False, "ep_size": 1, "ep_rank": 0}},
    }
    clean = {**snapshot(), "hostname": worker["hostname"]}
    live = {**clean, "live_processes": [worker]}
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    args = argparse.Namespace(
        output_dir=tmp_path,
        target_rank=1,
        model=tmp_path,
        launcher=["srun"],
        python="python3",
        client_timeout_s=30,
        cleanup_timeout_s=60,
        startup_timeout_s=1,
        shutdown_timeout_s=30,
        ranks=1,
        graphs_requested=False,
    )
    process = Mock(returncode=0)
    process.poll.side_effect = lambda: None if clock[0] == 0 else 0

    def launch(*_: object, **__: object) -> Mock:
        directory = tmp_path / "initial"
        (directory / "workers").mkdir()
        (directory / "workers" / "0.json").write_text(json.dumps(worker))
        (directory / "healthy.json").write_text("{}")
        (directory / "shutdown.json").write_text("{}")
        return process

    monkeypatch.setattr(fault_injection.subprocess, "Popen", launch)
    monkeypatch.setattr(fault_injection, "_probe", Mock(side_effect=[[live], [clean], [clean]]))
    monkeypatch.setattr(fault_injection, "_steps", Mock(return_value=["123.1"]))
    monkeypatch.setattr(fault_injection, "_stop", Mock())
    monkeypatch.setattr(fault_injection.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        fault_injection.time, "sleep", lambda _: clock.__setitem__(0, clock[0] + 2000.0)
    )
    with pytest.raises(TimeoutError):
        fault_injection.run_phase(args, [clean], "initial", "healthy")
    assert (tmp_path / "initial" / "intervention.json").exists()


@pytest.mark.parametrize("reused", [False, True])
def test_known_identity_survives_empty_environment(
    monkeypatch: pytest.MonkeyPatch, reused: bool
) -> None:
    """Can an empty zombie environment hide an unreaped process, or retain a reused PID?"""
    known = {**identity(), "pid": os.getpid() + 1}
    actual = {**known, "state": "Z", "start_ticks": 0} if reused else {**known, "state": "Z"}
    monkeypatch.setattr(Path, "iterdir", lambda _: [Path(f"/proc/{known['pid']}")])
    monkeypatch.setattr(Path, "stat", lambda _: argparse.Namespace(st_uid=os.getuid()))
    monkeypatch.setattr(Path, "read_bytes", lambda _: b"")
    monkeypatch.setattr(fault_injector, "process_identity", lambda _: actual)
    assert fault_injector.owned_processes("run-a", [known]) == ([] if reused else [actual])


@pytest.mark.parametrize("error_type", [None, "RequestError", "TimeoutError"])
def test_process_loss_client_outcome_is_bounded_native_error_or_step_death(
    tmp_path: Path, error_type: str | None
) -> None:
    """Can a client timeout pass as native MPI error propagation?"""
    worker = identity()
    record(tmp_path, "trigger", {"run_id": "run-a", "event": "between_requests"})
    record(tmp_path, "injection_intent", {"scenario": "worker_sigkill_idle", "target": worker})
    (tmp_path / "launcher.log").write_text(
        f"srun: error: task {worker['rank']}: Exited with exit code 137"
    )
    if error_type:
        record(tmp_path, "client_error", {"type": error_type, "message": "MPI_ERR_OTHER"})
    if error_type == "TimeoutError":
        with pytest.raises(AssertionError, match="client error"):
            validate_fault_evidence(tmp_path, "worker_sigkill_idle", [worker])
    else:
        validate_fault_evidence(tmp_path, "worker_sigkill_idle", [worker])


@pytest.mark.parametrize("manager", ["systemd", "(sd-pam)"])
@pytest.mark.parametrize(
    "change",
    [{}, {"comm": "python"}, {"cgroup": "0::/\n"}, {"cmdline": b"python model_worker.py\0"}],
)
def test_user_manager_exemption_requires_full_identity(
    monkeypatch: pytest.MonkeyPatch, manager: str, change: dict
) -> None:
    """Can the host user manager block cleanup, or a model process be mistaken for it?"""
    process = Path("/proc") / str(os.getpid() + 1)
    known = {**identity(), "pid": int(process.name)}
    metadata = {
        "comm": manager,
        "cgroup": f"0::/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/init.scope\n",
        "cmdline": b"/usr/lib/systemd/systemd\0--user\0"
        if manager == "systemd"
        else b"(sd-pam)\0\0",
        **change,
    }
    with monkeypatch.context() as patch:
        patch.setattr(Path, "iterdir", lambda _: [process])
        patch.setattr(Path, "stat", lambda _: argparse.Namespace(st_uid=os.getuid()))
        patch.setattr(Path, "read_text", lambda path: metadata[path.name])

        def read_bytes(path: Path) -> bytes:
            if path.name == "environ":
                raise PermissionError("not dumpable")
            return metadata[path.name]

        patch.setattr(Path, "read_bytes", read_bytes)
        patch.delenv("SLURM_JOB_ID", raising=False)
        if not change:
            assert fault_injector.owned_processes("run-a") == []
        else:
            with pytest.raises(PermissionError):
                fault_injector.owned_processes("run-a")
        patch.setattr(fault_injector, "process_identity", lambda _: known)
        assert fault_injector.owned_processes("run-a", [known]) == [known]


@pytest.mark.parametrize("interrupt", [False, True])
def test_cleanup_backstop_waits_for_owned_step_after_control_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt: bool
) -> None:
    """Can clean GPU observations hide a Slurm step still draining after cancellation?"""
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    args = argparse.Namespace(
        output_dir=tmp_path,
        target_rank=1,
        model=tmp_path,
        launcher=["srun"],
        python="python3",
        client_timeout_s=30,
        cleanup_timeout_s=60,
        startup_timeout_s=1,
        shutdown_timeout_s=30,
        ranks=2,
        graphs_requested=False,
    )
    process = Mock()
    process.poll.return_value = 1
    monkeypatch.setattr(fault_injection.subprocess, "Popen", lambda *_, **__: process)

    def stop(*_: object) -> None:
        if interrupt:
            os.kill(os.getpid(), signal.SIGINT)
            os.kill(os.getpid(), signal.SIGTERM)
        raise TimeoutError("cancel budget")

    def interrupted(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt("cleanup interrupted")

    handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    monkeypatch.setattr(fault_injection, "_stop", stop)
    monkeypatch.setattr(fault_injection, "_probe", lambda *_, **__: [snapshot()])
    steps = Mock(side_effect=[["123.0"], []])
    monkeypatch.setattr(fault_injection, "_steps", steps)
    monkeypatch.setattr(fault_injection.time, "sleep", Mock())
    try:
        with pytest.raises(AssertionError, match="healthy readiness"):
            fault_injection.run_phase(args, [snapshot()], "initial", "worker_sigkill_idle")
        assert all(signal.getsignal(sig) == interrupted for sig in handlers)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    assert steps.call_count == 2
    assert (tmp_path / "initial" / "cleanup_control_error.json").exists()
    assert not (tmp_path / "initial" / "cleanup_failed.json").exists()
    proof = json.loads((tmp_path / "initial" / "forced_cleanup.json").read_text())
    assert proof["forced"] and proof["owned_steps"] == []


def test_worker_identity_probe_does_not_inspect_environments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Can a read-only death witness stay independent of unrelated process environments?"""
    worker = identity()
    monkeypatch.setattr(Path, "read_bytes", Mock(side_effect=AssertionError("environment read")))
    assert fault_injector.worker_processes([worker]) == [process_identity(os.getpid())]
    monkeypatch.setattr(fault_injector, "process_identity", lambda _: {**worker, "start_ticks": 0})
    assert fault_injector.worker_processes([worker]) == []
    monkeypatch.setattr(fault_injector, "process_identity", Mock(side_effect=ProcessLookupError))
    assert fault_injector.worker_processes([worker]) == []
    monkeypatch.setattr(fault_injector, "process_identity", lambda _: {**worker, "uid": -1})
    with pytest.raises(PermissionError):
        fault_injector.worker_processes([worker])


@pytest.mark.parametrize("missing", [None, "target_death", "client_error", "native_abort"])
def test_ulfm_kill_requires_death_error_and_abort_evidence(
    tmp_path: Path, missing: str | None
) -> None:
    """Can launcher exit alone qualify target death or bounded client failure under ULFM?"""
    worker = identity()
    record(tmp_path, "trigger", {"run_id": "run-a", "event": "between_requests"})
    record(tmp_path, "injection_intent", {"scenario": "worker_sigkill_idle", "target": worker})
    record(tmp_path, "target_death", {"target": worker, "alive": False})
    record(tmp_path, "client_error", {"type": "RequestError", "message": "MPI_ERR_PROC_FAILED"})
    (tmp_path / "launcher.log").write_text(
        f"Rank{worker['rank']} MGMN worker node exit code: 137\n"
        + ("" if missing == "native_abort" else "MPI_ABORT was invoked")
    )
    if missing in ("target_death", "client_error"):
        (tmp_path / f"{missing}.json").unlink()
    if missing:
        with pytest.raises((AssertionError, FileNotFoundError)):
            validate_fault_evidence(tmp_path, "worker_sigkill_idle", [worker], "ulfm")
    else:
        validate_fault_evidence(tmp_path, "worker_sigkill_idle", [worker], "ulfm")


@pytest.mark.parametrize("scenario", ["healthy", "worker_sigkill_idle", "fence_round_mismatch"])
@pytest.mark.parametrize("invalid", [None, "controller", "cleanup"])
def test_restart_benchmark_uses_one_clock_and_verified_cleanup(
    scenario: str, invalid: str | None
) -> None:
    """Does failure-to-response include cleanup and reject clocks or overlapping restarts?"""
    initial = dict(
        producer_host="node-a",
        controller_pid=1,
        clock_source="single parent CLOCK_MONOTONIC",
        started_monotonic_s=1000,
        seconds_to_received_event=dict(
            first_result=10,
            healthy=11,
            injection_intent=12,
            injection_result=12,
            client_error=13,
            step_exit=14,
            cleanup=15,
        ),
    )
    restarted = {**initial, "started_monotonic_s": 1016}
    if invalid == "controller":
        restarted["controller_pid"] = 2
    elif invalid == "cleanup":
        restarted["seconds_to_received_event"] = {
            **initial["seconds_to_received_event"],
            "cleanup": 1,
        }
    if invalid:
        with pytest.raises(ValueError):
            restart_benchmark(initial, restarted, scenario)
    else:
        result = restart_benchmark(initial, restarted, scenario)
        assert result["restart_first_response_s"] == 10
        if scenario != "healthy":
            assert result["seconds_from_injection_receipt"] == dict(
                client_error=1,
                step_exit=2,
                cleanup=3,
                restart_launch=4,
                restart_first_response=14,
                restart_ready=15,
            )

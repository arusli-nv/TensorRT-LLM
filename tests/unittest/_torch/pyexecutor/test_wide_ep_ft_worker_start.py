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
"""Tests for the startup of a Wide-EP fault-tolerant deployment that runs each rank in its own CPU process.

Each process runs the real worker start (``start_worker`` on rank 0, ``worker_node_main`` on the others) and the real
membership authority. A fake ``worker_main`` sleeps to stand in for the model load, calls ``wait_for_every_rank``,
and serves until it is ended.
"""

import os
import queue
import signal
import socket
import subprocess  # nosec B404
import sys
import threading
import time

import pytest

pytestmark = pytest.mark.cpu_only

STARTUP_S = 300  # each process imports tensorrt_llm
RENDEZVOUS_TIMEOUT_S = 20.0  # the deployment's, scaled down for the test
# A rank failure during startup ends every process within this bound, far below the model load time (LOAD_S).
PROMPT_EXIT_S = 15.0
LOAD_S = 3600.0

_WORKER = r"""
import os, sys, time
from types import SimpleNamespace

from tensorrt_llm._torch.pyexecutor import rank_failure_recovery as recovery
from tensorrt_llm._wide_ep_fault_tolerance import (
    set_wide_ep_fault_tolerance_enabled,
    wide_ep_fault_tolerance_rank,
)
from tensorrt_llm.executor import worker

load_s, recovery.RENDEZVOUS_TIMEOUT_S = float(sys.argv[1]), float(sys.argv[2])


def worker_main(**args):
    set_wide_ep_fault_tolerance_enabled(True)  # as worker_main does first
    recovery.start_worker(args)
    rank = wide_ep_fault_tolerance_rank().rank
    print(f"STARTED {rank} {args['marker']}", flush=True)
    if rank == 0:
        print(f"AUTHORITY {recovery._authority.pid}", flush=True)
    time.sleep(load_s)  # the model load
    recovery.wait_for_every_rank()
    # Bind the client as the executor does, with the real handler for a failed authority.
    executor = SimpleNamespace(rank=rank, _exit=lambda status, reason: recovery._end_process(rank, status, reason))
    recovery._client.bind(
        recovery._on_record_at_startup,
        on_error=lambda error: recovery.RankFailureRecovery._on_client_error(executor, error),
    )
    print(f"SERVING {rank}", flush=True)
    time.sleep(3600)


worker.worker_main = worker_main
if os.environ["TLLM_FAULT_TOLERANCE_RANK"].startswith("0,"):
    worker_main(hf_model_dir=None, marker="rank-0-arguments", worker_queues=None, rpc_addr=None, hmac_key=b"")
else:
    recovery.worker_node_main()
"""

_LAUNCHER_PREFIXES = ("OMPI_", "PMIX_", "PMI_", "PRTE_", "SLURM_")


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class Deployment:
    """Run the processes of a deployment of ``world`` ranks and queue each printed line in :attr:`lines`."""

    def __init__(self, world, rendezvous_timeout_s=RENDEZVOUS_TIMEOUT_S):
        self.world = world
        self.rendezvous_timeout_s = rendezvous_timeout_s
        self.port = _free_port()
        self.processes = {}
        self.lines = queue.Queue()
        self.seen = []
        self._readers = []

    def start(self, rank, load_s=LOAD_S):
        env = {k: v for k, v in os.environ.items() if not k.startswith(_LAUNCHER_PREFIXES)}
        env.update(
            TLLM_FAULT_TOLERANCE_SHARED_SECRET="ab" * 32,
            TLLM_FAULT_TOLERANCE_RENDEZVOUS=f"127.0.0.1:{self.port}",
            TLLM_FAULT_TOLERANCE_RANK=f"{rank},{self.world},{self.world},0",
            TRT_LLM_NO_LIB_INIT="1",
            MPI4PY_RC_INITIALIZE="false",
            CUDA_VISIBLE_DEVICES="",
        )
        process = subprocess.Popen(  # nosec B603
            [sys.executable, "-c", _WORKER, str(load_s), str(self.rendezvous_timeout_s)],
            env=env,
            stdout=subprocess.PIPE,
            text=True,
        )
        self.processes[rank] = process
        reader = threading.Thread(target=self._read, args=(rank, process), daemon=True)
        reader.start()
        self._readers.append(reader)
        return process

    def _read(self, rank, process):
        for line in process.stdout:
            self.lines.put((rank, line.strip()))

    def wait_for(self, predicate, timeout_s):
        """Return the first printed (rank, line) pair for which ``predicate`` holds."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                item = self.lines.get(timeout=1)
            except queue.Empty:
                continue
            self.seen.append(item)
            if predicate(*item):
                return item
        raise AssertionError(f"timed out; printed {self.seen}")

    def wait_for_all(self, word, ranks, timeout_s=STARTUP_S):
        pending = set(ranks)
        deadline = time.monotonic() + timeout_s
        while pending:
            rank, _ = self.wait_for(
                lambda r, line: line.startswith(word) and r in pending, deadline - time.monotonic()
            )
            pending.discard(rank)

    def authority_pid(self):
        return int(next(line for _, line in self.seen if line.startswith("AUTHORITY")).split()[1])

    def statuses_within(self, ranks, timeout_s):
        """Return the exit status of each of ``ranks`` and fail unless all end within ``timeout_s``."""
        deadline = time.monotonic() + timeout_s
        statuses = {}
        for rank in ranks:
            try:
                statuses[rank] = self.processes[rank].wait(max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                raise AssertionError(f"rank {rank} did not end within {timeout_s} s") from None
        return statuses

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.end()

    def end(self):
        for process in self.processes.values():
            if process.poll() is None:
                process.kill()
            process.wait()
        for reader in self._readers:  # each ends at its process's end of output
            reader.join(10)


def _process_ended(pid, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with open(f"/proc/{pid}/stat") as stat:
                if stat.read().rsplit(")", 1)[1].split()[0] == "Z":
                    return True  # ended, not reaped yet
        except FileNotFoundError:
            return True
        time.sleep(0.2)
    return False


@pytest.mark.timeout(900)
def test_ranks_wait_for_late_rank_0():
    """Run worker_main on every rank with rank 0's arguments, even when rank 0 starts last.

    The other ranks wait for rank 0 at the rendezvous. The deployment serves once every rank has started.
    """
    with Deployment(3) as ranks:
        ranks.start(1, load_s=1)
        ranks.start(2, load_s=1)
        time.sleep(5)
        ranks.start(0, load_s=1)
        ranks.wait_for_all("STARTED", range(3))
        assert {line for _, line in ranks.seen if line.startswith("STARTED")} == {
            f"STARTED {rank} rank-0-arguments" for rank in range(3)
        }
        ranks.wait_for_all("SERVING", range(3))


@pytest.mark.timeout(900)
@pytest.mark.parametrize("failed_rank", [0, 1])
def test_startup_failure_ends_every_process(failed_rank):
    """Abort as soon as the failed rank's connection closes, without waiting for a timeout."""
    with Deployment(3) as ranks:
        for rank in range(3):
            ranks.start(rank)
        ranks.wait_for_all("STARTED", range(3))
        authority = ranks.authority_pid()
        ranks.processes[failed_rank].kill()
        others = [rank for rank in range(3) if rank != failed_rank]
        assert set(ranks.statuses_within(others, PROMPT_EXIT_S).values()) == {1}
        assert _process_ended(authority, PROMPT_EXIT_S)


@pytest.mark.timeout(900)
def test_late_rendezvous_ends_within_bound():
    """Abort at the rendezvous timeout when a rank starts too late.

    The late rank finds no store and ends at its own timeout.
    """
    with Deployment(3) as ranks:
        ranks.start(0)
        ranks.wait_for(lambda rank, line: line.startswith("STARTED"), STARTUP_S)
        ranks.start(1)
        assert ranks.statuses_within([0], 2 * RENDEZVOUS_TIMEOUT_S)[0] == 1
        ranks.start(2)
        statuses = ranks.statuses_within([1, 2], STARTUP_S + 2 * RENDEZVOUS_TIMEOUT_S)
        assert 0 not in statuses.values()


@pytest.mark.timeout(900)
def test_rank_without_rank_0_times_out():
    with Deployment(2) as ranks:
        ranks.start(1)
        assert ranks.statuses_within([1], STARTUP_S + RENDEZVOUS_TIMEOUT_S)[1] != 0


@pytest.mark.timeout(900)
def test_ranks_serve_without_authority():
    """Keep serving without fault tolerance when the authority dies, since ranks hold no lease.

    If rank 0 also ends, node 0's supervisor ends the job, as test_node_supervisor.py checks.
    """
    with Deployment(3) as ranks:
        for rank in range(3):
            ranks.start(rank, load_s=1)
        ranks.wait_for_all("SERVING", range(3))
        os.kill(ranks.authority_pid(), signal.SIGKILL)
        time.sleep(5)
        assert all(process.poll() is None for process in ranks.processes.values())

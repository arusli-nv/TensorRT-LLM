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
"""CPU-only targeting tests for the one-rank fault injector."""

from __future__ import annotations

import json
import signal
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fault_injector import capture_process_identity, kill_rank


class FaultInjectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.directory = Path(self.temp_dir.name)
        self.processes: list[subprocess.Popen[bytes]] = [
            subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            for _ in range(2)
        ]

    def tearDown(self) -> None:
        for process in self.processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)

    def _write_rank_map(self) -> Path:
        records = [
            capture_process_identity(rank, process.pid)
            for rank, process in enumerate(self.processes)
        ]
        path = self.directory / "ranks.json"
        path.write_text(json.dumps({"ranks": records}))
        return path

    def test_kills_only_selected_rank_once(self) -> None:
        rank_map = self._write_rank_map()
        output_dir = self.directory / "result"

        kill_rank(rank_map, 1, output_dir)

        self.assertEqual(self.processes[1].wait(timeout=5), -signal.SIGKILL)
        self.assertIsNone(self.processes[0].poll())
        intent = json.loads((output_dir / "injection_intent.json").read_text())
        result = json.loads((output_dir / "injection_result.json").read_text())
        self.assertEqual(intent["pid"], self.processes[1].pid)
        self.assertEqual(result, {"status": "signal_sent"})
        with self.assertRaisesRegex(FileExistsError, "already has an injection record"):
            kill_rank(rank_map, 1, output_dir)

    def test_command_line_inspect_and_kill(self) -> None:
        script = Path(__file__).with_name("fault_injector.py")
        rank_map = self.directory / "ranks.json"
        inspected = subprocess.run(
            [
                sys.executable,
                str(script),
                "inspect",
                "--world-rank",
                "1",
                "--pid",
                str(self.processes[1].pid),
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        rank_map.write_text(inspected.stdout)
        injected = subprocess.run(
            [
                sys.executable,
                str(script),
                "kill",
                "--rank-map",
                str(rank_map),
                "--target-rank",
                "1",
                "--output-dir",
                str(self.directory / "result"),
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )

        self.assertEqual(injected.returncode, 0)
        self.assertEqual(self.processes[1].wait(timeout=5), -signal.SIGKILL)
        self.assertIsNone(self.processes[0].poll())

    def test_refuses_rank_zero(self) -> None:
        rank_map = self._write_rank_map()
        with self.assertRaisesRegex(ValueError, "non-rank-0"):
            kill_rank(rank_map, 0, self.directory / "result")
        self.assertTrue(all(process.poll() is None for process in self.processes))

    def test_refuses_stale_process_identity(self) -> None:
        rank_map = self._write_rank_map()
        data = json.loads(rank_map.read_text())
        data["ranks"][1]["start_time_ticks"] += 1
        rank_map.write_text(json.dumps(data))

        with self.assertRaisesRegex(ValueError, "identity changed"):
            kill_rank(rank_map, 1, self.directory / "result")
        self.assertTrue(all(process.poll() is None for process in self.processes))

    def test_records_signal_error_without_killing_target(self) -> None:
        rank_map = self._write_rank_map()
        output_dir = self.directory / "result"
        with patch(
            "fault_injector.signal.pidfd_send_signal", side_effect=PermissionError("denied")
        ):
            with self.assertRaisesRegex(PermissionError, "denied"):
                kill_rank(rank_map, 1, output_dir)

        result = json.loads((output_dir / "injection_result.json").read_text())
        self.assertEqual(result, {"status": "error", "error": "denied"})
        self.assertTrue(all(process.poll() is None for process in self.processes))

    def test_refuses_duplicate_or_remote_rank(self) -> None:
        rank_map = self._write_rank_map()
        data = json.loads(rank_map.read_text())
        data["ranks"].append(dict(data["ranks"][1]))
        rank_map.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "exactly one record"):
            kill_rank(rank_map, 1, self.directory / "result")

        data["ranks"].pop()
        data["ranks"][1]["hostname"] = socket.gethostname() + "-other-node"
        rank_map.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "not on this node"):
            kill_rank(rank_map, 1, self.directory / "result")
        self.assertTrue(all(process.poll() is None for process in self.processes))


if __name__ == "__main__":
    unittest.main()

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
"""CPU-only checks for process-loss observation and injection gating."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from process_loss_harness import _request_result, _wait_for_model, run_case


class ProcessLossHarnessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.directory = Path(self.temp_dir.name)

    @patch("process_loss_harness.kill_rank")
    @patch("process_loss_harness._request_result")
    @patch("process_loss_harness._wait_for_model", return_value="test-model")
    def test_records_post_failure_error_without_claiming_recovery(
        self, wait_for_model, request_result, kill_rank
    ) -> None:
        request_result.side_effect = [
            {"status": "response", "response": {"choices": [{"text": " Paris"}]}},
            {"status": "error", "error_type": "RemoteDisconnected", "error": "closed"},
        ]
        output_dir = self.directory / "run"
        summary = run_case(
            "http://localhost:8000", self.directory / "ranks.json", 1, output_dir, 1, 1
        )

        self.assertEqual(summary["status"], "observation_complete")
        self.assertEqual(kill_rank.call_count, 1)
        self.assertEqual(
            json.loads((output_dir / "post_failure_request.json").read_text())["status"],
            "error",
        )

    @patch("process_loss_harness.kill_rank")
    @patch("process_loss_harness._request_result")
    @patch("process_loss_harness._wait_for_model", return_value="test-model")
    def test_post_failure_response_is_only_an_observation(
        self, wait_for_model, request_result, kill_rank
    ) -> None:
        request_result.side_effect = [
            {"status": "response", "response": {"choices": [{"text": " Paris"}]}},
            {"status": "response", "response": {"choices": [{"text": " Rome"}]}},
        ]
        output_dir = self.directory / "run"
        summary = run_case(
            "http://localhost:8000", self.directory / "ranks.json", 1, output_dir, 1, 1
        )

        self.assertEqual(summary["status"], "observation_complete")
        self.assertEqual(kill_rank.call_count, 1)
        self.assertEqual(
            json.loads((output_dir / "post_failure_request.json").read_text())["response"][
                "choices"
            ][0]["text"],
            " Rome",
        )

        with self.assertRaises(FileExistsError):
            run_case("http://localhost:8000", self.directory / "ranks.json", 1, output_dir, 1, 1)
        self.assertEqual(kill_rank.call_count, 1)

    @patch("process_loss_harness.kill_rank")
    @patch("process_loss_harness._request_result", return_value={"status": "error"})
    @patch("process_loss_harness._wait_for_model", return_value="test-model")
    def test_never_injects_before_healthy_request(
        self, wait_for_model, request_result, kill_rank
    ) -> None:
        output_dir = self.directory / "run"
        summary = run_case(
            "http://localhost:8000", self.directory / "ranks.json", 1, output_dir, 1, 1
        )

        self.assertEqual(summary["status"], "not_injected")
        self.assertFalse((output_dir / "post_failure_request.json").exists())
        kill_rank.assert_not_called()

    @patch("process_loss_harness.kill_rank")
    @patch("process_loss_harness._wait_for_model", side_effect=TimeoutError("not ready"))
    def test_readiness_timeout_never_injects(self, wait_for_model, kill_rank) -> None:
        output_dir = self.directory / "run"
        summary = run_case(
            "http://localhost:8000", self.directory / "ranks.json", 1, output_dir, 1, 1
        )

        self.assertEqual(summary["status"], "not_injected")
        self.assertEqual(summary["error_type"], "TimeoutError")
        kill_rank.assert_not_called()

    @patch("process_loss_harness.kill_rank")
    @patch(
        "process_loss_harness._request_result",
        return_value={"status": "response", "response": {"choices": [{"text": ""}]}},
    )
    @patch("process_loss_harness._wait_for_model", return_value="test-model")
    def test_empty_healthy_completion_never_injects(
        self, wait_for_model, request_result, kill_rank
    ) -> None:
        summary = run_case(
            "http://localhost:8000", self.directory / "ranks.json", 1, self.directory / "run", 1, 1
        )

        self.assertEqual(summary["status"], "not_injected")
        kill_rank.assert_not_called()

    @patch("process_loss_harness.kill_rank", side_effect=PermissionError("denied"))
    @patch(
        "process_loss_harness._request_result",
        return_value={"status": "response", "response": {"choices": [{"text": " Paris"}]}},
    )
    @patch("process_loss_harness._wait_for_model", return_value="test-model")
    def test_injection_error_is_not_recorded_as_an_observation(
        self, wait_for_model, request_result, kill_rank
    ) -> None:
        output_dir = self.directory / "run"
        summary = run_case(
            "http://localhost:8000", self.directory / "ranks.json", 1, output_dir, 1, 1
        )

        self.assertEqual(summary["status"], "injection_uncertain")
        self.assertEqual(summary["stage"], "injection")
        self.assertEqual(summary["error_type"], "PermissionError")
        self.assertFalse((output_dir / "post_failure_request.json").exists())

    @patch("process_loss_harness.kill_rank")
    def test_rejects_rank_zero_before_creating_run(self, kill_rank) -> None:
        output_dir = self.directory / "run"
        with self.assertRaisesRegex(ValueError, "non-rank-0"):
            run_case("http://localhost:8000", self.directory / "ranks.json", 0, output_dir, 1, 1)
        self.assertFalse(output_dir.exists())
        kill_rank.assert_not_called()

    def test_readiness_polling_has_a_deadline(self) -> None:
        with self.assertRaisesRegex(TimeoutError, "readiness deadline"):
            _wait_for_model("http://127.0.0.1:0", 0.05)

    @patch("process_loss_harness._completion", side_effect=ConnectionResetError("closed"))
    def test_client_disconnect_is_recorded(self, completion) -> None:
        result = _request_result("http://localhost:8000", "test-model", "prompt", 1)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "ConnectionResetError")


if __name__ == "__main__":
    unittest.main()

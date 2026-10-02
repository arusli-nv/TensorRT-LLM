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
"""Record one between-request model process loss in an externally launched job."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from fault_injector import kill_rank


def _write_json(path: Path, record: dict) -> None:
    with path.open("x") as output:
        json.dump(record, output, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())


def _get_model(base_url: str, timeout_s: float) -> str:
    with urllib.request.urlopen(f"{base_url}/v1/models", timeout=timeout_s) as response:
        return json.load(response)["data"][0]["id"]


def _wait_for_model(base_url: str, timeout_s: float) -> str:
    deadline = time.monotonic() + timeout_s
    while True:
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            raise TimeoutError("model did not become ready before the readiness deadline")
        try:
            return _get_model(base_url, min(remaining_s, 2.0))
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))


def _completion(base_url: str, model: str, prompt: str, timeout_s: float) -> dict:
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps({"model": model, "prompt": prompt, "max_tokens": 16}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.load(response)


def _request_result(base_url: str, model: str, prompt: str, timeout_s: float) -> dict:
    started_at_utc = datetime.now(timezone.utc).isoformat()
    try:
        response = _completion(base_url, model, prompt, timeout_s)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {
            "status": "error",
            "started_at_utc": started_at_utc,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
        }
    return {
        "status": "response",
        "started_at_utc": started_at_utc,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "response": response,
    }


def run_case(
    base_url: str,
    rank_map_path: Path,
    target_rank: int,
    output_dir: Path,
    readiness_timeout_s: float,
    request_timeout_s: float,
) -> dict:
    """Observe one worker loss; do not claim recovery from a completed observation."""
    for name, value in (
        ("readiness_timeout_s", readiness_timeout_s),
        ("request_timeout_s", request_timeout_s),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if target_rank <= 0:
        raise ValueError("the first experiment requires a non-rank-0 target")
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "status": "not_injected",
        "stage": "waiting_for_model",
        "target_rank": target_rank,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        model = _wait_for_model(base_url, readiness_timeout_s)
        summary["model"] = model
        summary["stage"] = "healthy_request"
        healthy = _request_result(base_url, model, "The capital of France is", request_timeout_s)
        _write_json(output_dir / "healthy_request.json", healthy)
        response = healthy.get("response")
        choices = response.get("choices") if isinstance(response, dict) else None
        if (
            healthy["status"] != "response"
            or not isinstance(choices, list)
            or not any(isinstance(choice, dict) and choice.get("text") for choice in choices)
        ):
            raise RuntimeError("healthy model request did not produce a completion")
        summary["stage"] = "injection"
        summary["status"] = "injection_uncertain"
        kill_rank(rank_map_path, target_rank, output_dir / "injection")
        summary["status"] = "signal_sent"
        summary["stage"] = "post_failure_request"
        post_failure = _request_result(
            base_url, model, "The capital of Italy is", request_timeout_s
        )
        _write_json(output_dir / "post_failure_request.json", post_failure)
        summary["status"] = "observation_complete"
        summary["stage"] = "done"
    except (OSError, ValueError, RuntimeError) as error:
        summary["error_type"] = type(error).__name__
        summary["error"] = str(error)
    finally:
        summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(output_dir / "run_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--rank-map", type=Path, required=True)
    parser.add_argument("--target-rank", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--readiness-timeout-s", type=float, default=600)
    parser.add_argument("--request-timeout-s", type=float, default=60)
    args = parser.parse_args()
    summary = run_case(
        args.base_url.rstrip("/"),
        args.rank_map,
        args.target_rank,
        args.output_dir,
        args.readiness_timeout_s,
        args.request_timeout_s,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary["status"] == "observation_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

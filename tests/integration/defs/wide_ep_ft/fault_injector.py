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
"""Kill one manually identified local model rank for a fault-tolerance experiment."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path


def _read_process_start_time_ticks(pid: int) -> int:
    stat = Path(f"/proc/{pid}/stat").read_text()
    fields = stat.rpartition(") ")[2].split()
    return int(fields[19])


def capture_process_identity(world_rank: int, pid: int) -> dict[str, int | str]:
    """Record the identity of a manually identified local model process.

    Args:
        world_rank: Model world rank assigned to this process by the caller.
        pid: Linux PID visible from this process's PID namespace.

    Returns:
        Rank, host, PID, and process start time for the rank map.
    """
    if world_rank < 0 or pid <= 0:
        raise ValueError("world_rank must be non-negative and pid must be positive")
    return {
        "world_rank": world_rank,
        "hostname": socket.gethostname(),
        "pid": pid,
        "start_time_ticks": _read_process_start_time_ticks(pid),
    }


def _load_target_rank(rank_map_path: Path, target_rank: int) -> dict[str, int | str]:
    if target_rank <= 0:
        raise ValueError("the first experiment only permits a non-rank-0 target")
    ranks = json.loads(rank_map_path.read_text())["ranks"]
    if not isinstance(ranks, list):
        raise ValueError("rank map must contain a ranks list")
    matches = [record for record in ranks if record["world_rank"] == target_rank]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one record for world rank {target_rank}")
    record = matches[0]
    if record["hostname"] != socket.gethostname():
        raise ValueError("target process is not on this node")
    if (
        type(record["pid"]) is not int
        or record["pid"] <= 0
        or type(record["start_time_ticks"]) is not int
        or record["start_time_ticks"] <= 0
    ):
        raise ValueError("target PID and start time must be positive integers")
    return record


def _write_json(path: Path, record: dict[str, int | str]) -> None:
    with path.open("x") as output:
        json.dump(record, output, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())


def kill_rank(rank_map_path: Path, target_rank: int, output_dir: Path) -> None:
    """Send SIGKILL once to a local rank and save the injection outcome.

    Args:
        rank_map_path: JSON map of manually identified model ranks.
        target_rank: Nonzero model world rank to kill.
        output_dir: Directory for one-shot intent and result records.
    """
    target = _load_target_rank(rank_map_path, target_rank)
    output_dir.mkdir(parents=True, exist_ok=True)
    intent_path = output_dir / "injection_intent.json"
    result_path = output_dir / "injection_result.json"
    if intent_path.exists() or result_path.exists():
        raise FileExistsError("this output directory already has an injection record")

    pidfd = os.pidfd_open(target["pid"])
    try:
        if _read_process_start_time_ticks(target["pid"]) != target["start_time_ticks"]:
            raise ValueError("target process identity changed since inspection")
        intent = {
            **target,
            "signal": "SIGKILL",
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(intent_path, intent)
        try:
            signal.pidfd_send_signal(pidfd, signal.SIGKILL)
        except OSError as exc:
            _write_json(result_path, {"status": "error", "error": str(exc)})
            raise
        _write_json(result_path, {"status": "signal_sent"})
    finally:
        os.close(pidfd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_command = commands.add_parser("inspect", help="print one rank-map record")
    inspect_command.add_argument("--world-rank", type=int, required=True)
    inspect_command.add_argument("--pid", type=int, required=True)
    kill_command = commands.add_parser("kill", help="kill one manually identified local rank")
    kill_command.add_argument("--rank-map", type=Path, required=True)
    kill_command.add_argument("--target-rank", type=int, required=True)
    kill_command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            print(
                json.dumps(
                    {"ranks": [capture_process_identity(args.world_rank, args.pid)]}, indent=2
                )
            )
        else:
            kill_rank(args.rank_map, args.target_rank, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"fault injector refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

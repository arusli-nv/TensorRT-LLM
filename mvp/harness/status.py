"""One line per case of the given jobs: state, readiness, rows, kill/recovery times, counters, errors.

Usage: status.py mvp/runs/<JOB>-<id> [...]
"""

import json
import sys
from pathlib import Path


def counters(run: Path, after: float) -> str:
    """Graph captures, compiles and autotunes logged by any rank after ``after`` (wall seconds)."""
    counts = {"WIDEEP_CAPTURE": 0, "WIDEEP_COMPILE": 0, "WIDEEP_AUTOTUNE": 0}
    for log in run.glob("rank-*.log"):
        for line in log.read_text(errors="replace").splitlines():
            for key in counts:
                if line.startswith(key) and float(line.split("t=")[1].split()[0]) > after:
                    counts[key] += 1
    return " ".join(f"{k[7:].lower()}={v}" for k, v in counts.items())


def main() -> None:
    for job in map(Path, sys.argv[1:]):
        print(job.name)
        for run in sorted(p for p in job.iterdir() if p.is_dir()):
            path = run / "result.json"
            if not path.exists():
                exits = [l for l in (run / "timeline").read_text().splitlines() if "RANK_EXIT" in l] \
                    if (run / "timeline").exists() else []
                print(f"  {run.name:16} running" + (f" ({len(exits)} rank exits)" if exits else ""))
                continue
            r = json.loads(path.read_text())
            parts = [f"{r.get('state', 'RUNNING'):7}", f"ready={r.get('ready_s', 0):.0f}s"]
            for row in r.get("rows", []):
                if row["label"].startswith("warmup"):
                    continue
                tput = (row.get("output_throughput") or 0) / row.get("gpus", 32)
                parts.append(f"{row['label']}:{tput:.0f}/gpu,tpot={row.get('median_tpot_ms') or 0:.1f},"
                             f"fail={row.get('failed_requests')}")
            if "kill_wall_s" in r or "trigger_wall_s" in r:
                kill = r.get("kill_wall_s") or r.get("trigger_wall_s")
                if "all_resumed_seen_wall_s" in r:
                    parts.append(f"resumed_seen={1e3 * (r['all_resumed_seen_wall_s'] - kill):.0f}ms")
                if "first_token_after_kill_s" in r:
                    parts.append(f"ttft_after={1e3 * r['first_token_after_kill_s']:.0f}ms")
                parts.append(counters(run, kill))
            if r.get("error"):
                parts.append("ERROR " + r["error"][:160])
            print(f"  {run.name:16} " + " ".join(parts))


if __name__ == "__main__":
    main()

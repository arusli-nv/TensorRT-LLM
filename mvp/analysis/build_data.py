"""Turn mvp/runs into the numbers the write-up uses (JSON on stdout). Reads only mvp/runs.

Usage: build_data.py mvp/runs > figures/data.json

lane1: per mode (FT off, FT on, FT on after a loss, standard FT-off reference) and load, one entry per
       independent server run: output tokens/s per GPU, TPOT and TTFT p50/p99, rank 0's median decode step
       (print_iter_log), plus each sweep's drift (its repeated first load against the first).
lane2: per kill setting, one entry per kill: phase times from the recovery markers, first token of a new
       request, request outcomes, and the captures/compiles/autotunes after the kill.
ft_off: per abort+restart pair: detect+abort, teardown, restart stages, first token, total downtime.
"""

import datetime
import json
import re
import statistics
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

MODES = {"sweep-noft": "FT off", "sweep-ft": "FT on", "sweep-ft-loss": "FT on",
         "reference-noft": "FT off, standard config", "ulfm-noft": "FT off, ULFM launcher",
         "check-noft": "FT off, check 1"}
PHASES = ["watcher_detect", "recover_start", "stopped_sent", "install_recv", "resumed"]


def rows(run: Path) -> list:
    path = run / "result.json"
    return json.loads(path.read_text()).get("rows", []) if path.exists() else []


def point(row: dict) -> dict:
    return {"tok_s_per_gpu": row["output_throughput"] / row["gpus"],
            "tpot_p50_ms": row["median_tpot_ms"], "tpot_p99_ms": row["p99_tpot_ms"],
            "ttft_p50_ms": row["median_ttft_ms"], "ttft_p99_ms": row["p99_ttft_ms"],
            "failed": row.get("failed_requests"), "run": None}


ITER = re.compile(r"^\[(\S+)\] .* iter = \d+, .*prev_device_step_time = ([\d.]+)ms, .*"
                  r"'num_ctx_requests': (\d+), .*'num_generation_tokens': (\d+)", re.M)
LOG_TZ = ZoneInfo("America/Los_Angeles")  # the cluster's local time, which the log stamps use


def iterations(run: Path) -> list:
    """Rank 0's iteration log: (wall time, previous step's device time in ms, context requests, decode tokens)."""
    log = run / "rank-0.log"
    if not log.exists():
        return []
    out = []
    for stamp, dev, ctx, gen in ITER.findall(log.read_text(errors="replace")):
        t = datetime.datetime.strptime(stamp, "%m/%d/%Y-%H:%M:%S").replace(tzinfo=LOG_TZ).timestamp()
        out.append((t, float(dev), int(ctx), int(gen)))
    return out


def decode_step(its: list, row: dict):
    """Median device time of rank 0's decode-only steps during a load (the step after a decode-only step,
    so idle waits, which the log folds into the next step, are excluded)."""
    t0, t1 = row["start_wall_s"], row["start_wall_s"] + row["elapsed_s"]
    steps = [b[1] for a, b in zip(its, its[1:])
             if t0 <= a[0] <= t1 and a[2] == 0 and a[3] > 0 and b[2] == 0 and b[3] > 0]
    return round(statistics.median(steps), 2) if steps else None


def lane1(root: Path) -> dict:
    out = {}
    for job in sorted(root.glob("L1-*")):
        for run in sorted(job.iterdir()):
            # Repeated cases of one mode in a job are numbered: sweep-ft-1, sweep-ft-2.
            mode = MODES.get(re.sub(r"-\d+$", "", run.name))
            if mode is None:
                continue
            first = {}
            its = iterations(run)
            for row in rows(run):
                if row["label"].startswith("warmup") or row.get("rc") != 0:
                    continue
                # A load whose client lost requests (connection resets, seen at 512/GPU with one client process)
                # would bias throughput over the completed ones; L1-512 reran that load with the client split.
                if row.get("failed_requests"):
                    continue
                label = "FT on, after a loss" if row["label"].startswith("after-loss") else mode
                entry = point(row) | {"run": f"{job.name}/{run.name}", "decode_step_ms": decode_step(its, row)}
                key = str(row["load_per_gpu"])
                if row["label"].endswith("-repeat"):
                    out.setdefault("drift", []).append(
                        {"run": entry["run"], "mode": label, "load": key,
                         "repeat_vs_first": entry["tok_s_per_gpu"] / first[label]})
                    continue
                first.setdefault(label, entry["tok_s_per_gpu"])
                out.setdefault(label, {}).setdefault(key, []).append(entry)
    return out


# A recovery takes well under a second; markers later than this after the kill belong to the case's
# teardown (stopping mpirun is a second "loss" every rank notices).
RECOVERY_WINDOW_S = 30.0


def marks(run: Path, kill: float) -> dict:
    """Recovery markers of the kill: name -> (earliest, latest) wall time over the survivors."""
    seen = {}
    for log in run.glob("rank-*.log"):
        for name, t in re.findall(r"WIDEEP_PHASE (\w+) ([\d.]+)", log.read_text(errors="replace")):
            t = float(t)
            if not kill <= t <= kill + RECOVERY_WINDOW_S:
                continue
            lo, hi = seen.get(name, (float("inf"), 0.0))
            seen[name] = (min(lo, t), max(hi, t))
    return seen


def self_kill(run: Path):
    for log in run.glob("rank-*.log"):
        m = re.search(r"WIDEEP_SELF_KILL rank \d+ t=([\d.]+)", log.read_text(errors="replace"))
        if m:
            return float(m.group(1))
    return None


def counters(run: Path, after: float) -> dict:
    counts = {"captures": 0, "compiles": 0, "autotunes": 0}
    names = {"WIDEEP_CAPTURE": "captures", "WIDEEP_COMPILE": "compiles", "WIDEEP_AUTOTUNE": "autotunes"}
    for log in run.glob("rank-*.log"):
        for event, t in re.findall(r"^(WIDEEP_CAPTURE|WIDEEP_COMPILE|WIDEEP_AUTOTUNE) rank \d+ t=([\d.]+)",
                                   log.read_text(errors="replace"), re.M):
            if float(t) > after:
                counts[names[event]] += 1
    return counts


def lane2(root: Path) -> dict:
    out = {}
    for job in sorted(root.glob("FAIL-*")):
        for run in sorted(job.iterdir()):
            path = run / "result.json"
            if not path.exists():
                continue
            r = json.loads(path.read_text())
            setting = re.sub(r"-\d+$", "", run.name)
            entry = {"run": f"{job.name}/{run.name}", "state": r.get("state"), "error": r.get("error")}
            kill = r.get("kill_wall_s") or self_kill(run)
            m = marks(run, kill) if kill else {}
            if "resumed" in m:
                detect = max(kill, m.get("watcher_detect", m["recover_start"])[0])
                points = [kill, detect, m["stopped_sent"][1], m["install_recv"][1], m["resumed"][1]]
                entry["phases_ms"] = dict(zip(["detect", "stop", "agree", "install_resume"],
                                              [round((b - a) * 1e3, 1) for a, b in zip(points, points[1:])]))
                entry["recovery_ms"] = round((points[-1] - kill) * 1e3, 1)
                entry["counters_after_kill"] = counters(run, kill)
            entry["first_token_after_kill_ms"] = 1e3 * r["first_token_after_kill_s"] \
                if "first_token_after_kill_s" in r else None
            load = [row for row in r.get("rows", []) if row["label"] == "load"]
            if load:
                entry["load_failed_requests"] = load[0].get("failed_requests")
                entry["load_completed"] = load[0].get("completed")
                # benchmark_serving counts a stream that ends with the server's error event as completed;
                # with --ignore-eos a request that failed is the one that came back short.
                detail = run / "bench" / "load.json"
                if detail.exists():
                    lens = json.loads(detail.read_text())["output_lens"]
                    entry["load_cut_short"] = sum(n != max(lens) for n in lens)
            out.setdefault(setting, []).append(entry)
    return out


def startup_stages(run: Path) -> dict:
    done = dict(re.findall(r"\] ([\w/]+): done in ([\d.]+)s", (run / "rank-0.log").read_text(errors="replace")))
    pick = lambda *keys: sum(float(done.get(f"executor_creation/{k}", 0)) for k in keys)
    return {"load_weights_s": pick("model"),
            "warmup_capture_s": pick("profiling_executor_creation", "final_executor_creation"),
            "kv_cache_s": pick("profiling_kv_cache_allocation", "final_kv_cache_allocation")}


def ft_off(root: Path) -> list:
    out = []
    for job in sorted(root.glob("ABORT-*")):
        for abort in sorted(job.glob("abort-*")):
            restart = job / abort.name.replace("abort", "restart")
            timeline = (abort / "timeline").read_text() if (abort / "timeline").exists() else ""
            exit_m = re.search(r"MPIRUN_EXIT ([\d.]+)", timeline)
            if not (abort / "result.json").exists() or exit_m is None:
                continue  # Still running.
            a = json.loads((abort / "result.json").read_text())
            exit_t = float(exit_m.group(1))
            clean_t = max(float(t) for t in re.findall(r"NODE_CLEAN \d+ ([\d.]+)", timeline))
            entry = {"run": f"{job.name}/{abort.name}", "detect_abort_s": exit_t - a["kill_wall_s"],
                     "teardown_s": clean_t - exit_t}
            if (restart / "result.json").exists():
                r = json.loads((restart / "result.json").read_text())
                start_t = min(float(t) for t in re.findall(r"MPIRUN_START ([\d.]+)", (restart / "timeline").read_text()))
                entry |= {"relaunch_gap_s": start_t - clean_t, "restart_to_healthy_s": r["healthy_wall_s"] - start_t,
                          "first_token_s": r["first_token_s"],
                          "downtime_s": r["first_token_wall_s"] - a["kill_wall_s"]} | startup_stages(restart)
            out.append(entry)
    return out


def summary(values: list) -> dict:
    return {"n": len(values), "median": statistics.median(values), "min": min(values), "max": max(values)}


def main(root: str) -> None:
    root = Path(root)
    data = {"lane1": lane1(root), "lane2": lane2(root), "ft_off": ft_off(root)}
    data["lane2_summary"] = {k: summary([e["recovery_ms"] for e in v if "recovery_ms" in e])
                             for k, v in data["lane2"].items() if any("recovery_ms" in e for e in v)}
    json.dump(data, sys.stdout, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])

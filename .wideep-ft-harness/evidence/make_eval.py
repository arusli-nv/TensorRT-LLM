"""Evaluation charts from raw run outputs (result.json files and nsys-derived JSON). Nothing is typed in by hand.

Usage: python3 make_eval.py RUNS_DIR
"""

import glob
import json
import re
import os
import statistics as st
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OFF, ON, LOSS = "tab:blue", "tab:orange", "tab:green"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def load_sweep(run_dir: str) -> dict:
    """{mode: {req_per_gpu: [tokens/s per round]}} from one sweep job directory."""
    out = {}
    for mode, key in (("noft", "sweep"), ("ft", "sweep"), ("ft-loss", "degraded_sweep")):
        path = f"{run_dir}/{mode}/result.json"
        if os.path.exists(path) and key in (data := json.load(open(path))):
            out[mode] = {int(n): [r["tokens_per_s"] for r in rounds] for n, rounds in data[key].items()}
    return out


def sweep_chart(runs: str) -> None:
    # max_batch_size 64 run: only loads that all run concurrently (above 64 requests queue).
    merged = {mode: {n: v for n, v in points.items() if n <= 64}
              for mode, points in load_sweep(f"{runs}/sweep-7923718").items()}
    series = [("noft", "FT off (32 GPUs)", OFF), ("ft", "FT on (32 GPUs)", ON),
              ("ft-loss", "FT on, after a loss (31 GPUs)", LOSS)]
    fig, (ax, rel) = plt.subplots(1, 2, figsize=(12, 4.2))
    for mode, label, color in series:
        loads = sorted(merged[mode])
        med = [st.median(merged[mode][n]) / 1e3 for n in loads]
        err = [[m - min(merged[mode][n]) / 1e3 for m, n in zip(med, loads)],
               [max(merged[mode][n]) / 1e3 - m for m, n in zip(med, loads)]]
        ax.errorbar(loads, med, yerr=err, color=color, marker="o", capsize=3, label=label)
    loads = sorted(merged["noft"])
    for a in (ax, rel):
        a.set_xscale("log", base=2)
        a.set_xticks(loads, [str(n) for n in loads])
        a.set_xlabel("requests per GPU")
        a.grid(alpha=0.3)
    ax.set_ylabel("output tokens/s (thousands)")
    ax.set_title("Throughput")
    ax.legend(frameon=False)
    on_off = [st.median(merged["ft"][n]) / st.median(merged["noft"][n]) * 100 for n in loads]
    per_gpu = [(st.median(merged["ft-loss"][n]) / 31) / (st.median(merged["ft"][n]) / 32) * 100 for n in loads]
    rel.plot(loads, on_off, color=ON, marker="o", label="FT on vs FT off")
    rel.plot(loads, per_gpu, color=LOSS, marker="o", label="after loss vs before, per GPU")
    rel.set_ylim(90, 100.5)
    rel.set_ylabel("percent")
    rel.set_title("Relative throughput")
    rel.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig("eval_load_sweep.png", dpi=160)
    json.dump({m: {str(n): v for n, v in p.items()} for m, p in merged.items()}, open("eval_load_sweep.json", "w"),
              indent=1)


def overhead_chart() -> None:
    data = json.load(open("eval_overhead.json"))
    fig, (step_ax, layer_ax) = plt.subplots(1, 2, figsize=(12, 4.2))
    labels = [f"{load} req/GPU\nFT {mode}" for load in ("8", "64") for mode in ("off", "on")]
    rows = [data[f"{load}/{mode}"] for load in ("8", "64") for mode in ("off", "on")]
    x = range(len(rows))
    busy = [r["gpu_busy"] / 1e3 for r in rows]
    idle = [r["gpu_idle"] / 1e3 for r in rows]
    step_ax.bar(x, busy, color=OFF, label="GPU busy")
    step_ax.bar(x, idle, bottom=busy, color="lightgray", label="GPU idle")
    step_ax.set_xticks(list(x), labels)
    step_ax.set_ylabel("ms per decode step")
    step_ax.set_title("Decode step time")
    step_ax.legend(frameon=False, loc="upper left")
    step_ax.grid(axis="y", alpha=0.3)

    parts = [("dispatch", "dispatch"), ("fixup_after_dispatch", "sanitize/gather"), ("combine", "combine")]
    width = 0.2
    for j, (load, mode, color) in enumerate((("8", "off", OFF), ("8", "on", ON), ("64", "off", OFF),
                                             ("64", "on", ON))):
        d = data[f"{load}/{mode}"]
        offs = [k + (j - 1.5) * width for k in range(len(parts))]
        layer_ax.bar(offs, [d[key] for key, _ in parts], width, color=color,
                     hatch="//" if load == "64" else None, edgecolor="white",
                     label=f"FT {mode}, {load} req/GPU")
    layer_ax.set_xticks(range(len(parts)), [name for _, name in parts])
    layer_ax.set_ylabel("µs per MoE layer")
    layer_ax.set_title("MoE all-to-all kernels")
    layer_ax.legend(frameon=False, fontsize=9)
    layer_ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("eval_overhead.png", dpi=160)


FT_STAGES = [  # (label, from-marker, to-marker); "kill" is the frontend's SIGKILL time
    ("detect + stop\n(watcher, abort, Revoke)", "kill", "revoked"),
    ("agree on the lost rank\n(ULFM Shrink)", "revoked", "shrink_done"),
    ("drain GPU + KV rollback", "shrink_done", "kv_rollback_done"),
    ("authority: Stopped → Propose\n→ Ready → Install", "kv_rollback_done", "install_recv"),
    ("install: survivor comm,\nreroute, rank mask", "install_recv", "install_applied"),
    ("authority: Installed → Commit", "install_applied", "commit_recv"),
    ("settle requests, resume", "commit_recv", "resumed"),
]


def ft_recovery(run: str) -> dict | None:
    """Median over survivors of each recovery step, in seconds after the kill; None if recovery did not finish."""
    res = json.load(open(f"{run}/result.json"))
    marks = {}
    for log in glob.glob(f"{run}/rank-*.log"):
        for name, t in re.findall(r"WIDEEP_PHASE (\w+) ([\d.]+)", open(log, errors="ignore").read()):
            marks.setdefault(name, []).append(float(t) - res["kill_wall_s"])
    if len(marks.get("resumed", [])) < 31:
        return None
    out = {name: st.median(v) for name, v in marks.items()}
    out["kill"] = 0.0
    out["first_token"] = res.get("first_token_after_kill_s")
    out["ranks"] = len(marks["resumed"])
    return out


def failover_chart(runs: str) -> None:
    cases = [(f"idle kill {i}", f"{runs}/failover-7923692/ft-idle-{i}") for i in (1, 2, 3)]
    cases += [(f"in-flight kill {i}", d) for i, d in
              enumerate(sorted(glob.glob(f"{runs}/inflight-fix-*/inflight-*")), 1)]
    rows = [(label, ft_recovery(d)) for label, d in cases]
    rows = [(label, r) for label, r in rows if r is not None]
    short = ["detect + stop", "Shrink", "GPU drain + KV rollback", "authority: prepare",
             "install", "authority: commit", "resume"]
    colors = plt.get_cmap("tab10").colors
    fig, ax = plt.subplots(figsize=(11, 0.45 * len(rows) + 1.8))
    for y, (label, r) in enumerate(rows):
        for k, (_, a, b) in enumerate(FT_STAGES):
            ax.barh(y, (r[b] - r[a]) * 1e3, left=r[a] * 1e3, color=colors[k], height=0.6,
                    label=short[k] if y == 0 else None)
        if r.get("first_token"):
            ax.barh(y, (r["first_token"] - r["resumed"]) * 1e3, left=r["resumed"] * 1e3, color=colors[7],
                    height=0.6, label="new request → first token" if y == 0 else None)
    ax.set_yticks(range(len(rows)), [label for label, _ in rows])
    ax.invert_yaxis()
    ax.set_xlabel("ms after the kill (median over 31 survivors)")
    ax.set_title("With FT: steps back to serving new requests")
    ax.legend(frameon=False, ncol=4, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig("eval_ft_recovery.png", dpi=160)
    json.dump({label: r for label, r in rows}, open("eval_ft_recovery.json", "w"), indent=1)


RESTART_PHASES = [("model load", "executor_creation/model"),
                  ("warmup + memory profiling", "executor_creation/profiling_executor_creation"),
                  ("KV cache allocation", "executor_creation/final_kv_cache_allocation"),
                  ("final warmup + CUDA graphs", "executor_creation/final_executor_creation")]


def timeline(run: str) -> dict:
    out = {}
    for line in open(f"{run}/timeline"):
        parts = line.split()
        # NODE_CLEAN lines carry the node id before the timestamp.
        out.setdefault(parts[0], []).append(float(parts[2] if parts[0] == "NODE_CLEAN" else parts[1]))
    return out


def restart(run: str) -> dict:
    """Seconds per startup phase, from mpirun start to the LLM API being ready (rank 0's startup log)."""
    res, tl = json.load(open(f"{run}/result.json")), timeline(run)
    log = open(f"{run}/rank-0.log", errors="ignore").read()
    phases = {"launch (processes, imports)": res["start_wall_s"] - tl["MPIRUN_START"][0]}
    for label, key in RESTART_PHASES:
        phases[label] = float(re.search(re.escape(key) + r": done in ([\d.]+)s", log).group(1))
    phases["other"] = res["ready_wall_s"] - tl["MPIRUN_START"][0] - sum(phases.values())
    return phases


def teardown(run: str) -> dict:
    """Seconds from the terminate signal until every node has no rank or GPU process left."""
    tl = json.load(open(f"{run}/result.json")), timeline(run)
    term = tl[1]["TEARDOWN_TERM"][0]
    return {"terminate to all nodes clean": max(tl[1]["NODE_CLEAN"]) - term,
            "nodes": len(tl[1]["NODE_CLEAN"]), "kill_to_terminate": term - tl[0]["kill_wall_s"]}


def noft_chart(runs: str) -> None:
    starts = [(f"restart {i}", restart(d)) for i, d in enumerate(sorted(glob.glob(f"{runs}/noft-7924227/start-*")), 1)]
    downs = [(f"teardown {i}", teardown(d)) for i, d in enumerate(sorted(glob.glob(f"{runs}/teardown-*/kill-*")), 1)
             if os.path.exists(f"{d}/timeline") and "NODE_CLEAN" in open(f"{d}/timeline").read()]
    colors = plt.get_cmap("tab10").colors
    rows = [(label, {"terminate → all processes gone": t["terminate to all nodes clean"]}) for label, t in downs]
    rows += starts
    names = list(dict.fromkeys(k for _, r in rows for k in r))
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(rows) + 1.8))
    for y, (label, r) in enumerate(rows):
        left = 0.0
        for name, value in r.items():
            ax.barh(y, value, left=left, color=colors[names.index(name) % 10], height=0.6,
                    label=name if name not in ax.get_legend_handles_labels()[1] else None)
            left += value
    ax.set_yticks(range(len(rows)), [label for label, _ in rows])
    ax.invert_yaxis()
    ax.set_xlabel("seconds")
    ax.set_title("Without FT: teardown and restart")
    ax.legend(frameon=False, ncol=3, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig("eval_noft_failover.png", dpi=160)
    json.dump({"teardown": dict(downs), "restart": dict(starts)}, open("eval_noft_failover.json", "w"), indent=1)


if __name__ == "__main__":
    sweep_chart(sys.argv[1])
    overhead_chart()
    failover_chart(sys.argv[1])
    noft_chart(sys.argv[1])

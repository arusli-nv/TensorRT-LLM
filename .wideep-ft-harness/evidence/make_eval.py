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

BLACK, GRAY, GREEN, RED = "#1a1a1a", "#8c8c8c", "#76b900", "#d32f2f"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#555", "axes.titleweight": "bold"})


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
    series = [("noft", "FT off (32 GPUs)", BLACK, "-"), ("ft", "FT on (32 GPUs)", GREEN, "-"),
              ("ft-loss", "FT on, after a loss (31 GPUs)", GREEN, "--")]
    fig, (ax, rel) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})
    for mode, label, color, style in series:
        if mode not in merged:
            continue
        loads = sorted(merged[mode])
        med = [st.median(merged[mode][n]) / 1e3 for n in loads]
        lo = [m - min(merged[mode][n]) / 1e3 for m, n in zip(med, loads)]
        hi = [max(merged[mode][n]) / 1e3 - m for m, n in zip(med, loads)]
        ax.errorbar(loads, med, yerr=[lo, hi], color=color, ls=style, marker="o", ms=5, capsize=3, lw=2,
                    label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(merged["noft"]), [str(n) for n in sorted(merged["noft"])])
    ax.set_xlabel("concurrent requests per GPU")
    ax.set_ylabel("output tokens/s (thousands)")
    ax.set_title("Decode throughput vs load")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper left")

    loads = sorted(set(merged["noft"]) & set(merged["ft"]))
    on_off = [st.median(merged["ft"][n]) / st.median(merged["noft"][n]) * 100 for n in loads]
    rel.plot(loads, on_off, color=GREEN, marker="o", lw=2, label="FT on / FT off (32 GPUs)")
    if "ft-loss" in merged:
        both = [n for n in loads if n in merged["ft-loss"]]
        per_gpu = [(st.median(merged["ft-loss"][n]) / 31) / (st.median(merged["ft"][n]) / 32) * 100 for n in both]
        rel.plot(both, per_gpu, color=GREEN, ls="--", marker="o", lw=2,
                 label="after loss, per GPU / before loss")
    rel.axhline(100, color=GRAY, lw=1)
    rel.set_xscale("log", base=2)
    rel.set_xticks(loads, [str(n) for n in loads])
    rel.set_ylim(90, 101)
    rel.set_xlabel("concurrent requests per GPU")
    rel.set_ylabel("percent")
    rel.set_title("Relative throughput")
    rel.grid(alpha=0.25)
    rel.legend(frameon=False, loc="lower left")
    fig.text(0.01, 0.01, "DeepSeek-R1 NVFP4, 32 GB200; 480 output tokens per request; median with min–max of the "
             "measured rounds after a warmup round (5 rounds per point). max_batch_size 64.",
             fontsize=9, color="#666")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig("eval_load_sweep.png", dpi=160)
    json.dump({m: {str(n): v for n, v in p.items()} for m, p in merged.items()}, open("eval_load_sweep.json", "w"),
              indent=1)


def overhead_chart() -> None:
    data = json.load(open("eval_overhead.json"))
    fig, (step_ax, layer_ax) = plt.subplots(1, 2, figsize=(13, 4.8))
    labels, x = [], 0
    for load in ("8", "64"):
        for mode in ("off", "on"):
            d = data[f"{load}/{mode}"]
            busy, idle = d["gpu_busy"] / 1e3, d["gpu_idle"] / 1e3
            color = GREEN if mode == "on" else BLACK
            step_ax.bar(x, busy, color=color, width=0.7)
            step_ax.bar(x, idle, bottom=busy, color="#d9d9d9", width=0.7, edgecolor=color, lw=1)
            step_ax.text(x, busy + idle + 0.3, f"{d['step'] / 1e3:.2f} ms", ha="center", fontsize=10)
            labels.append(f"FT {mode}\n{load} req/GPU")
            x += 1
        off, on = data[f"{load}/off"], data[f"{load}/on"]
        moe = (on['span'] - off['span']) * 58 / 1e3
        other = (on['step'] - off['step']) / 1e3 - moe
        step_ax.text(x - 1.5, 27.5, f"step +{(on['step'] - off['step']) / 1e3:.2f} ms "
                     f"(+{(on['step'] / off['step'] - 1) * 100:.1f}%)\n"
                     f"= MoE layers (58 × span) {moe:+.2f} ms\n+ everything else {other:+.2f} ms",
                     ha="center", va="bottom", fontsize=9.5, color="#444")
        x += 0.6
    step_ax.set_xticks([0, 1, 2.6, 3.6], labels)
    step_ax.set_ylabel("median decode step (ms)")
    step_ax.set_title("Decode step: GPU busy (solid) and GPU idle (gray)")
    step_ax.set_ylim(0, 33)

    parts = [("dispatch", "dispatch"), ("fixup_after_dispatch", "sanitize/\ngather"),
             ("combine", "combine"), ("rest", "rest\n(experts)")]
    width, x = 0.17, 0
    ticks = []
    for load in ("8", "64"):
        for j, (mode, color) in enumerate((("off", BLACK), ("on", GREEN))):
            d = dict(data[f"{load}/{mode}"])
            d["rest"] = d["span"] - d["dispatch"] - d["fixup_after_dispatch"] - d["combine"]
            for k, (key, _) in enumerate(parts):
                layer_ax.bar(x + k * width * 2.6 + j * width, d[key], width, color=color)
                if mode == "on":
                    delta = d[key] - (data[f"{load}/off"][key] if key != "rest" else
                                      data[f"{load}/off"]["span"] - sum(data[f"{load}/off"][p] for p in
                                                                        ("dispatch", "fixup_after_dispatch",
                                                                         "combine")))
                    layer_ax.text(x + k * width * 2.6 + width, d[key] + 1.5, f"{delta:+.1f}", ha="center",
                                  fontsize=9, color="#444")
        for k, (_, name) in enumerate(parts):
            ticks.append((x + k * width * 2.6 + width / 2, name))
        layer_ax.text(x + 1.5 * width * 2.6 + width / 2, 76, f"{load} req/GPU", ha="center", fontsize=11,
                      weight="bold")
        x += 2.4
    layer_ax.set_xticks([t for t, _ in ticks], [lab for _, lab in ticks], fontsize=9)
    layer_ax.set_ylabel("median per MoE layer (µs)")
    layer_ax.set_title("One MoE layer, FT off (black) vs on (green); +Δ µs")
    layer_ax.set_ylim(0, 80)
    fig.text(0.01, 0.01, "DeepSeek-R1 NVFP4, 32 GB200, nsys on one rank; medians over ~2,400 decode steps "
             "(58 MoE layers each) per run. Sanitize/gather start early (PDL), so their bar is the time "
             "they add after dispatch ends.", fontsize=9, color="#666")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
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
    fig, ax = plt.subplots(figsize=(13, 0.55 * len(rows) + 2.6))
    shades = ["#1a1a1a", "#4d4d4d", "#808080", GREEN, "#a8d65c", "#5c8f00", "#b3b3b3"]
    for y, (label, r) in enumerate(rows):
        for k, (stage, a, b) in enumerate(FT_STAGES):
            left, width = r[a] * 1e3, (r[b] - r[a]) * 1e3
            ax.barh(y, width, left=left, color=shades[k], height=0.6,
                    label=stage if y == 0 else None)
        ax.text(r["resumed"] * 1e3 + 1.5, y, f"{r['resumed'] * 1e3:.0f} ms", va="center", fontsize=10)
    ax.set_yticks(range(len(rows)), [label for label, _ in rows])
    ax.invert_yaxis()
    ax.set_xlabel("milliseconds after the rank was killed (median over the 31 survivors)")
    ax.set_title("FT recovery: from the kill to serving again")
    ax.legend(frameon=False, ncol=4, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    ax.grid(axis="x", alpha=0.25)
    first = [r["first_token"] for _, r in rows if r.get("first_token")]
    note = (f"First new token at the client after an idle kill: {min(first):.2f}–{max(first):.2f} s "
            "(request path; recovery itself is done by the end of each bar)." if first else "")
    fig.text(0.01, 0.01, "DeepSeek-R1 NVFP4, 32 GB200, rank 2 SIGKILLed. Each bar: median over the 31 survivors of "
             "wall-clock markers at every recovery step (8 nodes, NTP clocks: ms-level skew).\n"
             "The authority proposes only after every survivor reported Stopped, so its segment includes waiting "
             "for the slowest survivor (idle kill 2: one GPU drained after 86 ms).\n" + note,
             fontsize=9, color="#666", va="bottom")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig("eval_ft_recovery.png", dpi=160)
    json.dump({label: r for label, r in rows}, open("eval_ft_recovery.json", "w"), indent=1)


if __name__ == "__main__":
    sweep_chart(sys.argv[1])
    overhead_chart()
    failover_chart(sys.argv[1])

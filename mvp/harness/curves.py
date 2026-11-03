"""Throughput-latency curves of an interleaved A/B job, and the gap at matched concurrency and matched latency.

Per mode (cases <mode>-<round>), per load: the mean over rounds of the steady-state per-GPU output throughput
and per-user rate (steady_state.py). Plots per-GPU tok/s against per-user tok/s, one line per mode, and prints
per load: the throughput gap at matched concurrency (same load) and at matched latency (the test mode's
throughput interpolated, on its own curve, at the base mode's per-user rate).
Usage: python3 curves.py RUN_DIR OUT.png MODE [MODE ...]   (the first mode is the base)
"""

import collections
import json
import pathlib
import statistics
import sys

import numpy as np

import steady_state


def curve(run: pathlib.Path, mode: str) -> dict:
    """load -> (per-user tok/s, per-GPU tok/s), means over rounds."""
    points = collections.defaultdict(list)
    for result in sorted(run.glob(f"{mode}-[0-9]*/result.json")):
        rows = json.loads(result.read_text()).get("rows", [])
        for row in steady_state.annotate(result.parent, rows):
            if row.get("label") == f"healthy-{row['load_per_gpu']}" and "steady_tok_s_gpu" in row:
                points[row["load_per_gpu"]].append((row["steady_user_tok_s"], row["steady_tok_s_gpu"]))
    return {load: tuple(statistics.mean(v) for v in zip(*pts)) for load, pts in sorted(points.items())}


def at_user_rate(points: dict, user: float) -> float:
    """Per-GPU throughput on the curve at a per-user rate (linear between measured loads; nan outside)."""
    users, gpus = zip(*sorted(points.values()))
    return float(np.interp(user, users, gpus, left=np.nan, right=np.nan))


def main() -> None:
    run, out, modes = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    curves = {mode: curve(run, mode) for mode in modes}
    base = curves[modes[0]]
    for mode in modes[1:]:
        print(f"\n{mode} vs {modes[0]}: throughput gap % (positive = {mode} lower)")
        print(f"{'load':>5} {'user tok/s':>10} {'matched conc.':>13} {'matched lat.':>12}")
        for load, (user, gpu) in base.items():
            if load not in curves[mode]:
                continue
            conc = (gpu - curves[mode][load][1]) / gpu * 100
            lat = (gpu - at_user_rate(curves[mode], user)) / gpu * 100
            print(f"{load:>5} {user:>10.1f} {conc:>13.2f} {lat:>12.2f}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mode, points in curves.items():
        users, gpus = zip(*points.values())
        ax.plot(users, gpus, marker="o", markersize=8, linewidth=2, label=mode)
        for load, (user, gpu) in points.items():
            if mode == modes[0]:
                ax.annotate(str(load), (user, gpu), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("per-user output tok/s (steady state)")
    ax.set_ylabel("per-GPU output tok/s (steady state)")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(f"{run.name}: labels = concurrent requests per GPU")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


if __name__ == "__main__":
    main()

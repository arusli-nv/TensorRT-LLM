"""Result charts and tables for DESIGN.md, from the JSON build_data.py writes.

A chart whose data is missing is skipped. Captions belong in the document, never in the image.
Usage: python3 figs.py data.json OUT_DIR
"""

import json
import statistics
import sys
from pathlib import Path

import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
for font in (HERE / "fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(font))
plt.rcParams.update({
    "font.family": "Lato", "font.size": 15, "axes.edgecolor": "#BDBDBD", "axes.linewidth": 1.2,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": "#5F5F5F",
    "ytick.color": "#5F5F5F", "xtick.major.size": 0, "ytick.major.size": 0, "savefig.dpi": 200,
})

GREEN, LIGHT_GREEN, GRAY, DARK, MUTED = "#76B900", "#B9DB7F", "#A9A9A9", "#1A1A1A", "#6E6E6E"
BLUE = "#6A8EAE"
# Breakdown segments (phases, stages): one neutral ramp, dark to light.
RAMP = ["#2F4B66", "#4D6E8F", "#7393B3", "#A3BBD3", "#D3E0EC"]
GRAYS = ["#5A5A5A", "#7A7A7A", "#A3A3A3", "#C7C7C7", "#E0E0E0"]
SERIES = {"FT off": (GRAY, "o"), "FT on": (GREEN, "D"), "FT on, after a loss": (LIGHT_GREEN, "s"),
          "FT off, standard config": (BLUE, "^")}
LEGEND = {"FT off": "FT off, 32 GPUs", "FT on": "FT on, 32 GPUs", "FT on, after a loss": "FT on, 31 GPUs after a loss",
          "FT off, standard config": "FT off, standard config"}
KILLS = {"idle": "Idle", "decode8": "In a decode step, 8/GPU", "random8": "Random time, 8/GPU",
         "decode256": "In a decode step, 256/GPU", "prefill256": "In a prefill step, 256/GPU",
         "random256": "Random time, 256/GPU"}
PHASES = ["detect", "stop", "agree", "install_resume"]
PHASE_NAMES = ["Detect", "Stop, shrink, discard", "Agree on placement", "Install and resume"]


def med(values):
    return statistics.median(values) if values else None


def barh(ax, y, width, height, left=0.0, color=GREEN):
    """One bar segment: soft corners and a thin white gap to its left neighbor."""
    ax.barh(y, width, height, left=left, color=color, edgecolor=color, linewidth=3, joinstyle="round")
    if left:
        ax.plot([left, left], [y - height / 2 - 0.02, y + height / 2 + 0.02], color="white", lw=2.5,
                solid_capstyle="butt")


def finish_barh(ax):
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color="#E9E9E9")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)


def downtime(data):
    """Top: kill -> first token of a new request, restart (FT off) vs recover in place (FT on), one seconds
    axis, medians. Bottom: FT on recovery by kill setting, median per phase."""
    off = [e for e in data["ft_off"] if "downtime_s" in e]
    settings = [(k, data["lane2"][k]) for k in KILLS
                if any("phases_ms" in e for e in data["lane2"].get(k, ()))]
    if not off or not settings:
        return None
    stages = [("Detect and abort", lambda e: e["detect_abort_s"]), ("Tear down", lambda e: e["teardown_s"]),
              ("Load weights", lambda e: e["load_weights_s"]),
              ("Warm up, capture graphs", lambda e: e["warmup_capture_s"]),
              ("Other startup, first token", lambda e: e["downtime_s"] - e["detect_abort_s"] - e["teardown_s"]
               - e["load_weights_s"] - e["warmup_capture_s"])]
    off_parts = [med([f(e) for e in off]) for _, f in stages]
    # The restarted FT-off server answers its first request unloaded; under load a new request also queues
    # behind admission (seconds at 256/GPU), so compare with kills of an idle FT-on server.
    idle = [e for k, v in settings if k == "idle" for e in v if e.get("first_token_after_kill_ms")]
    ft_first = med([e["first_token_after_kill_ms"] / 1e3 for e in idle])
    n = len(settings)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 2.6 + 0.6 * n),
                                      gridspec_kw={"height_ratios": [2, n], "hspace": 0.95})
    x = 0
    for value, color in zip(off_parts, GRAYS):
        barh(top, 1, value, 0.6, left=x, color=color)
        x += value
    top.text(x * 1.01, 1, f"{x:.0f} s", va="center", fontsize=14, fontweight="bold", color=DARK)
    if ft_first is not None:
        barh(top, 0, ft_first, 0.6, color=GREEN)
        top.text(ft_first + x * 0.01, 0, f"{ft_first:.1f} s", va="center", fontsize=14, fontweight="bold",
                 color=DARK)
    n_ft = len(idle)
    top.set_yticks([1, 0], [f"FT off: abort and restart (256/GPU, n={len(off)})", f"FT on: recover in place (idle, n={n_ft})"],
                   fontsize=13)
    top.set_xlim(0, x * 1.15)
    top.xaxis.set_major_formatter(lambda v, _: f"{v:.0f} s" if v else "0")
    top.set_title("Losing 1 of 32 GPUs: kill to first token of a new request (median)", loc="left", fontsize=15,
                  fontweight="bold", color=DARK)
    top.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in GRAYS], [s for s, _ in stages],
               loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=3, frameon=False, fontsize=11,
               handlelength=1.0)
    worst = 0
    for i, (name, entries) in enumerate(settings):
        y = n - 1 - i
        # The kill with the median recovery time, so that each bar's total is that median (medians of the
        # phases would not add up to it).
        done = sorted((e for e in entries if "phases_ms" in e), key=lambda e: e["recovery_ms"])
        typical = done[(len(done) - 1) // 2]
        x = 0
        for phase, color in zip(PHASES, RAMP):
            value = typical["phases_ms"][phase]
            barh(bottom, y, value, 0.6, left=x, color=color)
            x += value
        worst = max(worst, x)
        bottom.text(x, y, f"  {x:.0f} ms", va="center", fontsize=13, fontweight="bold", color=DARK)
    bottom.set_yticks(range(n), [f"{KILLS.get(k, k)} (n={sum('phases_ms' in e for e in v)})"
                                 for k, v in settings[::-1]], fontsize=13)
    bottom.set_xlim(0, worst * 1.2)
    bottom.xaxis.set_major_formatter(lambda v, _: f"{v:.0f} ms" if v else "0")
    bottom.set_title("FT on: recovery by when the GPU was lost (the median kill)", loc="left", fontsize=15,
                     fontweight="bold", color=GREEN)
    bottom.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in RAMP[:4]], PHASE_NAMES, loc="upper center",
                  bbox_to_anchor=(0.45, -0.22), ncol=4, frameon=False, fontsize=11, handlelength=1.0)
    for ax in (top, bottom):
        finish_barh(ax)
    fig.subplots_adjust(left=0.33, right=0.97, bottom=0.14, top=0.93)
    return fig


def loads_of(lane1):
    return sorted({int(k) for mode, d in lane1.items() if mode in SERIES for k in d})


def throughput(lane1):
    """Output tokens/s per GPU vs concurrency: a dot per independent server run, a line through the medians."""
    loads = loads_of(lane1)
    if not loads:
        return None
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for mode, (color, marker) in SERIES.items():
        d = lane1.get(mode, {})
        xs, ys = [], []
        for i, load in enumerate(loads):
            values = [e["tok_s_per_gpu"] for e in d.get(str(load), [])]
            if not values:
                continue
            ax.scatter([i] * len(values), values, s=22, color=color, edgecolors=DARK, linewidths=0.5, zorder=4)
            xs.append(i)
            ys.append(med(values))
        if xs:
            ax.plot(xs, ys, color=color, lw=2.2, marker=marker, ms=8, markeredgecolor="white",
                    markeredgewidth=1.4, zorder=3, label=LEGEND[mode])
    ax.set_xticks(range(len(loads)), [str(n) for n in loads])
    ax.set_xlabel("Concurrent requests per GPU", fontsize=15)
    ax.set_ylabel("Output tokens/s per GPU", fontsize=15)
    ax.set_ylim(0, None)
    ax.yaxis.grid(True, color="#EDEDED")
    ax.set_axisbelow(True)
    fig.legend(loc="lower center", ncol=2, frameon=False, fontsize=13)
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.26, top=0.97)
    return fig


def relative(lane1):
    """Total throughput of FT on (32 GPUs, and 31 after a loss) as % of FT off on 32 GPUs at the same
    concurrency per GPU; dashed line = 31/32, what 31 GPUs serve at FT off's per-GPU rate."""
    off = lane1.get("FT off", {})
    loads = [n for n in loads_of(lane1) if str(n) in off]
    if not loads or "FT on" not in lane1:
        return None
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axhline(100, color=GRAY, lw=1.4, zorder=1)
    ax.axhline(100 * 31 / 32, color=GRAY, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(0.01, 100 * 31 / 32 + 0.15, "31/32", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=12.5, color=MUTED)
    lowest = 100
    for mode in ("FT on", "FT on, after a loss"):
        color, marker = SERIES[mode]
        xs, ys = [], []
        for i, load in enumerate(loads):
            values = [e["tok_s_per_gpu"] for e in lane1.get(mode, {}).get(str(load), [])]
            if not values:
                continue
            base = med([e["tok_s_per_gpu"] for e in off[str(load)]])
            gpus = 31 if mode == "FT on, after a loss" else 32
            pct = [100 * v * gpus / (base * 32) for v in values]
            ax.scatter([i] * len(pct), pct, s=22, color=color, edgecolors=DARK, linewidths=0.5, zorder=4)
            xs.append(i)
            ys.append(med(pct))
            lowest = min(lowest, *pct)
        if xs:
            ax.plot(xs, ys, color=color, lw=2.2, marker=marker, ms=8, markeredgecolor="white",
                    markeredgewidth=1.4, zorder=3, label=LEGEND[mode])
    ax.set_xticks(range(len(loads)), [str(n) for n in loads])
    ax.set_xlabel("Concurrent requests per GPU", fontsize=15)
    ax.set_ylabel("Total output tokens/s,\n% of FT off on 32 GPUs", fontsize=15)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0f}%")
    ax.set_ylim(lowest - 2, 102)
    ax.set_xlim(-0.4, len(loads) - 0.6)
    ax.yaxis.grid(True, color="#EDEDED")
    ax.set_axisbelow(True)
    fig.legend(loc="lower center", ncol=2, frameon=False, fontsize=13)
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.24, top=0.97)
    return fig


def latency_table(lane1):
    """Markdown table: per mode and concurrency, medians over runs of each run's p50/p99."""
    keys = [("tok_s_per_gpu", "tok/s/GPU", "{:.0f}"), ("decode_step_ms", "step ms", "{:.1f}"),
            ("tpot_p50_ms", "TPOT p50 ms", "{:.1f}"), ("tpot_p99_ms", "TPOT p99 ms", "{:.1f}"),
            ("ttft_p50_ms", "TTFT p50 ms", "{:.0f}"), ("ttft_p99_ms", "TTFT p99 ms", "{:.0f}")]
    lines = ["| Mode | Requests/GPU | n | " + " | ".join(k[1] for k in keys) + " |",
             "|---|---:|---:|" + "---:|" * len(keys)]
    for mode in SERIES:
        for load in sorted(lane1.get(mode, {}), key=int):
            entries = lane1[mode][load]
            cells = []
            for key, _, fmt in keys:
                value = med([e[key] for e in entries if e.get(key) is not None])
                cells.append("–" if value is None else fmt.format(value))
            lines.append(f"| {LEGEND[mode]} | {load} | {len(entries)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(data_path, out_dir):
    data = json.loads(Path(data_path).read_text())
    out = Path(out_dir)
    for name, fig in (("fig_downtime", downtime(data)), ("fig_throughput", throughput(data["lane1"])),
                      ("fig_relative", relative(data["lane1"]))):
        if fig is not None:
            fig.savefig(out / f"{name}.png")
        plt.close("all")
    (out / "table_latency.md").write_text(latency_table(data["lane1"]))


if __name__ == "__main__":
    main(*sys.argv[1:3])

"""Appendix data and slide charts drawn straight from run files (each writes a CSV next to its figure).

ttft_timeline: median and p90 time to first token of the requests that arrived in each 10 s bin, from 2 min before to
5 min after a rank kill under realistic traffic; the same three losses as Figure 10 / slide 14 (S2 8299473 rank 14,
S3 8314070 ranks 29 and 14). Requests with status 200 and every expected token only (the lost rank's cut requests are
counted per bin as failed, not timed).
usage: /usr/bin/python3 make_appendix.py   (writes appendix_ttft_timeline.csv and ttft_timeline_talk.png/.svg)
"""
import csv
import json
import math
import os
import statistics

from make_talk_figs import (FS_NOTE, GRID, INK, LABEL, ON_EDGE, figure, header, plt, save)
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

ROOT = "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude"
RUNS = f"{ROOT}/mvp/runs"
OUT = os.path.dirname(os.path.abspath(__file__))
BIN, PRE, POST = 10, 120, 300
CASES = [(f"{RUNS}/STRESS-8299473/s2-rank14", "S2", 1), (f"{RUNS}/STRESS-8314070/s3-rank29-rank14", "S3", 1),
         (f"{RUNS}/STRESS-8314070/s3-rank29-rank14", "S3", 2)]


def load(case):
    recs = []
    for row in ("stress", "stress-chat"):
        with open(f"{case}/bench/{row}.sse.jsonl") as fh:
            for line in fh:
                try:
                    recs.append(json.loads(line))
                except ValueError:
                    pass
    return recs, json.load(open(f"{case}/result.json"))["kills"]


def q(values, p):
    v = sorted(values)
    return v[min(len(v) - 1, int(math.ceil(p * len(v))) - 1)] if v else float("nan")


rows, series = [], []
for case, test, k in CASES:
    recs, kills = load(case)
    t0 = kills[k - 1]["kill_wall_s"]
    edges = list(range(-PRE, POST, BIN))
    med = []
    for e in edges:
        b = [r for r in recs if r.get("start") is not None and t0 + e <= r["start"] < t0 + e + BIN]
        ok = [r["ttft"] * 1000 for r in b if r.get("status") == 200 and r.get("ttft") is not None
              and r.get("tok") == r.get("exp")]
        failed = len(b) - len(ok)
        m, p90 = (statistics.median(ok) if ok else float("nan")), q(ok, 0.9)
        med.append(m)
        rows.append([test, k, f"{t0:.3f}", e, e + BIN, len(b), failed, f"{m:.1f}", f"{p90:.1f}"])
    series.append((test, k, edges, med))
with open(f"{OUT}/appendix_ttft_timeline.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["test", "kill", "kill_wall_s", "bin_start_s_from_kill", "bin_end_s_from_kill", "requests", "failed",
                "ttft_p50_ms", "ttft_p90_ms"])
    w.writerows(rows)

fig, ax = figure(5.0)
x = [(e + BIN / 2) / 60 for e in series[0][2]]
for _, _, _, med in series:
    ax.plot(x, [v / 1000 for v in med], color="#C9CFD6", lw=1.0, zorder=2)
mean = [statistics.mean(v) / 1000 for v in zip(*(s[3] for s in series))]
ax.plot(x, mean, color=ON_EDGE, lw=3.2, zorder=4)
pre = [v for xx, v in zip(x, mean) if xx < 0]
base = statistics.mean(pre)
ax.axhline(base, color="#4A5568", lw=1.4, ls=(0, (5, 3)), zorder=1)
ax.axvline(0, color=INK, lw=1.4, zorder=2)
ax.text(0.06, ax.get_ylim()[1] * 0.97, "kill", ha="left", va="top", fontsize=FS_NOTE, color=INK)
ax.set_xlim(-2, 5)
ax.set_xticks(range(-2, 6), [f"{v:+d}".replace("-", "\u2212") if v else "0" for v in range(-2, 6)])
ax.set_ylim(0, max(mean) * 1.25)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} s"))
ax.set_xlabel("minutes from the kill (requests binned by arrival, 10 s)")
ax.set_ylabel("median time to first token")
header(fig, "TTFT around a kill", None,
       [Line2D([], [], color=ON_EDGE, lw=3.2, label="mean of 3 losses"),
        Line2D([], [], color="#C9CFD6", lw=1.0, label="each loss"),
        Line2D([], [], color="#4A5568", lw=1.4, ls=(0, (5, 3)), label="before the kill (mean)")], ncol=3,
       left=0.1, bottom=0.14)
save(fig, "ttft_timeline")
print("ttft_timeline", f"pre-kill mean {base * 1000:.0f} ms, peak {max(mean) * 1000:.0f} ms")

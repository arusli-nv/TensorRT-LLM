"""Per-pair FT-on vs FT-off % change of client latency metrics, with a 95% t-CI per mode (healthy-cost readouts).

usage: python3 tools/pair_latency_ci.py --load 128 PAIRS_CSV...   (PAIRS_CSV = tools/finals_stats.py --csv output)
Each pair's arms are found as make_figs.py does: the bench.log healthy-<load> row whose output tok/s per GPU equals the
pair's client_off / client_on. Prints rows '| mode | load | metric | n | mean | [lo, hi] |' (the v5f_tpot_ttft.txt
format make_figs.py reads).
"""
import argparse
import csv
import glob
import json
import statistics
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "mvp" / "runs"
GPUS = 32
METRICS = ("median_tpot_ms", "median_ttft_ms", "output_throughput", "p99_tpot_ms")
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201}


def bench_rows(load):
    rows = []
    for log in glob.glob(f"{RUNS}/STD*/std-*-1/bench.log"):
        for line in open(log):
            if line.startswith("{"):
                r = json.loads(line)
                if r.get("label") == f"healthy-{load}" and "output_throughput" in r:
                    rows.append(r)
    return rows


def arm(rows, client):
    hits = [r for r in rows if abs(r["output_throughput"] / GPUS - client) < 1e-6]
    assert len(hits) == 1, (client, len(hits))
    return hits[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--load", type=int, required=True)
    p.add_argument("pairs", nargs="+")
    args = p.parse_args()
    rows = bench_rows(args.load)
    pct = {}
    for path in args.pairs:
        for pr in csv.DictReader(open(path)):
            if pr["load"] != str(args.load):
                continue
            off, on = arm(rows, float(pr["client_off"])), arm(rows, float(pr["client_on"]))
            for m in METRICS:
                pct.setdefault((pr["mode"], m), []).append((on[m] / off[m] - 1) * 100)
    print("| Mode | Load | Metric | n | mean % FT vs off | 95% CI |")
    print("|---|---|---|---|---|---|")
    for (mode, m), v in sorted(pct.items()):
        mean = statistics.mean(v)
        h = T975[len(v) - 1] * statistics.stdev(v) / len(v) ** 0.5
        print(f"| {mode} | {args.load} | {m} | {len(v)} | {mean:+.2f} | [{mean - h:+.2f}, {mean + h:+.2f}] |")


if __name__ == "__main__":
    main()

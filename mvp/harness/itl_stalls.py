"""Client-side stall readout (no server log needed): the streamed chunk gaps of a benchmark_serving --save-detailed
result. Per case and row: median gap, p99 / p99.9 / max, gaps above 2x the median and the time they add (excess
over the median, % of all gap time), and output tok/s per GPU. The first gap of each request (right after the
first token, while the request's own batch fills) is left out.
Usage: python3 itl_stalls.py [--row LABEL] [--gpus N] RUN_DIR [RUN_DIR ...]
"""
import argparse
import json
import pathlib
import statistics

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("runs", nargs="+", type=pathlib.Path)
parser.add_argument("--row", default="healthy-32")
parser.add_argument("--gpus", type=int, default=32)
args = parser.parse_args()
for run in args.runs:
    for case in sorted(p for p in run.iterdir() if (p / "bench" / f"{args.row}.json").exists()):
        result = json.loads((case / "bench" / f"{args.row}.json").read_text())
        gaps = sorted(g * 1e3 for itl in result["itls"] for g in itl[1:])
        if not gaps:
            continue
        med = statistics.median(gaps)
        slow = [g for g in gaps if g > 2 * med]
        excess = sum(g - med for g in slow)
        q = lambda f: gaps[min(len(gaps) - 1, int(f * len(gaps)))]
        print(f"{run.name:24s} {case.name:28s} gaps={len(gaps):6d} med={med:7.1f} p99={q(0.99):7.1f} "
              f"p99.9={q(0.999):7.1f} max={gaps[-1]:7.1f} >2xmed={len(slow):5d} excess={100 * excess / sum(gaps):5.1f}% "
              f"tok/s/gpu={result['output_throughput'] / args.gpus:7.1f}")

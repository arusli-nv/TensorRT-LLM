"""Late-admitted requests per case and row: requests whose TTFT falls in a band (default 55-75 s, a connect retried
after SYN retransmits) out of all requests of benchmark_serving's --save-detailed result (protocol amendment 4: a
512/GPU A/B is invalid for client throughput when either arm has more than 1% of requests in the band).
Usage: python3 ttft_band.py [--row healthy-512] [--band 55,75] RUN_DIR [RUN_DIR ...]
"""
import argparse
import json
import pathlib

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("runs", nargs="+", type=pathlib.Path)
parser.add_argument("--row", default="healthy-512")
parser.add_argument("--band", default="55,75", type=lambda s: tuple(float(x) for x in s.split(",")))
args = parser.parse_args()
lo, hi = args.band
for run in args.runs:
    for case in sorted(p for p in run.iterdir() if p.is_dir()):
        parts = sorted((case / "bench").glob(f"{args.row}.part*.json")) or sorted((case / "bench").glob(f"{args.row}.json"))
        ttfts = [t for part in parts for t in json.loads(part.read_text()).get("ttfts", []) if t]
        if not ttfts:
            continue
        late = sum(lo <= t <= hi for t in ttfts)
        share = 100 * late / len(ttfts)
        print(f"{run.name:22s} {case.name:24s} {args.row} requests={len(ttfts):6d} ttft_{lo:.0f}_{hi:.0f}s={late:6d} "
              f"({share:5.2f}%) {'INVALID_FOR_CLIENT_THROUGHPUT' if share > 1 else 'ok'}")

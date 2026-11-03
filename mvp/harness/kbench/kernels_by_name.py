"""Per-kernel median duration and launches per layer in each "tokens=T" NVTX range of an a2a_bench nsys report,
plus the layer time (range length / layers) and the GPU idle time between kernels per layer.

Usage: python3 kernels_by_name.py LABEL=SQLITE [LABEL=SQLITE ...]   (one table per token count, a column per label)
"""

import sqlite3
import statistics as st
import sys
from collections import defaultdict

LAYERS = 200  # a2a_bench_mpi.py ITERS: graph replays inside each range


def analyze(path: str) -> dict:
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    ranges = db.execute("select text, start, end from NVTX_EVENTS where text like 'tokens=%'").fetchall()
    kernels = db.execute(
        "select s.value, k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id = k.shortName "
        "order by k.start"
    ).fetchall()
    out = {}
    for text, start, end in ranges:
        rows = [k for k in kernels if start <= k[1] < end]
        by_name = defaultdict(list)
        for name, s, e in rows:
            by_name[name].append(e - s)
        busy = sum(e - s for _, s, e in rows)
        span = rows[-1][2] - rows[0][1] if rows else 0
        out[text] = {
            "kernels": {n: (len(d) / LAYERS, st.median(d) / 1e3) for n, d in by_name.items()},
            "layer_us": span / LAYERS / 1e3,
            "idle_us": (span - busy) / LAYERS / 1e3,
        }
    return out


def main() -> None:
    labels = [arg.split("=", 1) for arg in sys.argv[1:]]
    results = {label: analyze(path) for label, path in labels}
    for tokens in sorted({t for r in results.values() for t in r}, key=lambda t: int(t.split("=")[1])):
        print(f"\n{tokens}  (per layer: launches x median us)")
        names = sorted({n for r in results.values() for n in r.get(tokens, {}).get("kernels", {})})
        print(f"{'kernel':48s}" + "".join(f"{label:>18s}" for label, _ in labels))
        for name in names:
            cells = []
            for label, _ in labels:
                count, med = results[label].get(tokens, {}).get("kernels", {}).get(name, (0, 0))
                cells.append(f"{count:5.1f} x {med:7.2f}")
            print(f"{name[:48]:48s}" + "".join(f"{c:>18s}" for c in cells))
        for key in ("layer_us", "idle_us"):
            print(f"{key:48s}" + "".join(f"{results[l].get(tokens, {}).get(key, 0):18.2f}" for l, _ in labels))


if __name__ == "__main__":
    main()

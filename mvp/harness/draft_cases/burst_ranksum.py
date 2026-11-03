"""BENCHMARK_STANDARD proposal metric per load: median steady step and burst count per row, base vs test arm.

Reads burst_counts.py output (one or more files, e.g. two jobs of 3+3 rounds pooled) and prints, per load, each
arm's rows (median step ms, bursts = steady steps > 2x the row median), and an exact two-sided Mann-Whitney
rank-sum test on the burst counts (ties get mid-ranks; the p-value enumerates every split of the pooled rows).
Usage: python3 burst_ranksum.py BASE TEST burst_counts.txt [burst_counts.txt ...]
"""

import collections
import itertools
import re
import sys

LINE = re.compile(r"^(\S+)\s+(\S+)-(\d+)\s+load=\s*(\d+) steps=\s*\d+ med=\s*([\d.]+) >40ms=\s*\d+ >2xmed=\s*(\d+)")


def ranks(values):
    order = sorted(values)
    return [sum(i + 1 for i, v in enumerate(order) if v == x) / order.count(x) for x in values]


def ranksum_p(a, b):
    pooled = a + b
    r = ranks(pooled)
    expected = len(a) * (len(pooled) + 1) / 2
    observed = abs(sum(r[:len(a)]) - expected)
    splits = list(itertools.combinations(range(len(pooled)), len(a)))
    extreme = sum(abs(sum(r[i] for i in s) - expected) >= observed - 1e-9 for s in splits)
    return extreme / len(splits)


def main(base, test, *paths):
    rows = collections.defaultdict(list)  # (arm, load) -> [(job, round, median, bursts)]
    for path in paths:
        for line in open(path):
            m = LINE.match(line)
            if m and m[2] in (base, test):
                rows[(m[2], int(m[4]))].append((m[1], int(m[3]), float(m[5]), int(m[6])))
    for load in sorted({load for _, load in rows}):
        a, b = rows.get((base, load), []), rows.get((test, load), [])
        print(f"load={load}/GPU")
        for arm, arm_rows in ((base, a), (test, b)):
            print(f"  {arm:12s} n={len(arm_rows)} median step ms {[r[2] for r in arm_rows]} "
                  f"bursts {[r[3] for r in arm_rows]} (sum {sum(r[3] for r in arm_rows)})")
        if len(a) >= 2 and len(b) >= 2:
            print(f"  rank-sum p (two-sided, exact) on bursts = {ranksum_p([r[3] for r in a], [r[3] for r in b]):.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])

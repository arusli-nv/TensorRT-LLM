#!/usr/bin/env python3
"""Decode-only device-step readout for a WideEP FT gate-pack run dir, per arm and load.

RUN holds case dirs <arm>-<rep> (noft-1, ft-1, ...), each with rank-<r>.log. Every iteration line a rank logs
(print_iter_log) carries prev_device_step_time: the device time of the iteration before it. That time is
attributed to the previous iteration, and an iteration counts when it is decode-only (num_ctx_requests == 0),
lies in a load window and serves at least FLOOR x load active requests on its rank (generation tokens would also count MTP draft tokens).
Load windows are the case's result.json rows (start_wall_s .. + elapsed_s, load_per_gpu). On a live run, the
bench/<label>.log of a row not yet in result.json opens a window from the last finished row to now, load taken
from the label's trailing number. --load N instead keeps iterations with num_active == N, without windows.

Reports per arm and load: per-rank median, p10, p90, p99 (ms) pooled over replicates; the cross-rank median of
medians; each arm's delta vs the first arm in µs and %, paired on (replicate, rank) with a sign count; and the
period-60 check: the median step by iteration index mod 60, flagging phases above 1.5x the overall median.

Usage: step_readout.py RUN [--arms BASE,OTHER,...] [--load N] [--floor 0.9] [--rows REGEX] [--per-rank] [-j 16]
"""

import argparse
import concurrent.futures
import datetime
import json
import pathlib
import re
import statistics
import time
from collections import defaultdict

LINE = re.compile(r"iter = (\d+),.*?prev_device_step_time = ([\d.]+|N/A)(?:ms)?,.*?"
                  r"timestamp = ([\d-]+ [\d:]+), states = \{'num_ctx_requests': (\d+), .*?"
                  r"'num_generation_tokens': (\d+)[^}]*\}, num_active = (\d+)")
CASE = re.compile(r"^(.+)-(\d+)$")
LOAD = re.compile(r"(\d+)$")
PERIOD = 60
SPIKE = 1.5


def pct(sorted_values: list, q: float) -> float:
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


def windows(case: pathlib.Path, rows_re: re.Pattern) -> list:
    """(label, load, start, end) per benchmark row of the case, the running row open-ended."""
    rows = []
    try:
        rows = json.loads((case / "result.json").read_text()).get("rows", [])
    except (OSError, json.JSONDecodeError):
        pass
    out, done, last_end = [], set(), 0.0
    for row in rows:
        if "start_wall_s" not in row or "elapsed_s" not in row:
            continue
        done.add(row["label"])
        last_end = max(last_end, row["start_wall_s"] + row["elapsed_s"])
        if rows_re.search(row["label"]):
            out.append((row["label"], row["load_per_gpu"], row["start_wall_s"], row["start_wall_s"] + row["elapsed_s"]))
    if not (case / "frontend.done").exists():
        for log in sorted((case / "bench").glob("*.log")):
            label = log.stem.split(".part")[0]
            match = LOAD.search(label)
            if label not in done and match and rows_re.search(label):
                out.append((label + "*", int(match.group(1)), last_end, time.time() + 3600))
                done.add(label)
    return out


def scan(log: pathlib.Path, wins: list, load: int, floor: float) -> dict:
    """{label: [(iter, device ms), ...]} of the counted iterations in one rank log, streamed."""
    out = defaultdict(list)
    prev = None  # (iter, wall, ctx, gen tokens, active) of the previous iteration line
    with open(log, errors="replace") as lines:
        for line in lines:
            if "prev_device_step_time" not in line:
                continue
            match = LINE.search(line)
            if not match:
                continue
            it, dev = int(match.group(1)), match.group(2)
            if prev is not None and dev != "N/A" and it == prev[0] + 1 and prev[2] == 0:
                p_it, wall, _, gen, active = prev
                if load is not None:
                    if active == load:
                        out[f"active={load}"].append((p_it, float(dev)))
                else:
                    for label, w_load, start, end in wins:
                        if start <= wall <= end and active >= floor * w_load:
                            out[label].append((p_it, float(dev)))
                            break
            wall = datetime.datetime.strptime(match.group(3), "%Y-%m-%d %H:%M:%S").timestamp()
            prev = (it, wall, int(match.group(4)), int(match.group(5)), int(match.group(6)))
    return dict(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=pathlib.Path)
    parser.add_argument("--arms", help="comma-separated arm order; the first is the baseline")
    parser.add_argument("--load", type=int, help="keep iterations with num_active == LOAD instead of row windows")
    parser.add_argument("--floor", type=float, default=0.9, help="min active requests per step / load (generation tokens count draft tokens too under MTP)")
    parser.add_argument("--rows", default=r"^(?!warm)", help="regex of row labels to report")
    parser.add_argument("--per-rank", action="store_true", help="print every rank's stats")
    parser.add_argument("-j", type=int, default=16, help="parallel log readers")
    args = parser.parse_args()
    rows_re = re.compile(args.rows)

    cases = {}  # (arm, rep) -> case dir
    for case in sorted(args.run.iterdir()):
        match = CASE.match(case.name)
        if case.is_dir() and match and any(case.glob("rank-*.log")):
            cases[(match.group(1), int(match.group(2)))] = case
    arms = args.arms.split(",") if args.arms else sorted({a for a, _ in cases}, key=lambda a: (a.startswith("ft"), a))

    # samples[arm][label][(rep, rank)] = [(iter, ms), ...]
    samples = defaultdict(lambda: defaultdict(dict))
    with concurrent.futures.ProcessPoolExecutor(args.j) as pool:
        jobs = {}
        for (arm, rep), case in cases.items():
            if arm not in arms:
                continue
            wins = windows(case, rows_re)
            for log in case.glob("rank-*.log"):
                rank = int(log.stem.split("-")[1])
                jobs[pool.submit(scan, log, wins, args.load, args.floor)] = (arm, rep, rank)
        for fut in concurrent.futures.as_completed(jobs):
            arm, rep, rank = jobs[fut]
            for label, steps in fut.result().items():
                samples[arm][label][(rep, rank)] = steps

    labels = sorted({lb for arm in samples for lb in samples[arm]},
                    key=lambda lb: (int(m.group(1)) if (m := LOAD.search(lb)) else 0, lb))
    if not labels:
        print("no decode-only iterations found (are iteration logs enabled and rows started?)")
        return
    live = [str(c.name) for c in cases.values() if not (c / "frontend.done").exists()]
    print(f"run {args.run.name}  arms {','.join(arms)}  "
          f"{'filter num_active=' + str(args.load) if args.load is not None else 'windows: result.json rows'}"
          f"{'  LIVE: ' + ','.join(sorted(live)) if live else ''}")
    for label in labels:
        print(f"\n== {label}  (device step ms, decode-only; per-rank stats pooled over reps) ==")
        print(f"{'arm':<14}{'reps':>5}{'ranks':>6}{'steps':>8}{'med-of-med':>11}{'p10':>8}{'p90':>8}{'p99':>8}"
              f"{'rank-med min..max':>20}")
        rank_meds = {}  # arm -> {rank: median}
        unit_meds = {}  # arm -> {(rep, rank): median}
        for arm in arms:
            units = samples[arm].get(label, {})
            if not units:
                print(f"{arm:<14}  (no data)")
                continue
            by_rank = defaultdict(list)
            for (rep, rank), steps in units.items():
                by_rank[rank].extend(ms for _, ms in steps)
            stats = {}
            for rank, ms in sorted(by_rank.items()):
                ms.sort()
                stats[rank] = (len(ms), statistics.median(ms), pct(ms, 0.1), pct(ms, 0.9), pct(ms, 0.99))
            rank_meds[arm] = {r: s[1] for r, s in stats.items()}
            unit_meds[arm] = {u: statistics.median(ms for _, ms in st) for u, st in units.items() if st}
            col = list(zip(*stats.values()))
            meds = sorted(col[1])
            print(f"{arm:<14}{len({u[0] for u in units}):>5}{len(stats):>6}{sum(col[0]):>8}"
                  f"{statistics.median(meds):>11.3f}{statistics.median(col[2]):>8.3f}{statistics.median(col[3]):>8.3f}"
                  f"{statistics.median(col[4]):>8.3f}{meds[0]:>10.3f}..{meds[-1]:<8.3f}")
            if args.per_rank or len(stats) <= 4:
                for rank, (n, med, p10, p90, p99) in stats.items():
                    print(f"  rank {rank:<7}{'':>11}{n:>8}{med:>11.3f}{p10:>8.3f}{p90:>8.3f}{p99:>8.3f}")
        base = arms[0]
        for arm in arms[1:]:
            if base not in rank_meds or arm not in rank_meds:
                continue
            b, o = statistics.median(rank_meds[base].values()), statistics.median(rank_meds[arm].values())
            pairs = [unit_meds[arm][u] - unit_meds[base][u] for u in unit_meds[base] if u in unit_meds[arm]]
            line = f"delta {arm} - {base}: {(o - b) * 1e3:+.1f} us ({(o - b) / b * 100:+.2f}%) med-of-med"
            if pairs:
                pos, neg = sum(d > 0 for d in pairs), sum(d < 0 for d in pairs)
                line += (f"; paired (rep,rank) n={len(pairs)} median {statistics.median(pairs) * 1e3:+.1f} us"
                         f" sign +{pos}/-{neg}/={len(pairs) - pos - neg}")
            print(line)
        print(f"period-{PERIOD} (iter mod {PERIOD}; phases > {SPIKE}x overall median; occ = steps > {SPIKE}x"
              f" in that phase / steps in phase):")
        for arm in arms:
            units = samples[arm].get(label, {})
            steps = [s for st in units.values() for s in st]
            if not steps:
                continue
            overall = statistics.median(ms for _, ms in steps)
            phase = defaultdict(list)
            for it, ms in steps:
                phase[it % PERIOD].append(ms)
            flagged = [(p, statistics.median(v), sum(m > SPIKE * overall for m in v), len(v))
                       for p, v in sorted(phase.items()) if statistics.median(v) > SPIKE * overall]
            top = max(phase.items(), key=lambda kv: statistics.median(kv[1]))
            text = ", ".join(f"phase {p}: med {m:.2f} ms ({m / overall:.1f}x) occ {k}/{n}" for p, m, k, n in flagged)
            print(f"  {arm:<14}overall {overall:.3f} ms  {text or 'none flagged'}"
                  f"{'' if flagged else f'  (max phase {top[0]}: {statistics.median(top[1]):.3f} ms)'}")


if __name__ == "__main__":
    main()

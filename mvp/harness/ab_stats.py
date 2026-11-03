"""Test vs base from an interleaved A/B job: cases named <mode>-<round>, each a sweep (default: base noft =
FT off, test ft = FT on).

Per load and metric (client output throughput and median TPOT; steady-state per-GPU throughput and step
time from the server's iteration log, steady_state.py): the mean of each mode, FT's cost in % (positive = FT
slower), and its 95% interval from the per-round paired differences (Student t). The noise floor is the
spread of FT off across rounds (its coefficient of variation and the largest pairwise gap). A load's last
healthy row counts (a load listed twice, e.g. 8,8,32: the second, warmed run); the drift-check repeat does not.
A row with failed client requests (connection resets at 512/GPU) counts if they are at most MAX_FAILED_FRACTION of
its requests: client throughput is over the completed requests only; each mode's failed count is its own column.
Usage: python3 ab_stats.py RUN_DIR [BASE_MODE TEST_MODE]
"""

import json
import pathlib
import statistics
import sys

import steady_state

# Most failed client requests a healthy row may have, as a fraction of its requests.
MAX_FAILED_FRACTION = 0.05
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}
METRICS = (("output_throughput", "tok/s", 1), ("median_tpot_ms", "ms", -1),  # sign: +1 higher is better
           ("steady_mean_tok_s_gpu", "tok/s/GPU", 1), ("window_tok_s_gpu", "tok/s/GPU", 1),
           ("steady_tok_s_gpu", "tok/s/GPU", 1), ("steady_step_ms", "ms", -1),
           ("steady_mean_step_ms", "ms", -1), ("steady_p99_step_ms", "ms", -1))


def clean(row: dict) -> bool:
    """A measured row: rc 0, failed client requests at most MAX_FAILED_FRACTION of its requests, and no cut-off
    stream (one ends without a client error, so it is short but not failed; a failed request is short too)."""
    failed = row.get("failed_requests") or 0
    requests = (row.get("completed") or 0) + failed
    return (row.get("rc") == 0 and requests > 0 and failed <= MAX_FAILED_FRACTION * requests
            and (row.get("short_requests") or 0) == failed)


def load_rows(path: pathlib.Path) -> dict:
    rows = {}
    for row in steady_state.annotate(path.parent, json.loads(path.read_text()).get("rows", [])):
        # healthy-<load> rows only: not the warmup, not the drift-check repeat.
        if row.get("label") != f"healthy-{row['load_per_gpu']}":
            continue
        if clean(row):
            rows[row["load_per_gpu"]] = row
        else:
            print(f"dropped {path.parent.name} {row['label']}: rc={row.get('rc')} completed={row.get('completed')} "
                  f"failed={row.get('failed_requests')} short={row.get('short_requests')}")
    return rows


def main() -> None:
    run = pathlib.Path(sys.argv[1])
    base, test = sys.argv[2:4] if len(sys.argv) > 3 else ("noft", "ft")
    cases = {}
    for result in sorted(run.glob("*-[0-9]*/result.json")):
        mode, _, rnd = result.parent.name.rpartition("-")
        if mode in (base, test):
            cases[(mode, int(rnd))] = load_rows(result)
    rounds = sorted({r for _, r in cases})
    loads = sorted({load for rows in cases.values() for load in rows})
    for metric, unit, sign in METRICS:
        print(f"\n{metric} ({unit}); cost % = {test} slower than {base} by")
        print(f"{'load':>5} {'n':>2} {base:>10} {test:>10} {'cost %':>7} {'95% CI':>15} {'off CV %':>8} {'off max gap %':>13}"
              f" {'failed ' + base:>12} {'failed ' + test:>12}")
        for load in loads:
            off = {r: cases[(base, r)][load][metric] for r in rounds if metric in cases.get((base, r), {}).get(load, {})}
            on = {r: cases[(test, r)][load][metric] for r in rounds if metric in cases.get((test, r), {}).get(load, {})}
            paired = [r for r in off if r in on]
            if not paired:
                continue
            costs = [sign * (off[r] - on[r]) / off[r] * 100 for r in paired]
            mean_cost = statistics.mean(costs)
            half = T95.get(len(costs) - 1, 2.0) * statistics.stdev(costs) / len(costs) ** 0.5 if len(costs) > 1 else float("nan")
            offs = list(off.values())
            cv = statistics.stdev(offs) / statistics.mean(offs) * 100 if len(offs) > 1 else float("nan")
            gap = (max(offs) - min(offs)) / statistics.mean(offs) * 100
            failed = [sum(cases[(mode, r)][load].get("failed_requests") or 0 for r in paired) for mode in (base, test)]
            print(f"{load:>5} {len(paired):>2} {statistics.mean(offs):>10.2f} {statistics.mean(on.values()):>10.2f} "
                  f"{mean_cost:>7.2f} {mean_cost - half:>7.2f}..{mean_cost + half:<6.2f} {cv:>8.2f} {gap:>13.2f}"
                  f" {failed[0]:>12} {failed[1]:>12}")


if __name__ == "__main__":
    main()

"""Decode-step cost split from serving nsys profiles (sqlite exports).

A decode step is one replay of the decode CUDA graph. Per replay: the interval to the next replay (what
throughput sees), the GPU span of its kernels (first start to last end, matched by correlation id), and
the difference, which is time the GPU waits on the host. Per kernel name: the median total time per step.
Only the middle half of the replays of the most frequent large graph is used, so warmup and ramp-down are
excluded.
Per host NVTX range (the executor's phases): the median total time per step.
Usage: python3 profile_steps.py BASE.sqlite[@WALL_S] [OTHER.sqlite[@WALL_S]]
  One profile: its step split and top kernels. Two: both splits and the kernels whose time differs most.
"""

import bisect
import collections
import json
import sqlite3
import statistics
import sys


def analyze(path: str, after_wall_s: float | None = None) -> dict:
    """``after_wall_s``: only steps launched after this wall-clock time (for example a kill) count."""
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    since = 0
    if after_wall_s is not None:
        session = db.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
        since = int(after_wall_s * 1e9) - session
    launches = db.execute(
        "select r.correlationId, r.start from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id = r.nameId "
        "where s.value like 'cudaGraphLaunch%' and r.start >= ? order by r.start", (since,)).fetchall()
    # Graph kernels traced per node do not carry the launch's correlation id: give each kernel to the latest
    # graph launch that precedes it.
    launch_starts = [start for _, start in launches]
    kernels, graph_of = collections.defaultdict(list), {}
    for corr, name, start, end, graph in db.execute(
            "select k.correlationId, s.value, k.start, k.end, k.graphId from CUPTI_ACTIVITY_KIND_KERNEL k "
            "join StringIds s on s.id = k.shortName"):
        if graph is not None:
            i = bisect.bisect_right(launch_starts, start) - 1
            if i < 0:
                continue
            corr = launches[i][0]
        kernels[corr].append((name, start, end))
        graph_of[corr] = graph
    # The decode graph is the most-replayed large graph; a step interval is counted only between two
    # back-to-back replays of it.
    replays = [(start, graph_of[c], kernels[c]) for c, start in launches if c in kernels]
    graph = collections.Counter(g for _, g, k in replays if g and len(k) > 100).most_common(1)[0][0]
    pairs = [(a, b) for a, b in zip(replays, replays[1:]) if a[1] == graph and b[1] == graph]
    pairs = pairs[len(pairs) // 4: 3 * len(pairs) // 4]
    size = len(pairs[0][0][2])
    mid = [a for a, _ in pairs]
    intervals = [b[0] - a[0] for a, b in pairs]
    spans = [max(e for _, _, e in k) - min(s for _, s, _ in k) for _, _, k in mid]
    # GPU busy time per step from every kernel that starts in the step, graph or eager.
    everything = sorted((s, e, n) for k in kernels.values() for n, s, e in k)
    starts = [s for s, _, _ in everything]
    busy = []
    for a, b in pairs:
        lo, hi = bisect.bisect_left(starts, a[0]), bisect.bisect_left(starts, b[0])
        total, end = 0, 0
        for s, e, n in everything[lo:hi]:
            total += max(0, e - max(s, end))
            end = max(end, e)
        busy.append(total)
    per_kernel = collections.defaultdict(list)
    for _, _, k in mid:
        totals = collections.Counter()
        for name, s, e in k:
            totals[name] += e - s
        for name, total in totals.items():
            per_kernel[name].append(total)
    # Host phases: NVTX ranges of the executor thread, summed per step interval.
    ranges = db.execute("select coalesce(n.text, s.value), n.start, n.end from NVTX_EVENTS n "
                        "left join StringIds s on s.id = n.textId where n.end > n.start and n.start >= ? "
                        "order by n.start", (pairs[0][0][0],)).fetchall()
    # Per-step ranges carry the step number ("_forward_step 2826: ..."); keep the phase name only.
    ranges = [(text.split(" ")[1] if text and text.startswith("[Executor]") else text, s, e)
              for text, s, e in ranges if text]
    per_range = collections.defaultdict(list)
    range_starts = [r[1] for r in ranges]
    for a, b in pairs:
        totals = collections.Counter()
        for text, s, e in ranges[bisect.bisect_left(range_starts, a[0]):bisect.bisect_left(range_starts, b[0])]:
            totals[text] += e - s
        for text, total in totals.items():
            per_range[text].append(total)
    us = lambda values: round(statistics.median(values) / 1e3, 1)  # noqa: E731
    return {"profile": path, "kernels_per_step": size, "steps": len(intervals),
            "step_us": us(intervals), "gpu_us": us(spans),
            "outside_graph_us": us([i - g for i, g in zip(intervals, spans)]),
            "gpu_busy_us": us(busy), "gpu_idle_us": us([i - b for i, b in zip(intervals, busy)]),
            "kernels_us": {name: us(v) for name, v in per_kernel.items()},
            "host_ranges_us": {text: us(v) for text, v in per_range.items() if len(v) > len(pairs) // 2}}


def main(paths: list[str]) -> None:
    # PROFILE[@WALL_SECONDS] analyzes only the steps after that wall-clock time.
    results = [analyze(*([p.split("@")[0], float(p.split("@")[1])] if "@" in p else [p])) for p in paths]
    for r in results:
        print(json.dumps({k: v for k, v in r.items() if k not in ("kernels_us", "host_ranges_us")}))
    if len(results) == 1:
        top = sorted(results[0]["kernels_us"].items(), key=lambda kv: -kv[1])[:15]
        print(json.dumps({"top_kernels_us": top}))
        return
    base, other = (r["kernels_us"] for r in results)
    diff = {n: round(other.get(n, 0) - base.get(n, 0), 1) for n in set(base) | set(other)}
    print(json.dumps({"kernel_delta_us": sorted(diff.items(), key=lambda kv: -abs(kv[1]))[:15]}))
    base, other = (r["host_ranges_us"] for r in results)
    diff = {n: round(other.get(n, 0) - base.get(n, 0), 1) for n in set(base) | set(other)}
    print(json.dumps({"host_range_delta_us": sorted(diff.items(), key=lambda kv: -abs(kv[1]))[:15]}))


if __name__ == "__main__":
    main(sys.argv[1:])

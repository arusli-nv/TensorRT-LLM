"""Throughput through a failure from KILL-FINAL runs: rank-0 generated tokens per second (1 s bins) x live ranks.

usage: kill_timeline.py OUT.json CASEDIR=MODE:KIND [...]
Uses result.json trigger_wall_s and rank-0 iteration logs (timestamp, num_generation_tokens). The estimate assumes
rank 0's share is representative (attention DP spreads requests evenly); it counts draft tokens (MTP) too.
"""
import datetime as dt
import json
import re
import sys
from collections import Counter
from zoneinfo import ZoneInfo

out = {}
for spec in sys.argv[2:]:
    case, tag = spec.split("=")
    mode, kind = tag.split(":")
    r = json.load(open(f"{case}/result.json"))
    t0 = r["trigger_wall_s"]
    lost = 4 if kind == "node1" else 1
    bins = Counter()
    for line in open(f"{case}/rank-0.log"):
        m = re.search(r"timestamp = (\S+ \S+), states = \{.*'num_generation_tokens': (\d+)", line)
        if not m:
            continue
        ts = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        bins[ts] += int(m.group(2))
    best = None
    for tz in ("UTC", "America/Los_Angeles"):
        secs = {k.replace(tzinfo=ZoneInfo(tz)).timestamp(): v for k, v in bins.items()}
        if min(secs) <= t0 <= max(secs) and (best is None or True):
            best = secs
    rel = sorted((s - t0, v) for s, v in best.items() if -30 <= s - t0 <= 60)
    series = [(t, v * (32 if t < 0 else 32 - lost)) for t, v in rel]
    out.setdefault(mode, {}).setdefault(kind, []).append({"case": case.rstrip("/").split("/")[-2] + "/" + case.rstrip("/").split("/")[-1], "series": series})
json.dump(out, open(sys.argv[1], "w"))
for m, v in out.items():
    for k, cs in v.items():
        s = cs[0]["series"]
        before = [x for t, x in s if -20 <= t < -1]; after = [x for t, x in s if 10 <= t < 40]
        print(m, k, len(cs), "before", round(sum(before) / max(1, len(before))), "after", round(sum(after) / max(1, len(after))))

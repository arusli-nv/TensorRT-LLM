"""Collect per-arm, per-load client metrics from STD runs into one JSON for the evaluation figures.

usage: std_collect.py OUT.json RUNDIR [RUNDIR ...]
Each RUNDIR holds <arm>-<round>/result.json with rows labelled healthy-<load> (warm-up rows are skipped).
tok/s per user is 1000 / median TPOT.
"""
import glob
import json
import os
import re
import sys

out, dirs = sys.argv[1], sys.argv[2:]
# Rows invalid under the protocol (e.g. >1% of requests in the 55-75 s TTFT band at 512): "RUNDIR/arm-N:load" in $STD_EXCLUDE.
exclude = set(os.environ.get("STD_EXCLUDE", "").split())
data = {}
for d in dirs:
    for f in sorted(glob.glob(f"{d}/*-[0-9]*/result.json")):
        m = re.match(r"(?:std-)?(.+)-(\d+)$", os.path.basename(os.path.dirname(f)))
        if not m:
            continue
        for row in json.load(open(f)).get("rows", []):
            if not re.fullmatch(r"healthy-\d+", row.get("label", "")):
                continue
            if row.get("rc") or row.get("failed_requests") or row.get("completed", 0) == 0:
                print(f"skip {f} {row.get('label')}: rc={row.get('rc')} failed={row.get('failed_requests')}")
                continue
            rec = {k: row.get(k) for k in ("output_throughput", "median_tpot_ms", "p99_tpot_ms", "median_ttft_ms",
                                           "p99_itl_ms", "elapsed_s", "completed")}
            rec["user_throughput"] = 1000.0 / row["median_tpot_ms"]
            rec["run"] = f"{os.path.basename(d)}/{m.group(1)}-{m.group(2)}"
            if f"{rec['run']}:{row['load_per_gpu']}" in exclude:
                print(f"excluded {rec['run']} at {row['load_per_gpu']}")
                continue
            data.setdefault(m.group(1), {}).setdefault(str(row["load_per_gpu"]), []).append(rec)
json.dump(data, open(out, "w"), indent=1)
for arm in sorted(data):
    print(arm, {L: len(v) for L, v in sorted(data[arm].items(), key=lambda x: int(x[0]))})

"""Collect FT recovery timings from KILL-FINAL runs into one JSON for the recovery figure.

usage: kill_collect.py OUT.json RUNDIR=MODE [RUNDIR=MODE ...]
Reads RUNDIR/kill_final.txt (KILL_FINAL_VERDICT lines). Phases, in seconds from the failure:
detect, stop (detect->stopped), plan (stopped->ready, incl. ghost mapping), install (ready->resumed),
first step (resumed->first_step).
"""
import json
import re
import sys

out, specs = sys.argv[1], sys.argv[2:]
data = {}
for spec in specs:
    rundir, mode = spec.split("=")
    for line in open(f"{rundir}/kill_final.txt"):
        if not line.startswith("KILL_FINAL_VERDICT"):
            continue
        case = line.split()[1]
        kv = dict(re.findall(r"(\w+)=(\S+)", line))
        if "PASS" not in line or "phases1" not in kv:
            print(f"skip {rundir} {case}: not PASS or no phases")
            continue
        ph = dict(p.split("=") for p in kv["phases1"].split(","))
        t = {k: float(v) for k, v in ph.items()}
        rec = {"detect": t["detect"], "stop": t["stopped"] - t["detect"], "plan": t["ready"] - t["stopped"],
               "install": t["resumed"] - t["ready"], "first step": t["first_step"] - t["resumed"],
               "total": t["first_step"], "resumed": t["resumed"], "client_failed": int(float(kv.get("client_failed", 0))),
               "run": f"{rundir.rstrip('/').split('/')[-1]}/{case}"}
        kind = re.sub(r"-\d+$", "", case)
        data.setdefault(mode, {}).setdefault(kind, []).append(rec)
json.dump(data, open(out, "w"), indent=1)
for mode, v in data.items():
    for kind, rs in v.items():
        print(mode, kind, [round(r["total"], 2) for r in rs], "resumed", [round(r["resumed"], 2) for r in rs])

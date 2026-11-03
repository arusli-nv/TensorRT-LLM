#!/usr/bin/env python3
"""History sweep of rank 0's cold-start and steady stall statistics over every served arm (FT on vs off).

Reuses tools/cold_steps.py (first 150 iterations after ready) and mvp/analysis/dist32.case_stats (trim mean,
stall ms/step, as tools/trim_readout.py). Kill arms are read only up to their first recovery line (pre-kill).
Usage: PYTHONPATH=tools/pylib python3 -I review/HISTORY_STALL_SWEEP.py OUT.csv WHEELMAP.txt [ROOT...]
       python3 -I review/HISTORY_STALL_SWEEP.py --enrich RAW.csv OUT.csv   (served arms only; FT state, model, trust)
       python3 -I review/HISTORY_STALL_SWEEP.py --pairs HISTORY_STALL_SWEEP.csv PAIRS.csv  (same-run FT on vs off)
WHEELMAP lines: "<sha256 first 16 hex> <build dir>" for every build-*/dist/*.whl (install_wheel.sh's marker).
"""
import csv
import importlib.util
import multiprocessing
import pathlib
import re
import sys

C = pathlib.Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cold = _load("cold_steps", C / "tools" / "cold_steps.py")
dist32 = _load("dist32", C / "mvp" / "analysis" / "dist32.py")

DATE = re.compile(r"\[(\d\d)/(\d\d)/(\d{4})-(\d\d:\d\d)")
HOST = re.compile(r"Rank0 host: ([\w.-]+)")
CONFIG = re.compile(r"--config (\S+)")
SLOTS = re.compile(r"appends (\d+) (\w+) slots per rank to EPLB's (\d+)")
SETTINGS = re.compile(r"Recovery authority: fault-tolerance settings \S+: (.*)")
RECOVERY = re.compile(r"WideEP recovery gen=|could not end failed rank|peer failure|Peer failure|ranks? failed")
FT_LOG = re.compile(r"registering with the WideEP recovery authority|WideEP peer-loss recovery|peer-loss recovery")
BUILD = re.compile(r"(build-[A-Za-z0-9_.-]+)/dist")
INSTALLED = re.compile(r"already installed \(([0-9a-f]{16})\)")


def kv(text, key):
    m = re.search(rf"^{key}=(.*)$", text, re.M)
    return m.group(1) if m else None


def settings_field(s, key):
    m = re.search(rf"\b{key}=('?[^ ']*'?)", s or "")
    return m.group(1).strip("'") if m else ""


def read(p):
    try:
        return p.read_text(errors="replace")
    except OSError:
        return ""


def cold_stats(steps):
    rest = [s for s in steps if s[0] != 1]
    times = sorted(ms for _, ms in rest)
    if not times:
        return {}
    big = [ms for ms in times if ms > 1000]
    return {"cold_n": len(rest), "cold_gt1s": len(big), "cold_gt1s_sum_s": round(sum(big) / 1000, 1),
            "cold_gt200ms": sum(ms > 200 for ms in times), "cold_sum_gt200_s": round(sum(ms for ms in times if ms > 200) / 1000, 2),
            "cold_p50": round(times[len(times) // 2], 1), "cold_p90": round(times[int(0.9 * (len(times) - 1))], 1),
            "cold_max": round(times[-1]), "cold_iter1": round(steps[0][1]) if steps[0][0] == 1 else ""}


def prekill_steps(log, first):
    """cold_steps.first_steps, stopping at the first recovery line."""
    steps = []
    with open(log, errors="replace") as lines:
        for line in lines:
            if RECOVERY.search(line):
                break
            m = cold.ITER.search(line)
            if not m:
                continue
            it = int(m.group(1))
            if it == 1:
                steps = []
            if it <= first:
                steps.append((it, float(m.group(2))))
    return steps


def mode_of(cfg):
    if not cfg:
        return ""
    if "load_balancer" not in cfg:
        return "noeplb"
    m = re.search(r"layer_updates_per_iter:\s*(\d+)", cfg)
    if m and int(m.group(1)) > 0:
        return "online"
    return "offline"


def scan(case):
    case = pathlib.Path(case)
    row = {"run": case.parent.name, "arm": case.name, "path": str(case.relative_to(C))}
    env = read(case / "env-0.txt")
    head = ""
    with open(case / "rank-0.log", errors="replace") as f:
        for i, line in enumerate(f):
            if i > 3000:
                break
            head += line
    m = DATE.search(head)
    row["date"] = f"{m.group(3)}-{m.group(1)}-{m.group(2)} {m.group(4)}" if m else ""
    m = HOST.search(head)
    row["rank0_host"] = m.group(1) if m else ""
    # Build.
    build = kv(env, "MVP_BUILD") or ""
    src = "env" if build else ""
    inst = read(case / "install-0.log")[:4000]
    if not build:
        m = BUILD.search(inst) or BUILD.search(kv(env, "dist") or "")
        if m:
            build, src = m.group(1), "install-log"
    m = INSTALLED.search(inst)
    row["wheel_sha16"] = m.group(1) if m else ""
    if not build and m and m.group(1) in WHEELS:
        build, src = WHEELS[m.group(1)], "wheel-sha"
    row["build"], row["build_src"] = build, src
    row["overlay"] = kv(env, "MVP_OVERLAY") or ""
    # Config / mode.
    cfg = read(case / "config-0.yaml") or read(case / "effective-config-0.yaml")
    cfg_src = "case" if cfg else ""
    m = CONFIG.search(head)
    if not cfg and m:
        p = pathlib.Path(m.group(1))
        cfg = read(p) if p.exists() else read(C / "mvp" / "harness" / "configs" / p.name)
        cfg_src = "harness-copy" if cfg else ""
    row["config"] = pathlib.Path(m.group(1)).name if m else ""
    row["config_src"] = cfg_src
    row["mode"] = mode_of(cfg)
    ms = re.search(r"num_slots:\s*(\d+)", cfg or "")
    row["num_slots"] = ms.group(1) if ms else ""
    # FT state.
    auth = read(case / "authority.log")
    sm = SETTINGS.search(auth)
    s = sm.group(1) if sm else ""
    row["authority_settings"] = s
    ftenv = kv(env, "TLLM_FAULT_TOLERANCE_MODE")
    ftlog = bool(FT_LOG.search(head))
    row["ft"] = "on" if (ftenv == "1" or ftlog) else ("off" if env or head else "")
    row["ft_src"] = "env" if ftenv is not None else ("log" if ftlog else "no-marker")
    ftc = read(case / "ft-config-0.yaml")
    hd = settings_field(s, "hang_detection") or kv(env, "TLLM_FAULT_TOLERANCE_HANG_DETECTION") or \
        (re.search(r"hang_detection:\s*(\S+)", ftc).group(1) if "hang_detection" in ftc else "")
    row["hang_detection"] = {"1": "True", "true": "True", "0": "False", "false": "False"}.get(hd, hd)
    row["peer_failure_poll_s"] = settings_field(s, "peer_failure_poll_s")
    row["extra_slots_cfg"] = settings_field(s, "extra_slots") or kv(env, "TLLM_FAULT_TOLERANCE_EXTRA_SLOTS") or ""
    row["failure_domain"] = settings_field(s, "failure_domain")
    m = SLOTS.search(head)
    row["slots_log"] = f"{m.group(1)} {m.group(2)} + {m.group(3)}" if m else ""
    row["ghost_not_host_accessible"] = "Ghost slot memory cannot be host accessible" in head
    row["transport"] = kv(env, "WIDEEP_SM_MCA") or kv(env, "WIDEEP_MPI_MCA") or ""
    row["pin_host_copies"] = kv(env, "TLLM_TEST_PIN_HOST_COPIES") or ""
    row["thp_disable"] = kv(env, "TLLM_TEST_THP_DISABLE") or ""
    # Host copies: rank 0 node's peak shared memory (free's 'shared' column) during the case.
    shared = []
    for line in read(case / "hostmem-0.log").splitlines():
        if line.startswith("Mem:"):
            parts = line.split()
            if len(parts) >= 5 and parts[4].isdigit():
                shared.append(int(parts[4]) / 2**30)
    row["node0_shared_gib_max"] = round(max(shared), 1) if shared else ""
    kill = bool(kv(env, "TLLM_TEST_KILL_RANK") or "KILL" in case.parent.name.upper() or "kill" in case.name.lower())
    row["kill_case"] = kill
    steps = prekill_steps(case / "rank-0.log", 150) if kill else cold.first_steps(case / "rank-0.log", 150)
    row["cold_scope"] = "pre-kill" if kill else "full"
    row.update(cold_stats(steps))
    # Steady windows.
    if (case / "result.json").exists() and not kill:
        try:
            for label, st in sorted(dist32.case_stats(case).items()):
                if st:
                    load = label.split("-", 1)[-1]
                    row[f"trim_{load}"] = round(st["trim_mean"], 3)
                    row[f"stall_{load}"] = round(st["stall_ms_per_step"], 3)
                    row[f"n_{load}"] = st["n"]
        except Exception as e:  # noqa: BLE001 - a malformed historical result must not stop the sweep
            row["steady_error"] = type(e).__name__
    return row


WHEELS = {}


def init(wheelmap):
    for line in pathlib.Path(wheelmap).read_text().splitlines():
        h, b = line.split()
        WHEELS[h] = b.split("/")[-1]


def safe_scan(case):
    try:
        return scan(case)
    except Exception as e:  # noqa: BLE001
        return {"path": str(case), "scan_error": repr(e)[:200]}


CLEAN = {"build-v2-cef4670789", "build-v3-bcea8da496", "build-v5-2ccc3e8e1c", "build-v5r-b643ad8a29",
         "build-v5r-b0ad558bcb", "build-v5x-eb3950a3f8", "build-v5land-9f2f1892f0", "build-v5zero-6645ce1848",
         "build-pr0-fc52f0adee"}


def trust(build):
    if build == "build-main-0829689c52":
        return "NOT-clean"
    if build in CLEAN or build.startswith("build-freeze-"):
        return "clean"
    return "lower(unaudited)" if build else "lower(build unknown)"


def enrich(raw, out):
    """FT on = the env says so, the authority logged its settings, or rank 0 registered with it. The emptied-slot
    layout testhook logs 'keeps N EPLB slots empty' with FT off, so that line alone does not count."""
    rows = [r for r in csv.DictReader(open(raw)) if r.get("cold_n")]
    for r in rows:
        head = ""
        with open(C / r["path"] / "rank-0.log", errors="replace") as f:
            for i, line in enumerate(f):
                if i > 4000:
                    break
                head += line
        reg = "registering with the WideEP recovery authority" in head
        env = read(C / r["path"] / "env-0.txt")
        ftenv = kv(env, "TLLM_FAULT_TOLERANCE_MODE")
        if ftenv is not None:
            r["ft"] = "on" if ftenv == "1" else "off"
        else:
            r["ft"] = "on" if (reg or r["authority_settings"]) else "off"
        r["ft_src"] = "env" if ftenv is not None else ("authority" if r["authority_settings"] else
                                                       ("rank0-register" if reg else "no-marker"))
        r["layout_testhook_ft_off"] = r["ft"] == "off" and "WideEP peer-loss recovery" in head
        m = re.search(r"trtllm-serve (\S+)", head)
        r["model"] = pathlib.Path(m.group(1)).name if m else ""
        m = re.search(r"tllm_mpi_size: (\d+)", head)
        r["world"] = m.group(1) if m else ""
        if r["ft"] == "on" and not r["hang_detection"]:
            r["hang_detection"] = "n/a(settings not logged)"
        if r["ft"] == "on" and r["authority_settings"] and "hang_detection" not in r["authority_settings"]:
            r["hang_detection"] = "absent(pre-hang build)"
        if r["ft"] == "off":
            r["hang_detection"] = r["peer_failure_poll_s"] = r["extra_slots_cfg"] = ""
        r["build_trust"] = trust(r["build"])
        # Deployment mode for pairing: static one-slot-per-expert (256) placements are the no-EPLB FT layout.
        r["mode_cmp"] = ("online" if r["mode"] == "online" else "noeplb" if r["mode"] == "noeplb" or
                         r["num_slots"] == "256" else "static320" if r["num_slots"] == "320" else "offline")
        g = r["node0_shared_gib_max"]
        r["shm"] = ("unknown" if g == "" else "none(<5G)" if float(g) < 5 else "ring(5-150G)" if float(g) < 150
                    else "full(>150G)")
        kind = r["extra_slots_cfg"] if r["extra_slots_cfg"] not in ("", "None") else ""
        r["extra_kind"] = kind or (r["slots_log"].split()[1] if r["slots_log"] else "")
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "served arms")


def pairs(arms_csv, out):
    """Every non-kill FT-on arm against each non-kill FT-off arm of the same run dir (same job group, same nodes)
    and deployment mode (no-EPLB counts static 256/320 placements)."""
    rows = [r for r in csv.DictReader(open(arms_csv)) if r["world"] == "32" and "R1" in r["model"]
            and r["kill_case"] != "True" and not r["arm"].startswith("tune")]
    static = ("noeplb", "static320")
    by_run = {}
    for r in rows:
        by_run.setdefault(r["run"], []).append(r)

    def diff(a, o, k):
        try:
            return round(float(a[k]) - float(o[k]), 3)
        except (KeyError, ValueError):
            return ""
    out_rows = []
    for run, arms in by_run.items():
        for a in (x for x in arms if x["ft"] == "on"):
            for o in (x for x in arms if x["ft"] == "off"):
                if not (o["mode_cmp"] == a["mode_cmp"] or (a["mode_cmp"] in static and o["mode_cmp"] in static)):
                    continue
                row = {"run": run, "on": a["arm"], "off": o["arm"], "date": a["date"][:10], "build": a["build"],
                       "trust": a["build_trust"], "mode": a["mode_cmp"], "hang": a["hang_detection"],
                       "poll": a["peer_failure_poll_s"], "extra": a["extra_kind"], "shm_on": a["shm"],
                       "shm_off": o["shm"]}
                for k in ("cold_gt200ms", "cold_sum_gt200_s", "cold_p90", "stall_32", "stall_128"):
                    row[k + "_on"], row[k + "_off"] = a[k], o[k]
                for k in ("stall_32", "stall_128", "trim_32", "trim_128"):
                    row["d_" + k] = diff(a, o, k)
                out_rows.append(row)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(len(out_rows), "pairs")


def main():
    if sys.argv[1] == "--pairs":
        pairs(sys.argv[2], sys.argv[3])
        return
    if sys.argv[1] == "--enrich":
        enrich(sys.argv[2], sys.argv[3])
        return
    out, wheelmap, *roots = sys.argv[1:]
    init(wheelmap)
    cases = sorted({p.parent for r in roots for p in pathlib.Path(r).rglob("rank-0.log")})
    with multiprocessing.Pool(16, initializer=init, initargs=(wheelmap,)) as pool:
        rows = pool.map(safe_scan, cases, chunksize=4)
    # Fill a missing build from the run dir's other arms (one job installs one wheel unless arms alternate).
    by_run = {}
    for r in rows:
        if r.get("build"):
            by_run.setdefault(r["run"], set()).add(r["build"])
    for r in rows:
        if not r.get("build") and len(by_run.get(r.get("run"), ())) == 1:
            r["build"], r["build_src"] = next(iter(by_run[r["run"]])), "run-sibling"
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "arms")


if __name__ == "__main__":
    main()

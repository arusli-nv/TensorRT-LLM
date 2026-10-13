"""Recovery-step timeline of FT kill runs: per step, seconds after the kill (min/median/max over survivors)."""
import glob, json, re, statistics as st, sys

ORDER = ["watcher_detect", "watcher_revoked", "recover_start", "revoked", "shrink_done", "gpu_drained",
         "kv_rollback_done", "stopped_sent", "propose_recv", "ready_sent", "install_recv", "install_applied",
         "installed_sent", "commit_recv", "requests_settled", "resumed"]

def case(run):
    res = json.load(open(f"{run}/result.json"))
    kill = res["kill_wall_s"]
    marks = {}
    for log in glob.glob(f"{run}/rank-*.log"):
        rank = int(re.search(r"rank-(\d+)", log).group(1))
        for name, t in re.findall(r"WIDEEP_PHASE (\w+) ([\d.]+)", open(log, errors="ignore").read()):
            marks.setdefault(name, {}).setdefault(rank, float(t))  # first occurrence per rank
    rows = {n: [t - kill for t in marks[n].values()] for n in ORDER if n in marks}
    return res, rows

for run in sys.argv[1:]:
    res, rows = case(run)
    print(f"== {run.split('/')[-1]}  state={res.get('state')}  "
          f"first_token_after_kill={res.get('first_token_after_kill_s')}  requests_done={res.get('recovered_requests_done_s'):.2f}s")
    for n, v in rows.items():
        print(f"   {n:18s} ranks={len(v):2d}  min={min(v):7.3f}  med={st.median(v):7.3f}  max={max(v):7.3f}")

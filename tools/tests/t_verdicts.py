# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Usage: python3 -I t_verdicts.py mvp/harness/draft_cases/val_verdicts.py   (C10 deadline and C17 exact verdicts)
import importlib.util, sys
spec = importlib.util.spec_from_file_location("vv", sys.argv[1]); vv = importlib.util.module_from_spec(spec); spec.loader.exec_module(vv)

# C10 deadline: per rank, no MPI call after its own first deadline line; job ended <= 180 s.
call = lambda r, t, n="shrink": f"WIDEEP_MPI_CALL rank {r} t={t} name={n}\n"
dl = lambda r, t: f"WIDEEP_DEADLINE rank {r} t={t} 'The recovery deadline passed before x'\n"
good = {0: call(0, 1, "watch_peer_failures") + call(0, 10, "revoke") + dl(0, 50), 9: call(9, 46) + call(9, 60) + dl(9, 70)}
ok, f = vv.deadline_checks(good, 60.0); assert ok and f["deadline_ranks"] == "0,9", f
ok, f = vv.deadline_checks({**good, 3: dl(3, 50) + call(3, 51, "revoke")}, 60.0); assert not ok and f["late_call_ranks"] == "3", f
ok, f = vv.deadline_checks(good, 181.0); assert not ok and f["ended"] is False, f
ok, f = vv.deadline_checks(good, None); assert not ok, f
ok, f = vv.deadline_checks({0: call(0, 1)}, 60.0); assert not ok and f["deadline_seen"] is False, f
# An event glued to another thread's output still counts.
ok, f = vv.deadline_checks({0: dl(0, 5) + "x " + call(0, 6)}, 60.0); assert not ok and f["late_call_ranks"] == "0", f

# C17 exact: pre-kill exact; after the kill no earlier/more divergence than the perturbed (victim on rank 0) control.
import os, tempfile
row = lambda toks: {"token_ids": list(toks)}
ref_rows = [row(range(10)), row(range(10, 20)), row(range(20, 30))]
assert vv.first_diffs(ref_rows, ref_rows) == [10, 10, 10]
assert vv.first_diffs(ref_rows, [ref_rows[0], row([10, 11, 99] + list(range(13, 20))), ref_rows[2]]) == [10, 2, 10]
assert vv.first_diffs(ref_rows, [ref_rows[0], row(range(10, 15)), ref_rows[2]]) == [10, 5, 10]  # truncated
assert vv.first_diffs(ref_rows, ref_rows[:2]) is None and vv.first_diffs(ref_rows, [{"error": "x"}] * 3) is None
victim = row(range(50, 60))
def diverge(r, at):
    t = list(r["token_ids"]); t[at] = 999; return row(t)
def run(recovered=ref_rows, vic=victim, **kw):
    return {"state": "PROBE", "deterministic": True, "healthy": ref_rows, "recovered": recovered, "victim_request": vic, **kw}
ref = run()
pert = run(recovered=[diverge(ref_rows[0], 6), ref_rows[1], ref_rows[2]], vic=diverge(victim, 4), state="FAIL")
kill = run(state="DONE", victim_finished_before_kill=False); del kill["deterministic"]
placed = (0, 300)
ok, p, f = vv.exact_verdict(ref, pert, kill, 1, placed); assert ok, p                              # kill exact
assert f["pert_differ"] == "2/4=50%" and sorted(f["pert_first_diff"]) == [4, 6, 10, 10] and f["kill_differ"] == "0/4=0%", f
k2 = {**kill, "recovered": [diverge(ref_rows[0], 7), ref_rows[1], ref_rows[2]], "victim_request": diverge(victim, 5)}
ok, p, f = vv.exact_verdict(ref, pert, k2, 1, placed); assert ok, (p, f)                            # later, same count
k3 = {**kill, "recovered": [diverge(ref_rows[0], 7), ref_rows[1], ref_rows[2]], "victim_request": diverge(victim, 3)}
ok, p, f = vv.exact_verdict(ref, pert, k3, 1, placed)                                              # one earlier
assert not ok and p == ["after the kill: diverges more or earlier than the perturbed control"], p
k4 = {**k2, "recovered": [diverge(ref_rows[0], 7), diverge(ref_rows[1], 9), ref_rows[2]]}
ok, p, f = vv.exact_verdict(ref, pert, k4, 1, placed); assert not ok and f["kill_differ"] == "3/4=75%", (p, f)  # more requests
ok, p, f = vv.exact_verdict(ref, pert, {**kill, "healthy": [diverge(ref_rows[0], 9), ref_rows[1], ref_rows[2]]}, 1, placed)
assert not ok and any(x.startswith("pre-kill") for x in p), p                                     # pre-kill must be exact
ok, p, f = vv.exact_verdict(ref, ref, kill, 1, placed); assert ok, p
ok, p, f = vv.exact_verdict(ref, pert, kill, 1, (0, 0)); assert not ok and any("placement unproven" in x for x in p), p
ok, p, f = vv.exact_verdict(ref, pert, kill, 1, (5, 300)); assert not ok, p
ok, p, f = vv.exact_verdict(ref, pert, kill, 0, placed); assert p == ["reruns_logged=0"], p
ok, p, f = vv.exact_verdict(ref, pert, {**kill, "victim_finished_before_kill": True}, 1, placed); assert not ok, p
ok, p, f = vv.exact_verdict(ref, {}, kill, 1, placed); assert not ok, p
ok, p, f = vv.exact_verdict(ref, pert, {**kill, "victim_request": {"error": "lost"}}, 1, placed); assert not ok, p
ok, p, f = vv.exact_verdict(ref, pert, {}, 1, placed); assert not ok, p
with tempfile.TemporaryDirectory() as d:
    log = os.path.join(d, "rank-0.log")
    with open(log, "w") as out:
        out.write("iter = 1, num_scheduled_requests = 4, currank_total_requests = 4/512,\n"
                  + "iter = 9, currank_total_requests = 2/512,\n" * 3 + "iter = 12, currank_total_requests = 1/512,\n")
    assert vv.rank0_two_request_iters(log) == 3 and vv.rank0_two_request_iters(d + "/none.log") is None
print("PASS t_verdicts")

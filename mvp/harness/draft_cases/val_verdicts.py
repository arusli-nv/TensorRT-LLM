# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze-validation verdicts (job.sh), kept here so tools/tests/t_verdicts.py can test them on a CPU.

deadline (C10, KILL-FINAL row ``deadline``): the job ends on its own within 180 s of the kill, a recovery deadline
passed on some rank (WIDEEP_DEADLINE), and no rank starts revoke/shrink/watch (WIDEEP_MPI_CALL) after its own first
deadline line (testhooks TLLM_TEST_LOG_DEADLINE).

exact (C17, EXACT-FINAL): runs of the same greedy prompts (tokens/frontend.py result.json): ref (the victim prompt on
rank 1), pert (the victim prompt on rank 0, beside a compared request: rank 0's batch goes 1 -> 2, the batch-shape
change a survivor sees when the lost rank's request runs again there), kill (rank 5 killed mid-decode). Pre-kill
tokens must equal ref exactly. After the kill (the survivors' requests plus the victim's rerun), the kill run's
per-request first-diff positions vs ref (a request that does not differ counts as its length) must be no earlier,
sorted position by position, than pert's: so neither more differing requests nor earlier divergence than a no-kill
batch-shape change. Both distributions are printed, so a FAIL reads as "how much more than the control".

h23 (every recoverable KILL-FINAL row, user decision H23): serving resumes within 10 s of the loss; the deployment's
decode rate after the loss over before is at least 0.8 x live ranks / total ranks; at least 20 requests ran in full
after the loss, so the ratio covers real traffic. A frozen rank (row stop, SIGSTOP) is not a loss until the hang
confirm window (hang_confirm_s, 30 s) confirms it: its resume is timed from that confirmation, reported apart as
h23_confirm_s.
"""
import heapq
import json
import re
import statistics
import sys
import time

# Not anchored at a line start: another thread's output can precede an event on its line.
_EVENT = re.compile(r"(WIDEEP_DEADLINE|WIDEEP_MPI_CALL) rank (\d+) t=([0-9.]+)")


def deadline_checks(rank_logs, ended_s):
    """rank_logs: {rank: log text}; ended_s: kill -> mpirun exit (None if it never ended). Returns (ok, fields)."""
    deadlines, late = {}, []
    for rank, text in rank_logs.items():
        events = [(float(t), kind) for kind, _, t in _EVENT.findall(text)]
        first = min((t for t, kind in events if kind == "WIDEEP_DEADLINE"), default=None)
        if first is not None:
            deadlines[rank] = first
            late += [rank for t, kind in events if kind == "WIDEEP_MPI_CALL" and t > first][:1]
    checks = dict(ended=ended_s is not None and ended_s <= 180, deadline_seen=bool(deadlines), no_call_after=not late)
    fields = dict(ended_s=ended_s, deadline_ranks=",".join(map(str, sorted(deadlines))) or None,
                  late_call_ranks=",".join(map(str, sorted(late))) or None)
    return all(checks.values()), {**fields, **{k: v for k, v in checks.items() if not v}}


H23_RESUME_S, H23_TPUT_FRACTION, H23_MIN_COMPLETED = 10.0, 0.8, 20


def h23_checks(resume_s, tok_s_before, tok_s_after, live, total, completed_after_kill, confirm_s=None):
    """resume_s: loss -> serving again, one per loss (a frozen rank: hang confirmed -> serving again); tok_s_*: the
    deployment's decode rate before / after the loss; live / total: ranks after the loss / at start; confirm_s: a
    frozen rank's freeze -> hang confirmed, reported only. An empty after-window (tok_s_after None: the load ended
    before enough full decode steps after the resume) is not a throughput loss: h23_tput_ratio=N/A, not checked.
    Returns (checks, fields)."""
    floor = H23_TPUT_FRACTION * live / total
    ratio = tok_s_after / tok_s_before if tok_s_before and tok_s_after else None
    checks = dict(h23_resume=bool(resume_s) and all(s is not None and s <= H23_RESUME_S for s in resume_s),
                  h23_tput=tok_s_after is None or (ratio is not None and ratio >= floor - 1e-9),  # 0.8 x 28/32 is 0.7000...1
                  h23_completed=(completed_after_kill or 0) >= H23_MIN_COMPLETED)
    fields = dict(h23_resume_s=",".join(str(s) for s in resume_s),
                  **({"h23_confirm_s": confirm_s} if confirm_s is not None else {}),
                  h23_tput_ratio="N/A" if tok_s_after is None else ratio and round(ratio, 3),
                  h23_tput_floor=round(floor, 3), h23_live=f"{live}/{total}",
                  h23_completed_after_kill=completed_after_kill)
    return checks, fields


_HUNG = re.compile(r"^\[(\d\d/\d\d/\d{4}-\d\d:\d\d:\d\d)\].* is hung .*fencing it", re.M)


def hang_confirmed_wall_s(authority_log, detect_t):
    """When a frozen rank's hang was confirmed (wall s): the earlier of the authority's first "is hung ... fencing it"
    line (a whole-second local stamp, so up to 1 s early) and rank 0's gen-1 detect phase (*detect_t*; Open MPI's
    detector, when on, detects first). None without either."""
    m = _HUNG.search(authority_log or "")
    hung = time.mktime(time.strptime(m.group(1), "%m/%d/%Y-%H:%M:%S")) if m else None
    return min((t for t in (hung, detect_t) if t is not None), default=None)


def completed_after(parts, anchor_s, t0, osl):
    """Requests that ran in full (no error, osl tokens) and finished after wall time t0, or None without the data.

    parts: one benchmark_serving result per client process (input order, with max_concurrency); anchor_s: the
    clients' benchmark start (wall s). Each client starts its first max_concurrency requests at once and every later
    one when a slot frees (FIFO semaphore), so replaying the per-request latencies gives each request's finish time.
    """
    if not parts or anchor_s is None or t0 is None:
        return None
    count = 0
    for result in parts:
        e2els, errors, lens = (result.get(k) or [] for k in ("e2els", "errors", "output_lens"))
        conc = result.get("max_concurrency")
        if not e2els or not conc or not len(e2els) == len(errors) == len(lens):
            return None
        finishes = []
        for i, (e2el, error, out) in enumerate(zip(e2els, errors, lens)):
            start = heapq.heappop(finishes) if i >= conc else 0.0
            heapq.heappush(finishes, start + e2el)
            count += not error and out >= osl and anchor_s + start + e2el > t0
    return count


_MAPPED = re.compile(r"Mapped (\d+) bytes behind ghost expert slots")


def restored_load(rank_logs):
    """Readout: device memory each rank mapped behind ghost slots at its recoveries (the restored experts it now
    serves), as max@ranks/median/min MB and max over median; None when no rank logged a mapping."""
    mb = {rank: sum(map(int, found)) / 2**20 for rank, text in rank_logs.items() if (found := _MAPPED.findall(text))}
    if not mb:
        return None
    top, med = max(mb.values()), statistics.median(mb.values())
    hot = ",".join(str(r) for r in sorted(mb) if mb[r] == top)
    return f"max{top:.0f}@{hot}/med{med:.0f}/min{min(mb.values()):.0f}/x{top / med:.2f}" if med else f"max{top:.0f}@{hot}"


def first_diffs(want_rows, got_rows):
    """Per request, the first differing token position (its length when equal); None on a count mismatch or an
    errored request."""
    if not want_rows or len(want_rows) != len(got_rows) or any("error" in row for row in got_rows):
        return None
    out = []
    for want, got in zip(want_rows, got_rows):
        a, b = want.get("token_ids") or [], got.get("token_ids") or []
        out.append(next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))))
    return out


def _differing(want_rows, positions):
    return [p for p, row in zip(positions, want_rows) if p < len(row.get("token_ids") or [])]


def exact_verdict(ref, pert, kill, reruns, rank0_pairs=(None, None)):
    """rank0_pairs: iterations where rank 0 held 2 requests in ref and pert (placement proof). Returns
    (ok, problems, fields)."""
    problems = []
    if ref.get("state") != "PROBE" or not ref.get("deterministic"):
        problems.append(f"ref_state={ref.get('state')} ref_deterministic={ref.get('deterministic')}")
    if not pert.get("deterministic"):
        problems.append(f"pert_deterministic={pert.get('deterministic')}")
    if kill.get("victim_finished_before_kill") is not False:
        problems.append(f"victim_finished_before_kill={kill.get('victim_finished_before_kill')}")
    if reruns != 1:
        problems.append(f"reruns_logged={reruns}")
    ref_pairs, pert_pairs = rank0_pairs
    if not (ref_pairs == 0 and (pert_pairs or 0) > 0):
        problems.append(f"control placement unproven: rank-0 two-request iterations ref={ref_pairs} pert={pert_pairs}")
    reference = ref.get("healthy") or []
    pre = first_diffs(reference, kill.get("healthy") or [])
    fields = {"pre_kill_first_diff": pre}
    if pre is None or _differing(reference, pre):
        problems.append("pre-kill tokens differ from ref (or failed)")
    want = reference + ([ref["victim_request"]] if ref.get("victim_request") else [])
    after = {}
    for name, run in (("pert", pert), ("kill", kill)):
        rows = (run.get("recovered") or []) + ([run["victim_request"]] if run.get("victim_request") else [])
        after[name] = first_diffs(want, rows)
        d = None if after[name] is None else _differing(want, after[name])
        fields[f"{name}_differ"] = None if d is None else f"{len(d)}/{len(want)}={100 * len(d) // len(want)}%"
        fields[f"{name}_first_diff"] = after[name]
    if len(want) != len(reference) + 1 or after["pert"] is None or after["kill"] is None:
        problems.append("after the kill: missing or failed requests (pert or kill)")
    elif any(k < p for k, p in zip(sorted(after["kill"]), sorted(after["pert"]))):
        problems.append("after the kill: diverges more or earlier than the perturbed control")
    fields["rank0_two_request_iters"] = f"ref={ref_pairs},pert={pert_pairs}"
    return not problems, problems, fields


def rank0_two_request_iters(log_path):
    """Iterations rank 0's print_iter_log shows with exactly 2 of its own requests scheduled (None without a log).

    num_scheduled_requests is per iteration; currank_total_requests is a running count, so it cannot show placement.
    """
    try:
        with open(log_path, errors="replace") as f:
            return len(re.findall(r"\biter = \d+,.*?\bnum_scheduled_requests = 2,", f.read()))
    except OSError:
        return None


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


if __name__ == "__main__":
    # exact DEP REF_DIR PERT_DIR KILL_DIR RERUNS: one EXACT_FINAL_VERDICT line (dirs hold result.json, rank-0.log).
    dep, ref_dir, pert_dir, kill_dir, reruns = sys.argv[2:7]
    pairs = tuple(rank0_two_request_iters(f"{d}/rank-0.log") for d in (ref_dir, pert_dir))
    ok, problems, fields = exact_verdict(*(_load(f"{d}/result.json") for d in (ref_dir, pert_dir, kill_dir)),
                                         int(reruns or 0), pairs)
    print("EXACT_FINAL_VERDICT", dep, "PASS" if ok else "FAIL", *(f"{k}={v}" for k, v in fields.items()),
          *(["problems=" + "; ".join(problems)] if problems else []))

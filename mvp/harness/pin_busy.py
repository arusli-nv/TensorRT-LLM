"""Give every busy thread of a case's ranks its own core (WIDEEP_PIN_BUSY=1, the PIN-AB tail test).

Usage: pin_busy.py PID...  (the case's processes on this node, from serve.sh's tagged_pids)

A rank is a process with more than 150 threads. A thread is busy if it used more than BUSY_FRACTION of a CPU over
SAMPLE_S seconds. Each busy thread gets one core of its rank's socket (the rank's current affinity), from the top of
the socket down, never shared with another thread. Every other thread of the case's processes may run on the node's
remaining cores. One PIN line per busy thread and one PIN_SUMMARY line; the thread snapshot taken later shows
whether the affinities held.
"""

import os
import sys
import time

SAMPLE_S = 2.0
BUSY_FRACTION = 0.2
# Serving under load runs three busy threads per rank (main loop, executor, EPLB worker); sampling repeats until
# every rank shows them (the load may not have started yet), at most ATTEMPTS times.
BUSY_PER_RANK = 3
ATTEMPTS = 10
TICKS = os.sysconf("SC_CLK_TCK")


def threads(pid):
    try:
        return [int(t) for t in os.listdir(f"/proc/{pid}/task")]
    except FileNotFoundError:
        return []


def cpu_ticks(pid, tid):
    try:
        with open(f"/proc/{pid}/task/{tid}/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        return int(fields[11]) + int(fields[12])  # utime + stime
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return None


def set_affinity(tid, cpus):
    try:
        os.sched_setaffinity(tid, cpus)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def sample(ranks):
    before = {(p, t): cpu_ticks(p, t) for p in ranks for t in threads(p)}
    time.sleep(SAMPLE_S)
    busy = {}
    for (p, t), start in before.items():
        end = cpu_ticks(p, t)
        if start is not None and end is not None and (end - start) / TICKS / SAMPLE_S > BUSY_FRACTION:
            busy.setdefault(p, []).append((t, (end - start) / TICKS / SAMPLE_S))
    return busy


def main():
    pids = [int(p) for p in sys.argv[1:]]
    ranks = [p for p in pids if len(threads(p)) > 150]
    for attempt in range(ATTEMPTS):
        busy = sample(ranks)
        if all(len(busy.get(p, [])) >= BUSY_PER_RANK for p in ranks):
            break
        time.sleep(5)
    print(f"PIN_SAMPLE attempts={attempt + 1} busy_per_rank={[len(busy.get(p, [])) for p in ranks]}", flush=True)
    node = os.sched_getaffinity(0)
    taken = set()
    for p in ranks:
        try:
            socket = sorted(os.sched_getaffinity(p), reverse=True)
        except ProcessLookupError:
            continue
        for t, load in busy.get(p, []):
            core = next((c for c in socket if c not in taken), None)
            if core is None:
                print(f"PIN_FAILED pid={p} tid={t}: no free core on its socket", flush=True)
                sys.exit(1)
            taken.add(core)
            ok = set_affinity(t, {core})
            print(f"PIN pid={p} tid={t} load={load:.2f} core={core} ok={int(ok)}", flush=True)
    pool = node - taken
    others = 0
    for p in pids:
        mine = {t for t, _ in busy.get(p, [])}
        for t in threads(p):
            if t not in mine:
                others += set_affinity(t, pool)
    print(f"PIN_SUMMARY ranks={len(ranks)} busy={len(taken)} dedicated={sorted(taken)} pool={len(pool)} "
          f"others_confined={others}", flush=True)


if __name__ == "__main__":
    main()

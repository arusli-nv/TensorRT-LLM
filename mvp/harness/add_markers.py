"""Insert benchmark-only timing markers into the installed peer_loss_recovery.py.

Each marker logs `WIDEEP_PHASE <name> <wall time>` right after an anchor statement; every anchor must occur exactly
once, so a code change that moves or renames one fails loudly instead of measuring the wrong thing.
Usage: python3 add_markers.py PATH_TO_peer_loss_recovery.py
"""

import sys

MARKERS = [  # (anchor statement, marker name)
    ("self._peer_failed.set()", "watcher_detect"),
    ("executor.dist.revoke()", "watcher_revoked"),
    ("rank = dist.rank", "recover_start"),
    # Since the survivor watch also revokes (fix-review), the recovery's revoke is the one after self._watch.stop().
    (("self._watch.stop()\ndist.revoke()", "dist.revoke()"), "revoked"),
    (("failed = dist.shrink()", "failed = self._shrink()"), "shrink_done"),
    ("self._join_device_work(deadline)", "gpu_drained"),
    # The MVP discards one step; wideep-ft-next (overlap scheduler) discards every uncommitted step.
    (("executor.discard_interrupted_step(self._uncommitted_step)",
      "executor.discard_interrupted_steps(reversed(self._uncommitted_steps))"), "kv_rollback_done"),
    # wideep-ft-node reports the failed set.
    (("client.send(Stopped(rank=rank, failed_rank=failed_rank))",
      "client.send(Stopped(rank=rank, failed_ranks=failed_ranks))",
      "client.send(Stopped(rank=rank, failed_ranks=failed_ranks, placement=placement))",
      'self._log_phase("stopped", self._generation + 1)'), "stopped_sent"),
    # Since the re-plan on a further loss (wideep-ft-all), the waits take an interrupt.
    (("proposal = client.wait(Propose, deadline)",
      "proposal = client.wait(Propose, deadline, interrupt=further_loss, skip=skip)",
      "proposal = self._wait_for_proposal(deadline, further_loss, skip)"), "propose_recv"),
    (("client.send(Ready(rank=rank, generation=generation, placement_digest=placement_digest))",
      "client.send(ready)"), "ready_sent"),
    (("_expect_generation(client.wait(Install, deadline), generation)",
      "_expect_generation(install, generation)"), "install_recv"),
    # The stack on main installs the placement in a multi-line call right after the survivor communicators;
    # its marker follows those (the placement takes microseconds).
    (("engine.moe_execution_state.install_resident_placement(engine.moe_load_balancer, failed)",
      "dist.install_survivors()"), "install_applied"),
    (("client.send(Installed(rank=rank, generation=generation))",
      "client.send(Installed(rank=rank, generation=generation, failed_requests=failed_requests))"), "installed_sent"),
    ("_expect_generation(client.wait(Commit, deadline), generation)", "commit_recv"),
    ("self._readmit_undelivered_requests(failed, held)", "requests_settled"),
    ("self._peer_failed.clear()", "resumed"),
]
# Self-contained: the module it is inserted into need not import os, time or logger.
HELPER = '''

import os as _mark_os
import time as _mark_time

from tensorrt_llm.logger import logger as _mark_logger

_MARKS = []
_BUFFER_MARKS = _mark_os.environ.get("TLLM_TEST_BUFFER_MARKS") == "1"


def _mark(name: str) -> None:
    """Benchmark only: a wall-clock timestamp for one recovery step.

    With TLLM_TEST_BUFFER_MARKS=1 the timestamps stay in memory until the rank resumes serving, so writing the
    rank log cannot delay the recovery being measured.
    """
    if not _BUFFER_MARKS:
        _mark_logger.warning(f"WIDEEP_PHASE {name} {_mark_time.time():.6f}")
        return
    _MARKS.append((name, _mark_time.time()))
    if name == "resumed":
        for mark, wall in _MARKS:
            _mark_logger.warning(f"WIDEEP_PHASE {mark} {wall:.6f}")
        _MARKS.clear()
'''


def main(path: str) -> None:
    text = open(path).read()
    if "WIDEEP_PHASE" in text:
        return  # Already patched by an earlier case of this job.
    lines = text.split("\n")
    for anchors, name in MARKERS:
        # An anchor is one statement or a tuple of alternatives in order of preference; the first
        # alternative present must occur exactly once.
        anchors = (anchors,) if isinstance(anchors, str) else anchors
        for anchor in anchors:
            # "prev\nstatement": the statement right after the line prev (for a statement that repeats).
            *before, last = anchor.split("\n")
            hits = [i for i, line in enumerate(lines) if line.strip() == last
                    and all(lines[i - len(before) + k].strip() == p for k, p in enumerate(before))]
            if hits:
                break
        if len(hits) != 1:
            sys.exit(f"anchor {anchors!r} occurs {len(hits)} times")
        i = hits[0]
        indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
        lines.insert(i + 1, f'{indent}_mark("{name}")')
    text = "\n".join(lines)
    first_def = text.index("\ndef ")
    open(path, "w").write(text[:first_def] + HELPER + text[first_def:])


if __name__ == "__main__":
    main(sys.argv[1])

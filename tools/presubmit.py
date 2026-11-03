#!/usr/bin/env python3
"""Presubmit for every Slurm job (RETRO's presubmit lessons and mvp/PREFLIGHT.md, as code).

Checks the job spec before it is submitted. If they pass, it writes a token, logs/presubmit/<token>.pass, and
prints the token. The job's command must start with `bash /art/tools/require_presubmit.sh <token> &&`, so a
spec that skipped presubmit fails at once instead of wasting a slot.

Usage:
  presubmit.py --name NAME --nodes N [--segment S] [--partition batch|cpu] [--job CASE] [--build MVP_BUILD]
               [--overlay MVP_OVERLAY (a directory or a .list file)] [--branch WORKTREE] [--checkout DIR]
               [--build-job]
Exit status 1 and a FAIL line for each failed check.
"""

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

C = Path(__file__).resolve().parent.parent
FAILS = []


def fail(message: str) -> None:
    FAILS.append(message)
    print(f"FAIL {message}")


def ok(message: str) -> None:
    print(f"ok   {message}")


def git(worktree: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(worktree), *args], capture_output=True, text=True, check=True).stdout


def wheel_commit(build: str) -> str:
    """The commit a wheel was built from: build-<label>-<sha> (or a COMMIT file in the build dir)."""
    commit_file = C / build / "COMMIT"
    if commit_file.exists():
        return commit_file.read_text().strip()
    match = re.search(r"-([0-9a-f]{10,40})$", build)
    return match.group(1) if match else ""


def check_job_case(case: str) -> None:
    job_sh = (C / "mvp/harness/job.sh").read_text()
    # job.sh runs every block whose label matches, so a reused label runs two cases (BORROW-KILL, pack1 8054444).
    labels = re.findall(rf"^(?:[\w-]+\|)*{re.escape(case)}(?:\|[\w-]+)*\)", job_sh, re.M)
    if len(labels) > 1:
        fail(f"job case {case} is defined {len(labels)} times in job.sh")
    # A case label, alone or in an alternation such as "A|B)".
    if re.search(rf"^(?:[\w-]+\|)*{re.escape(case)}(?:\|[\w-]+)*\)", job_sh, re.M):
        ok(f"job case {case} exists in job.sh")
    else:
        fail(f"job case {case} is not in mvp/harness/job.sh")
        return
    label = re.search(rf"^(?:[\w-]+\|)*{re.escape(case)}(?:\|[\w-]+)*\)", job_sh, re.M)
    block = job_sh[label.start() :][:6000].split("\n    ;;")[0]
    if "--mode accuracy" in block:
        bench = (C / "mvp/harness/bench.py").read_text()
        if re.search(r"lm_eval\[api\]==\d", bench):
            ok("accuracy case: bench.py installs the pinned lm_eval")
        else:
            fail("accuracy case, but bench.py does not install a pinned lm_eval")


def _same_code(branch: Path, base: str, head: str, path: str) -> bool:
    """Whether ``path`` differs between ``base`` and ``head`` only in comments and whitespace (C/C++), or for CMake
    files also in line order (source lists)."""
    import re

    def strip(text: str) -> str:
        if path.endswith((".txt", ".cmake")):
            return "".join(sorted("".join(line.split()) for line in re.sub(r"#[^\n]*", "", text).splitlines()))
        else:
            text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
            text = re.sub(r"//[^\n]*", "", text)
        return "".join(text.split())

    try:
        return strip(git(branch, "show", f"{base}:{path}")) == strip(git(branch, "show", f"{head}:{path}"))
    except subprocess.CalledProcessError:
        return False


def check_overlay(build: str, overlay: str, branch: Path, checkout: Path | None) -> None:
    base = wheel_commit(build)
    if not base:
        fail(f"cannot tell the commit of {build}: name it build-<label>-<sha> or add a COMMIT file")
        return
    head = git(branch, "rev-parse", "HEAD").strip()
    changed = [p for p in git(branch, "diff", "--name-only", base, head).split() if p]
    cpp = [p for p in changed if p.startswith("cpp/") and not p.startswith("cpp/tests/")]
    code = [p for p in cpp if not _same_code(branch, base, head, p)]
    if code:
        fail(f"{len(code)} C++ files changed after the wheel's commit {base[:10]} (e.g. {code[0]}): build a wheel")
    elif cpp:
        ok(f"{len(cpp)} C++ files differ from the wheel's commit only in comments/whitespace (CMake: line order)")
    launcher = "tensorrt_llm/llmapi/trtllm-llmapi-launch"
    expected = sorted(p for p in changed if p.startswith("tensorrt_llm/") and (p.endswith(".py") or p == launcher))
    path = Path(overlay if overlay.startswith("/") else C / overlay)
    path = Path(str(path).replace("/art/", f"{C}/", 1)) if str(path).startswith("/art/") else path
    if path.suffix == ".list":
        listed = sorted(line.strip() for line in path.read_text().splitlines() if line.strip())
        source = checkout
    else:
        listed = sorted(str(p.relative_to(path)) for p in path.rglob("*") if p.suffix == ".py" or p.name == "trtllm-llmapi-launch")
        source = path
    if listed != expected:
        fail(
            f"overlay {overlay} does not cover exactly the Python files changed in {base[:10]}..{head[:10]}: "
            f"missing {sorted(set(expected) - set(listed))}, extra {sorted(set(listed) - set(expected))}"
        )
    else:
        ok(f"overlay covers the {len(expected)} Python files changed after the wheel's commit {base[:10]}")
    if source is not None:
        stale = [p for p in listed if (source / p).exists() and (source / p).read_text() != git(branch, "show", f"{head}:{p}")]
        if stale:
            fail(f"overlay files differ from {head[:10]}: {stale}")
        else:
            ok(f"overlay contents match {head[:10]}")
        recovery = source / "tensorrt_llm/_torch/pyexecutor/peer_loss_recovery.py"
        if recovery.exists():
            check_markers(recovery)


def module_level_undefined(source: str) -> list[str]:
    """Names that module-level code (run at import, outside functions and classes) uses before any top-level
    statement binds them."""
    import ast
    import builtins

    bound = set(dir(builtins)) | {"__name__", "__file__", "__doc__"}
    missing = []
    for node in ast.parse(source).body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for name in ast.walk(node):
                if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Load) and name.id not in bound:
                    missing.append(name.id)
        for name in ast.walk(node):
            if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store):
                bound.add(name.id)
            elif isinstance(name, ast.alias):
                bound.add((name.asname or name.name).split(".")[0])
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
    return sorted(set(missing))


def check_markers(recovery: Path) -> None:
    scratch = Path(tempfile.mkdtemp(prefix="presubmit-markers-"))
    copy = scratch / "peer_loss_recovery.py"
    shutil.copy(recovery, copy)
    result = subprocess.run([sys.executable, str(C / "mvp/harness/add_markers.py"), str(copy)], capture_output=True, text=True)
    if result.returncode:
        fail(f"add_markers.py does not apply to {recovery}: {(result.stdout + result.stderr).strip()[-300:]}")
        return
    # The marked module must still compile and have no undefined module-level names (C-LAYOUT 8048228 died on
    # `os` after the module stopped importing it).
    check = subprocess.run([sys.executable, "-m", "pyflakes", str(copy)], capture_output=True, text=True)
    undefined = [line for line in check.stdout.splitlines() if "undefined name" in line]
    if check.returncode and "No module named pyflakes" in check.stderr:
        undefined = module_level_undefined(copy.read_text())
        if undefined:
            fail(f"marked peer_loss_recovery.py uses undefined names at import time: {undefined}")
        else:
            ok("add_markers.py anchors apply; no undefined name runs at import time")
    elif undefined:
        fail(f"marked peer_loss_recovery.py has undefined names: {undefined[:3]}")
    else:
        ok("add_markers.py anchors apply and the marked module has no undefined names")


def check_signatures(build: str, branch: Path) -> None:
    """Functions whose signature changed after the wheel's commit, and the tests that still call the old form."""
    base = wheel_commit(build)
    if not base:
        return
    diff = git(branch, "diff", "-U0", base, "HEAD", "--", "tensorrt_llm")
    removed = {m.group(1) for m in re.finditer(r"^-\s*def (\w+)\(", diff, re.M)}
    added = {m.group(1) for m in re.finditer(r"^\+\s*def (\w+)\(", diff, re.M)}
    # Distinctive names only (generic ones such as `wait` match unrelated code).
    gone = sorted(name for name in removed - added if "_" in name.strip("_"))
    if not gone:
        ok("no function removed or renamed after the wheel's commit")
        return
    tests = subprocess.run(
        ["git", "-C", str(branch), "grep", "-l", "-w", "-E", "|".join(gone), "--", "tests/unittest", "tests/integration"],
        capture_output=True, text=True,
    ).stdout.split()
    if tests:
        fail(f"tests still name functions removed after {base[:10]} ({gone}): {tests}")
    else:
        ok(f"no test names the removed functions {gone}")
    changed = sorted(removed & added)
    if changed:
        print(f"note signatures changed (check their fakes in tests/): {changed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--segment", type=int)
    parser.add_argument("--partition", default="batch")
    parser.add_argument("--job")
    parser.add_argument("--build")
    parser.add_argument("--overlay")
    parser.add_argument("--branch", default=str(C / "dev"))
    parser.add_argument("--checkout", help="the tree a .list overlay is copied from")
    parser.add_argument("--build-job", action="store_true", help="the job compiles a wheel")
    args = parser.parse_args()
    branch = Path(args.branch)

    # Packed jobs: 8-node groups, one segment each (bundle.sh, 8-aligned). 32 nodes only with the user's OK (2026-10-19).
    packed = args.segment == 8 and args.nodes in (16, 32)
    if packed:
        ok(f"packed {args.nodes}-node job: {args.nodes // 8} 8-node groups, one segment each (bundle.sh, 8-aligned groups)")
    elif args.nodes > 1 and args.segment != args.nodes:
        fail(f"{args.nodes}-node job without segment={args.nodes} (Slurm spreads it across NVLink racks)")
    elif args.nodes > 1:
        ok(f"segment={args.segment}")
    if args.build_job and args.partition != "cpu" and args.nodes != 1:
        fail("a build in a GPU slot (one per milestone, user decision 2026-10-18) uses one node")
    if args.job:
        check_job_case(args.job)
    if args.build and not args.build_job:
        wheels = list((C / args.build / "dist").glob("*.whl"))
        if wheels:
            ok(f"{args.build}/dist has {wheels[0].name}")
        else:
            fail(f"{args.build}/dist has no wheel")
        if args.overlay:
            check_overlay(args.build, args.overlay, branch, Path(args.checkout) if args.checkout else None)
        check_signatures(args.build, branch)

    if FAILS:
        print(f"PRESUBMIT_FAILED {len(FAILS)}")
        sys.exit(1)
    spec = " ".join(sys.argv[1:])
    token = f"{args.name}-{hashlib.sha256(spec.encode()).hexdigest()[:10]}"
    out = C / "logs/presubmit"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{token}.pass").write_text(spec + "\n")
    print(f"PRESUBMIT_OK token={token}")


if __name__ == "__main__":
    main()

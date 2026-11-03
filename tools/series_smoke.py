#!/usr/bin/env python3
"""Check that every commit of a series stands alone (CPU, no torch): per commit, for each .py file it changes,
  - ruff F821/F822/F823 (undefined names) on the file as it is at that commit, and
  - every absolute or relative `from tensorrt_llm... import X` resolves at that commit: the module file exists and X is
    defined at its top level (def, class, assignment, import) or is a submodule.

Usage: series_smoke.py REPO BASE TIP [RUFF]
Prints one line per problem and SERIES_SMOKE PASS/FAIL.
"""

import ast
import pathlib
import subprocess
import sys
import tempfile


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=True).stdout


def show(repo, commit, path):
    r = subprocess.run(["git", "-C", repo, "show", f"{commit}:{path}"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def module_paths(module):
    base = module.replace(".", "/")
    return [f"{base}.py", f"{base}/__init__.py"]


def top_level_names(src):
    names = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for node in tree.body:
        targets = []
        if isinstance(node, ast.FunctionDef) and node.name == "__getattr__":
            return None  # lazily exported names: not checkable statically
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                if a.name == "*":
                    return None  # re-exports a module's names: not checkable statically
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, (ast.If, ast.Try)):
            # Conditional definitions (e.g. optional imports) count as defined.
            for sub in ast.walk(node):
                if isinstance(sub, (ast.FunctionDef, ast.ClassDef)):
                    names.add(sub.name)
                elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                    for a in sub.names:
                        names.add((a.asname or a.name).split(".")[0])
                elif isinstance(sub, ast.Assign):
                    targets += sub.targets
        for t in targets:
            for n in ast.walk(t):
                if isinstance(n, ast.Name):
                    names.add(n.id)
    return names


def resolve(path, node):
    if node.level == 0:
        return node.module
    pkg = pathlib.PurePosixPath(path).parent.parts
    pkg = pkg[: len(pkg) - (node.level - 1)]
    return ".".join(list(pkg) + ([node.module] if node.module else []))


def check_imports(repo, commit, path, src, cache):
    problems = []
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"{path}: syntax error {e}"]
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = resolve(path, node)
        if not module or not module.startswith("tensorrt_llm") or module.startswith("tensorrt_llm.bindings"):
            continue  # compiled extension modules are not in git
        key = (commit, module)
        if key not in cache:
            src_m = next((s for p in module_paths(module) if (s := show(repo, commit, p)) is not None), None)
            cache[key] = "missing" if src_m is None else top_level_names(src_m)
        names = cache[key]
        if names is None:
            continue
        if names == "missing":
            if show(repo, commit, module.replace(".", "/") + "/") is None and not any(
                show(repo, commit, p) is not None for p in module_paths(module)
            ):
                problems.append(f"{path}:{node.lineno}: module {module} missing")
            continue
        for a in node.names:
            if a.name == "*" or a.name in names:
                continue
            if any(show(repo, commit, p) is not None for p in module_paths(f"{module}.{a.name}")):
                continue
            problems.append(f"{path}:{node.lineno}: {module} has no {a.name}")
    return problems


def ruff_f82(ruff, files_src):
    """Undefined-name findings (file, message) for {path: source}."""
    with tempfile.TemporaryDirectory() as tmp:
        for f, src in files_src.items():
            dst = pathlib.Path(tmp, f)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(src)
        r = subprocess.run([ruff, "check", "--no-cache", "--isolated", "--select", "F821,F822,F823",
                            "--output-format", "concise", tmp], capture_output=True, text=True)
    out = set()
    for line in r.stdout.splitlines():
        if "F82" in line and ":" in line:
            path, rest = line.replace(tmp + "/", "").split(":", 1)
            out.add((path, rest.split(" ", 1)[1] if " " in rest else rest, line.replace(tmp + "/", "")))
    return out


def main():
    repo, base, tip = sys.argv[1:4]
    ruff = sys.argv[4] if len(sys.argv) > 4 else "ruff"
    commits = git(repo, "rev-list", "--reverse", f"{base}..{tip}").split()
    cache, bad = {}, 0
    for c in commits:
        title = git(repo, "log", "-1", "--format=%h %s", c).strip()
        files = [f for f in git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", "--diff-filter=AM", c).split()
                 if f.endswith(".py")]
        problems, srcs, base_srcs = [], {}, {}
        for f in files:
            src = show(repo, c, f)
            if src is None:
                continue
            srcs[f] = src
            b = show(repo, base, f)
            if b is not None:
                base_srcs[f] = b
            if f.startswith(("tensorrt_llm/", "tests/")):
                # Unresolvable imports the file already has on BASE (generated or lazily exported names) are not new.
                known = {m.split(": ", 1)[1] for m in check_imports(repo, base, f, b, cache)} if b is not None else set()
                problems += [m for m in check_imports(repo, c, f, src, cache) if m.split(": ", 1)[1] not in known]
        if srcs:
            known = {(p, m.split(" ", 1)[1]) for p, m, _ in ruff_f82(ruff, base_srcs)}
            problems += [line for p, m, line in ruff_f82(ruff, srcs) if (p, m.split(" ", 1)[1]) not in known]
        for p in problems:
            print(f"{title}: {p}")
        bad += len(problems)
    print(f"SERIES_SMOKE {'PASS' if bad == 0 else 'FAIL'} commits={len(commits)} problems={bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

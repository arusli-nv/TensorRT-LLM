import os, sys, types, importlib.util, io, contextlib
spec = importlib.util.spec_from_file_location("th", sys.argv[1]); th = importlib.util.module_from_spec(spec); spec.loader.exec_module(th)
plan = iter([[], [], ["a"], [], []])
class PyExecutor:
    def _balance_adp_requests(self, ctx, gen): return next(plan)
    def _fetch_new_requests(self, queue, active): return queue
m = types.SimpleNamespace(PyExecutor=PyExecutor); th._patch_admit_log(m)
e, out = PyExecutor(), io.StringIO()
with contextlib.redirect_stdout(out):
    assert e._balance_adp_requests(["a"], [1]) == []      # newly held -> log
    assert e._balance_adp_requests(["a"], [1]) == []      # still held -> quiet
    assert e._balance_adp_requests(["a"], [1]) == ["a"]  # released -> log
    assert e._balance_adp_requests([], [1]) == []         # nothing -> quiet
    assert e._balance_adp_requests([], [1]) == []
    assert e._fetch_new_requests([], []) == [] and e._fetch_new_requests(["r1", "r2"], []) == ["r1", "r2"]
lines = [l for l in out.getvalue().splitlines() if "WIDEEP_ADMIT" in l]
fetch_lines = [l for l in out.getvalue().splitlines() if "WIDEEP_FETCH" in l]
assert len(fetch_lines) == 1 and "call=2 new=2" in fetch_lines[0], fetch_lines
assert len(lines) == 2 and "call=1 ctx=0 held=1" in lines[0] and "call=3 ctx=1 held=0" in lines[1], lines
print("PASS", lines)

import os, sys, types, importlib.util, io, contextlib, time
os.environ["TLLM_TEST_PHASE_TIMES"] = "2"; os.environ["OMPI_COMM_WORLD_RANK"] = "0"
spec = importlib.util.spec_from_file_location("th", sys.argv[1]); th = importlib.util.module_from_spec(spec); spec.loader.exec_module(th)
class PyExecutor:
    def _prepare_and_schedule_batch(self): time.sleep(0.001); return "b"
    def _wait_for_model_engine_input_copy(self): return None
class RecoveryHook:
    def commit_step(self, batch, state): return None
class PeerLossRecovery(RecoveryHook):
    def commit_step(self, batch, state): time.sleep(0.0005)
ex = types.SimpleNamespace(PyExecutor=PyExecutor); rec = types.SimpleNamespace(RecoveryHook=RecoveryHook, PeerLossRecovery=PeerLossRecovery)
th._patch_phase_executor(ex); th._patch_phase_recovery(rec)
out = io.StringIO()
with contextlib.redirect_stdout(out):
    e = PyExecutor(); r = PeerLossRecovery()
    for _ in range(4):
        assert e._prepare_and_schedule_batch() == "b"; e._wait_for_model_engine_input_copy(); r.commit_step(1, 2)
names = sorted(set(l.split("name=")[1].split()[0] for l in out.getvalue().splitlines() if "WIDEEP_PHASE" in l))
assert names == ["_prepare_and_schedule_batch", "_wait_for_model_engine_input_copy", "commit_step", "period"], names
print("PASS", names)

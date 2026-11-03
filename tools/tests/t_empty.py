import os, sys, types, importlib.util
os.environ.pop("TLLM_FAULT_TOLERANCE_MODE", None)
os.environ["TLLM_TEST_EMPTY_EXTRA_SLOTS"] = "1"
spec = importlib.util.spec_from_file_location("th", sys.argv[1]); th = importlib.util.module_from_spec(spec)
spec.loader.exec_module(th)
seen = []
class Layer:
    def set_initial_weight_assignments(self, a): seen.append(("assign", os.environ.get("TLLM_FAULT_TOLERANCE_MODE")))
class LB:
    def finalize_model(self): seen.append(("finalize", os.environ.get("TLLM_FAULT_TOLERANCE_MODE"))); Layer().set_initial_weight_assignments([-1])
    def extra_slots_mode(self): return "ghost"
m = types.SimpleNamespace(SingleLayerMoeLoadBalancer=Layer, MoeLoadBalancer=LB, peer_loss_recovery_enabled=None)
th._patch_empty_extra_slots(m)
m.MoeLoadBalancer().finalize_model()
assert seen == [("finalize", "1"), ("assign", "1")], seen
assert "TLLM_FAULT_TOLERANCE_MODE" not in os.environ
assert m.peer_loss_recovery_enabled() is True
print("PASS", seen)

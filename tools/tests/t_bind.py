"""testhooks TLLM_TEST_FT_UNBOUND / _NO_INPLACE_SLOTS / _NO_GHOST / _NO_SLOT_TABLE against a fake TRT-LLM tree imported through the hook.

Run: python3 -I t_bind.py <testhooks.py>  (each mode runs in its own child process: the hooks read env at import)."""
import os, sys, subprocess, tempfile, textwrap, importlib, importlib.util, io, contextlib

FILES = {
    "tensorrt_llm/_torch/moe/fused_moe/wide_ep_ft.py": """
        class EPGroupHealth:
            def get_mask_words(self): return [0xFFFFFFFF]
        def get_wide_ep_ft_options(model_config):
            health = model_config.extra_attrs.setdefault("wide_ep_ft_ep_group_health", EPGroupHealth())
            return health, 300.0, 0.25
        """,
    "tensorrt_llm/_torch/moe/fused_moe/communication/nvlink_one_sided.py": """
        import types
        class NVLinkOneSided:
            def __init__(self, ep_rank, ep_group_health=None, alltoall_watchdog_timeout_s=None, mnnvl_mem=None):
                self.ep_rank, self.ep_group_health = ep_rank, ep_group_health
                self._rank_mask_enabled = ep_group_health is not None
                self.watchdog_timeout_s = alltoall_watchdog_timeout_s
                self._execution_state = None
                self.mnnvl_mem = mnnvl_mem
                self.workspace = types.SimpleNamespace(device="cpu")
                self.moe_a2a_metainfo, self.workspace_size_per_rank = (7,), 1 << 20
                self.dispatched = []
            @staticmethod
            def source_write_ranges(metainfo, source): return [(source, source + metainfo[0])]
            def _reset_local_workspace(self): pass
            def reset_state(self): pass
            def dispatch_source_slots(self): return None if self._execution_state is None else "slots"
            @classmethod
            def bind_layers(cls, state, layers):
                for comm, _ in layers:
                    comm.bind_execution_state(state)
            def bind_execution_state(self, state):
                state.retain(self.mnnvl_mem, self._reset_local_workspace, lambda source: [])
                state.track_layer(self.ep_rank, self.reset_state)
                self._execution_state = state
            def dispatch(self, x, **kwargs):
                state = self._execution_state
                self.dispatched.append((state, self._rank_mask_enabled, kwargs.get("expert_to_slot")))
                return x
        """,
    "tensorrt_llm/_torch/moe/fused_moe/communication/execution_state.py": """
        from .nvlink_one_sided import NVLinkOneSided
        class MoeExecutionState:
            def __init__(self, ep_group_health, device):
                self.health, self.device = ep_group_health, device
                self._workspaces, self._layers, self.progress_comm = {}, [], None
            @classmethod
            def for_model(cls, model):
                comms = {id(m.comm): m.comm for m in model.modules() if isinstance(getattr(m, "comm", None), NVLinkOneSided)}
                comm = next(iter(comms.values()))
                if comm.ep_group_health is None:
                    raise RuntimeError("WideEP recovery requires one shared EPGroupHealth for all MoE layers")
                state = cls(comm.ep_group_health, comm.workspace.device)
                state.progress_comm = comm
                NVLinkOneSided.@BIND@(state, [(c, None) for c in comms.values()])
                return state
            def retain(self, memory, reset_workspace, write_ranges):
                self._workspaces.setdefault(id(memory), (memory, reset_workspace, write_ranges))
            def track_layer(self, ep_rank, reset_state):
                self._layers.append((ep_rank, reset_state))
        """,
    "tensorrt_llm/_torch/moe/fused_moe/communication/communication_factory.py": """
        from ..wide_ep_ft import get_wide_ep_ft_options
        from .nvlink_one_sided import NVLinkOneSided
        class CommunicationFactory:
            @staticmethod
            def create_strategy(model_config, ep_rank, memory):
                health, timeout_s, _ = get_wide_ep_ft_options(model_config)
                return NVLinkOneSided(ep_rank, ep_group_health=health, alltoall_watchdog_timeout_s=timeout_s, mnnvl_mem=memory)
            @staticmethod
            def _create_forced_method(model_config, ep_rank, memory):
                health, timeout_s, _ = get_wide_ep_ft_options(model_config)
                return NVLinkOneSided(ep_rank, ep_group_health=health, alltoall_watchdog_timeout_s=timeout_s, mnnvl_mem=memory)
        """,
    "tensorrt_llm/_torch/moe/fused_moe/moe_scheduler.py": """
        def peer_loss_recovery_enabled(): return True
        class ExternalCommMoEScheduler:
            def __init__(self, moe):
                self.moe = moe
                self._peer_loss_recovery = peer_loss_recovery_enabled()
            def _compact_receive_slots(self, x, x_sf):
                if self.moe.backend.supports_dispatch_receive_slots():
                    return x, x_sf
                return type(x)(x.shape[0]), x_sf  # a copy
            def _build_comm_plan(self):
                return self._peer_loss_recovery and self.moe.backend.supports_dispatch_receive_slots()
            def _dispatch_slot_table(self):
                return self.moe.layer_load_balancer.one_slot_table()
            def _forward_chunk_impl(self, x):
                expert_to_slot = None
                if self._peer_loss_recovery:
                    expert_to_slot = @SLOT_TABLE@
                if expert_to_slot is None:
                    pass  # route()
                return self.moe.comm.dispatch(x, expert_to_slot=expert_to_slot)
        """,
    "tensorrt_llm/_torch/moe/fused_moe/moe_load_balancer.py": """
        class Table:
            def __init__(self, n): self.n = n
            def numel(self): return self.n
        class SingleLayerMoeLoadBalancer:
            def __init__(self, expert_count, slot_count_per_rank, ghost_slot_count):
                self.expert_count, self.slot_count_per_rank, self.ghost_slot_count = expert_count, slot_count_per_rank, ghost_slot_count
                self.ghost_slots, self.storage = None, None
            def one_slot_table(self): return Table(self.expert_count)
            def make_tensor_host_accessible(self, wt):
                if self.ghost_slots is None:
                    self.storage = "host_accessible"
                else:
                    self.storage = ("ghost_memory", list(self.ghost_slots))
        class MoeLoadBalancer:
            def __init__(self, ep_size, hosts):
                self.ep_size, self.copy_nodes, self.single_layer_load_balancers = ep_size, None, []
                self.use_gpu_memcpy, self._ring_copy_nodes = False, "ring"
                self._largest_loss = self._largest_failure_domain(hosts)
            def _largest_failure_domain(self, hosts): return max(hosts.count(h) for h in hosts)
            def set_use_gpu_memcpy(self, use_gpu_memcpy): self.use_gpu_memcpy = use_gpu_memcpy
            def add_layer(self, expert_count, slot_count_per_rank):
                @SURVIVORS@
                ghost_slot_count = max(0, -(-expert_count // survivors) - slot_count_per_rank)
                if slot_count_per_rank * self.ep_size == expert_count:
                    self.copy_nodes = self._ring_copy_nodes
                layer = SingleLayerMoeLoadBalancer(expert_count, slot_count_per_rank + ghost_slot_count, ghost_slot_count)
                self.single_layer_load_balancers.append(layer)
                return layer
            def register_weight_slots_after_to_cuda(self):
                mode = "ghost" if any(l.ghost_slot_count for l in self.single_layer_load_balancers) else "replicas"
                if mode in ("ghost", "replicas+ghost"):
                    self.set_use_gpu_memcpy(True)
                    for layer in self.single_layer_load_balancers:
                        layer.ghost_slots = list(range(layer.slot_count_per_rank - layer.ghost_slot_count, layer.slot_count_per_rank))
                for layer in self.single_layer_load_balancers:
                    layer.make_tensor_host_accessible(None)
            def recovery_placements(self, dead_ranks, base=None):
                if self.single_layer_load_balancers[0].slot_count_per_rank * (self.ep_size - len(set(dead_ranks))) < 256:
                    raise ValueError("uncovered")
            def covers_loss_of(self, dead_ranks):
                try:
                    self.recovery_placements(dead_ranks)
                except ValueError:
                    return False
                return True
        """,
    "tensorrt_llm/_torch/moe/fused_moe/fused_moe_cute_dsl.py": """
        class CuteDslFusedMoE:
            has_nvfp4 = True
            def supports_dispatch_receive_slots(self):
                return self.has_nvfp4
        """,
}
# hook_ok's patterns (job.sh): the lines must match them.
ACTIVE = {"unbound": "WIDEEP_FT_UNBOUND_ACTIVE rank [0-9]* t=[0-9.]* layers=[1-9][0-9]* comms=[1-9][0-9]* bound=0 rank_mask=0 health=0 slot_table=0 ",
          "noinplace": r"WIDEEP_FT_NO_INPLACE_ACTIVE rank [0-9]* t=[0-9.]* layers=\([1-9][0-9]*\) bound=\1 inplace=0 ",
          "noghost": r"WIDEEP_FT_NO_GHOST_ACTIVE rank [0-9]* t=[0-9.]* layers=\([1-9][0-9]*\) bound=\1 num_slots=256 slots_per_rank=8 ghost=0 ghost_storage=\1 copy_nodes=ring largest_loss=0 ",
          "noslottable": r"WIDEEP_FT_NO_SLOT_TABLE_ACTIVE rank [0-9]* t=[0-9.]* layers=\([1-9][0-9]*\) bound=\1 recovery=\1 slot_table=0$"}
CALL = {"unbound": "WIDEEP_FT_UNBOUND_CALL rank [0-9]* t=[0-9.]* .*name=dispatch state=None rank_mask=0 expert_to_slot=0",
        "noinplace": "WIDEEP_FT_NO_INPLACE_CALL rank [0-9]* t=[0-9.]* .*compacted=1 ",
        "noghost": "WIDEEP_FT_NO_GHOST_CALL rank [0-9]* t=[0-9.]* .*name=dispatch state=bound num_slots=256 expert_to_slot=1 table=256",
        "noslottable": "WIDEEP_FT_NO_SLOT_TABLE_CALL rank [0-9]* t=[0-9.]* .*name=dispatch state=bound rank_mask=1 expert_to_slot=0"}
ENV = {"unbound": "TLLM_TEST_FT_UNBOUND", "refused": "TLLM_TEST_FT_UNBOUND", "noinplace": "TLLM_TEST_FT_NO_INPLACE_SLOTS",
       "noghost": "TLLM_TEST_FT_NO_GHOST", "noghost_refused": "TLLM_TEST_FT_NO_GHOST",
       "noslottable": "TLLM_TEST_FT_NO_SLOT_TABLE", "noslottable_refused": "TLLM_TEST_FT_NO_SLOT_TABLE"}
# The fake tree's patched code, and how each refused mode moves it.
TREE = {"@BIND@": "bind_layers", "@SURVIVORS@": "survivors = self.ep_size - self._largest_loss",
        "@SLOT_TABLE@": "self._dispatch_slot_table()"}
MOVED = {"refused": {"@BIND@": "bind_moved"}, "noghost_refused": {"@SURVIVORS@": "survivors = (self.ep_size - self._largest_loss)"},
         "noslottable_refused": {"@SLOT_TABLE@": "self._dispatch_slot_table(*())"}}


def write_tree(moved):
    root = tempfile.mkdtemp(prefix="t_bind.", dir=os.environ.get("TMPDIR"))
    for path, body in FILES.items():
        parts = path.split("/")
        for i in range(1, len(parts)):
            init = os.path.join(root, *parts[:i], "__init__.py")
            os.makedirs(os.path.dirname(init), exist_ok=True)
            open(init, "a").close()
        with open(os.path.join(root, path), "w") as f:
            body = textwrap.dedent(body)
            for key, value in {**TREE, **moved}.items():
                body = body.replace(key, value)
            f.write(body)
    return root


def grep_ok(pattern, text):
    return subprocess.run(["grep", "-q", pattern], input=text, text=True).returncode == 0


def child(hooks, mode):
    os.environ["OMPI_COMM_WORLD_RANK"] = "3"
    os.environ[ENV[mode]] = "1"
    sys.path.insert(0, write_tree(MOVED.get(mode, {})))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        spec = importlib.util.spec_from_file_location("th", hooks); th = importlib.util.module_from_spec(spec); spec.loader.exec_module(th)
        P = "tensorrt_llm._torch.moe.fused_moe."
        wft = importlib.import_module(P + "wide_ep_ft")
        fac = importlib.import_module(P + "communication.communication_factory")
        nvl = importlib.import_module(P + "communication.nvlink_one_sided")
        es = importlib.import_module(P + "communication.execution_state")
        sch = importlib.import_module(P + "moe_scheduler")
        cute = importlib.import_module(P + "fused_moe_cute_dsl")
        lb = importlib.import_module(P + "moe_load_balancer")
        # 32 ranks on 8 hosts: a node failure domain loses 4, so 256 experts need ceil(256/28) - 8 = 2 ghost slots.
        balancer = lb.MoeLoadBalancer(32, [f"h{r // 4}" for r in range(32)])
        cfg = type("Cfg", (), {"extra_attrs": {}})()
        memory = object()  # one workspace shared by both layers
        comms = [fac.CommunicationFactory.create_strategy(cfg, 3, memory), fac.CommunicationFactory._create_forced_method(cfg, 3, memory)]
        health = cfg.extra_attrs["wide_ep_ft_ep_group_health"]
        moes = []
        for comm in comms:
            moe = type("Moe", (), {})(); moe.comm, moe.backend = comm, cute.CuteDslFusedMoE()
            moe.layer_load_balancer = balancer.add_layer(256, 8)
            moe.num_slots = comm.num_experts = 32 * moe.layer_load_balancer.slot_count_per_rank
            moe.scheduler = sch.ExternalCommMoEScheduler(moe); moes.append(moe)
        balancer.register_weight_slots_after_to_cuda()
        covered = balancer.covers_loss_of([0, 1, 2, 3])  # the executor's startup check
        model = type("Model", (), {"modules": lambda self: [object(), *moes]})()
        if mode == "refused":
            try:
                es.MoeExecutionState.for_model(model)
                raise AssertionError("for_model must stay the original")
            except RuntimeError as error:  # the original, on comms built without health: startup fails loudly
                assert "EPGroupHealth" in str(error), error
        else:
            state = es.MoeExecutionState.for_model(model)
            original_dispatch = nvl.NVLinkOneSided.__dict__["dispatch"]
            for _ in range(2):  # as the scheduler dispatches (the new hooks' modes), or directly
                if mode in ("unbound", "noinplace"):
                    comms[0].dispatch("x", expert_to_slot=None)
                else:
                    moes[0].scheduler._forward_chunk_impl("x")
            x = type("T", (), {"__init__": lambda t, n: setattr(t, "shape", (n, 4))})(2)
            for _ in range(2):
                moes[0].scheduler._compact_receive_slots(x, None)
    text = out.getvalue()
    print(text, end="")
    layers = balancer.single_layer_load_balancers
    if mode == "noghost":
        assert covered and balancer._largest_loss == 0 and balancer.copy_nodes == "ring" and balancer.use_gpu_memcpy
        assert all(l.ghost_slot_count == 0 and l.slot_count_per_rank == 8 and l.storage == ("ghost_memory", []) for l in layers)
        assert all(m.num_slots == 256 for m in moes) and all(m.scheduler._peer_loss_recovery for m in moes)
    else:  # the ghost layout: 10 slots per rank, 2 of them ghost; 28 survivors hold 280 slots
        assert covered and balancer._largest_loss == 4 and balancer.copy_nodes == "ring" and balancer.use_gpu_memcpy
        assert all(l.ghost_slot_count == 2 and l.slot_count_per_rank == 10 and l.storage == ("ghost_memory", [8, 9]) for l in layers)
    if mode.endswith("_refused"):
        event = "WIDEEP_FT_NO_GHOST" if mode == "noghost_refused" else "WIDEEP_FT_NO_SLOT_TABLE"
        moved = "add_layer.ghost_slot_count" if mode == "noghost_refused" else "_forward_chunk_impl.slot_table"
        assert f"{event}_REFUSED rank 3 t=" in text and f"moved={moved}" in text, text
        assert f"{event} rank 3" not in text, text  # no applied line
        assert all(d[2] is not None for d in comms[0].dispatched)  # the table, as without the hook
        return
    if mode == "refused":
        assert "WIDEEP_FT_UNBOUND_REFUSED rank 3 t=" in text and "module=execution_state moved=for_model.bind_layers" in text, text
        assert "applied module=execution_state" not in text and "WIDEEP_FT_UNBOUND_ACTIVE" not in text, text
        return
    assert grep_ok(ACTIVE[mode], text), (ACTIVE[mode], text)
    assert grep_ok(CALL[mode], text), (CALL[mode], text)
    assert "REFUSED" not in text, text
    if mode == "unbound":
        assert all(c.ep_group_health is None and not c._rank_mask_enabled and c.watchdog_timeout_s is None for c in comms)
        assert all(c._execution_state is None for c in comms) and state.health is health and state.progress_comm is comms[0]
        assert len(state._workspaces) == 1 and [r for r, _ in state._layers] == [3, 3]
        assert state._workspaces[id(memory)][2](2) == [(2, 9)]  # the comm's source write ranges, for the staging check
        assert all(m.scheduler._peer_loss_recovery is False for m in moes)
        assert [d[0] for d in comms[0].dispatched] == [None, None]
        assert nvl.NVLinkOneSided.__dict__["dispatch"] is not original_dispatch or text.count("WIDEEP_FT_UNBOUND_CALL") == 1
        assert text.count("WIDEEP_FT_UNBOUND_CALL") == 1 and text.count("WIDEEP_FT_UNBOUND rank 3") == 3, text
        assert "layers=2 comms=2 bound=0 rank_mask=0 health=0 slot_table=0 workspaces=1 progress_comm=1" in text, text
        assert moes[0].backend.supports_dispatch_receive_slots() is True  # hook 2 is not on
    elif mode == "noinplace":
        assert all(c._execution_state is state for c in comms) and all(c._rank_mask_enabled for c in comms)
        assert all(m.scheduler._peer_loss_recovery is True for m in moes)
        assert not moes[0].backend.supports_dispatch_receive_slots() and moes[0].scheduler._build_comm_plan() is False
        assert "layers=2 bound=2 inplace=0 backends=CuteDslFusedMoE" in text, text
        assert "compacted=1 rows_in=2 rows_out=2" in text and text.count("WIDEEP_FT_NO_INPLACE_CALL") == 1, text
        assert text.count("WIDEEP_FT_NO_INPLACE rank 3") == 2, text
        assert "WIDEEP_FT_UNBOUND" not in text
    if mode in ("noghost", "noslottable"):
        assert "WIDEEP_FT_UNBOUND" not in text and "WIDEEP_FT_NO_INPLACE" not in text, text
        assert all(d[0] is state for d in comms[0].dispatched) and all(c._rank_mask_enabled for c in comms)
        tables = [d[2] for d in comms[0].dispatched]
        assert (all(t is not None and t.numel() == 256 for t in tables) if mode == "noghost" else tables == [None, None])
    if mode == "noghost":
        assert "layers=2 bound=2 num_slots=256 slots_per_rank=8 ghost=0 ghost_storage=2 copy_nodes=ring largest_loss=0 real_largest_loss=4" in text, text
        assert text.count("WIDEEP_FT_NO_GHOST_CALL") == 1 and text.count("WIDEEP_FT_NO_GHOST rank 3") == 1, text
    if mode == "noslottable":
        assert "layers=2 bound=2 recovery=2 slot_table=0" in text and moes[0].backend.supports_dispatch_receive_slots(), text
        assert text.count("WIDEEP_FT_NO_SLOT_TABLE_CALL") == 1 and text.count("WIDEEP_FT_NO_SLOT_TABLE rank 3") == 1, text


if len(sys.argv) > 2:
    child(sys.argv[1], sys.argv[2])
else:
    for mode in ("unbound", "noinplace", "refused", "noghost", "noslottable", "noghost_refused", "noslottable_refused"):
        run = subprocess.run([sys.executable, "-I", __file__, sys.argv[1], mode], capture_output=True, text=True)
        assert run.returncode == 0, (mode, run.stdout, run.stderr)
        print("PASS", mode, "|", " / ".join(l.split(" t=")[0] + " " + l.split(" ", 4)[-1] for l in run.stdout.splitlines()
                                             if "ACTIVE" in l or "REFUSED" in l))

# Fix candidate "no peer prefault" (branch fix-nopopulate, /tmp/claude-161238/wt-fix on v5-final 6688790d56): FT-off path

Change (tensorrt_llm/_torch/moe/fused_moe/moe_load_balancer.py, -35 lines): removes `_start_populate` (a background
madvise(MADV_POPULATE_READ) over every peer's /dev/shm host-copy segment), `HostTensorSharer.populate_thread`, the call in
`HostTensorSharer.finalize_host_tensor_sharing` (`if peer_loss_recovery_enabled(): self.populate_thread = _start_populate(...)`),
the join loop in `MoeLoadBalancer.finalize_model`, the `_MADV_POPULATE_READ` constant and the then-unused `import ctypes`.

FT off: `peer_loss_recovery_enabled()` is False, so `populate_thread` stayed None and the join loop joined nothing; no other
code reads `populate_thread` or `_start_populate` (grep over tensorrt_llm/: only these sites; tests: test_ring_host_copies.py,
updated). The FT-off execution path is therefore unchanged statement for statement; the only difference is one fewer
module import (ctypes, a stdlib module already loaded by the process). The FT-off arms need no rerun.

FT on: peers' host copies are mapped but not prefaulted; a restore takes page faults (64 KiB pages on these nodes) on the
pages it copies. Restore-time cost measured by POPULATE-AB's rank5 rows on the diagnostic overlay diag-nopopulate-0eb6e82ac0,
which removes only the call (runtime-equivalent to this change).

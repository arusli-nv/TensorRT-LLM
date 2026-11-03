# c33: land "the launcher selects and verifies the FT transport" (design + red team)

Code read: /tmp/claude-161238/wt-v5 @ c64945a524 (includes ac9ea4e776). No worktree edited. Data cited: CLAIMS 158, job 8145956.

## 1. Where TRT-LLM can select the btl

The premise "the launcher starts after mpirun chose the components" holds only for the PRRTE side (`--with-ft ulfm`,
errmgr, oob). Open MPI picks pml/btl/smsc inside each process's `MPI_Init`, which mpi4py runs at `from mpi4py import
MPI` (tensorrt_llm/_utils.py:38). It reads `OMPI_MCA_*` from the environment at that point. So the launcher shell
can still select them: it runs before any `MPI_Init` on every rank. The rank-0 leader, the comm server and the
exec'd worker all inherit its environment (trtllm-llmapi-launch:274, :308). TRT-LLM already relies on this:
tensorrt_llm/_bootstrap.py:44 sets `OMPI_MCA_coll_ucc_enable=0` before MPI_Init. So does the existing FT check, which
reads `OMPI_MCA_mpi_ft_enable` (communicator.py:959) and works under our mpirun (all FT runs).

Options:
- (a) doc only: an existing deployment that copied the old doc command (`btl self,tcp`, wide-ep-fault-tolerance.md:55-60)
  stays slow without notice, and a launch without `--mca pml` can load pml ucx, which FT does not support.
- (b) verify only: catches a wrong pml but selects nothing.
- (c) **recommended**: the launcher sets defaults, Python verifies, and the doc shows the command. The defaults go in
  the launcher, set only when unset, and only in FT recovery mode (`peer_loss_recovery_enabled`, launcher:172).
  `enable_peer_loss_recovery` (communicator.py:933-980) checks the pml after the ft_enable check. The doc command
  drops the btl/pml flags or keeps them explicit (both work).

How to read the transport, and how reliable each source is:
- `OMPI_MCA_*` env: what mpirun `--mca`, `-x` and the launcher set. It misses MCA param files
  (`~/.openmpi/mca-params.conf`, `<prefix>/etc/openmpi-mca-params.conf`, `mca_base_param_files`). The env wins
  over the files, so after the launcher sets a value, the env is what Open MPI uses. With an explicit include
  list, `pml=ob1` in the env means ob1 is loaded, or MPI_Init fails, so a pass is a real pass.
- MPI_T: mpi4py 4.x has no MPI_T bindings (ctypes into libmpi would be needed). The `pml`/`btl` cvars hold only
  the requested include string, the same information as the env plus the param files, not the selected
  component. Not worth it.
- `ompi_info`: lists built components and params; it does not report what this process selected, and it costs a
  subprocess per rank. Not worth it.
- Truly selected btls per peer: only through `btl_base_verbose` logs. Use them in the validation smoke, not in
  product code.

If the transport is wrong:
- pml not `ob1`: **raise**. A ucx/cm pml under ULFM is a correctness risk (recovery unsupported), and the startup
  check already fails closed for `mpi_ft_enable`.
- btl without `sm`: **warn on rank 0** with the cost. tcp-only is correct, just slower (CLAIMS 158: FT
  host-collective overhead about 2-4x higher with ob1/tcp than with sm).

## 2. Minimal diff (sketch)

trtllm-llmapi-launch: top level, after `peer_loss_recovery_enabled` is defined, before both rank branches:
```bash
# Peer-loss recovery needs Open MPI's ob1 messaging layer; shared memory within a node detects a node-mate's
# loss only because the worker below replaces this launcher (exec).
if peer_loss_recovery_enabled; then
    export OMPI_MCA_pml=${OMPI_MCA_pml:-ob1} OMPI_MCA_btl=${OMPI_MCA_btl:-self,sm,tcp}
fi
```
communicator.py `enable_peer_loss_recovery`, right after the ft_enable check (:959-962):
```python
if os.environ.get("OMPI_MCA_pml") != "ob1":
    raise RuntimeError("Peer-loss recovery requires Open MPI's ob1 messaging layer; "
                       "launch with `--mca pml ob1` (trtllm-llmapi-launch sets it by default)")
btl = os.environ.get("OMPI_MCA_btl", "")
if mpi_rank() == 0 and (btl.startswith("^") or "sm" not in btl.split(",")):
    logger.warning(f"btl {btl or '<default>'}: without shared memory within a node "
                   "(`--mca btl self,sm,tcp`) peer-loss recovery's host collectives are slower")
```
smsc is not set: Open MPI 5 picks xpmem > cma > knem, and our containers have no xpmem, so the default is cma.
Confirm this once with `smsc_base_verbose` in the smoke. If a cma probe fails (yama `ptrace_scope` >= 2, or a
container seccomp without process_vm_readv), smsc should fall back and sm copy through its shared buffers; this is
unverified. Neither the launcher nor the check covers ptrace: correctness is the same and only bandwidth changes.
Name it in the doc.

Unit tests (CPU; tests/unittest/_torch/distributed/test_mpi_survivors.py, `fake_mpi` fixture :197):
- fixture: also `setenv("OMPI_MCA_pml","ob1")` and `setenv("OMPI_MCA_btl","self,sm,tcp")`;
- extend `test_recovery_requires_mpi4py_ulfm_launch_and_thread_multiple` (:559): delenv pml raises "ob1", and
  `pml=ucx` raises;
- new test: `btl=self,tcp`, `^openib` and unset each warn on rank 0 (caplog or a patched logger); `self,sm,tcp`
  does not warn.
- launcher: in tests/unittest/llmapi/test_trtllm_llmapi_launch_exit.py, run the script with a stub
  `python3` on PATH that dumps its env. Check that FT recovery mode exports the defaults, that a preset
  `OMPI_MCA_btl=self,tcp` is kept, and that non-FT mode exports nothing.

Doc edits (docs/source/features/wide-ep-fault-tolerance.md):
- :55 command: `mpirun --with-ft ulfm --mca coll ^hcoll,ucc ...`, and note that trtllm-llmapi-launch selects
  `pml ob1` and `btl self,sm,tcp` unless set.
- Replace :59-60 with: ULFM needs ob1 (other pmls are refused at startup). Within a node it uses shared memory,
  which detects a node-mate's loss only when each worker is the process mpirun started: trtllm-llmapi-launch
  replaces itself with the worker. A wrapper of your own must `exec` too. `btl_tcp_if_include` now only selects
  the interface between nodes.
- Requirements (:35): mpirun from the same ULFM Open MPI. Single-copy shared memory (cma) needs ptrace between
  same-user processes (`/proc/sys/kernel/yama/ptrace_scope` <= 1); without it, only bandwidth drops.
- Known limits: srun/PMIx launch untested (below).

## 3. Red team

- Non-Open-MPI MPI: `OMPI_MCA_mpi_ft_enable` is absent, so the existing check raises first (:959). The new checks
  are not reached. The launcher's `OMPI_MCA_*` exports are inert for MPICH/Intel MPI. No change.
- Open MPI built without the sm btl: sm is core in Open MPI 5 (ULFM's only line), so this is unlikely. With an
  explicit include list that names a missing component, Open MPI reports "requested component was not found"
  at MPI_Init, so the failure is loud, not silent. Whether it aborts or continues on tcp is unverified.
- `btl_tcp_if_include`: unaffected and now matters only between nodes. Excluding the interface that ranks use
  between nodes is the same failure as today. The oob interface is separate (`oob_tcp_if_include`), unchanged.
- Single node: every peer is a node-mate, so the gain is largest and every detection goes through sm. A node kill
  is the whole job, which is fine. The rank kill measured in job 8145956 covers this path.
- srun/PMIx: `--with-ft ulfm` is an mpirun/PRRTE option, and under srun the user must set
  `OMPI_MCA_mpi_ft_enable` themselves. The launcher and check still apply, because the env is the mechanism.
  Whether Slurm's PMIx server delivers ULFM failure events is untested. Document it as unsupported and untested.
- False pass: the env says sm, but sm is disqualified at runtime (for example /dev/shm full), so ranks silently use
  tcp. This costs performance, not correctness. A `pml=ob1` pass is reliable (explicit include).
- False fail: pml set only in an MCA param file. The check raises with the fix (`--mca pml ob1`). With the
  launcher, its default overrides the file anyway, so only direct launches without the launcher hit this.
  Acceptable, and the existing ft_enable check has the same exposure.
- Override hazard: the launcher may override a param-file btl choice (env > file). This is intended in FT mode
  and documented.
- **Biggest open risk: rank 0 is not exec'd.** The MPI rank 0 is `mgmn_leader_node`, a child of the launcher shell
  (launcher:274), which then waits on the trtllm-serve subshell. Under sm, rank 0's node-mates may not learn of its
  death, the same mechanism as the pre-exec sm hang. Rank 0's loss is terminal by design, but the job must still
  end promptly rather than hang. Before landing, run a rank-0 kill smoke with sm and check that the job ends within
  `recovery_timeout_s` and the hang detection bounds. tcp avoided this through socket close.
- A user wrapper that runs the worker under its own shell and uses sm: shrink hangs (pre-exec evidence, CLAIMS 158).
  Product code cannot detect this cheaply, because a parent-process check is fragile across prted and slurmstepd.
  Doc only.
- Validation before finals (validate-before-finals): one job with three cases: the doc command without
  btl/pml flags, a rank kill plus a node kill, and a rank-0 kill. Run it with `btl_base_verbose` and
  `smsc_base_verbose` to confirm sm+cma are selected and that the launcher's defaults reached every rank.

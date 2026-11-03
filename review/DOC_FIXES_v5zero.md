# Stale docs/docstrings vs today's design (wt-zero, v5-zero @ 7a03252e92)

Read-only audit. Line numbers are at 7a03252e92. Design checked against the code: packed fence a2a + rank mask,
recovery-time same-address remap (`PeerLossRecovery._remap_workspaces`, `MoeExecutionState.release/remap_workspaces`,
`MnnvlMemory.release_backing/remap`), KV lent to the remap and returned at commit or idle (`serving_idle`), CFT
refused with FT, `peer_failure_poll_s` 0.02 s, no recapture.

Already correct (no change): doc lines 16-19 (CFT/FORCE_CFT=0), 81 and config.py:67-71 (poll 0.02 s, le 1.0),
274 and feature-combination-matrix.md:48 (no recapture), `nvlink_one_sided.py` `__init__` CFT docstring
(419-425), `mnnvl_memory.py` `release_backing`/`remap` docstrings (938-970), `execution_state.py`
`release_workspaces`/`remap_workspaces` (120-144), `peer_loss_recovery.py` `_remap_workspaces` (1377-1393).
No remaining mention of per-sender receive rows, receive CTAs, compact copy, token-major layout or
per-signature regions anywhere in docs/, examples/wide_ep, or the named sources (grep hits for
"compact"/"signature" are main's top-k compaction and the comm-layout signature, unrelated).
moeAlltoAllKernels.{cu,h}: the FT comments (abort record, released round, rank mask) match the code; the
"skip dead ranks ... this is the bug the rank-mask is here to prevent" comments (cu:704-725, 1792-1813) are
main's text, so left alone.

## A. Wrong or incomplete statements

### A1. docs/source/features/wide-ep-fault-tolerance.md:133-134 (further loss during a recovery)
Current: "Several such ranks lost together, or one more during a recovery (before the survivors install the new
communicator), are recovered as one loss if the survivors' slots still hold every expert."
Proposed: "Several such ranks lost together, or one more during a recovery before the survivors remap the
all-to-all workspaces, are recovered as one loss if the survivors' slots still hold every expert. A loss from
the start of the remap to the install ends the job (see Known limits)."
(Confirm the window's exact start: `_remap_workspaces` runs after Propose, peer_loss_recovery.py:1026.)

### A2. wide-ep-fault-tolerance.md:145-146 (Not covered)
Current: "a further loss after the survivors installed a recovery and before it committed (a loss before the
install re-plans the recovery, see above), ..."
Proposed: "a further loss from the start of the all-to-all workspace remap until the recovery committed (an
earlier loss re-plans the recovery, see above), ..."

### A3. wide-ep-fault-tolerance.md:93-99 (late writes of a failed rank are not described)
Current paragraph says a failed rank "can still hold its GPU and its mappings of the survivors' memory" and only
covers ending it. Add after its first sentence:
"Its kernels can still write the all-to-all workspace pages it mapped. Each recovery therefore releases the
survivors' workspace pages and maps fresh ones at the same addresses, shared only among the survivors: the
failed rank's late writes land in pages no survivor maps, and the CUDA graphs keep their addresses. The old
pages are freed only when the failed process ends; until then the fresh ones are taken from the KV cache if the
device lacks the memory (see below)."

### A4. wide-ep-fault-tolerance.md:159-160 (KV cache after a recovery)
Current: "each loss lowers capacity (and may take more of the KV cache for restored experts)."
Proposed: "each loss lowers capacity and may take KV cache: for restored experts until the deployment restarts,
and for the remapped all-to-all workspaces only until the failed processes have ended and the executor is idle
or a later recovery commits."

### A5. wide-ep-fault-tolerance.md:275-277 (resume log line)
Current lists "the KV cache bytes given to restored experts" (correct for the resume line,
peer_loss_recovery.py:1113-1116). Add: "Each rank also logs the bytes of the remapped all-to-all workspaces and
how many came from the KV cache, and later each return of them ('Returned N bytes the all-to-all workspace remap
took to the KV cache')."

### A6. wide-ep-fault-tolerance.md Known limits (207-230): two gaps missing
Add:
"- A further loss while the survivors remap the all-to-all workspaces (up to the commit) ends the job.
- KV cache lent to a remap is returned only when the failed processes have ended and the executor has an idle
  iteration (checked at most once a second) or a later recovery commits; under steady load with no idle
  iteration that KV cache stays lent."

### A7. wide-ep-fault-tolerance.md:256 and 178 (recovery duration)
Current 256: "resume serving well under a second later"; 178: "a recovery takes milliseconds after detection".
The remap (release, barrier, KV resize, collective remap, reset) is now on that path. Proposed 178: "...; a
recovery takes TODO(measure: detection-to-resume incl. remap, DEP32 v5-zero) after detection". Keep 256 only if
the measured value supports it.

### A8. execution_state.py:98 `MoeExecutionState.retain`
Current: "Keep a bound layer's workspace mapped until :meth:`close`."
Proposed: "Keep a bound layer's workspace addresses until :meth:`close`; a recovery replaces its pages at those
addresses (:meth:`release_workspaces`, :meth:`remap_workspaces`)."

### A9. execution_state.py:36-40 class docstring
Current: "Both are only edited in place, so CUDA graphs captured once keep running after :meth:`install_mask`
changes the membership they observe."
Proposed: append "The bound workspaces keep their addresses across a recovery's remap, so the same graphs replay
on the fresh pages."

### A10. nvlink_one_sided.py:735-739 `bind_execution_state`
Current: "... stay valid when the state installs a new rank mask or requests an abort. The state keeps the
workspace mapped."
Proposed: "... stay valid when the state installs a new rank mask, requests an abort, or remaps the workspace's
pages at the same addresses during a recovery."

### A11. nvlink_one_sided.py:441-446 `ep_group_health` arg
Current: "so captured CUDA graphs follow membership changes made between captures."
Proposed: "so CUDA graphs captured once follow every later membership change." (graphs are never recaptured)

### A12. mnnvl_memory.py:353-357 `MnnvlMemory.pin`
Current: "Keep the allocation mapped until the matching :meth:`unpin`. Captured CUDA graphs address the mapping
directly, so checkpointing and closing do not unmap a pinned allocation."
Proposed: "Keep the allocation's addresses until the matching :meth:`unpin`. Captured CUDA graphs address them
directly, so checkpointing and closing do not unmap a pinned allocation; only :meth:`release_backing` and
:meth:`remap` replace its pages, at the same addresses."

### A13. peer_loss_recovery.py:974-975 comment
Current: "Until Install, a further loss restarts the plan with every rank lost so far, off the bytes the plans
proposed so far had the lost ranks copy."
Proposed: "Until the workspace remap, a further loss restarts the plan with every rank lost so far; from the
remap on it ends the job." (Confirm against the `except RecoveryInterrupted` path at 1046-1049: it still
re-plans after a remap; if a second remap of an already-remapped allocation fails, that is the job-ending path.
If the intent is a clean refusal, the code should raise explicitly rather than rely on that failure.)

### A14. Ring-restore staging rationale (possibly stale rationale and redundant code)
- peer_loss_recovery.py:1467-1471 `_stage_ring_restore`: "the plan keeps off the bytes their all-to-alls may
  still write, and their copies of the chunks ``earlier_in_flight`` ..."
- moe_load_balancer.py:1523-1526 `restore_copy_plan`: "The halves leave out what ranks of ``dead_ranks`` may
  still write ..."
- nvlink_one_sided.py:784-792 `source_write_ranges`, execution_state.py:113-114 `staging_write_ranges`.
Staging now runs after `_remap_workspaces` (peer_loss_recovery.py:1026-1030), into fresh pages the failed
ranks never mapped, so their late all-to-all writes and in-flight staging copies cannot reach them. The text is
still true of the code, but the protection it describes is now redundant. Proposed: decide with the owner
whether to drop the dead-rank write-range exclusion (keep only the startup room check against live ranks'
ranges, peer_loss_recovery.py:474-480); if kept, the docstrings should say why. Not a doc-only fix.

## B. Claims needing a measured number on the current design (leave TODO, do not invent)

- doc:176 "Measured ... at DEP32 under an earlier rule: 10.6 / 15.9 / 36.7 s" -> TODO(re-measure SIGSTOP-to-serving
  on v5-zero, cite CLAIMS row) or drop.
- doc:178, 256 recovery "milliseconds" / "well under a second" -> TODO(measure incl. remap).
- doc:307 "a recovery returns 503 (`recovering`) for about a second" -> TODO(measure).
- doc:204-205 "At DEP32, 45 s host stalls and interpreter-lock stalls cost nothing" -> TODO(re-verify on v5-zero).
- doc:271-273 GSM8K within two standard errors after a kill -> TODO(re-run on v5-zero, cite run).
- doc:325-331 maturity list ("tested most, including repeated kills under load at DEP32") was established on
  the earlier a2a design -> TODO(restate per v5-zero runs; add remap and remap-KV-return to the list).
- doc:64-65 sm transport "no measured serving throughput gain" -> TODO(cite run or drop).
- doc:78 "an estimated 1.4 GiB per GPU for DeepSeek-R1 NVFP4" -> estimate, mark TODO(measure) or keep "estimated".
- doc:333-338 Cost: no number. Add "FT on vs off at the standard config: TODO(measured %, CLAIMS row)", and state
  the comparison uses `TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT=0` on both sides.
- New: remap duration and bytes per recovery (DeepSeek-R1 DEP32) -> TODO(measure from the "Remapped N bytes ... in
  X ms" log line).

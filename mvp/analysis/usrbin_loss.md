# Container "lost /usr/bin" (rc 127) in jobs 8133265 G1 and 8137627 G0

## Verdict
In both jobs the tools disappeared only after the Slurm step had already ended. Pyxis/enroot removed the
container's rootfs, but harness processes left over from the step kept running and writing to their Lustre logs.
The rc 127 errors are a symptom of the teardown. They are not its cause. No harness code deletes, unmounts or
overlays anything under / or /usr.

## Evidence
- 8137627 (G0, T02,T04,T07,T12-T16): logs/slurm/slurm-8137627.out has
  `JOB 8137627 ON nvl72090-T02 CANCELLED AT 2026-10-21T18:13:01 DUE to SIGNAL Terminated` at 18:13:01.26 PT
  (the same message for STEP .0). The case std-noeplb-ulfm-1 started at 18:11:12. mpirun started at 18:11:22,
  about 100 s into a startup that normally takes about 13 min. The timeline then shows MPIRUN_EXIT rc=255 at
  18:13:01.73 (after the cancel), RANK_EXIT 1 with a date at 18:13:03, and RANK_EXIT 2/3 with no date (date was
  already missing). Seven of the eight nodes lost /usr/bin. Node 5 also lost /dev/null, which means the whole
  rootfs was gone. Node 4 (T13) reached NODE_CLEAN and BUNDLE_TASK rc=0 at 18:13:05 before its own teardown.
  The run_job.sh watchdog on node 5 kept going and printed `JOB_STALL : no output for 30 min` 30 min later.
  G1 (nodes 8-15) had finished at 17:46:41, so those 8 nodes sat idle for 27 min before the cancel. The likely
  sender is the idle-GPU reaper or an agent. No cancel record was found.
- 8133265 G1 (T10-T16,T18): bundle.sh's `timeout LIMIT_S` (85 min) ended the task at 16:54:29-16:55:10
  (BUNDLE_TASK rc=124) during std-online-ft-loc-1. The server had been ready since 16:53:55. Every node then
  logged BUNDLE_NODE_EXIT at 16:55:10-19, which ended the step. All of the "missing tool" lines in 8133265-8.log
  come after the rc=124 line. They come from task.sh's guard and stall subshells (lines 80 and 106-109), which
  outlived the timeout. mpirun.log's "daemon failed to report back" was written at 16:55, after the teardown.
- PATH was not the problem: the failing calls use absolute paths (`/usr/bin/sleep`, `/usr/bin/grep`, `/usr/bin/cp`)
  and the files themselves were missing.
- Harness audit (bundle.sh, run_job.sh, job.sh, serve.sh, rank.sh, task.sh, remote_exec.py, node_agent.py,
  install_wheel.sh, drop_page_cache.py):
  - The only deletions are `rm -rf /tmp/mvp-harness` (fixed path), mpi4py in site-packages, and the
    `/dev/shm/moe_shared*` unlink.
  - drop_page_cache.py only calls fadvise DONTNEED.
  - There is no umount, fusermount, chroot, ldconfig or `find -delete`, and nothing touches /usr.
- The two failures do not share an arm type. One is -loc (node-local logs, ft), the other is ulfm (Lustre logs).
  The "between cases" cleanup ran normally in both.

## Fix
1. In 8137627, find out who sent SIGTERM at 18:13:01, most likely the reaper reacting to idle nodes 8-15.
   Then stop packing groups whose run times differ by 27 min or more, or release finished nodes (per the
   no-idle-GPU rule).
2. Set the bundle LIMIT_S for G1 above the case total. 4 cases of about 22-26 min plus startup came to
   more than 85 min.
3. Treat an rc 127 or rc 124 case as "killed by teardown" and rerun it. Optionally make bundle.sh kill the
   task's whole session on timeout, so that no orphans write to logs after the teardown.

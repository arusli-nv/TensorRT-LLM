# KNOB-AB 8240302 readout (diagnostic, offline EPLB, R1 NVFP4 DEP32 GB200, build-freeze-58caa0fde6, n=1 per arm per group)

Steady (tools/trim_readout.py, gap vs the group's std-offline-1, ms per decode step; short windows 32=14 / 128=6 waves):

| arm | G0 32 | G0 128 | G8 32 | G8 128 | stall ms/step G0 32/128 | G8 32/128 |
|---|---|---|---|---|---|---|
| off (std-offline-1) | 14.114 | 26.540 | 14.152 | 26.328 | 0.20/0.25 | 0.17/0.39 |
| ft (finals) | +0.078 | +0.062 | +0.047 | +0.105 | 0.39/0.39 | 0.24/0.15 |
| ft-nohang (hang detection off) | +0.088 | +0.083 | -0.008 | +0.071 | 0.16/0.20 | 0.15/0.28 |
| ft-poll1 (peer_failure_poll_s 1.0) | +0.144 | +0.182 | +0.041 | +0.040 | 0.26/0.14 | 0.16/0.26 |

Neither knob removes the steady gap: arm-to-arm differences (<= ~0.1 ms) are within the run-to-run spread at n=1.

Cold first-150 serving iterations after ready (tools/cold_steps.py; the window runs ready -> ~+45 s and contains the
post-ready page-cache drop, which starts within 1 s of ready): FT off p90 9.9 / 9.9 ms (0 GB shm); every FT arm (ghost,
390 GB shm) stalls: ft 518.5 / 102.6, nohang 186.0 / 115.2, poll1 320.1 / 341.5 ms (G0 / G8). Neither knob removes it.

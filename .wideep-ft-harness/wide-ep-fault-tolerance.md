# [WIP] Wide-EP Fault Tolerance (Experimental)

A Wide-EP deployment (attention data parallelism with expert-parallel MoE) can keep serving
after one worker process dies. The surviving ranks stop the interrupted step, agree on the new
membership, route around the lost rank and continue on the CUDA graphs captured at startup:
nothing is reloaded, recompiled or recaptured. Requests that the lost attention-DP rank was
serving fail, because their KV cache was lost, with an error asking the client to retry; all
other requests complete.

```{note}
This feature is experimental. It recovers from the loss of one process; a second loss, or the
loss of rank 0, ends the job.
```

## How it works

1. **Detection.** Every rank watches its peers through MPI User-Level Failure Mitigation
   (ULFM). When a peer dies, the survivors release MoE all-to-all kernels that wait for it and
   interrupt host collectives.
2. **Discard.** The step that was running is discarded: its KV cache reservations are returned
   and its results are not applied, so it runs again after recovery.
3. **Agreement.** A recovery authority, a separate CPU process, collects the survivors'
   observations and drives a commit protocol (stop, prepare, install, commit). Survivors resume
   only after every survivor installed the new membership.
4. **Install.** The lost rank is masked out of host collectives, EPLB routing and the
   NVLinkOneSided dispatch/combine. The mask lives in device memory read by the captured graphs.
5. **Resume.** Survivors deliver responses the interruption delayed, fail the lost rank's
   requests, and continue.

## Requirements

- Open MPI with ULFM, launched with `mpirun --with-ft ulfm`, and the mpi4py 4.1.2 that the
  TensorRT LLM container installs (with TensorRT LLM's patch).
- PyTorch backend, MPI orchestration, pure attention DP: `enable_attention_dp`, `PP = CP = 1`,
  expert TP 1 and TP size equal to the EP size and the world size.
- The NVLinkOneSided MoE all-to-all on its fence path. If your driver enables CFT counted
  writes, set `TRTLLM_MOE_A2A_FORCE_CFT=0`.
- Static EPLB (`layer_updates_per_iter: 0`) whose placement keeps a replica of every expert of a
  lost rank on another rank. Only ranks covered this way are recoverable; the authority reports
  which ranks are covered at startup.
- KV Cache Manager V2, with or without block reuse.
- No speculative decoding, guided decoding, beam search, PEFT, KV cache connector or
  transceiver, conversation manager, overlap scheduler, attention-DP balancing, sleep/wakeup,
  LM-head TP in attention DP, or encoder-decoder models.
- The executor's hang-detection timeout must exceed the recovery timeout
  (`TLLM_FAULT_TOLERANCE_RECOVERY_TIMEOUT_S`).
- Greedy sampling is assumed: penalties, seeded sampling and multi-token stop words are not
  rolled back with a discarded step.

Unsupported configurations are rejected at startup when fault tolerance is enabled.

## Usage

Start the authority before the job, on any host the workers can reach:

```bash
export TLLM_FAULT_TOLERANCE_AUTHORITY_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
python3 -m tensorrt_llm.executor.recovery.authority \
    --endpoint tcp://0.0.0.0:24680 --world-size 32
```

Launch the job with the same key and these variables on every rank:

| Variable | Value |
|----------|-------|
| `TLLM_FAULT_TOLERANCE_MODE` | `1` |
| `TLLM_FAULT_TOLERANCE_AUTHORITY` | `tcp://<authority host>:24680` |
| `TLLM_FAULT_TOLERANCE_AUTHORITY_KEY` | the hex key above |
| `TLLM_FAULT_TOLERANCE_RECOVERY_TIMEOUT_S` | optional; default 120 |

```bash
mpirun --with-ft ulfm --mca pml ob1 --mca btl self,tcp --mca coll ^hcoll,ucc -np 32 \
    -x TLLM_FAULT_TOLERANCE_MODE -x TLLM_FAULT_TOLERANCE_AUTHORITY -x TLLM_FAULT_TOLERANCE_AUTHORITY_KEY \
    trtllm-llmapi-launch trtllm-serve <model> --config config.yaml
```

ULFM needs MPI components that support it: the `ob1` point-to-point layer without `hcoll`
or `ucc` collectives. Only host-side control traffic uses MPI; MoE data moves over NVLink.

The key authenticates messages on a trusted network; it does not encrypt them.

## Behavior and limits

- **Recovery time.** Typically 1 to 3 seconds from the failure until the survivors' requests
  continue; every wait is bounded by the recovery timeout and the executor hang detector.
- **Own failures.** If a rank's event loop fails while the others can recover without it, that
  rank exits on its own and the others recover from its loss.
- **Capacity.** The job continues with one rank fewer; its throughput drops accordingly.
- **Shutdown.** A job that lost a rank ends with `MPI_Abort` once serving shuts down, because
  the MPI task executor cannot release a world with a missing member. Every rank exits, but on
  multiple nodes `mpirun` itself may keep running afterwards; stop it from your launcher once
  the server process has exited.
- **Health.** The server keeps reporting healthy while it recovers and after it lost a rank.
- **Cost.** With fault tolerance enabled, every MoE layer adds a gather kernel after dispatch and
  each step synchronizes before its results are applied. With it disabled, behavior and
  performance are unchanged.

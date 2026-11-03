# Product stress tests: FT as early users would run it (2026-10-25)

User (16:50 PT): stress-test FT as a feature, as if people are starting to use the MVP. Issues are OK, but we must
know about them and try to fix them. Collect good data on the product.

Today's tests used a lab pattern: exact 1k/1k requests, all released at once, one kill at a fixed point. Users will
have mixed lengths, arriving over time, failures at random moments, and long uptimes. Each test below has its pass
criteria written before data. Failures get root-caused (as CLAIMS 204 was) and listed; fixes are a separate step.

Runs on v5-final (overlays/fix-dropmap-9f1fd4cab4), online EPLB unless noted, 32 GB200, both mounts.

1. **Realistic traffic, healthy (S1).**
   - Load: Poisson arrivals at ~80% of the 32/GPU throughput, input 128-4k and output 64-2k tokens (random range
     ratio), FT on vs off, 1 h each, same nodes.
   - Pass: no stall > 60 s; FT-on throughput and p50/p99 TTFT/TPOT reported vs off; no memory growth (GPU and host
     sampled each minute).
2. **Random kill under realistic traffic (S2).**
   - Same load. One SIGKILL of a random non-0 rank at a random time in minutes 10-40, then 20 min of degraded
     serving. x4 (rank kills, plus 1 node kill).
   - Pass: recovery commits; no request other than the lost rank's fails; degraded throughput >= 0.85x for rank
     kills (0.75x for node kills); no stall afterwards; /health goes recovering -> degraded.
3. **Second loss later (S3).**
   - A kill, 15 min degraded, then a second kill of a different rank.
   - Pass: both recover (or the job ends cleanly if the slots cannot cover); no hang.
4. **Long healthy uptime (S4).** Covered by the overnight soak (CLAIMS 204 follow-up), plus memory/leak sampling.
5. **Client view (S5).**
   - Streamed requests through the OpenAI endpoint during a kill. Count the retryable errors the client sees, check
     their format, and check that /health transitions match the doc.
   - Pass: only the lost rank's in-flight requests error; each error is retryable and well formed.

Order:
- S2 and S5 first (one 8-node job: FT only, 4 kills + client view).
- Then S1 (one 16-node job, on/off pairs).
- Then S3.
- These run in the slot next to the soak chain tonight. Readouts go to review/diag/stress_<test>_<job>.md and to
  CLAIMS as PRODUCT rows.

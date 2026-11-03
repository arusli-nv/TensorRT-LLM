"""FT-off sanity on one node: DeepSeek-V3-Lite NVFP4, EP4 + attention DP with the integration test's
configuration and recovery off. Writes the greedy token ids of a fixed prompt set and the wall time of
timed rounds, to compare a main wheel against the stack's wheel.
Usage: python3 ftoff_sanity.py DRIVER_DIR MODEL OUTPUT_JSON
"""

import json
import sys
import time

driver_dir, model, output = sys.argv[1:4]
sys.path.insert(0, driver_dir)
from peer_loss_driver import llm_kwargs  # noqa: E402

from tensorrt_llm import LLM, SamplingParams  # noqa: E402

SUBJECTS = ["France", "Italy", "Spain", "Japan", "Brazil", "Canada", "Egypt", "India"]
PROMPTS = [
    template.format(subject)
    for subject in SUBJECTS
    for template in (
        "The capital of {} is",
        "Write one sentence about the history of {}:",
        "List three foods from {}:",
        "The main language spoken in {} is",
    )
]
ROUNDS = 3


def main() -> None:
    llm = LLM(model=model, **llm_kwargs())
    params = SamplingParams(temperature=0, max_tokens=64, ignore_eos=True)
    try:
        llm.generate(PROMPTS, params)  # warmup
        seconds = []
        for _ in range(ROUNDS):
            start = time.perf_counter()
            outputs = llm.generate(PROMPTS, params)
            seconds.append(time.perf_counter() - start)
        result = dict(
            seconds=seconds,
            tokens=[list(out.outputs[0].token_ids) for out in outputs],
            text=[out.outputs[0].text for out in outputs],
        )
    finally:
        llm.shutdown()
    with open(output, "w") as f:
        json.dump(result, f, indent=1)
    print(f"FTOFF_DONE seconds={[round(s, 3) for s in seconds]}")


if __name__ == "__main__":
    main()

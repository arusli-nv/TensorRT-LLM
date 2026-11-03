"""TRT-LLM's benchmark_serving, also recording when each request's tokens arrived (wall clock), so the client's token rate
can be read in fixed windows around a failure (a token's time = start + ttft + the cumulative inter-token latencies).

Usage: python3 timed_client.py OUT.jsonl <benchmark_serving args...>
OUT.jsonl gets one line per finished request: {"start": wall s, "ok": bool, "tokens": {"<wall second>": count, ...}}.
"""

import json
import runpy
import sys
import threading
import time

from tensorrt_llm.serve.scripts import backend_request_func

out_path = sys.argv.pop(1)
out = open(out_path, "w")
lock = threading.Lock()


def _timed(request_func):
    async def timed(*args, **kwargs):
        start = time.time()
        result = await request_func(*args, **kwargs)
        tokens = {}
        if result.ttft:
            # One latency per streamed chunk (stream_interval tokens each): spread the request's tokens over them.
            chunks = [0.0, *result.itl]
            per_chunk = (result.output_tokens or len(chunks)) / len(chunks)
            at = start + result.ttft
            for gap in chunks:
                at += gap
                tokens[int(at)] = round(tokens.get(int(at), 0) + per_chunk, 2)
        record = {"start": start, "ok": result.success, "tokens": tokens}
        with lock:
            out.write(json.dumps(record) + "\n")
            out.flush()
        return result

    return timed


# benchmark_serving looks its request function up in this dict when it runs.
for backend, request_func in list(backend_request_func.ASYNC_REQUEST_FUNCS.items()):
    backend_request_func.ASYNC_REQUEST_FUNCS[backend] = _timed(request_func)
runpy.run_module("tensorrt_llm.serve.scripts.benchmark_serving", run_name="__main__", alter_sys=True)

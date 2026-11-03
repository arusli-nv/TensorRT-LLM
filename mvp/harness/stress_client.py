"""TRT-LLM's benchmark_serving with a request function that records what an OpenAI streaming client sees (stress tests,
review/PRODUCT_STRESS_PLAN.md S5). Benchmark-only (never part of TRT-LLM).

Usage: python3 stress_client.py OUT.jsonl <benchmark_serving args...>   (--backend openai or openai-chat)
Env STRESS_LOAD_START=path: the wall time of the first request is written there (the load's start).

main's request functions skip an SSE error event (no "choices") and stop at the first [DONE], so a stream the server ends
with an error counts as a short success. This one parses every event to the end of the stream and marks a request
failed on an error event, a non-200 status, an exception, or a stream without [DONE]. OUT.jsonl gets one line per request:
  ep (completions | chat), start / end (wall s), ttft (s), exp (requested tokens), tok (usage.completion_tokens),
  chunks, status, retry_after, body (non-200, first 300 chars), errors (the error objects of error events), done
  (number of [DONE]), after_done (events after the first [DONE]), exc, ts ({wall second: chunks}).
"""

import json
import os
import runpy
import sys
import threading
import time
import traceback

import aiohttp

from tensorrt_llm.serve.scripts import backend_request_func as brf

out = open(sys.argv.pop(1), "w")
lock = threading.Lock()
load_start = os.environ.get("STRESS_LOAD_START")
TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


def _record(row: dict) -> None:
    with lock:
        out.write(json.dumps(row, separators=(",", ":")) + "\n")
        out.flush()


async def _events(content):
    """Every SSE data payload to the end of the stream ([DONE] included)."""
    buf = b""
    async for chunk in content.iter_any():
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.decode("utf-8", errors="replace").strip()
            if line.startswith("data:"):
                yield line[5:].strip()


async def recorded_request(request_func_input, streaming=True, pbar=None, session=None, chat=False):
    global load_start
    if load_start:
        with open(load_start + ".tmp", "w") as mark:
            mark.write(f"{time.time():.3f}\n")
        os.replace(load_start + ".tmp", load_start)
        load_start = None
    inp = request_func_input
    payload = {"model": inp.model_name or inp.model, "temperature": 0.0, "stream": True,
               "stream_options": {"include_usage": True}}
    if chat:
        payload.update(messages=[], max_completion_tokens=inp.output_len)
        if isinstance(inp.prompt, list):
            payload["prompt_token_ids"] = inp.prompt
        else:
            payload["messages"].append({"role": "user", "content": inp.prompt})
    else:
        payload.update(prompt=inp.prompt, max_tokens=inp.output_len)
    if inp.ignore_eos:
        payload["ignore_eos"] = True
    if inp.extra_body:
        payload.update(inp.extra_body)
    output = brf.RequestFuncOutput(prompt_len=inp.prompt_len)
    row = dict(ep="chat" if chat else "completions", start=time.time(), ttft=None, exp=inp.output_len, tok=None,
               chunks=0, status=None, retry_after=None, body=None, errors=[], done=0, after_done=0, exc=None, ts={})
    st = last = time.perf_counter()
    text = []
    own = session is None
    session = session or aiohttp.ClientSession(trust_env=True, timeout=TIMEOUT,
                                               connector=aiohttp.TCPConnector(limit=0, limit_per_host=0))
    try:
        async with session.post(inp.api_url, json=payload, headers={"Content-Type": "application/json"}) as response:
            row["status"], row["retry_after"] = response.status, response.headers.get("Retry-After")
            if response.status != 200:
                row["body"] = (await response.text())[:300]
            else:
                async for data in _events(response.content):
                    now = time.perf_counter()
                    if row["done"]:
                        row["after_done"] += 1
                    if data == "[DONE]":
                        row["done"] += 1
                        continue
                    try:
                        event = json.loads(data)
                    except ValueError:
                        row["errors"].append({"unparsed": data[:300]})
                        continue
                    if "error" in event:
                        row["errors"].append(event["error"])
                    elif event.get("choices"):
                        choice = event["choices"][0]
                        piece = choice.get("delta", {}).get("content") if chat else choice.get("text")
                        text.append(piece or "")
                        if row["chunks"] == 0:
                            output.ttft = row["ttft"] = now - st
                        else:
                            output.itl.append(now - last)
                        last = now
                        row["chunks"] += 1
                        second = str(int(time.time()))
                        row["ts"][second] = row["ts"].get(second, 0) + 1
                    if event.get("usage"):
                        row["tok"] = event["usage"].get("completion_tokens")
    except Exception as error:  # the client's view of any failure is the record
        row["exc"] = "".join(traceback.format_exception_only(type(error), error)).strip()[:300]
        output.exception_type = type(error).__name__
    finally:
        if own:
            await session.close()
    row["end"] = time.time()
    output.generated_text = "".join(text)
    output.latency = last - st
    output.output_tokens = row["tok"] or 0
    problems = ([f"HTTP {row['status']}"] if row["status"] not in (None, 200) else []) + \
        [f"SSE_ERROR {e.get('code')}: {str(e.get('message'))[:200]}" for e in row["errors"]] + \
        ([row["exc"]] if row["exc"] else []) + \
        (["no [DONE]"] if row["status"] == 200 and not row["done"] and not row["exc"] else [])
    output.success = not problems and row["chunks"] > 0
    output.error = "; ".join(problems) or ("" if output.success else "no token")
    _record(row)
    if pbar:
        pbar.update(1)
    return output


async def _chat(request_func_input, streaming=True, pbar=None, session=None):
    return await recorded_request(request_func_input, streaming, pbar, session, chat=True)


# benchmark_serving looks its request function up in this dict when it runs.
brf.ASYNC_REQUEST_FUNCS["openai"] = recorded_request
brf.ASYNC_REQUEST_FUNCS["openai-chat"] = _chat
runpy.run_module("tensorrt_llm.serve.scripts.benchmark_serving", run_name="__main__", alter_sys=True)

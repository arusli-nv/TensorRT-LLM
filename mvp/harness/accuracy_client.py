"""GSM8K through a running trtllm-serve, with lm_eval's task as TensorRT-LLM's GSM8K evaluator runs it.

The dataset is the local copy the repo's accuracy tests use. Writes OUTPUT (JSON) with the task metrics and
OUTPUT.samples.jsonl with every question's result per filter (doc_id, filter, exact_match, answer), to diff two runs
question by question (accuracy_diff.py).
Usage: python3 accuracy_client.py PORT MODEL OUTPUT [--concurrency N] [--limit N] [--max-retries N]
"""

import argparse
import json

DATASET = "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/common/datasets/openai/gsm8k"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=int)
    parser.add_argument("model")
    parser.add_argument("output")
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=1,
                        help="attempts per request (lm_eval retries a failed request with backoff)")
    parser.add_argument("--tokenizer", default=None, help="tokenizer path (default: the model path)")
    args = parser.parse_args()

    import lm_eval
    import lm_eval.api.task

    # As tensorrt_llm.evaluate.lm_eval does for --dataset_path: every task reads the local copy.
    post_init = lm_eval.api.task.TaskConfig.__post_init__

    def local_dataset(task_config, *a, **kw):
        task_config.dataset_path = DATASET
        post_init(task_config, *a, **kw)

    lm_eval.api.task.TaskConfig.__post_init__ = local_dataset
    model_args = (f"base_url=http://127.0.0.1:{args.port}/v1/completions,model={args.model},"
                  f"tokenizer={args.tokenizer or args.model},num_concurrent={args.concurrency},max_retries={args.max_retries},"
                  "tokenized_requests=False")
    results = lm_eval.simple_evaluate(model="local-completions", model_args=model_args, tasks=["gsm8k"],
                                      limit=args.limit, log_samples=True)
    with open(args.output + ".samples.jsonl", "w") as f:
        for sample in results.get("samples", {}).get("gsm8k", []):
            answer = sample.get("filtered_resps") or [""]
            print(json.dumps({"doc_id": sample.get("doc_id"), "filter": sample.get("filter"),
                              "exact_match": sample.get("exact_match"),
                              "answer": str(answer[0] if isinstance(answer, list) else answer)[:200]}), file=f)
    metrics = results["results"]["gsm8k"]
    out = {key: value for key, value in metrics.items() if isinstance(value, (int, float))}
    out["n_samples"] = results.get("n-samples", {}).get("gsm8k")
    with open(args.output, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()

"""
evaluation/run_critic_eval.py -- prompted Critic on the held-out test set (Step 5).

Environment: LOCAL (Mac), Ollama menu-bar app running.
  Smoke test (5 rows, not saved):  python -m evaluation.run_critic_eval --limit 5
  Full run (100 rows, saved):      python -m evaluation.run_critic_eval

Uses the PRODUCTION path (agents.critic._get_llm + judge_with_llm), so the result
describes the Critic the pipeline actually runs. This local Ollama number is the
prompted BASELINE; the final 4-row table is re-run in ONE Colab environment.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

from openai import APIConnectionError, APITimeoutError

import agents.critic as critic  # module import: _get_llm is looked up at call time
import config
from evaluation.metrics import critic_metrics

TEST_PATH = Path("evaluation/data/critic_test.jsonl")
RESULTS_DIR = Path("evaluation/results")


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run(examples: list[dict], llm, judge: Callable = critic.judge_with_llm) -> list[dict]:
    """Judge every example. Malformed output is recorded (prediction None), never fatal."""
    rows = []
    for i, ex in enumerate(examples, start=1):
        start = time.perf_counter()
        try:
            judgment = judge(llm, ex["evidence"])
            prediction, reason, error = judgment.verdict, judgment.reason, None
        except APIConnectionError as exc:
            if not isinstance(exc, APITimeoutError):
                # Ollama isn't reachable: abort rather than record 100 fake "malformed" rows.
                raise RuntimeError("LLM endpoint unreachable. Is the Ollama app running?") from exc
            prediction, reason, error = None, "", f"timeout: {exc}"
        except Exception as exc:  # unparseable output / no tool call -> malformed
            prediction, reason, error = None, "", f"{type(exc).__name__}: {exc}"[:300]
        seconds = round(time.perf_counter() - start, 2)
        rows.append({
            "id": ex["id"], "scenario": ex["scenario"], "strategy": ex["strategy"],
            "borderline": ex["borderline"], "label": ex["label"],
            "prediction": prediction, "reason": reason, "error": error, "seconds": seconds,
        })
        mark = "MALFORMED" if prediction is None else ("ok" if prediction == ex["label"] else "WRONG")
        print(f"[{i:3d}/{len(examples)}] {ex['id']} {ex['strategy']:<22} "
              f"label={ex['label']:<6} pred={str(prediction):<6} {mark} ({seconds}s)")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the prompted Critic.")
    parser.add_argument("--limit", type=int, default=None, help="only the first N rows; not saved")
    parser.add_argument("--name", default="prompted_ollama", help="results file name")
    args = parser.parse_args()

    examples = load_jsonl(TEST_PATH)
    if args.limit:
        examples = examples[: args.limit]

    provider = config.LLM_PROVIDER.strip().lower()
    model = config.OLLAMA_MODEL if provider == "ollama" else config.LLM_MODEL
    print(f"Evaluating {len(examples)} examples with {provider}:{model}\n")

    started = time.perf_counter()
    rows = run(examples, critic._get_llm(model))
    metrics = critic_metrics(rows)
    print("\n" + json.dumps(metrics, indent=2))

    if args.limit:
        print("\n--limit run: results NOT saved.")
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{args.name}.json"
    out_path.write_text(json.dumps({
        "condition": args.name,
        "provider": provider,
        "model": model,
        "test_file": str(TEST_PATH),
        "total_seconds": round(time.perf_counter() - started, 1),
        "metrics": metrics,
        "predictions": rows,
    }, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
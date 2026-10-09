"""
evaluation/run_critic_eval.py -- Critic on the held-out test set (Steps 5-6).

Environment: LOCAL (Mac), Ollama menu-bar app running.
  Prompted baseline:            python -m evaluation.run_critic_eval
  Prompted + hybrid retrieval:  python -m evaluation.run_critic_eval --retrieval
  Smoke test (not saved):       add --limit 5
  Re-score a saved run:         python -m evaluation.run_critic_eval --rescore prompted_ollama
                                (recomputes metrics from saved predictions; no LLM calls)

Uses the PRODUCTION path (agents.critic._get_llm + judge_with_llm), so the result
describes the Critic the pipeline actually runs. These local Ollama numbers are the
prompted baselines; the final 4-row table is re-run in ONE Colab environment.
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


def _rule(example: dict) -> str:
    # "rule 7: drop_rows_missing affected 12.7% > 10%" -> "rule 7"
    return example.get("label_rule", "").split(":")[0]


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
            "borderline": ex["borderline"], "rule": _rule(ex), "label": ex["label"],
            "prediction": prediction, "reason": reason, "error": error, "seconds": seconds,
        })
        mark = "MALFORMED" if prediction is None else ("ok" if prediction == ex["label"] else "WRONG")
        print(f"[{i:3d}/{len(examples)}] {ex['id']} {ex['strategy']:<22} "
              f"label={ex['label']:<6} pred={str(prediction):<6} {mark} ({seconds}s)")
    return rows


def rescore(name: str) -> None:
    """Recompute metrics for a saved run (e.g. after adding a new metric). No LLM calls."""
    path = RESULTS_DIR / f"{name}.json"
    data = json.loads(path.read_text())
    rules = {ex["id"]: _rule(ex) for ex in load_jsonl(TEST_PATH)}
    for row in data["predictions"]:
        row["rule"] = rules[row["id"]]
    data["metrics"] = critic_metrics(data["predictions"])
    path.write_text(json.dumps(data, indent=2))
    print(json.dumps(data["metrics"], indent=2))
    print(f"\nRe-scored {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Critic on the test set.")
    parser.add_argument("--limit", type=int, default=None, help="only the first N rows; not saved")
    parser.add_argument("--retrieval", action="store_true", help="add hybrid-retrieved cases")
    parser.add_argument("--name", default=None, help="results file name")
    parser.add_argument("--rescore", default=None, help="re-score a saved run by name")
    args = parser.parse_args()

    if args.rescore:
        rescore(args.rescore)
        return

    # CRITIC_MODEL set -> results are saved as the fine-tuned run, never over a baseline.
    base_name = "finetuned_ollama" if config.CRITIC_MODEL else "prompted_ollama"
    name = args.name or (base_name.replace("_ollama", "_rag_ollama") if args.retrieval else base_name)
    examples = load_jsonl(TEST_PATH)
    if args.limit:
        examples = examples[: args.limit]

    judge = critic.judge_with_llm
    if args.retrieval:
        from vector_store.retrieval import HybridRetriever, default_embedder, load_cases
        print("Building the retrieval store from the TRAIN split (first run downloads the embedding model)...")
        retriever = HybridRetriever(load_cases(), default_embedder())
        judge = lambda llm, evidence: critic.judge_with_llm(llm, evidence, retriever.retrieve(evidence))

    provider = config.LLM_PROVIDER.strip().lower()
    model = config.CRITIC_MODEL or (config.OLLAMA_MODEL if provider == "ollama" else config.LLM_MODEL)
    print(f"Evaluating {len(examples)} examples with {provider}:{model} "
          f"(retrieval={'on' if args.retrieval else 'off'})\n")

    started = time.perf_counter()
    rows = run(examples, critic._get_llm(model), judge)
    metrics = critic_metrics(rows)
    print("\n" + json.dumps(metrics, indent=2))

    if args.limit:
        print("\n--limit run: results NOT saved.")
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{name}.json"
    out_path.write_text(json.dumps({
        "condition": name,
        "provider": provider,
        "model": model,
        "retrieval": args.retrieval,
        "test_file": str(TEST_PATH),
        "total_seconds": round(time.perf_counter() - started, 1),
        "metrics": metrics,
        "predictions": rows,
    }, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
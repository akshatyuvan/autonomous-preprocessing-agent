"""
evaluation/compare.py -- is the difference between two runs on the SAME test rows real?

Environment: LOCAL (Mac).
  python -m evaluation.compare prompted_ollama prompted_rag_ollama

McNemar's exact test. Both runs judged the same 100 examples, so compare them
PAIRWISE: rows both got right (or both got wrong) say nothing about which is better.
Only the disagreements count:
  a_only = rows run A got right and run B got wrong
  b_only = rows run B got right and run A got wrong
If the runs were equally good, each disagreement is a coin flip, so
p = two-sided binomial probability of a split at least this lopsided.
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

RESULTS_DIR = Path("evaluation/results")


def _correct(row: dict) -> bool:
    # Malformed (None) counts as "accept", exactly like the pipeline and the main metrics.
    return (row["prediction"] or "accept") == row["label"]


def mcnemar(rows_a: list[dict], rows_b: list[dict]) -> dict:
    a = {r["id"]: _correct(r) for r in rows_a}
    b = {r["id"]: _correct(r) for r in rows_b}
    if a.keys() != b.keys():
        raise ValueError("runs were not on the same examples; a paired test is meaningless")
    a_only = sum(a[i] and not b[i] for i in a)
    b_only = sum(b[i] and not a[i] for i in a)
    n = a_only + b_only
    k = min(a_only, b_only)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"n_examples": len(a), "accuracy_a": round(sum(a.values()) / len(a), 3),
            "accuracy_b": round(sum(b.values()) / len(b), 3),
            "a_only_correct": a_only, "b_only_correct": b_only, "p_value": round(p, 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired McNemar test between two saved runs.")
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    args = parser.parse_args()
    rows = [json.loads((RESULTS_DIR / f"{name}.json").read_text())["predictions"]
            for name in (args.run_a, args.run_b)]
    print(json.dumps({"a": args.run_a, "b": args.run_b, **mcnemar(*rows)}, indent=2))


if __name__ == "__main__":
    main()
"""
evaluation/critic_dataset.py -- labelled Critic dataset, built WITHOUT hand-labelling.

Environment: LOCAL (Mac).   Run:  python -m evaluation.critic_dataset

For each example:
  1. Generate a CLEAN synthetic table (numpy, fixed seed).
  2. Inject ONE known corruption.
  3. Run the REAL detectors to get the DataIssue the pipeline would see.
     If they don't flag it, the example is discarded (and the discard is counted).
  4. Apply ONE strategy through the REAL dispatcher: sometimes a correct fix,
     sometimes a deliberately wrong one.
  5. Build the Critic's input with the SAME build_evidence the live Critic uses.
  6. Label it with agents/critic_policy.label_from_evidence on the MEASURED stats.

Because steps 3-5 reuse production code, the eval measures the exact task the
Critic performs inside the pipeline -- not a lookalike.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from agents.critic_policy import (
    MAX_CHANGE_SHARE, MOSTLY_MISSING_SHARE, SHARE_BAND, SKEW_BAND, SKEW_LIMIT,
    label_from_evidence,
)
from agents.detectors import PLACEHOLDER_TOKENS, run_all_detectors
from agents.dispatch import StrategyNotApplicableError, apply_strategy
from agents.evidence import build_evidence, summarize_column
from agents.profiler import findings_to_issues
from agents.registries import CLEANING_REGISTRY

SEED = 42
N_EXAMPLES = 400
TEST_EVERY = 4                      # every 4th example of the stratified order -> 100 of 400
OUT_DIR = Path("evaluation/data")
ALL_COLUMNS = "__all_columns__"
JUNK_TOKENS = ["abc", "approx", "12..5", "x7", "tbd!", "see note"]  # real junk, not placeholders
CATEGORY_SETS = [["Male", "Female"], ["Pune", "Delhi", "Mumbai", "Chennai"],
                 ["Basic", "Premium", "Enterprise"]]


@dataclass
class Case:
    scenario: str
    df: pd.DataFrame
    column: Optional[str]           # None = row-level issue (duplicates)
    issue_type: str
    strategy: str
    params: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- helpers
def _rows(rng) -> int:
    return int(rng.integers(150, 401))


def _pick(rng, options: list[str]) -> str:
    # str(): rng.choice returns numpy.str_, which we don't want leaking into JSON.
    return str(options[int(rng.integers(len(options)))])


def _frame(rng, n: int, column: str, values) -> pd.DataFrame:
    # Two distractor columns, like a real table. record_id is unique, so rows are
    # only duplicated when the duplicates scenario does it on purpose.
    return pd.DataFrame({
        "record_id": np.arange(1, n + 1),
        "segment": rng.choice(["retail", "online", "wholesale"], size=n),
        column: values,
    })


def _lognormal(rng, n: int, sigma: float) -> np.ndarray:
    # sigma controls skew: ~0.1 is nearly symmetric, 0.5 is clearly skewed, 2+ is heavy-tailed.
    return np.round(rng.lognormal(np.log(50), sigma, n), 2)


def _blank(rng, values: np.ndarray, share: float) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    k = max(1, int(round(share * len(out))))
    out[rng.choice(len(out), size=k, replace=False)] = np.nan
    return out


# ---------------------------------------------------------------- scenarios
def missing_numeric(rng) -> Case:
    n = _rows(rng)
    values = _blank(rng, _lognormal(rng, n, rng.uniform(0.1, 0.5)), rng.uniform(0.03, 0.30))
    strategy = _pick(rng, ["impute_mean", "impute_mean", "impute_median",
                           "drop_rows_missing", "drop_column", "impute_constant"])
    params = {}
    if strategy == "impute_constant":
        # Half the time 0 (outside the observed range: wrong), half the median (inside: fine).
        params["fill_value"] = 0.0 if rng.random() < 0.5 else float(np.round(np.nanmedian(values)))
    return Case("missing_numeric", _frame(rng, n, "amount", values), "amount",
                "missing_values", strategy, params)


def missing_mostly(rng) -> Case:
    n = _rows(rng)
    values = _blank(rng, _lognormal(rng, n, 0.3), rng.uniform(0.55, 0.85))
    strategy = _pick(rng, ["drop_column", "drop_column", "impute_median", "impute_mean"])
    return Case("missing_mostly", _frame(rng, n, "amount", values), "amount",
                "missing_values", strategy)


def missing_categorical(rng) -> Case:
    n = _rows(rng)
    cats = CATEGORY_SETS[int(rng.integers(len(CATEGORY_SETS)))]
    values = rng.choice(cats, size=n).astype(object)
    k = max(1, int(round(rng.uniform(0.03, 0.25) * n)))
    values[rng.choice(n, size=k, replace=False)] = None
    strategy = _pick(rng, ["impute_mode", "impute_constant", "drop_rows_missing", "drop_column"])
    params = {"fill_value": "Unknown"} if strategy == "impute_constant" else {}
    return Case("missing_categorical", _frame(rng, n, "category", values), "category",
                "missing_values", strategy, params)


def numbers_as_text(rng) -> Case:
    n = _rows(rng)
    nums = _lognormal(rng, n, rng.uniform(0.2, 0.6)) * 20
    text = [f"${v:,.2f}" if rng.random() < 0.5 else f"{v:.2f}" for v in nums]
    k = int(round(rng.uniform(0.0, 0.18) * n))  # <= 18% junk keeps the detector's 80% rule satisfied
    for i in rng.choice(n, size=k, replace=False):
        text[int(i)] = _pick(rng, JUNK_TOKENS)
    strategy = _pick(rng, ["cast_numeric", "cast_numeric", "cast_numeric", "no_action"])
    return Case("numbers_as_text", _frame(rng, n, "price", text), "price",
                "type_mismatch", strategy)


def dates_as_text(rng) -> Case:
    n = _rows(rng)
    dates = pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 1000, n), unit="D")
    text = list(dates.strftime("%Y-%m-%d"))
    k = int(round(rng.uniform(0.0, 0.15) * n))
    for i in rng.choice(n, size=k, replace=False):
        text[int(i)] = _pick(rng, ["abc", "unknown date", "tbd!"])
    strategy = _pick(rng, ["parse_datetime", "parse_datetime", "no_action"])
    return Case("dates_as_text", _frame(rng, n, "signup_date", text), "signup_date",
                "type_mismatch", strategy)


def outlier_errors(rng) -> Case:
    # A well-behaved column plus a few injected data-entry errors (x8 to x20).
    n = _rows(rng)
    values = np.round(rng.normal(50, 8, n), 2)
    m = max(2, int(round(rng.uniform(0.01, 0.03) * n)))
    idx = rng.choice(n, size=m, replace=False)
    values[idx] = np.round(values[idx] * rng.uniform(8, 20, m), 2)
    strategy = _pick(rng, ["cap_iqr", "drop_outlier_rows", "flag_outlier", "no_action"])
    return Case("outlier_errors", _frame(rng, n, "reading", values), "reading",
                "outlier", strategy)


def heavy_tail(rng) -> Case:
    # NO injected errors: the "outliers" are a genuine heavy tail. Depending on sigma,
    # capping touches ~5-15% of values, i.e. right around the 10% policy threshold.
    n = _rows(rng)
    values = _lognormal(rng, n, rng.uniform(0.6, 2.5))
    strategy = _pick(rng, ["cap_iqr", "cap_iqr", "drop_outlier_rows", "flag_outlier"])
    return Case("heavy_tail", _frame(rng, n, "income", values), "income", "outlier", strategy)


def categories(rng) -> Case:
    n = _rows(rng)
    cats = CATEGORY_SETS[int(rng.integers(len(CATEGORY_SETS)))]
    values = rng.choice(cats, size=n).astype(object)
    k = max(1, int(round(rng.uniform(0.03, 0.30) * n)))
    for i in rng.choice(n, size=k, replace=False):
        v = values[int(i)]
        values[int(i)] = _pick(rng, [v.lower(), v.upper(), v + " ", " " + v])
    strategy = _pick(rng, ["standardize_category", "standardize_category", "drop_column", "no_action"])
    return Case("categories", _frame(rng, n, "category", values), "category",
                "inconsistent_category", strategy)


def duplicates(rng) -> Case:
    n = _rows(rng)
    df = _frame(rng, n, "amount", _lognormal(rng, n, 0.3))
    d = max(1, int(round(rng.uniform(0.02, 0.20) * n)))
    df = pd.concat([df, df.iloc[rng.choice(n, size=d, replace=False)]], ignore_index=True)
    df = df.sample(frac=1, random_state=int(rng.integers(1_000_000_000))).reset_index(drop=True)
    strategy = _pick(rng, ["drop_duplicates", "drop_duplicates", "no_action"])
    return Case("duplicates", df, None, "duplicate_rows", strategy)


SCENARIOS = [
    (missing_numeric, 3.0), (missing_mostly, 1.0), (missing_categorical, 1.5),
    (numbers_as_text, 2.0), (dates_as_text, 1.0), (outlier_errors, 1.5),
    (heavy_tail, 2.0), (categories, 1.0), (duplicates, 1.0),
]


# ---------------------------------------------------------------- building
def make_example(case: Case, example_id: str) -> tuple[Optional[dict], str]:
    """Returns (example, "") or (None, discard_reason)."""
    issue_column = case.column if case.column is not None else ALL_COLUMNS
    issues = findings_to_issues(run_all_detectors(case.df))
    issue = next((i for i in issues
                  if i["issue_type"] == case.issue_type and i["column"] == issue_column), None)
    if issue is None:
        return None, "not_detected"

    params = {"placeholder_tokens": sorted(PLACEHOLDER_TOKENS), **case.params}
    try:
        new_df, stats = apply_strategy(CLEANING_REGISTRY, case.strategy, case.df, case.column, params)
    except StrategyNotApplicableError:
        return None, "not_applicable"

    # Exactly the decision shape cleaner_node produces.
    decision = {
        "action": case.strategy,
        "column": issue_column,
        "stats": {**stats, "before": summarize_column(case.df, case.column),
                  "after": summarize_column(new_df, case.column)},
    }
    evidence = build_evidence(decision, issue, CLEANING_REGISTRY[case.strategy]["description"])
    label, rule, borderline = label_from_evidence(evidence)
    return {
        "id": example_id,
        "scenario": case.scenario,
        "issue_type": case.issue_type,
        "strategy": case.strategy,
        "label": label,
        "borderline": borderline,
        "label_rule": rule,       # for analysis only; NEVER shown to the model
        "evidence": evidence,     # the ONLY thing the model sees (plus the system prompt)
    }, ""


def build_dataset(n: int = N_EXAMPLES, seed: int = SEED) -> tuple[list[dict], Counter]:
    rng = np.random.default_rng(seed)  # one generator, so the whole dataset is reproducible
    fns = [fn for fn, _ in SCENARIOS]
    weights = np.array([w for _, w in SCENARIOS])
    weights = weights / weights.sum()
    examples: list[dict] = []
    discards: Counter = Counter()
    attempts = 0
    while len(examples) < n:
        attempts += 1
        if attempts > n * 5:
            raise RuntimeError(f"too many discards: {dict(discards)}")
        case = fns[int(rng.choice(len(fns), p=weights))](rng)
        example, reason = make_example(case, f"ex_{len(examples) + 1:04d}")
        if example is None:
            discards[f"{case.scenario}:{reason}"] += 1
            continue
        examples.append(example)
    return examples, discards


def split(examples: list[dict], seed: int = SEED) -> tuple[list[dict], list[dict]]:
    """Stratified split: shuffle, then stable-sort by (scenario, label) and send every
    TEST_EVERY-th example to test, so the test set keeps the same mix as the whole."""
    rng = np.random.default_rng(seed + 1)
    order = [int(i) for i in rng.permutation(len(examples))]
    order = sorted(order, key=lambda i: (examples[i]["scenario"], examples[i]["label"]))
    test_ids = {examples[i]["id"] for i in order[::TEST_EVERY]}
    train = [e for e in examples if e["id"] not in test_ids]
    test = [e for e in examples if e["id"] in test_ids]
    return train, test


def _counts(rows: list[dict]) -> dict[str, Any]:
    return {
        "total": len(rows),
        "label": dict(Counter(e["label"] for e in rows)),
        "borderline": sum(e["borderline"] for e in rows),
        "scenario": dict(sorted(Counter(e["scenario"] for e in rows).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the labelled Critic dataset.")
    parser.add_argument("--n", type=int, default=N_EXAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    examples, discards = build_dataset(args.n, args.seed)
    train, test = split(examples, args.seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in (("critic_train.jsonl", train), ("critic_test.jsonl", test)):
        with open(OUT_DIR / name, "w") as f:
            for row in rows:
                f.write(json.dumps(row, default=str) + "\n")

    manifest = {
        "seed": args.seed,
        "n_examples": len(examples),
        "train": _counts(train),
        "test": _counts(test),
        "discarded": dict(sorted(discards.items())),
        "policy": {"skew_limit": SKEW_LIMIT, "max_change_share": MAX_CHANGE_SHARE,
                   "mostly_missing_share": MOSTLY_MISSING_SHARE,
                   "skew_band": SKEW_BAND, "share_band": SHARE_BAND},
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
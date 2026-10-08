"""
tests/test_evidence.py -- the Critic's evidence. Environment: LOCAL (Mac).
"""
import pandas as pd

from agents.cleaner import cleaner_node
from agents.evidence import build_evidence, summarize_column
from tests.test_cleaner import _profiled_state, _write_messy_csv


def test_summarize_numeric_column():
    df = pd.DataFrame({"age": [1.0, 2.0, 3.0, None]})
    s = summarize_column(df, "age")
    assert s["present"] is True
    assert s["missing"] == 1
    assert s["mean"] == 2.0 and s["median"] == 2.0


def test_summarize_rows_and_missing_column():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    assert summarize_column(df, None)["duplicate_rows"] == 1
    assert summarize_column(df, "nope") == {"present": False, "rows": 3}


def test_cleaner_records_before_and_after(tmp_path):
    csv = tmp_path / "messy.csv"
    _write_messy_csv(csv)
    decisions = cleaner_node(_profiled_state(csv))["cleaner"]["decisions"]
    assert all("before" in d["stats"] and "after" in d["stats"] for d in decisions)
    dup = decisions[0]  # drop_duplicates runs first
    assert dup["stats"]["before"]["duplicate_rows"] == 1
    assert dup["stats"]["after"]["duplicate_rows"] == 0


def test_evidence_excludes_justification():
    decision = {"action": "impute_median", "column": "age", "justification": "trust me",
                "stats": {"strategy": "impute_median", "column": "age", "values_imputed": 2,
                          "before": {"missing": 2}, "after": {"missing": 0}}}
    evidence = build_evidence(decision, None, "Fills missing values with the column median.")
    assert "justification" not in evidence
    assert "trust me" not in str(evidence)
    assert evidence["change"] == {"values_imputed": 2}
    assert evidence["before"] == {"missing": 2}
"""
tests/test_cleaner.py -- the Cleaning agent node. LLM always faked (no model needed).

Environment: LOCAL (Mac).
"""
import json

import pandas as pd

from agents import cleaner as cleaner_mod
from agents.cleaner import StrategyChoice, cleaner_node, fill_value_for
from agents.profiler import profiler_node
from state.schema import make_initial_state

# Priority order: duplicates, types, categories, outliers, then missing (age, price).
EXPECTED_ACTIONS = ["drop_duplicates", "cast_numeric", "standardize_category",
                    "cap_iqr", "impute_median", "impute_median"]


def _write_messy_csv(path):
    df = pd.DataFrame({
        "age": [25, 30, None, 40, 35, 28, 33, 31, 29, 1000, 27, 26],      # missing + outlier
        "city": ["Pune", "pune ", "Pune", "Delhi", "delhi", "Delhi",
                 "Pune", "Delhi", "Pune", "Delhi", "Pune", "Pune"],         # spelling variants
        "price": ["$1,200", "300", "N/A", "450", "500", "700",
                  "650", "800", "900", "1000", "1100", "1050"],             # numbers as text + missing
    })
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)                  # one exact duplicate
    df.to_csv(path, index=False)


def _profiled_state(csv_path):
    state = make_initial_state(dataset_path=str(csv_path), dataset_name="test")
    state["profiler"] = profiler_node(state)["profiler"]  # conftest fakes the Profiler LLM
    return state


class _FixedChoiceLLM:
    def __init__(self, strategy):
        self.strategy = strategy
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return StrategyChoice(strategy=self.strategy, fill_value="", justification="test")


class _BrokenLLM:
    def invoke(self, messages):
        raise RuntimeError("connection refused")


class _RecordingLLM:
    def __init__(self):
        self.payloads = []

    def invoke(self, messages):
        payload = json.loads(messages[-1][1])
        self.payloads.append(payload)
        return StrategyChoice(strategy=payload["allowed_strategies"][0]["key"],
                              fill_value="", justification="test")


def _use_llm(monkeypatch, llm):
    monkeypatch.setattr(cleaner_mod, "_get_llm", lambda *args, **kwargs: llm)


def test_cleaner_fixes_every_issue_in_priority_order(tmp_path):
    csv = tmp_path / "messy.csv"
    _write_messy_csv(csv)
    state = _profiled_state(csv)
    assert {i["issue_type"] for i in state["profiler"]["issues"]} == {
        "duplicate_rows", "type_mismatch", "inconsistent_category", "outlier", "missing_values"}

    out = cleaner_node(state)
    decisions = out["cleaner"]["decisions"]
    assert [d["action"] for d in decisions] == EXPECTED_ACTIONS
    assert all(d["chosen_by"] == "llm" and d["status"] == "applied" for d in decisions)
    assert out["errors"] == []

    cleaned = pd.read_parquet(out["cleaner"]["dataset_snapshot_path"])
    assert len(cleaned) == 12
    assert cleaned.duplicated().sum() == 0
    assert cleaned["age"].isna().sum() == 0
    assert cleaned["age"].max() < 1000
    assert pd.api.types.is_numeric_dtype(cleaned["price"])
    assert cleaned["price"].isna().sum() == 0
    assert set(cleaned["city"]) == {"Pune", "Delhi"}


def test_invalid_strategy_name_falls_back_deterministically(tmp_path, monkeypatch):
    csv = tmp_path / "messy.csv"
    _write_messy_csv(csv)
    _use_llm(monkeypatch, _FixedChoiceLLM("teleport"))
    out = cleaner_node(_profiled_state(csv))
    decisions = out["cleaner"]["decisions"]
    assert [d["action"] for d in decisions] == EXPECTED_ACTIONS
    assert all(d["chosen_by"] == "fallback" for d in decisions)
    assert all(d["error"].startswith("invalid_choice") for d in decisions)
    assert len(out["errors"]) == 1


def test_valid_but_unfitting_choice_falls_back(tmp_path, monkeypatch):
    csv = tmp_path / "text_missing.csv"
    pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8],
        "city": ["Pune", "Delhi", None, "Pune", "Delhi", "Pune", "Pune", "Delhi"],
    }).to_csv(csv, index=False)
    _use_llm(monkeypatch, _FixedChoiceLLM("impute_median"))  # a real key, wrong for text
    out = cleaner_node(_profiled_state(csv))
    [decision] = out["cleaner"]["decisions"]
    assert decision["action"] == "impute_mode"
    assert decision["chosen_by"] == "fallback"
    assert decision["error"].startswith("not_applicable")


def test_llm_outage_still_cleans(tmp_path, monkeypatch):
    csv = tmp_path / "messy.csv"
    _write_messy_csv(csv)
    _use_llm(monkeypatch, _BrokenLLM())
    out = cleaner_node(_profiled_state(csv))
    assert [d["action"] for d in out["cleaner"]["decisions"]] == EXPECTED_ACTIONS
    assert len(out["errors"]) == 1
    assert "llm_error" in out["errors"][0]["message"]


def test_raw_file_is_never_modified(tmp_path):
    csv = tmp_path / "messy.csv"
    _write_messy_csv(csv)
    before = csv.read_bytes()
    cleaner_node(_profiled_state(csv))
    assert csv.read_bytes() == before


def test_redo_appends_history_and_passes_critic_feedback(tmp_path, monkeypatch):
    csv = tmp_path / "messy.csv"
    _write_messy_csv(csv)
    llm = _RecordingLLM()
    _use_llm(monkeypatch, llm)
    state = _profiled_state(csv)

    first = cleaner_node(state)
    state["cleaner"] = first["cleaner"]
    state["metadata"] = first["metadata"]
    state["critic"]["current_verdict"] = {
        "verdict_id": "v1", "decision_ids_reviewed": [], "verdict": "reject",
        "reasoning": "median was wrong", "distribution_ok": False,
        "model_score_delta": None, "confidence": 0.9,
    }
    second = cleaner_node(state)

    decisions = second["cleaner"]["decisions"]
    assert len(decisions) == 12  # 6 from round 1 carried forward + 6 new
    assert len({d["decision_id"] for d in decisions}) == 12
    assert all(d["redo_count"] == 1 for d in decisions[6:])
    assert second["cleaner"]["current_round"] == 2
    assert second["cleaner"]["dataset_snapshot_path"].endswith("_r2.parquet")
    round_two = llm.payloads[6:]
    assert all(p["previous_attempt"]["critic_feedback"] == "median was wrong" for p in round_two)
    assert round_two[0]["previous_attempt"]["strategy"] == "drop_duplicates"


def test_strategy_name_formatting_is_repaired():
    choice = StrategyChoice(strategy=' "Impute_Median" ', fill_value=None, justification=None)
    assert choice.strategy == "impute_median"
    assert choice.fill_value == ""


def test_fill_value_is_coerced_to_the_column_type():
    assert fill_value_for(pd.Series([1.0, None]), "0") == 0.0
    assert fill_value_for(pd.Series([1.0, None]), "Unknown") is None
    assert fill_value_for(pd.Series(["a", None]), "Unknown") == "Unknown"
    assert fill_value_for(pd.Series([1.0]), "") is None


def test_cleaner_skips_when_profiler_failed(tmp_path, monkeypatch):
    llm = _FixedChoiceLLM("no_action")
    _use_llm(monkeypatch, llm)
    state = make_initial_state(dataset_path=str(tmp_path / "missing.csv"), dataset_name="x")
    out = cleaner_node(state)  # profiler never ran: run_complete is False
    assert out["cleaner"]["decisions"] == []
    assert out["cleaner"]["dataset_snapshot_path"] is None
    assert llm.calls == 0
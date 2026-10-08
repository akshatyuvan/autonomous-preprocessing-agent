"""
tests/test_profiler.py — Profiler Agent (Day 2).

Environment: LOCAL (Mac). No API key needed: the LLM is always faked.
"""
import json

import numpy as np
import pandas as pd
import pytest

import agents.profiler as profiler
from agents.detectors import (
    compute_column_stats, detect_duplicates, detect_inconsistent_categories,
    detect_missing, detect_outliers, detect_type_mismatch, outlier_mask, run_all_detectors,
)
from agents.profiler import (
    MAX_ROWS_PER_EVENT, ProfileSynthesis, findings_to_issues, load_dataset, profiler_node,
)
from state.schema import make_initial_state

SCHEMA_ISSUE_TYPES = {
    "missing_values", "type_mismatch", "outlier",
    "duplicate_rows", "inconsistent_category", "unknown",
}


class FakeLLM:
    def __init__(self, summary="LLM summary", top=None, raise_exc=None):
        self.summary, self.top, self.raise_exc = summary, top or [], raise_exc
        self.calls, self.last_messages = 0, None

    def invoke(self, messages):
        self.calls += 1
        self.last_messages = messages
        if self.raise_exc is not None:
            raise self.raise_exc
        return ProfileSynthesis(summary=self.summary, top_concerns=self.top)


def use_llm(monkeypatch, llm):
    monkeypatch.setattr(profiler, "_get_llm", lambda *a, **k: llm)
    return llm


@pytest.fixture
def messy_df():
    """One of each of the five issue types, planted deliberately."""
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({
        "age": rng.integers(20, 60, n).astype(float),
        "gender": rng.choice(["Male", "Female"], n),
        "spend": [f"${v:,}" for v in rng.integers(100, 5000, n)],
    })
    df.loc[[1, 2, 3], "age"] = np.nan                      # missing_values
    df.loc[10, "age"] = 250.0                               # outlier
    df.loc[[20, 21], "gender"] = ["male ", "MALE"]          # inconsistent_category
    return pd.concat([df, df.iloc[[0, 5]]], ignore_index=True)  # duplicate_rows
    # "spend" is numbers stored as text -> type_mismatch


def write_csv(tmp_path, df, name="data.csv"):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path)


def run_node(tmp_path, df):
    return profiler_node(make_initial_state(dataset_path=write_csv(tmp_path, df), dataset_name="t"))


# ---------------- missing values ----------------
def test_missing_counts_nulls_and_positions():
    [f] = detect_missing(pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan]}))
    assert f.issue_type == "missing_values"
    assert f.affected_rows == 2
    assert f.rows == [1, 3]


@pytest.mark.parametrize("n_missing,expected", [(2, "low"), (10, "medium"), (40, "high")])
def test_missing_severity_thresholds(n_missing, expected):
    values = [1.0] * 100
    values[:n_missing] = [np.nan] * n_missing
    [f] = detect_missing(pd.DataFrame({"a": values}))
    assert f.severity == expected


def test_missing_counts_placeholder_tokens():
    [f] = detect_missing(pd.DataFrame({"city": ["Pune", "N/A", "?", " ", "Mumbai"]}))
    assert f.affected_rows == 3
    assert "placeholder" in f.detail


def test_missing_clean_column_has_no_finding():
    assert detect_missing(pd.DataFrame({"a": [1, 2, 3]})) == []


# ---------------- type mismatch ----------------
def test_type_mismatch_numeric_with_symbols():
    [f] = detect_type_mismatch(pd.DataFrame({"price": ["$1,200", "$350", "$90", "$4,000", "$15"]}))
    assert f.issue_type == "type_mismatch"
    assert f.severity == "low"


def test_type_mismatch_mostly_numeric_with_bad_values():
    [f] = detect_type_mismatch(pd.DataFrame({"qty": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "ten"]}))
    assert f.severity == "medium"
    assert "cannot be parsed" in f.detail


def test_type_mismatch_dates_as_text():
    [f] = detect_type_mismatch(pd.DataFrame({"signup": ["2024-01-05", "2024-02-11", "2024-03-20", "2024-04-02"]}))
    assert "date" in f.detail


def test_free_text_not_type_mismatch():
    df = pd.DataFrame({"note": ["great service", "slow delivery", "ok", "would buy again"]})
    assert detect_type_mismatch(df) == []


def test_true_numeric_column_not_type_mismatch():
    assert detect_type_mismatch(pd.DataFrame({"a": [1.0, 2.0, 3.0]})) == []


# ---------------- outliers ----------------
def test_outlier_mask_iqr_and_modified_z_both_flag_extreme():
    x = pd.Series([float(v) for v in range(10, 30)] + [1000.0])
    mask, methods = outlier_mask(x)
    assert int(mask.sum()) == 1 and bool(mask.iloc[-1])
    assert "IQR" in methods and "modified Z-score" in methods


def test_iqr_zero_falls_back_to_mode_comparison():
    mask, methods = outlier_mask(pd.Series([5.0] * 95 + [500.0] * 5))
    assert int(mask.sum()) == 5
    assert "mode-comparison" in methods


def test_iqr_zero_with_common_minority_is_not_flagged():
    # 20% non-mode values is a distribution, not outliers.
    mask, _ = outlier_mask(pd.Series([0.0] * 80 + [1.0] * 10 + [2.0] * 10))
    assert int(mask.sum()) == 0


def test_bimodal_distribution_not_flagged():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"v": np.concatenate([rng.normal(10, 1, 100), rng.normal(100, 1, 100)])})
    assert detect_outliers(df) == []


def test_binary_numeric_column_skipped():
    assert detect_outliers(pd.DataFrame({"flag": [0] * 30 + [1] * 10})) == []


def test_short_numeric_column_skipped():
    assert detect_outliers(pd.DataFrame({"v": [1.0, 2.0, 3.0, 1000.0]})) == []


def test_detect_outliers_reports_row_position():
    [f] = detect_outliers(pd.DataFrame({"v": [float(v) for v in range(10, 30)] + [1000.0]}))
    assert f.issue_type == "outlier"
    assert f.rows == [20]


# ---------------- duplicates ----------------
def test_duplicates_detected():
    [f] = detect_duplicates(pd.DataFrame({"a": [1, 2, 1, 3, 1], "b": ["x", "y", "x", "z", "x"]}))
    assert f.issue_type == "duplicate_rows"
    assert f.affected_rows == 2
    assert f.rows == [2, 4]


def test_no_duplicates():
    assert detect_duplicates(pd.DataFrame({"a": [1, 2, 3]})) == []


# ---------------- inconsistent categories ----------------
def test_inconsistent_categories_detected():
    df = pd.DataFrame({"gender": ["Male"] * 6 + ["male ", "MALE", "Female", "Female"]})
    [f] = detect_inconsistent_categories(df)
    assert f.issue_type == "inconsistent_category"
    assert f.affected_rows == 2
    assert f.rows == [6, 7]


def test_consistent_categories_not_flagged():
    assert detect_inconsistent_categories(pd.DataFrame({"g": ["a", "b", "a", "c"]})) == []


def test_high_cardinality_text_skipped():
    df = pd.DataFrame({"note": [f"Comment {i}" for i in range(60)] + ["comment 0"]})
    assert detect_inconsistent_categories(df) == []


def test_numeric_like_text_skipped_by_category_detector():
    assert detect_inconsistent_categories(pd.DataFrame({"p": ["1", " 1", "2", "3"]})) == []


# ---------------- column stats ----------------
def test_column_stats_contents():
    st = compute_column_stats(pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": ["x", "y", "x"]}))
    assert st["a"]["null_count"] == 1
    assert st["a"]["null_pct"] == round(1 / 3, 4)
    assert "mean" in st["a"]
    assert "top_values" in st["b"]


def test_column_stats_json_serialisable():
    json.dumps(compute_column_stats(pd.DataFrame({"a": [np.nan, np.nan], "b": [1.0, np.nan]})))


# ---------------- runner ----------------
def test_run_all_detectors_finds_all_five_types(messy_df):
    types = {f.issue_type for f in run_all_detectors(messy_df)}
    assert types == {"missing_values", "type_mismatch", "outlier", "duplicate_rows", "inconsistent_category"}


def test_issue_ids_sequential_and_types_valid(messy_df):
    issues = findings_to_issues(run_all_detectors(messy_df))
    assert [i["issue_id"] for i in issues] == [f"issue_{k:03d}" for k in range(1, len(issues) + 1)]
    assert all(i["issue_type"] in SCHEMA_ISSUE_TYPES for i in issues)


def test_detectors_do_not_mutate_input(messy_df):
    snapshot = messy_df.copy()
    run_all_detectors(messy_df)
    pd.testing.assert_frame_equal(messy_df, snapshot)


# ---------------- loading ----------------
def test_load_dataset_csv(tmp_path):
    assert load_dataset(write_csv(tmp_path, pd.DataFrame({"a": [1, 2]}))).shape == (2, 1)


def test_load_dataset_rejects_unknown_extension(tmp_path):
    path = tmp_path / "x.xlsx"
    path.write_text("x")
    with pytest.raises(ValueError):
        load_dataset(str(path))


# ---------------- node ----------------
def test_node_returns_partial_state_only(monkeypatch, tmp_path, messy_df):
    use_llm(monkeypatch, FakeLLM())
    assert set(run_node(tmp_path, messy_df)) == {"profiler", "visualization_events", "errors"}


def test_node_counts_rows_and_columns(monkeypatch, tmp_path, messy_df):
    use_llm(monkeypatch, FakeLLM())
    p = run_node(tmp_path, messy_df)["profiler"]
    assert p["run_complete"] is True
    assert p["row_count"] == len(messy_df)
    assert p["column_count"] == 3


def test_node_populates_issues(monkeypatch, tmp_path, messy_df):
    use_llm(monkeypatch, FakeLLM())
    assert len(run_node(tmp_path, messy_df)["profiler"]["issues"]) >= 5


def test_node_emits_one_flag_event_per_issue_plus_round_complete(monkeypatch, tmp_path, messy_df):
    use_llm(monkeypatch, FakeLLM())
    result = run_node(tmp_path, messy_df)
    events = result["visualization_events"]
    flags = [e for e in events if e["event_type"] == "cell_status_change"]
    assert len(flags) == len(result["profiler"]["issues"])
    assert all(e["payload"]["status"] == "flagged" for e in flags)
    assert events[-1]["event_type"] == "round_complete"


def test_viz_rows_are_capped(monkeypatch, tmp_path):
    use_llm(monkeypatch, FakeLLM())
    df = pd.DataFrame({"a": [np.nan] * (MAX_ROWS_PER_EVENT + 50) + [1.0]})
    flag = run_node(tmp_path, df)["visualization_events"][0]
    assert len(flag["payload"]["rows"]) == MAX_ROWS_PER_EVENT
    assert flag["payload"]["rows_truncated"] is True


def test_llm_summary_used_and_bogus_ids_dropped(monkeypatch, tmp_path, messy_df):
    llm = use_llm(monkeypatch, FakeLLM(summary="Custom LLM summary", top=["issue_001", "bogus"]))
    p = run_node(tmp_path, messy_df)["profiler"]
    assert p["profile_summary"].startswith("Custom LLM summary")
    assert "issue_001" in p["profile_summary"]
    assert "bogus" not in p["profile_summary"]
    assert llm.calls == 1


def test_llm_prompt_has_issues_but_no_raw_top_values(monkeypatch, tmp_path, messy_df):
    llm = use_llm(monkeypatch, FakeLLM())
    run_node(tmp_path, messy_df)
    content = llm.last_messages[-1][1]
    assert "issue_001" in content
    assert "top_values" not in content


def test_llm_failure_falls_back_with_recoverable_error(monkeypatch, tmp_path, messy_df):
    use_llm(monkeypatch, FakeLLM(raise_exc=RuntimeError("timeout")))
    result = run_node(tmp_path, messy_df)
    assert "issue(s)" in result["profiler"]["profile_summary"]
    assert result["errors"][0]["recoverable"] is True


def test_llm_construction_failure_falls_back(monkeypatch, tmp_path, messy_df):
    def no_key(*args, **kwargs):
        raise RuntimeError("OPENAI_API_KEY missing")
    monkeypatch.setattr(profiler, "_get_llm", no_key)
    result = run_node(tmp_path, messy_df)
    assert result["profiler"]["run_complete"] is True
    assert result["errors"][0]["error_type"] == "llm_error"


def test_clean_dataset_skips_llm(monkeypatch, tmp_path):
    llm = use_llm(monkeypatch, FakeLLM())
    p = run_node(tmp_path, pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]}))["profiler"]
    assert p["issues"] == []
    assert llm.calls == 0
    assert "No data-quality issues" in p["profile_summary"]


def test_missing_file_records_non_recoverable_error(tmp_path):
    state = make_initial_state(dataset_path=str(tmp_path / "nope.csv"), dataset_name="t")
    result = profiler_node(state)
    assert result["profiler"]["run_complete"] is False
    assert result["errors"][0]["recoverable"] is False

# ---------------- structured-output robustness (small local models) ----------------
def test_synthesis_accepts_list_returned_as_json_string():
    s = ProfileSynthesis(summary="x", top_concerns='["issue_002", "issue_001"]')
    assert s.top_concerns == ["issue_002", "issue_001"]


def test_synthesis_extracts_ids_from_plain_text_string():
    s = ProfileSynthesis(summary="x", top_concerns="issue_003, then issue_001")
    assert s.top_concerns == ["issue_003", "issue_001"]
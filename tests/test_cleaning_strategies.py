"""
tests/test_cleaning_strategies.py -- the cleaning catalogue and the generic dispatcher.
Pure pandas, no LLM. Environment: LOCAL (Mac).
"""
import pandas as pd
import pytest

from agents.dispatch import (
    StrategyNotApplicableError,
    UnknownStrategyError,
    apply_strategy,
    options_for,
)
from agents.registries import CLEANING_REGISTRY

DETECTOR_ISSUE_TYPES = [
    "missing_values", "type_mismatch", "outlier", "duplicate_rows", "inconsistent_category",
]


def make_messy_df() -> pd.DataFrame:
    # Small enough to check by hand. Each column has deliberately planted problems.
    return pd.DataFrame({
        "age": [25.0, 30.0, None, 40.0, 35.0, 1000.0, 28.0, 33.0],           # 1 missing, 1 outlier
        "income_text": ["50", "60", "N/A", "70", "abc", "80", "90", "100"],   # numbers as text
        "city": ["Pune", "pune ", "Pune", "Delhi", "delhi", None, "Delhi", "Pune"],
        "joined": ["2024-01-05", "2024-02-10", "not a date", "2024-03-01",
                   "2024-04-12", "2024-05-20", "2024-06-30", "2024-07-15"],
    })


def run(strategy, column, params=None, df=None):
    frame = df if df is not None else make_messy_df()
    return apply_strategy(CLEANING_REGISTRY, strategy, frame, column, params)


# One working call per strategy. The set-equality assert below forces anyone who
# adds a registry entry to add a case here too, so no technique goes untested.
NO_MUTATION_CASES = {
    "drop_duplicates": (None, {}),
    "impute_mean": ("age", {}),
    "impute_median": ("age", {}),
    "impute_mode": ("city", {}),
    "impute_constant": ("city", {"fill_value": "Unknown"}),
    "drop_rows_missing": ("age", {}),
    "drop_column": ("joined", {}),
    "cast_numeric": ("income_text", {"placeholder_tokens": ["N/A"]}),
    "parse_datetime": ("joined", {}),
    "standardize_category": ("city", {}),
    "cap_iqr": ("age", {}),
    "drop_outlier_rows": ("age", {}),
    "flag_outlier": ("age", {}),
    "no_action": ("age", {}),
}


def test_registry_entries_are_complete():
    required = {"name", "handles", "applies_when", "description", "apply"}
    for key, entry in CLEANING_REGISTRY.items():
        assert required <= entry.keys(), key
        assert callable(entry["apply"]), key
        assert entry["handles"], key


def test_every_detector_issue_type_has_a_real_strategy():
    for issue_type in DETECTOR_ISSUE_TYPES:
        real = [k for k in options_for(CLEANING_REGISTRY, issue_type) if k != "no_action"]
        assert real, f"no strategy handles {issue_type}"
        assert "no_action" in options_for(CLEANING_REGISTRY, issue_type)


def test_unknown_strategy_is_rejected():
    with pytest.raises(UnknownStrategyError, match="teleport"):
        run("teleport", "age")


def test_missing_column_is_rejected():
    with pytest.raises(StrategyNotApplicableError, match="salary"):
        run("impute_mean", "salary")


def test_every_strategy_leaves_the_input_dataframe_untouched():
    assert set(NO_MUTATION_CASES) == set(CLEANING_REGISTRY), "add a case for every new strategy"
    for strategy, (column, params) in NO_MUTATION_CASES.items():
        df = make_messy_df()
        snapshot = df.copy(deep=True)
        new_df, _ = apply_strategy(CLEANING_REGISTRY, strategy, df, column, params)
        pd.testing.assert_frame_equal(df, snapshot, obj=f"input after {strategy}")
        assert new_df is not df, strategy


def test_impute_mean_fills_with_column_mean():
    out, stats = run("impute_mean", "age")
    expected = (25 + 30 + 40 + 35 + 1000 + 28 + 33) / 7
    assert out.loc[2, "age"] == pytest.approx(expected)
    assert stats["values_imputed"] == 1


def test_impute_median_refuses_text_column():
    with pytest.raises(StrategyNotApplicableError, match="numeric"):
        run("impute_median", "income_text")


def test_impute_mode_uses_most_frequent_value():
    out, stats = run("impute_mode", "city")
    assert out.loc[5, "city"] == "Pune"  # "Pune" x3 beats "Delhi" x2
    assert stats["values_imputed"] == 1


def test_impute_constant_requires_fill_value():
    with pytest.raises(StrategyNotApplicableError, match="fill_value"):
        run("impute_constant", "city")
    out, _ = run("impute_constant", "city", {"fill_value": "Unknown"})
    assert out.loc[5, "city"] == "Unknown"


def test_drop_rows_missing_resets_index():
    out, stats = run("drop_rows_missing", "age")
    assert list(out.index) == list(range(7))
    assert stats["rows_before"] == 8 and stats["rows_after"] == 7


def test_drop_duplicates_keeps_first():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    out, stats = run("drop_duplicates", None, df=df)
    assert len(out) == 2
    assert list(out.index) == [0, 1]
    assert stats["duplicates_removed"] == 1


def test_cast_numeric_blanks_placeholders_and_counts_coercions():
    out, stats = run("cast_numeric", "income_text", {"placeholder_tokens": ["N/A"]})
    assert pd.api.types.is_numeric_dtype(out["income_text"])
    assert out.loc[0, "income_text"] == 50
    assert out["income_text"].isna().sum() == 2  # "N/A" and "abc"
    assert stats["placeholders_blanked"] == 1
    assert stats["coerced_to_missing"] == 1      # only "abc" was real data loss


def test_parse_datetime_converts_text_dates():
    out, stats = run("parse_datetime", "joined")
    assert pd.api.types.is_datetime64_any_dtype(out["joined"])
    assert out.loc[0, "joined"] == pd.Timestamp("2024-01-05")
    assert stats["coerced_to_missing"] == 1


def test_standardize_category_picks_most_frequent_spelling():
    out, stats = run("standardize_category", "city")
    assert out.loc[1, "city"] == "Pune"
    assert out.loc[4, "city"] == "Delhi"
    assert pd.isna(out.loc[5, "city"])  # missing stays missing
    assert stats["values_changed"] == 2
    assert stats["categories_before"] == 4 and stats["categories_after"] == 2


def test_cap_iqr_clips_to_tukey_fences():
    # sorted non-missing: 25 28 30 33 35 40 1000 -> Q1=29, Q3=37.5, IQR=8.5
    # upper fence = 37.5 + 1.5*8.5 = 50.25
    out, stats = run("cap_iqr", "age")
    assert out.loc[5, "age"] == pytest.approx(50.25)
    assert out.loc[0, "age"] == 25
    assert pd.isna(out.loc[2, "age"])
    assert stats["values_capped"] == 1


def test_cap_iqr_refuses_zero_iqr():
    df = pd.DataFrame({"x": [5, 5, 5, 5, 5, 99]})
    with pytest.raises(StrategyNotApplicableError, match="IQR is 0"):
        run("cap_iqr", "x", df=df)


def test_drop_outlier_rows_removes_only_extremes():
    out, stats = run("drop_outlier_rows", "age")
    assert len(out) == 7
    assert 1000.0 not in out["age"].tolist()
    assert out["age"].isna().sum() == 1  # the missing row is NOT an outlier
    assert stats["outlier_rows_dropped"] == 1


def test_flag_outlier_adds_column_without_changing_values():
    out, stats = run("flag_outlier", "age")
    assert out["age_is_outlier"].sum() == 1
    assert out.loc[5, "age"] == 1000.0
    assert stats["values_flagged"] == 1


def test_dispatcher_reports_shape_changes():
    out, stats = run("drop_column", "joined")
    assert "joined" not in out.columns
    assert stats["columns_before"] == 4 and stats["columns_after"] == 3
    assert stats["rows_before"] == stats["rows_after"] == 8


def test_cast_numeric_strips_currency_and_thousands():
    df = pd.DataFrame({"p": ["$1,200", "₹300", "45", "7"]})
    out, stats = run("cast_numeric", "p", df=df)
    assert out["p"].tolist() == [1200, 300, 45, 7]
    assert stats["coerced_to_missing"] == 0


def test_cast_numeric_refuses_to_destroy_a_date_column():
    with pytest.raises(StrategyNotApplicableError, match="wrong conversion"):
        run("cast_numeric", "joined")


def test_parse_datetime_refuses_plain_words():
    df = pd.DataFrame({"t": ["red", "blue", "green", "red"]})
    with pytest.raises(StrategyNotApplicableError, match="wrong conversion"):
        run("parse_datetime", "t", df=df)
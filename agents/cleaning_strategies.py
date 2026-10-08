"""
agents/cleaning_strategies.py -- implementations of the cleaning technique catalogue.

Pure functions: no LLM, no graph state. Every function has the same shape:
    fn(df, column, params) -> (new_df, stats)
  df      current dataset. NEVER modified (locked decision: copy first).
  column  the column the DataIssue is about (row-level strategies ignore it).
  params  strategy-specific knobs, e.g. {"fill_value": "Unknown"} or
          {"placeholder_tokens": ["N/A", "?"]}.
  stats   small JSON-friendly facts about what changed; the Critic reads them.

The identical signature is what lets agents/dispatch.py call ANY strategy the
same way, with no if/elif per technique.

Strategies raise StrategyNotApplicableError on the wrong kind of column. Failing
loudly beats quietly producing garbage that the next stage builds on.
"""
from __future__ import annotations

import re
from typing import Any, Callable

import pandas as pd

from agents.detectors import SYMBOL_PATTERN
from agents.dispatch import StrategyNotApplicableError

Params = dict[str, Any]
Result = tuple[pd.DataFrame, dict[str, Any]]


# ---------------------------------------------------------
# SHARED HELPERS
# ---------------------------------------------------------

def _is_text(s: pd.Series) -> bool:
    # pandas 3 reads text as the new "str" dtype (a StringDtype); some code paths
    # still produce plain object columns. Both count as text.
    return s.dtype == object or isinstance(s.dtype, pd.StringDtype)


def _plain(value: Any) -> Any:
    # numpy scalars (np.float64) aren't JSON-serialisable; .item() makes them plain
    # Python numbers so stats can go straight into state and into LLM prompts.
    return value.item() if hasattr(value, "item") else value


def _normalise(s: pd.Series) -> pd.Series:
    # strip + casefold strings, leave NaN and non-strings untouched.
    # .map instead of .str because .str crashes on object columns with non-strings.
    return s.map(lambda v: v.strip().casefold() if isinstance(v, str) else v)


def _blank_placeholders(s: pd.Series, params: Params) -> pd.Series:
    """Turn tokens like "N/A" or "?" into real NaN so pandas treats them as missing."""
    tokens = params.get("placeholder_tokens") or []
    if not tokens or not _is_text(s):
        return s
    wanted = {str(t).strip().casefold() for t in tokens}
    # .mask returns a NEW series where matching cells become NaN; s is untouched.
    return s.mask(_normalise(s).isin(wanted))


def _require_numeric(s: pd.Series, strategy: str) -> None:
    # pandas counts bool as numeric, but "the mean of True/False" is never intended.
    if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
        raise StrategyNotApplicableError(
            f"{strategy} needs a numeric column, got dtype '{s.dtype}'. "
            "If the numbers are stored as text, run cast_numeric first."
        )


def _require_values(s: pd.Series, strategy: str) -> None:
    # A statistic of zero values is NaN; filling with NaN would "succeed" and fix nothing.
    if s.notna().sum() == 0:
        raise StrategyNotApplicableError(f"{strategy}: column is entirely missing")


def _iqr_bounds(s: pd.Series, k: float, strategy: str) -> tuple[float, float]:
    """Tukey fences [Q1 - k*IQR, Q3 + k*IQR], the same rule as the detector's IQR half."""
    if s.notna().sum() < 4:
        # quartiles of 1-3 values are meaningless
        raise StrategyNotApplicableError(f"{strategy} needs at least 4 non-missing values")
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        # Over half the values are identical, so both fences collapse onto one value
        # and EVERY other value looks extreme. Refuse rather than flatten the column.
        raise StrategyNotApplicableError(
            f"{strategy}: IQR is 0 (most values identical); fences would flag every other value"
        )
    return float(q1 - k * iqr), float(q3 + k * iqr)


# ---------------------------------------------------------
# MISSING VALUES
# ---------------------------------------------------------

def _impute(df: pd.DataFrame, column: str, params: Params, strategy: str,
            pick_fill: Callable[[pd.Series], Any]) -> Result:
    """Shared body of every impute_* strategy; only the choice of fill value differs."""
    out = df.copy()
    s = _blank_placeholders(out[column], params)
    n_missing = int(s.isna().sum())
    fill = pick_fill(s)
    try:
        out[column] = s.fillna(fill)
    except (TypeError, ValueError) as exc:
        # e.g. filling a text column with the number 0: pandas 3's str dtype refuses it
        raise StrategyNotApplicableError(
            f"{strategy}: cannot fill '{column}' with {fill!r}: {exc}"
        ) from exc
    return out, {"values_imputed": n_missing, "fill_value": _plain(fill)}


def impute_mean(df: pd.DataFrame, column: str, params: Params) -> Result:
    def pick(s: pd.Series) -> Any:
        _require_numeric(s, "impute_mean")
        _require_values(s, "impute_mean")
        return s.mean()  # skips NaN by default
    return _impute(df, column, params, "impute_mean", pick)


def impute_median(df: pd.DataFrame, column: str, params: Params) -> Result:
    def pick(s: pd.Series) -> Any:
        _require_numeric(s, "impute_median")
        _require_values(s, "impute_median")
        return s.median()  # robust: one extreme value barely moves it
    return _impute(df, column, params, "impute_median", pick)


def impute_mode(df: pd.DataFrame, column: str, params: Params) -> Result:
    def pick(s: pd.Series) -> Any:
        _require_values(s, "impute_mode")
        # .mode() returns ALL tied values sorted; taking [0] makes ties deterministic.
        return s.mode(dropna=True).iloc[0]
    return _impute(df, column, params, "impute_mode", pick)


def impute_constant(df: pd.DataFrame, column: str, params: Params) -> Result:
    if "fill_value" not in params:
        raise StrategyNotApplicableError("impute_constant needs params['fill_value']")
    # No _require_values here: a constant doesn't depend on the observed data.
    return _impute(df, column, params, "impute_constant", lambda s: params["fill_value"])


def drop_rows_missing(df: pd.DataFrame, column: str, params: Params) -> Result:
    s = _blank_placeholders(df[column], params)
    keep = s.notna()
    # reset_index: after dropping rows, positions 0..n-1 must stay contiguous, or
    # positional code downstream (.iloc) silently points at the wrong rows.
    out = df[keep].reset_index(drop=True)
    return out, {"rows_with_missing": int((~keep).sum())}


def drop_column(df: pd.DataFrame, column: str, params: Params) -> Result:
    s = _blank_placeholders(df[column], params)
    out = df.drop(columns=[column])  # drop() returns a new frame
    # Recorded so the Critic can judge whether dropping was justified.
    return out, {"missing_fraction": round(float(s.isna().mean()), 4)}


# ---------------------------------------------------------
# DUPLICATE ROWS
# ---------------------------------------------------------

def drop_duplicates(df: pd.DataFrame, column: Any, params: Params) -> Result:
    # keep="first": the earliest occurrence survives, which is deterministic.
    out = df.drop_duplicates(keep="first").reset_index(drop=True)
    return out, {"duplicates_removed": len(df) - len(out)}


# ---------------------------------------------------------
# TYPE MISMATCH
# ---------------------------------------------------------

def _coercion_guard(strategy: str, before: pd.Series, after: pd.Series, params: Params) -> int:
    """Count values a conversion turned into missing, and refuse if it is most of them.

    errors="coerce" never crashes, so a WRONG conversion (cast_numeric on a date
    column) would "succeed" and silently wipe the column. Losing more than
    max_coerce_share of the real values means this is the wrong conversion.
    Judgment calls below that threshold are the Critic's job, not this guard's.
    """
    attempted = int(before.notna().sum())
    coerced = int((before.notna() & after.isna()).sum())
    max_share = float(params.get("max_coerce_share", 0.5))
    if attempted and coerced / attempted > max_share:
        raise StrategyNotApplicableError(
            f"{strategy} would turn {coerced} of {attempted} values ({coerced / attempted:.0%}) "
            "into missing; this is probably the wrong conversion for this column"
        )
    return coerced


def cast_numeric(df: pd.DataFrame, column: str, params: Params) -> Result:
    out = df.copy()
    original = out[column]
    s = _blank_placeholders(original, params)
    # Strip currency symbols, thousands separators, % and spaces with the SAME
    # pattern the detector used, so "$1,200" becomes 1200 instead of missing.
    # Caveat worth knowing: "45%" becomes 45, not 0.45.
    stripped = s.map(lambda v: re.sub(SYMBOL_PATTERN, "", v) if isinstance(v, str) else v)
    # errors="coerce": unparseable leftovers become NaN instead of crashing...
    converted = pd.to_numeric(stripped, errors="coerce")
    # ...acceptable only because the guard counts them and refuses mass loss.
    coerced = _coercion_guard("cast_numeric", s, converted, params)
    out[column] = converted
    return out, {
        "placeholders_blanked": int((original.notna() & s.isna()).sum()),
        "coerced_to_missing": coerced,
    }


def parse_datetime(df: pd.DataFrame, column: str, params: Params) -> Result:
    out = df.copy()
    original = out[column]
    if not _is_text(original):
        # to_datetime on integers would read them as nanoseconds since 1970: nonsense.
        raise StrategyNotApplicableError(f"parse_datetime expects a text column, got '{original.dtype}'")
    s = _blank_placeholders(original, params)
    # format="mixed" parses each value on its own, so mixed date formats in one
    # column are accepted. The caller can pass a strict format in params instead.
    converted = pd.to_datetime(s, errors="coerce", format=params.get("format", "mixed"))
    coerced = _coercion_guard("parse_datetime", s, converted, params)
    out[column] = converted
    return out, {"coerced_to_missing": coerced}
    
# ---------------------------------------------------------
# INCONSISTENT CATEGORY
# ---------------------------------------------------------

def standardize_category(df: pd.DataFrame, column: str, params: Params) -> Result:
    out = df.copy()
    s = out[column]
    nonnull = s.dropna()
    if nonnull.empty or pd.api.types.infer_dtype(nonnull, skipna=True) != "string":
        raise StrategyNotApplicableError("standardize_category expects a column of strings")

    keys = _normalise(s)  # "Pune", "pune ", "PUNE" -> "pune"
    canonical: dict[str, str] = {}
    for key, group in nonnull.groupby(keys.dropna()):
        # The canonical spelling is the MOST FREQUENT raw variant, so the data's own
        # dominant spelling wins rather than an arbitrary lowercase version.
        # sort_index() first makes ties break alphabetically (deterministic).
        canonical[key] = group.value_counts().sort_index().idxmax()

    # NaN keys map to NaN, so missing values stay missing (not this strategy's job).
    new = keys.map(canonical).astype(s.dtype)
    changed = int((s.notna() & (new != s)).sum())
    out[column] = new
    return out, {
        "values_changed": changed,
        "categories_before": int(s.nunique()),
        "categories_after": int(new.nunique()),
    }


# ---------------------------------------------------------
# OUTLIERS  (IQR fences; recomputed on the CURRENT data, because the Profiler's
# row positions are stale once any earlier step has dropped rows)
# ---------------------------------------------------------

def cap_iqr(df: pd.DataFrame, column: str, params: Params) -> Result:
    out = df.copy()
    _require_numeric(out[column], "cap_iqr")
    # float64 first: a capped bound like 50.25 can't be stored in an int column.
    s = out[column].astype("float64")
    low, high = _iqr_bounds(s, float(params.get("k", 1.5)), "cap_iqr")
    capped = int(((s < low) | (s > high)).sum())
    out[column] = s.clip(lower=low, upper=high)  # clip keeps NaN as NaN
    return out, {"values_capped": capped, "lower_bound": low, "upper_bound": high}


def drop_outlier_rows(df: pd.DataFrame, column: str, params: Params) -> Result:
    _require_numeric(df[column], "drop_outlier_rows")
    s = df[column].astype("float64")
    low, high = _iqr_bounds(s, float(params.get("k", 1.5)), "drop_outlier_rows")
    # Keep missing values: this strategy removes extremes, not gaps.
    keep = s.isna() | s.between(low, high)
    out = df[keep].reset_index(drop=True)
    return out, {"outlier_rows_dropped": int((~keep).sum()), "lower_bound": low, "upper_bound": high}


def flag_outlier(df: pd.DataFrame, column: str, params: Params) -> Result:
    out = df.copy()
    _require_numeric(out[column], "flag_outlier")
    s = out[column].astype("float64")
    low, high = _iqr_bounds(s, float(params.get("k", 1.5)), "flag_outlier")
    # Values stay unchanged; a boolean column lets a later model decide what to do.
    out[f"{column}_is_outlier"] = (s < low) | (s > high)
    return out, {"values_flagged": int(out[f"{column}_is_outlier"].sum())}


# ---------------------------------------------------------
# ANY ISSUE
# ---------------------------------------------------------

def no_action(df: pd.DataFrame, column: Any, params: Params) -> Result:
    # Explicitly choosing to leave data alone is a decision the Critic can review.
    return df.copy(), {}
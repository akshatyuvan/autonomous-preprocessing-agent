"""
agents/detectors.py — the Profiler's five deterministic detectors.

Environment: LOCAL (Mac).

Why detection is pure Python, not the LLM: detection must be exact,
reproducible and unit-testable. An LLM asked to count nulls can hallucinate;
pandas cannot. The LLM (agents/profiler.py) only SYNTHESISES these findings.

Every detector:
  - never mutates its input DataFrame (locked decision #2),
  - returns a list of Finding objects,
  - reports row POSITIONS so the Day 15 viz can flag individual cells amber.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --- Thresholds as named constants: every number is defensible and tunable ---
# Strings that mean "missing" but are not NaN. Real CSVs are full of these.
PLACEHOLDER_TOKENS = {"", "na", "n/a", "nan", "null", "none", "?", "-", "--", "missing"}
NUMERIC_PARSE_THRESHOLD = 0.8    # >=80% of text values parse as numbers -> numbers stored as text
DATETIME_PARSE_THRESHOLD = 0.8
SYMBOL_PATTERN = r"[,$€£₹%\s]"   # stripped before numeric parsing: "$1,200" -> "1200"
MIN_OUTLIER_N = 8                # with fewer points, quartiles are meaningless
IQR_K = 1.5                      # Tukey's fences
MODIFIED_Z_THRESHOLD = 3.5       # Iglewicz & Hoaglin's recommended cut-off
MODE_FALLBACK_MAX_SHARE = 0.10   # when IQR == 0, flag non-mode values only if they are rare
CATEGORY_MAX_UNIQUE = 50         # above this a text column is free text, not a category


@dataclass
class Finding:
    """Internal detector output; the Profiler converts it to the schema's DataIssue.
    `rows` lives here because DataIssue has no field for it: it feeds viz events."""
    column: str
    issue_type: str
    severity: str
    affected_rows: int
    detail: str
    rows: list[int] = field(default_factory=list)


# ---------------------------------------------------------------- helpers
def _reset(df: pd.DataFrame) -> pd.DataFrame:
    # Returns a NEW frame (no mutation). Afterwards index labels == row
    # positions, so reported rows are positions even if the caller's index
    # was shuffled or had duplicate labels.
    return df.reset_index(drop=True)


def _positions(mask) -> list[int]:
    return [int(i) for i in np.flatnonzero(np.asarray(mask, dtype=bool))]


def _severity(share: float, medium: float, high: float) -> str:
    if share >= high:
        return "high"
    if share >= medium:
        return "medium"
    return "low"


def _is_text(s: pd.Series) -> bool:
    # Pandas 3 reads strings as StringDtype, not object, so `dtype == object`
    # alone misses them. The object check still catches mixed-type columns.
    return pd.api.types.is_string_dtype(s) or s.dtype == object


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _placeholder_mask(s: pd.Series) -> pd.Series:
    return s.notna() & s.astype(str).str.strip().str.lower().isin(PLACEHOLDER_TOKENS)


def _valid_text_mask(s: pd.Series) -> pd.Series:
    return s.notna() & ~_placeholder_mask(s)


def _numeric_parse_ratio(s: pd.Series) -> tuple[float, int, int]:
    """(share of real text values that parse as numbers, parsed count, total)."""
    vals = s[_valid_text_mask(s)].astype(str).str.strip()
    if vals.empty:
        return 0.0, 0, 0
    parsed = pd.to_numeric(vals.str.replace(SYMBOL_PATTERN, "", regex=True), errors="coerce")
    ok = int(parsed.notna().sum())
    return ok / len(vals), ok, len(vals)


def _datetime_parse_ratio(s: pd.Series) -> float:
    vals = s[_valid_text_mask(s)].astype(str).str.strip()
    if vals.empty:
        return 0.0
    # Require a digit: dateutil happily parses words like "March" or "now",
    # which would mislabel a month-name category as dates.
    has_digit = vals.str.contains(r"\d", regex=True)
    try:
        # format="mixed" parses per value. infer_datetime_format was removed in Pandas 2+.
        parsed = pd.to_datetime(vals, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        return 0.0
    return float((has_digit & parsed.notna()).mean())


def _f(value) -> float | None:
    """JSON-safe float: NaN/inf become None (stats go into an LLM prompt and the API)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return round(v, 4) if math.isfinite(v) else None


# ---------------------------------------------------------------- detector 1
def detect_duplicates(df: pd.DataFrame) -> list[Finding]:
    """Exact duplicates only. Fuzzy/near-duplicate matching is out of scope (Section 9)."""
    df = _reset(df)
    if df.empty:
        return []
    mask = df.duplicated(keep="first")  # first occurrence = original; later copies flagged
    count = int(mask.sum())
    if count == 0:
        return []
    share = count / len(df)
    return [Finding(
        column="__all_columns__",
        issue_type="duplicate_rows",
        severity=_severity(share, 0.01, 0.05),
        affected_rows=count,
        detail=f"{count} exact duplicate rows ({share:.1%}); first occurrences treated as originals",
        rows=_positions(mask),
    )]


# ---------------------------------------------------------------- detector 2
def detect_missing(df: pd.DataFrame) -> list[Finding]:
    df = _reset(df)
    n = len(df)
    if n == 0:
        return []
    findings = []
    for col in df.columns:
        s = df[col]
        mask = s.isna()
        tokens: list[str] = []
        if _is_text(s):
            ph = _placeholder_mask(s)
            if ph.any():
                tokens = sorted(set(s[ph].astype(str).str.strip()))[:5]
                mask = mask | ph
        count = int(mask.sum())
        if count == 0:
            continue
        share = count / n
        detail = f"{count} of {n} values missing ({share:.1%})"
        if tokens:
            detail += f", including placeholder tokens {tokens}"
        findings.append(Finding(
            column=str(col), issue_type="missing_values",
            severity=_severity(share, 0.05, 0.30),
            affected_rows=count, detail=detail, rows=_positions(mask),
        ))
    return findings


# ---------------------------------------------------------------- detector 3
def detect_type_mismatch(df: pd.DataFrame) -> list[Finding]:
    """Numbers or dates stored as text. These block every numeric check
    downstream (outliers, stats, scaling), so they must surface early."""
    df = _reset(df)
    findings = []
    for col in df.columns:
        s = df[col]
        if not _is_text(s):
            continue
        valid = _valid_text_mask(s)
        ratio, ok, total = _numeric_parse_ratio(s)
        if total == 0:
            continue
        if ratio >= NUMERIC_PARSE_THRESHOLD:
            bad = total - ok
            detail = f"{ok} of {total} non-missing values are numbers stored as text"
            if bad:
                detail += f"; {bad} value(s) cannot be parsed as numbers"
            findings.append(Finding(
                column=str(col), issue_type="type_mismatch",
                severity="medium" if bad else "low",
                affected_rows=total, detail=detail, rows=_positions(valid),
            ))
        elif _datetime_parse_ratio(s) >= DATETIME_PARSE_THRESHOLD:
            findings.append(Finding(
                column=str(col), issue_type="type_mismatch", severity="low",
                affected_rows=total,
                detail=f"{total} non-missing values look like dates stored as text",
                rows=_positions(valid),
            ))
    return findings


# ---------------------------------------------------------------- detector 4
def outlier_mask(x: pd.Series) -> tuple[pd.Series, list[str]]:
    """Union of IQR and modified Z-score (locked decision #11).

    - Modified Z uses median and MAD, which the outliers cannot drag around.
      Classic mean/std Z-score is inflated by the very outliers it hunts.
    - IQR == 0 (over half the values identical) falls back to mode
      comparison and must NOT exit early, or the Z-score check is skipped.
    """
    mask = pd.Series(False, index=x.index)
    methods: list[str] = []

    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    if iqr > 0:
        m = (x < q1 - IQR_K * iqr) | (x > q3 + IQR_K * iqr)
        if m.any():
            methods.append("IQR")
        mask |= m
    else:
        mode = x.mode().iloc[0]
        off = x != mode
        # Only flag non-mode values if they are rare: an 80/20 split is a
        # distribution, not 20% outliers.
        if 0 < off.mean() <= MODE_FALLBACK_MAX_SHARE:
            methods.append("mode-comparison")
            mask |= off

    med = x.median()
    mad = (x - med).abs().median()
    if mad > 0:  # runs whether or not IQR was zero
        m = (0.6745 * (x - med) / mad).abs() > MODIFIED_Z_THRESHOLD
        if m.any():
            methods.append("modified Z-score")
        mask |= m
    return mask, methods


def detect_outliers(df: pd.DataFrame) -> list[Finding]:
    df = _reset(df)
    findings = []
    for col in df.columns:
        s = df[col]
        if not _is_numeric(s):
            continue
        x = s.dropna().astype(float)
        # Binary-like columns have no meaningful outliers.
        if len(x) < MIN_OUTLIER_N or x.nunique() <= 2:
            continue
        mask, methods = outlier_mask(x)
        count = int(mask.sum())
        if count == 0:
            continue
        flagged = x[mask]
        share = count / len(x)
        findings.append(Finding(
            column=str(col), issue_type="outlier",
            severity=_severity(share, 0.01, 0.05),
            affected_rows=count,
            detail=(f"{count} outlier(s) by {' + '.join(methods)}; flagged values range "
                    f"{flagged.min():g} to {flagged.max():g}; column median {x.median():g}"),
            rows=[int(i) for i in flagged.index],  # labels == positions after _reset
        ))
    return findings


# ---------------------------------------------------------------- detector 5
def detect_inconsistent_categories(df: pd.DataFrame) -> list[Finding]:
    """Same category written several ways ('Male', 'male ', 'MALE')."""
    df = _reset(df)
    findings = []
    for col in df.columns:
        s = df[col]
        if not _is_text(s):
            continue
        vals = s[_valid_text_mask(s)].astype(str)
        if vals.empty:
            continue
        raw_unique = vals.nunique()
        if raw_unique > CATEGORY_MAX_UNIQUE:
            continue  # free text, not a categorical
        if _numeric_parse_ratio(s)[0] >= NUMERIC_PARSE_THRESHOLD:
            continue  # numbers-as-text belong to the type_mismatch detector
        norm = vals.str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
        if norm.nunique() == raw_unique:
            continue
        # Canonical form = the most frequent raw spelling in each group.
        canonical = vals.groupby(norm).agg(lambda g: g.value_counts().idxmax())
        off = vals != norm.map(canonical)
        count = int(off.sum())
        examples = [repr(sorted(set(vals[norm == key]))) for key in pd.unique(norm[off])[:3]]
        findings.append(Finding(
            column=str(col), issue_type="inconsistent_category",
            severity=_severity(count / len(vals), 0.05, 0.20),
            affected_rows=count,
            detail=(f"{norm.nunique()} categories written {raw_unique} ways; variants: "
                    + "; ".join(examples)),
            rows=[int(i) for i in vals.index[off]],
        ))
    return findings


# ---------------------------------------------------------------- stats + runner
def compute_column_stats(df: pd.DataFrame) -> dict[str, dict]:
    """Per-column stats for ProfilerState.column_stats. All values JSON-safe."""
    n = len(df)
    stats: dict[str, dict] = {}
    for col in df.columns:
        s = df[col]
        entry = {
            "dtype": str(s.dtype),
            "null_count": int(s.isna().sum()),
            "null_pct": round(float(s.isna().mean()), 4) if n else 0.0,
            "unique_count": int(s.nunique(dropna=True)),
        }
        if _is_numeric(s) and s.notna().any():
            x = s.dropna().astype(float)
            entry.update(
                mean=_f(x.mean()), std=_f(x.std()), min=_f(x.min()),
                median=_f(x.median()), max=_f(x.max()), skew=_f(x.skew()),
            )
        elif _is_text(s):
            top = s.dropna().astype(str).value_counts().head(3)
            entry["top_values"] = {str(k): int(v) for k, v in top.items()}
        stats[str(col)] = entry
    return stats


# Fixed order so issue_ids are deterministic across runs.
DETECTORS = (
    detect_duplicates,
    detect_missing,
    detect_type_mismatch,
    detect_outliers,
    detect_inconsistent_categories,
)


def run_all_detectors(df: pd.DataFrame) -> list[Finding]:
    findings: list[Finding] = []
    for detector in DETECTORS:
        findings.extend(detector(df))
    return findings
"""
agents/evidence.py -- what the Critic sees about ONE decision.

The same functions build the evidence for BOTH the live Critic and the
labelled evaluation dataset (Step 4). If they differed, the evaluation would
measure a different task from the one the Critic actually does in the pipeline.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import pandas as pd

# Keys of a decision's stats that are NOT "what changed": they are shown separately
# (before/after) or are already in the evidence (strategy, column).
_NOT_CHANGE_KEYS = {"strategy", "column", "before", "after"}


def _num(value) -> Optional[float]:
    """JSON-safe rounded float: NaN/inf become None."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return round(v, 4) if math.isfinite(v) else None


def summarize_column(df: pd.DataFrame, column: Optional[str]) -> dict[str, Any]:
    """Aggregate statistics only -- never raw values (same privacy rule as the Profiler)."""
    if column is None:
        # Row-level decisions (duplicates): the whole dataset is the "column".
        return {"rows": len(df), "columns": int(df.shape[1]),
                "duplicate_rows": int(df.duplicated().sum())}
    if column not in df.columns:
        return {"present": False, "rows": len(df)}  # e.g. after drop_column
    s = df[column]
    summary: dict[str, Any] = {
        "present": True,
        "rows": len(df),
        "dtype": str(s.dtype),
        "missing": int(s.isna().sum()),
        "unique": int(s.nunique(dropna=True)),
    }
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and s.notna().any():
        x = s.dropna().astype(float)
        # mean vs median vs skew is exactly what reveals a bad imputation or capping.
        summary.update(mean=_num(x.mean()), median=_num(x.median()), std=_num(x.std()),
                       min=_num(x.min()), max=_num(x.max()), skew=_num(x.skew()))
    return summary


def build_evidence(decision: dict, issue: Optional[dict], strategy_description: str) -> dict[str, Any]:
    """
    The Critic's input for one decision.

    The Cleaner LLM's `justification` is deliberately EXCLUDED: the Critic judges
    what the transformation DID to the data, not how persuasively it was argued.
    Otherwise one LLM could talk another into accepting a bad change.
    """
    stats = decision.get("stats") or {}
    return {
        "issue": None if issue is None else {
            "type": issue["issue_type"], "severity": issue["severity"], "detail": issue["detail"],
        },
        "strategy": decision["action"],
        "strategy_description": strategy_description,
        "column": decision["column"],
        "change": {k: v for k, v in stats.items() if k not in _NOT_CHANGE_KEYS},
        "before": stats.get("before"),
        "after": stats.get("after"),
    }
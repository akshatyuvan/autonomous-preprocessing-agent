"""
agents/dispatch.py -- turns a technique NAME (chosen by the LLM) into a pandas operation.

"LLMs decide, Python executes": the LLM only ever outputs a key like "impute_median".
This module looks that key up in a registry and calls the function stored there.
It is shared by EVERY preprocessing stage, so it contains no stage-specific logic.
Adding a technique = one new registry entry; this file never changes (resume bullet 1).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import pandas as pd


class UnknownStrategyError(LookupError):
    """The name isn't in the registry -- e.g. the LLM invented 'impute_magic'."""


class StrategyNotApplicableError(ValueError):
    """The name is valid but doesn't fit this data -- e.g. mean imputation on a text column."""


def options_for(registry: Mapping[str, dict], issue_type: str) -> list[str]:
    """
    Registry keys that can handle this issue type, in registry order.
    This list is exactly what the LLM will be allowed to choose from, so a new
    registry entry automatically shows up in the prompt with no prompt edits.
    """
    return [key for key, entry in registry.items() if issue_type in entry["handles"]]


def apply_strategy(
    registry: Mapping[str, dict],
    strategy: str,
    df: pd.DataFrame,
    column: Optional[str],
    params: Optional[dict[str, Any]] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Look up `strategy` in `registry`, run it, and return (new_df, stats)."""
    if strategy not in registry:
        # Never silently fall back to "something sensible": an unknown name is a bug
        # (or an LLM hallucination) and must surface, not be guessed around.
        raise UnknownStrategyError(
            f"Unknown strategy '{strategy}'. Valid options: {sorted(registry)}"
        )

    entry = registry[strategy]

    # Row-level techniques (drop_duplicates) don't target one column; everything else does.
    if entry.get("needs_column", True) and column not in df.columns:
        raise StrategyNotApplicableError(f"{strategy}: column '{column}' is not in the dataset")

    # dict(...) copies params so a strategy can never edit the caller's dict by accident.
    new_df, specific_stats = entry["apply"](df, column, dict(params or {}))

    # Generic before/after shape facts are computed HERE, once, so no strategy can
    # forget them. The Critic will use these as evidence ("you dropped 40% of rows").
    stats: dict[str, Any] = {
        "strategy": strategy,
        "column": column,
        "rows_before": len(df),
        "rows_after": len(new_df),
        "columns_before": df.shape[1],
        "columns_after": new_df.shape[1],
        **specific_stats,
    }
    return new_df, stats
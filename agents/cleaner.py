"""
agents/cleaner.py -- Cleaning agent (Step 2).

Environment: LOCAL (Mac).

Flow, once per round:
  1. Reload the RAW dataset. A redo REPLACES the rejected attempt; it never
     stacks a second round of fixes on top of the first one.
  2. Order the Profiler's issues: duplicates -> types -> categories -> outliers -> missing.
  3. For each issue the LLM picks ONE key from the registry options for that
     issue type. Python validates the key and applies it via agents/dispatch.py.
  4. If the LLM is down, returns junk, or picks a strategy that doesn't fit the
     data, a deterministic fallback takes the first registry option that applies.
  5. Save the result as a parquet snapshot for the Critic and the next stage.

"LLMs decide, Python executes": the LLM never writes code and never sees raw rows.
Returns PARTIAL state: "cleaner", "metadata", "visualization_events", "errors".
"""
from __future__ import annotations

import json
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator

import config
from agents.detectors import PLACEHOLDER_TOKENS
from agents.dispatch import StrategyNotApplicableError, apply_strategy, options_for
from agents.llm_factory import make_chat_model
from agents.profiler import load_dataset
from agents.registries import CLEANING_REGISTRY
from state.schema import (
    AgentState, CleanerState, CleaningDecision, DataIssue, ErrorRecord, VizEvent,
)

ALL_COLUMNS = "__all_columns__"  # column name the duplicate detector uses

# Fix order. Each step needs the ones before it to have happened:
ISSUE_PRIORITY = {
    "duplicate_rows": 0,         # copies would be double-counted in every statistic below
    "type_mismatch": 1,          # numeric/date fixes need real dtypes first
    "inconsistent_category": 2,  # merge "Pune"/"pune " BEFORE mode imputation counts them
    "outlier": 3,                # tame extremes BEFORE they pull a mean/median fill value
    "missing_values": 4,         # fill last, from the cleanest version of each column
    "unknown": 5,
}


class StrategyChoice(BaseModel):
    """The ONLY thing the LLM produces. All fields required (same reason as ProfileSynthesis)."""
    strategy: str = Field(description="Exactly one key from allowed_strategies, copied as written")
    fill_value: str = Field(description="Only for impute_constant; otherwise an empty string")
    justification: str = Field(description="1-2 sentences citing the column statistics used")

    @field_validator("strategy", mode="before")
    @classmethod
    def _normalise_strategy(cls, value):
        # A 3B model adds quotes, spaces or capitals: ' "Impute_Median" ' -> 'impute_median'.
        # Repairing the FORMAT is safe. Whether the key is ALLOWED is checked separately.
        if isinstance(value, str):
            return value.strip().strip("'\"` ").lower()
        return value

    @field_validator("fill_value", "justification", mode="before")
    @classmethod
    def _none_to_empty(cls, value):
        # Small models sometimes send null for a field they consider unused.
        return "" if value is None else value


SYSTEM_PROMPT = """You are the Cleaning agent in a data-preprocessing pipeline.
You receive ONE data-quality issue, statistics for its column, and allowed_strategies.
Choose exactly ONE strategy and copy its "key" exactly as written. Python applies it; you only choose.
Base the choice on the statistics: a skewed column (large |skew|) or one with outliers should not
be mean-imputed; dropping rows or columns loses data and needs a strong reason.
If previous_attempt is present, a reviewer rejected that attempt: read critic_feedback and choose
differently unless you can justify the same choice.
Set fill_value only for impute_constant; otherwise use an empty string.
justification: 1-2 sentences that cite the statistics you used."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_llm(model: str):
    """Isolated so tests monkeypatch exactly this (all tests run with no model)."""
    return make_chat_model(model).with_structured_output(StrategyChoice, method="function_calling")


def fill_value_for(series: Optional[pd.Series], raw: str) -> Any:
    """Turn the LLM's TEXT fill_value into something the column can hold, or None.
    Filling a numeric column with "Unknown" would silently turn it into text."""
    raw = (raw or "").strip()
    if not raw or series is None:
        return None
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        try:
            return float(raw)
        except ValueError:
            return None
    return raw


def build_messages(issue: DataIssue, options: list[str], column_stats: dict,
                   previous: Optional[dict]) -> list[tuple[str, str]]:
    payload = {
        "issue": {k: issue[k] for k in
                  ("issue_id", "column", "issue_type", "severity", "affected_rows", "detail")},
        # Aggregate stats only; top_values (raw data values) stay out, as in the Profiler.
        "column_stats": {k: v for k, v in column_stats.get(issue["column"], {}).items()
                         if k != "top_values"},
        # Built from the registry, so a new technique appears here with no prompt edit.
        "allowed_strategies": [
            {"key": k,
             "applies_when": CLEANING_REGISTRY[k]["applies_when"],
             "description": CLEANING_REGISTRY[k]["description"]}
            for k in options
        ],
        "previous_attempt": previous,
    }
    return [("system", SYSTEM_PROMPT), ("human", json.dumps(payload, indent=2, default=str))]


def _ask_llm(llm, issue, options, column_stats, previous) -> tuple[Optional[StrategyChoice], Optional[str]]:
    """Returns (choice, None) or (None, reason). The error prefix is what the eval counts."""
    if llm is None:
        return None, "llm_error: model could not be created"
    try:
        choice = llm.invoke(build_messages(issue, options, column_stats, previous))
    except Exception as exc:  # timeout, Ollama down, or output that fails pydantic validation
        return None, f"llm_error: {type(exc).__name__}: {exc}"
    if choice.strategy not in options:
        # Hallucinated name, or a real key that doesn't handle this issue type.
        return None, f"invalid_choice: '{choice.strategy}' is not one of {options}"
    return choice, None


def _decision(decision_id, issue, action, params, justification, redo_count,
              chosen_by, stats, error) -> CleaningDecision:
    return CleaningDecision(
        decision_id=decision_id, issue_id=issue["issue_id"], column=issue["column"],
        action=action, parameters=params, justification=justification,
        status="applied", redo_count=redo_count, chosen_by=chosen_by,
        stats=stats, error=error,
    )


def clean_issue(df: pd.DataFrame, issue: DataIssue, llm, column_stats: dict,
                previous: Optional[dict], decision_id: str,
                redo_count: int) -> tuple[CleaningDecision, pd.DataFrame]:
    """Decide (LLM) and apply (Python) one fix. Never raises for bad LLM output."""
    column = None if issue["column"] == ALL_COLUMNS else issue["column"]
    params: dict[str, Any] = {"placeholder_tokens": sorted(PLACEHOLDER_TOKENS)}

    # An earlier decision this round (drop_column) may have removed the column.
    if column is not None and column not in df.columns:
        return _decision(decision_id, issue, "no_action", params,
                         "Column was already removed by an earlier decision in this round.",
                         redo_count, "fallback", {}, None), df

    options = options_for(CLEANING_REGISTRY, issue["issue_type"])
    choice, error = _ask_llm(llm, issue, options, column_stats, previous)
    llm_key = choice.strategy if choice else None

    if choice:
        fill = fill_value_for(df[column] if column is not None else None, choice.fill_value)
        if fill is not None:
            params["fill_value"] = fill  # strategies that don't use it simply ignore it

    # LLM's choice first, then registry order, then no_action (which always applies).
    # dict.fromkeys removes duplicates while keeping order.
    fallback_order = [k for k in options if k != "no_action"] + ["no_action"]
    candidates = list(dict.fromkeys(([llm_key] if llm_key else []) + fallback_order))

    for key in candidates:
        try:
            new_df, stats = apply_strategy(CLEANING_REGISTRY, key, df, column, params)
        except StrategyNotApplicableError as exc:
            if key == llm_key:
                # A real key that doesn't fit this data (median on text): a quality
                # failure of the LLM, counted separately from malformed output.
                error = f"not_applicable: {exc}"
            continue
        if key == llm_key and choice is not None:
            return _decision(decision_id, issue, key, params, choice.justification,
                             redo_count, "llm", stats, None), new_df
        return _decision(decision_id, issue, key, params,
                         f"Deterministic fallback: first registry option for "
                         f"{issue['issue_type']} that fits this column.",
                         redo_count, "fallback", stats, error), new_df

    raise RuntimeError("unreachable: no_action always applies")


def _critic_feedback(state: AgentState) -> Optional[str]:
    """The Critic's reasoning if its LAST verdict rejected the Cleaner, else None.
    Must be read BEFORE this node overwrites current_active_agent."""
    verdict = state["critic"]["current_verdict"]
    if state["cleaner"]["current_round"] == 0 or verdict is None:
        return None
    if state["metadata"]["current_active_agent"] != "cleaner" or verdict["verdict"] == "accept":
        return None
    return verdict["reasoning"]


def _save_snapshot(df: pd.DataFrame, session_id: str, round_no: int) -> str:
    # config.CLEANED_DIR is read at CALL time so tests can redirect it to a temp folder.
    out_dir = Path(config.CLEANED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)  # empty folders aren't tracked by git
    path = out_dir / f"{session_id}_cleaner_r{round_no}.parquet"
    # Parquet keeps dtypes (datetimes, floats), unlike CSV, so the next stage
    # sees exactly what this stage produced.
    df.to_parquet(path, index=False)
    return str(path)


def _round_event(message: str) -> VizEvent:
    return VizEvent(event_id=str(uuid.uuid4()), agent="cleaner", event_type="round_complete",
                    payload={"message": message}, timestamp=_now())


def cleaner_node(state: AgentState) -> dict:
    """LangGraph node. Returns PARTIAL state only."""
    round_no = state["cleaner"]["current_round"] + 1
    previous_decisions = list(state["cleaner"]["decisions"])
    print(f"[Cleaner] Round {round_no}")

    feedback = _critic_feedback(state)  # read before current_active_agent is overwritten
    metadata = dict(state["metadata"])
    metadata["current_active_agent"] = "cleaner"
    if "cleaner" not in metadata["pipeline_steps_run"]:
        metadata["pipeline_steps_run"] = metadata["pipeline_steps_run"] + ["cleaner"]

    def result(decisions, snapshot, events, errors, message) -> dict:
        return {
            "cleaner": CleanerState(current_round=round_no, decisions=decisions,
                                    dataset_snapshot_path=snapshot),
            "metadata": metadata,
            "visualization_events": events + [_round_event(message)],
            "errors": errors,
        }

    if not state["profiler"]["run_complete"]:
        # The Profiler already recorded why; nothing to clean without its issues.
        return result(previous_decisions, None, [], [], "Cleaner skipped: profiler did not complete.")

    try:
        df = load_dataset(state["input"]["dataset_path"])
    except Exception as exc:
        return result(previous_decisions, None, [], [ErrorRecord(
            agent="cleaner", error_type="dataset_load_error", message=str(exc),
            timestamp=_now(), recoverable=False)], "Cleaner could not load the dataset.")

    issues = sorted(state["profiler"]["issues"],
                    key=lambda i: ISSUE_PRIORITY.get(i["issue_type"], 99))  # stable sort

    llm = None
    if issues:
        try:
            llm = _get_llm(state["metadata"]["llm_model"])
        except Exception:
            llm = None  # every decision then records llm_error and uses the fallback

    previous_actions = {d["issue_id"]: d["action"] for d in previous_decisions
                        if d["decision_id"].startswith(f"clean_r{round_no - 1}_")}

    new_decisions: list[CleaningDecision] = []
    events: list[VizEvent] = []
    for n, issue in enumerate(issues, start=1):
        previous = None
        if feedback:
            previous = {"strategy": previous_actions.get(issue["issue_id"]),
                        "critic_feedback": feedback}
        decision, df = clean_issue(df, issue, llm, state["profiler"]["column_stats"],
                                   previous, f"clean_r{round_no}_{n:03d}", round_no - 1)
        new_decisions.append(decision)
        events.append(VizEvent(
            event_id=str(uuid.uuid4()), agent="cleaner", event_type="cell_status_change",
            payload={"decision_id": decision["decision_id"], "issue_id": issue["issue_id"],
                     "column": issue["column"], "action": decision["action"],
                     "status": "cleaning"},
            timestamp=_now()))

    errors: list[ErrorRecord] = []
    problems = Counter(d["error"].split(":")[0] for d in new_decisions if d["error"])
    if problems:
        errors.append(ErrorRecord(
            agent="cleaner", error_type="llm_fallback_used",
            message=(f"LLM choice not used for {sum(problems.values())} of {len(new_decisions)} "
                     f"issue(s) {dict(problems)}; deterministic fallback applied"),
            timestamp=_now(), recoverable=True))

    try:
        snapshot = _save_snapshot(df, metadata["session_id"], round_no)
    except Exception as exc:
        snapshot = None  # the Critic will treat a missing snapshot as a failed stage
        errors.append(ErrorRecord(agent="cleaner", error_type="snapshot_write_error",
                                  message=str(exc), timestamp=_now(), recoverable=False))

    print(f"[Cleaner] {len(new_decisions)} decision(s) applied")
    return result(previous_decisions + new_decisions, snapshot, events, errors,
                  f"Cleaner round {round_no}: {len(new_decisions)} decision(s).")
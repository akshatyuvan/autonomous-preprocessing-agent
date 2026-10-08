"""
agents/profiler.py — Profiler Agent (Day 2).

Environment: LOCAL (Mac).

Flow: load file -> 5 deterministic detectors -> per-column stats ->
LLM synthesises a prioritised summary -> partial state + viz events.

Reads:   state["input"]["dataset_path"], state["metadata"]["llm_model"]
Returns: "profiler", "visualization_events", "errors"   (partial state only)
"""
from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from agents.detectors import Finding, compute_column_stats, run_all_detectors
from agents.llm_factory import make_chat_model
from state.schema import AgentState, DataIssue, ErrorRecord, ProfilerState, VizEvent

MAX_ROWS_PER_EVENT = 200     # keeps viz events small on large datasets
MAX_ISSUES_IN_PROMPT = 40    # bounds LLM token cost on very messy data


class ProfileSynthesis(BaseModel):
    """Structured output. All fields required: OpenAI's strict JSON-schema
    mode rejects optional fields, and required fields force explicit answers."""
    summary: str = Field(description="4-6 sentence summary for the Cleaning agent")
    top_concerns: list[str] = Field(description="issue_ids ordered most important first")

    @field_validator("top_concerns", mode="before")
    @classmethod
    def _coerce_stringified_list(cls, value):
        # Small local models (Llama 3.2 3B) often return a list as a STRING,
        # e.g. '["issue_007", "issue_005"]' or 'issue_007, issue_005'.
        # mode="before" runs BEFORE type validation, so we can repair it here
        # instead of failing the whole response. Real lists pass through untouched.
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed]
            except json.JSONDecodeError:
                pass
            # Not valid JSON: pull out anything shaped like an issue ID.
            return re.findall(r"issue_\d{3}", value)
        return value


SYSTEM_PROMPT = """You are the Profiler in a data-cleaning pipeline.
Deterministic detectors have already found the data-quality issues below; their
counts are exact. Do NOT invent issues, counts or columns that are not in the input,
and do not restate every number.
Write a concise summary (4-6 sentences) for the Cleaning agent: what is wrong,
which issues matter most for downstream analysis and why, and any interactions
between issues (e.g. outliers in a column that also has missing values change how
it should be imputed; numbers stored as text block outlier detection).
Return top_concerns as issue_ids from the input, most important first."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_dataset(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type '{suffix}' (expected .csv or .parquet)")


def findings_to_issues(findings: list[Finding]) -> list[DataIssue]:
    """Convert detector output to the schema's DataIssue. IDs are positional and
    the detector order is fixed, so the same file always gives the same IDs."""
    return [
        DataIssue(
            issue_id=f"issue_{i:03d}",
            column=f.column,
            issue_type=f.issue_type,
            severity=f.severity,
            affected_rows=f.affected_rows,
            detail=f.detail,
            suggested_fix=None,  # the Cleaner decides, with retrieval on Day 4
        )
        for i, f in enumerate(findings, start=1)
    ]


def build_viz_events(issues: list[DataIssue], findings: list[Finding]) -> list[VizEvent]:
    """One 'flagged' event per issue: the Day 15 grid turns these cells amber."""
    events = []
    for issue, f in zip(issues, findings):
        events.append(VizEvent(
            event_id=str(uuid.uuid4()),
            agent="profiler",
            event_type="cell_status_change",
            payload={
                "issue_id": issue["issue_id"],
                "column": f.column,
                "issue_type": f.issue_type,
                "status": "flagged",
                "rows": f.rows[:MAX_ROWS_PER_EVENT],
                "rows_truncated": len(f.rows) > MAX_ROWS_PER_EVENT,
            },
            timestamp=_now(),
        ))
    return events


def _round_complete_event(n_issues: int) -> VizEvent:
    return VizEvent(
        event_id=str(uuid.uuid4()),
        agent="profiler",
        event_type="round_complete",
        payload={"message": f"Profiler complete: {n_issues} issue(s) detected."},
        timestamp=_now(),
    )


def fallback_summary(row_count: int, column_count: int, issues: list[DataIssue]) -> str:
    """Deterministic summary used when there are no issues or the LLM is unavailable.
    The pipeline never depends on the LLM being up."""
    if not issues:
        return (f"Dataset has {row_count} rows and {column_count} columns. "
                f"No data-quality issues detected.")
    counts = Counter(i["issue_type"] for i in issues)
    parts = ", ".join(f"{n} {t}" for t, n in sorted(counts.items()))
    summary = (f"Dataset has {row_count} rows and {column_count} columns "
               f"with {len(issues)} issue(s): {parts}.")
    high = [i["issue_id"] for i in issues if i["severity"] == "high"]
    if high:
        summary += f" High severity: {', '.join(high)}."
    return summary


def _get_llm(model: str):
    """Isolated so tests monkeypatch exactly this (locked decision #4).
    method="function_calling": the most widely supported structured-output mode,
    so it works on both GitHub Models and the OpenAI API."""
    return make_chat_model(model).with_structured_output(ProfileSynthesis, method="function_calling")


def synthesize_summary(issues, row_count, column_count, column_stats, llm) -> str:
    # The LLM gets issue metadata and aggregate stats only. top_values (raw
    # data values) are stripped: lower token cost and no raw rows sent to OpenAI.
    payload = {
        "row_count": row_count,
        "column_count": column_count,
        "issues": issues[:MAX_ISSUES_IN_PROMPT],
        "issues_truncated": len(issues) > MAX_ISSUES_IN_PROMPT,
        "column_stats": {
            col: {k: v for k, v in st.items() if k != "top_values"}
            for col, st in column_stats.items()
        },
    }
    result = llm.invoke([("system", SYSTEM_PROMPT), ("human", json.dumps(payload, indent=2, default=str))])
    summary = result.summary.strip() or fallback_summary(row_count, column_count, issues)
    # Keep only IDs that exist: a hallucinated issue_id never reaches state.
    valid_ids = {i["issue_id"] for i in issues}
    priority = [i for i in result.top_concerns if i in valid_ids]
    if priority:
        summary += f" Priority order: {', '.join(priority)}."
    return summary


def profiler_node(state: AgentState) -> dict:
    """LangGraph node. Returns PARTIAL state only (locked decision #3)."""
    dataset_path = state["input"]["dataset_path"]
    print(f"[Profiler] Running on: {state['input']['dataset_name']}")

    try:
        df = load_dataset(dataset_path)
    except Exception as exc:
        # Non-recoverable: nothing downstream can work without data.
        # Halting the graph on this is Day 17's error-handling work.
        return {
            "profiler": ProfilerState(
                run_complete=False, row_count=0, column_count=0, issues=[],
                profile_summary=f"Profiler could not load dataset: {exc}", column_stats={},
            ),
            "errors": [ErrorRecord(
                agent="profiler", error_type="dataset_load_error",
                message=str(exc), timestamp=_now(), recoverable=False,
            )],
            "visualization_events": [_round_complete_event(0)],
        }

    findings = run_all_detectors(df)
    issues = findings_to_issues(findings)
    column_stats = compute_column_stats(df)
    row_count, column_count = len(df), df.shape[1]
    errors: list[ErrorRecord] = []

    summary = fallback_summary(row_count, column_count, issues)
    if issues:  # clean data needs no LLM call: no cost, no latency
        try:
            llm = _get_llm(state["metadata"]["llm_model"])
            summary = synthesize_summary(issues, row_count, column_count, column_stats, llm)
        except Exception as exc:
            # Recoverable: the deterministic summary is already in place.
            errors.append(ErrorRecord(
                agent="profiler", error_type="llm_error",
                message=f"LLM synthesis failed, used fallback summary: {exc}",
                timestamp=_now(), recoverable=True,
            ))

    print(f"[Profiler] {len(issues)} issue(s) found")
    return {
        "profiler": ProfilerState(
            run_complete=True,
            row_count=row_count,
            column_count=column_count,
            issues=issues,
            profile_summary=summary,
            column_stats=column_stats,
        ),
        "visualization_events": build_viz_events(issues, findings) + [_round_complete_event(len(issues))],
        "errors": errors,
    }
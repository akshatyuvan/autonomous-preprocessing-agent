"""
agents/stage_runner.py -- ONE generic runner for the encoder, scaler,
imbalance_handler and feature_selector stages.

A stage is just data: a StageSpec (name, registry, which columns it acts on).
For every target column:
  1. options = registry entries whose "requires" all hold for this column
  2. the LLM picks ONE option key -- or a deterministic fallback takes the first that fits
  3. agents/dispatch.py applies it
The stage reads the PREVIOUS stage's snapshot, so the stages form a real pipeline,
and a redo restarts from that same input (it replaces the rejected attempt).
"""
from __future__ import annotations

import json
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator

import config
from agents.dispatch import StrategyNotApplicableError, apply_strategy
from agents.evidence import summarize_column
from agents.llm_factory import make_chat_model
from agents.profiler import load_dataset
from agents.stage_strategies import MAX_ONE_HOT_CATEGORIES, ordered_vocabulary
from state.schema import AgentState, ErrorRecord, VizEvent


def _is_text(s: pd.Series) -> bool:
    return s.dtype == object or isinstance(s.dtype, pd.StringDtype)


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _features(df: pd.DataFrame, target: Optional[str]) -> list[str]:
    return [c for c in df.columns if c != target]


# Precondition name -> predicate(df, column, target). Registry "requires" lists use
# these names, so options are filtered in Python BEFORE the LLM ever sees them.
REQUIREMENTS: dict[str, Callable[[pd.DataFrame, str, Optional[str]], bool]] = {
    "numeric": lambda df, c, t: _is_numeric(df[c]),
    "categorical": lambda df, c, t: _is_text(df[c]),
    "low_cardinality": lambda df, c, t: df[c].nunique() <= MAX_ONE_HOT_CATEGORIES,
    "high_cardinality": lambda df, c, t: df[c].nunique() > MAX_ONE_HOT_CATEGORIES,
    "has_order": lambda df, c, t: ordered_vocabulary(df[c]) is not None,
    "target_column_present": lambda df, c, t: bool(t) and t in df.columns,
    "all_numeric_features": lambda df, c, t: all(_is_numeric(df[x]) for x in _features(df, t)),
    "mixed_feature_types": lambda df, c, t: (any(_is_numeric(df[x]) for x in _features(df, t))
                                             and any(_is_text(df[x]) for x in _features(df, t))),
}


def options_that_fit(registry: dict, df: pd.DataFrame, column: str, target: Optional[str]) -> list[str]:
    """Registry keys whose preconditions all hold, in registry order."""
    return [key for key, entry in registry.items()
            if all(REQUIREMENTS[r](df, column, target) for r in entry.get("requires", []))]


# ---------------------------------------------------------------- which columns a stage acts on
def text_columns(df: pd.DataFrame, target: Optional[str]) -> list[str]:
    return [c for c in _features(df, target) if _is_text(df[c])]


def scalable_columns(df: pd.DataFrame, target: Optional[str]) -> list[str]:
    # nunique > 2: 0/1 dummy columns and binary flags stay as they are.
    return [c for c in _features(df, target) if _is_numeric(df[c]) and df[c].nunique() > 2]


def numeric_columns(df: pd.DataFrame, target: Optional[str]) -> list[str]:
    return [c for c in _features(df, target) if _is_numeric(df[c])]


def imbalanced_target(df: pd.DataFrame, target: Optional[str]) -> list[str]:
    """[target] only when it is a classification target AND actually imbalanced."""
    if not target or target not in df.columns:
        return []
    counts = df[target].value_counts()
    if not 2 <= len(counts) <= 20:
        return []
    return [target] if counts.min() / counts.max() < 0.5 else []


@dataclass(frozen=True)
class StageSpec:
    name: str        # state key AND graph node name, e.g. "scaler"
    id_prefix: str   # decision ids look like "scale_r2_003"
    registry: dict
    targets: Callable[[pd.DataFrame, Optional[str]], list[str]]
    purpose: str     # one sentence telling the LLM what this stage is for


class StageChoice(BaseModel):
    """The ONLY thing the stage LLM produces."""
    strategy: str = Field(description="Exactly one key from allowed_strategies, copied as written")
    justification: str = Field(description="1-2 sentences citing the column statistics used")

    @field_validator("strategy", mode="before")
    @classmethod
    def _normalise_strategy(cls, value):
        if isinstance(value, str):
            return value.strip().strip("'\"` ").lower()
        return value

    @field_validator("justification", mode="before")
    @classmethod
    def _none_to_empty(cls, value):
        return "" if value is None else value


SYSTEM_TEMPLATE = """You are the {name} agent in a data-preprocessing pipeline. {purpose}
You receive ONE column, its statistics, and allowed_strategies. Choose exactly ONE strategy and copy
its "key" exactly as written; Python applies it, you only choose.
If previous_attempt is present, a reviewer rejected that attempt: choose differently unless you can
justify the same choice.
justification: 1-2 sentences citing the statistics you used."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_llm(model: str):
    """Shared by all four stages. Isolated so tests monkeypatch exactly this."""
    return make_chat_model(model).with_structured_output(StageChoice, method="function_calling")


def current_dataset_path(state: AgentState, stage_name: str) -> str:
    """Latest snapshot from a stage that ran BEFORE this one; the raw file otherwise.
    Skipping this stage's own name is what makes a redo restart from its input."""
    for name in reversed(state["metadata"]["pipeline_steps_run"]):
        if name == stage_name:
            continue
        path = (state.get(name) or {}).get("dataset_snapshot_path")
        if path:
            return path
    return state["input"]["dataset_path"]


def _build_messages(spec: StageSpec, df: pd.DataFrame, column: str, target: Optional[str],
                    options: list[str], previous: Optional[dict]) -> list[tuple[str, str]]:
    payload = {
        "column": column,
        "target_column": target,
        "column_stats": summarize_column(df, column),
        "allowed_strategies": [{"key": k,
                                "applies_when": spec.registry[k]["applies_when"],
                                "description": spec.registry[k]["description"]} for k in options],
        "previous_attempt": previous,
    }
    system = SYSTEM_TEMPLATE.format(name=spec.name, purpose=spec.purpose)
    return [("system", system), ("human", json.dumps(payload, indent=2, default=str))]


def run_registry_stage(state: AgentState, spec: StageSpec) -> dict:
    """The body of every registry-driven stage node. Returns PARTIAL state only."""
    own = state[spec.name]
    round_no = own["current_round"] + 1
    previous_decisions = list(own["decisions"])
    print(f"[{spec.name}] Round {round_no}")

    verdict = state["critic"]["current_verdict"]
    feedback = (verdict["reasoning"] if verdict and verdict.get("agent") == spec.name
                and verdict["verdict"] == "reject" else None)

    metadata = dict(state["metadata"])
    metadata["current_active_agent"] = spec.name
    if spec.name not in metadata["pipeline_steps_run"]:
        metadata["pipeline_steps_run"] = metadata["pipeline_steps_run"] + [spec.name]

    target = state["input"]["target_column"]
    new_decisions: list[dict] = []
    events: list[VizEvent] = []
    errors: list[ErrorRecord] = []

    def result(snapshot: Optional[str]) -> dict:
        return {
            # {**own}: keeps stage-specific keys such as user_opted_in.
            spec.name: {**own, "run_complete": True, "current_round": round_no,
                        "decisions": previous_decisions + new_decisions,  # history carried forward
                        "dataset_snapshot_path": snapshot},
            "metadata": metadata,
            "visualization_events": events + [VizEvent(
                event_id=str(uuid.uuid4()), agent=spec.name, event_type="round_complete",
                payload={"message": f"{spec.name} round {round_no}: {len(new_decisions)} decision(s)."},
                timestamp=_now())],
            "errors": errors,
        }

    try:
        df = load_dataset(current_dataset_path(state, spec.name))
    except Exception as exc:
        errors.append(ErrorRecord(agent=spec.name, error_type="dataset_load_error",
                                  message=str(exc), timestamp=_now(), recoverable=False))
        return result(None)

    targets = spec.targets(df, target)
    llm = None
    if targets:
        try:
            llm = _get_llm(state["metadata"]["llm_model"])
        except Exception:
            llm = None  # every decision then falls back

    previous_actions = {d["column"]: d["action"] for d in previous_decisions
                        if d["decision_id"].startswith(f"{spec.id_prefix}_r{round_no - 1}_")}

    for n, column in enumerate(targets, start=1):
        if column not in df.columns:
            continue  # removed by an earlier decision this round (e.g. a dropped feature)
        options = options_that_fit(spec.registry, df, column, target)
        previous = ({"strategy": previous_actions.get(column), "critic_feedback": feedback}
                    if feedback else None)

        choice, error = None, None
        if llm is None:
            error = "llm_error: model could not be created"
        else:
            try:
                choice = llm.invoke(_build_messages(spec, df, column, target, options, previous))
            except Exception as exc:
                error = f"llm_error: {type(exc).__name__}: {exc}"
        llm_key = choice.strategy if choice is not None and choice.strategy in options else None
        if choice is not None and llm_key is None:
            error = f"invalid_choice: '{choice.strategy}' is not one of {options}"

        params = {"target": target}
        candidates = list(dict.fromkeys(([llm_key] if llm_key else []) + options))
        for key in candidates:
            try:
                new_df, stats = apply_strategy(spec.registry, key, df, column, params)
            except (StrategyNotApplicableError, ValueError, TypeError) as exc:
                if key == llm_key:
                    error = f"not_applicable: {exc}"
                continue
            break
        else:
            raise RuntimeError("unreachable: no_action always fits")

        chosen_by = "llm" if key == llm_key else "fallback"
        new_decisions.append({
            "decision_id": f"{spec.id_prefix}_r{round_no}_{n:03d}",
            "issue_id": None,
            "column": column,
            "action": key,
            "parameters": params,
            "justification": (choice.justification if chosen_by == "llm" and choice is not None
                              else "Deterministic fallback: first registry option that fits this column."),
            "status": "applied",
            "redo_count": round_no - 1,
            "chosen_by": chosen_by,
            "stats": {**stats, "before": summarize_column(df, column),
                      "after": summarize_column(new_df, column)},
            "error": None if chosen_by == "llm" else error,
        })
        events.append(VizEvent(
            event_id=str(uuid.uuid4()), agent=spec.name, event_type="cell_status_change",
            payload={"column": column, "action": key, "status": "cleaning"}, timestamp=_now()))
        df = new_df

    problems = Counter(d["error"].split(":")[0] for d in new_decisions if d["error"])
    if problems:
        errors.append(ErrorRecord(
            agent=spec.name, error_type="llm_fallback_used",
            message=(f"LLM choice not used for {sum(problems.values())} of {len(new_decisions)} "
                     f"column(s) {dict(problems)}; deterministic fallback applied"),
            timestamp=_now(), recoverable=True))

    try:
        out_dir = Path(config.CLEANED_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{metadata['session_id']}_{spec.name}_r{round_no}.parquet"
        df.to_parquet(path, index=False)
        snapshot: Optional[str] = str(path)
    except Exception as exc:
        snapshot = None  # the Critic rejects a stage that made decisions but saved nothing
        errors.append(ErrorRecord(agent=spec.name, error_type="snapshot_write_error",
                                  message=str(exc), timestamp=_now(), recoverable=False))

    print(f"[{spec.name}] {len(new_decisions)} decision(s) applied")
    return result(snapshot)
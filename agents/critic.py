"""
agents/critic.py -- the validation gate that runs after EVERY preprocessing stage.

Environment: LOCAL (Mac).

For each decision the stage made that has not been reviewed yet:
  1. Hard checks (pure Python, no LLM): missing before/after evidence, or more
     than MAX_ROW_LOSS of the rows removed in one step -> automatic reject.
  2. LLM judgment on the before/after evidence -> accept or reject + one-sentence reason.
  3. If the LLM output is unusable (unparseable, or the model is down), the decision
     is let through as UNVERIFIED and counted -- blocking on every bad parse of a
     3B model would halt most runs. That rate is part of the case for fine-tuning.

One rejected decision rejects the whole stage. main.critic_router then redoes the
stage, or -- once the stage has used max_rounds attempts -- the Critic sets
halted=True and the graph goes straight to END: a rejected transformation never
reaches the next stage.

Planned, NOT built: a Docker sandbox per decision and a downstream model-score check.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from agents.evidence import build_evidence
from agents.llm_factory import make_chat_model
from agents.registries import CLEANING_REGISTRY
from state.schema import AgentState, CriticState, CriticVerdict, ErrorRecord, VizEvent

# Which registry describes each stage's strategies. Stages are added here as they are built.
STAGE_REGISTRIES = {"cleaner": CLEANING_REGISTRY}

MAX_ROW_LOSS = 0.30   # removing >30% of rows in ONE step is never a routine fix
ROW_LOSS_EXEMPT_ISSUES = {"duplicate_rows"}  # removing rows IS the fix for duplicates


class DecisionJudgment(BaseModel):
    """The ONLY thing the Critic LLM produces, per decision."""
    verdict: Literal["accept", "reject"] = Field(description='Exactly "accept" or "reject"')
    reason: str = Field(description="One sentence naming the specific evidence")

    @field_validator("verdict", mode="before")
    @classmethod
    def _normalise_verdict(cls, value):
        # 3B models answer "Accept.", "REJECTED" or '"reject"'. Repair the format only;
        # anything that is neither still fails validation and counts as malformed.
        if isinstance(value, str):
            text = value.strip().strip("'\"`.! ").lower()
            if text.startswith("acc"):
                return "accept"
            if text.startswith("rej"):
                return "reject"
            return text
        return value

    @field_validator("reason", mode="before")
    @classmethod
    def _none_to_empty(cls, value):
        return "" if value is None else value


SYSTEM_PROMPT = """You are the Critic: the validation gate of a data-preprocessing pipeline.
You review ONE transformation that has already been applied. You receive the data-quality issue it
targeted, the strategy used, what changed, and statistics of the affected column BEFORE and AFTER.
Reject when the transformation damages the data, for example:
- mean imputation on a skewed column (|skew| > 1) or a column with outliers
- removing many rows, or a whole column, when a less destructive strategy fits the issue
- a conversion that turns many real values into missing
- a shift in mean, median or std far larger than fixing the issue requires
- a strategy that does not address the stated issue
Accept when the issue is fixed and the column's statistics stay plausible.
Return verdict ("accept" or "reject") and reason: one sentence citing the specific numbers."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_llm(model: str):
    """Isolated so tests monkeypatch exactly this (all tests run with no model)."""
    return make_chat_model(model).with_structured_output(DecisionJudgment, method="function_calling")


def build_critic_messages(evidence: dict, patterns: Optional[list[str]] = None) -> list[tuple[str, str]]:
    payload: dict = {"transformation": evidence}
    if patterns:
        payload["similar_cases"] = patterns  # Step 6: retrieved cleaning patterns go here
    return [("system", SYSTEM_PROMPT), ("human", json.dumps(payload, indent=2, default=str))]


def judge_with_llm(llm, evidence: dict, patterns: Optional[list[str]] = None) -> DecisionJudgment:
    """One LLM judgment. RAISES on unusable output; the evaluation calls this directly
    and counts those raises as the malformed-output rate."""
    result = llm.invoke(build_critic_messages(evidence, patterns))
    # With function calling, a model that never calls the tool gives None, not an error.
    if not isinstance(result, DecisionJudgment):
        raise ValueError(f"no usable judgment (got {type(result).__name__})")
    return result


def hard_check(decision: dict, issue: Optional[dict]) -> Optional[str]:
    """Deterministic limits applied BEFORE the LLM. Returns a reject reason or None."""
    stats = decision.get("stats") or {}
    if "before" not in stats or "after" not in stats:
        return "no before/after statistics were recorded, so the change cannot be verified"
    if issue is not None and issue["issue_type"] in ROW_LOSS_EXEMPT_ISSUES:
        return None
    rows_before, rows_after = stats.get("rows_before"), stats.get("rows_after")
    if rows_before and rows_after is not None:
        lost = rows_before - rows_after
        if lost / rows_before > MAX_ROW_LOSS:
            return (f"removed {lost} of {rows_before} rows ({lost / rows_before:.0%}), "
                    f"above the {MAX_ROW_LOSS:.0%} limit for one step")
    return None


def critic_node(state: AgentState) -> dict:
    """LangGraph node. Returns PARTIAL state only."""
    critic = state["critic"]
    agent = state["metadata"]["current_active_agent"] or "unknown"

    # Known issue 2: the redo cap is PER STAGE, not one global counter.
    rounds_per_agent = dict(critic["rounds_per_agent"])
    rounds_per_agent[agent] = rounds_per_agent.get(agent, 0) + 1
    attempt = rounds_per_agent[agent]
    print(f"[Critic] Reviewing {agent} (attempt {attempt})")

    stage = state.get(agent) or {}
    # Review only decisions no earlier verdict has seen: a redo produces new IDs.
    reviewed = {i for v in critic["verdicts"] for i in v["decision_ids_reviewed"]}
    decisions = [d for d in stage.get("decisions", []) if d["decision_id"] not in reviewed]
    issues_by_id = {i["issue_id"]: i for i in state["profiler"]["issues"]}
    registry = STAGE_REGISTRIES.get(agent, {})

    rejections: list[tuple[str, str]] = []
    unverified = 0
    if decisions and not stage.get("dataset_snapshot_path"):
        rejections.append(("(stage)", "decisions were applied but no output snapshot was saved"))
    else:
        llm = None
        for d in decisions:
            issue = issues_by_id.get(d.get("issue_id"))
            label = f"{d['action']} on {d['column']}"
            rule = hard_check(d, issue)
            if rule:
                rejections.append((d["decision_id"], f"{label}: {rule}"))
                continue
            description = registry.get(d["action"], {}).get("description", "")
            try:
                if llm is None:
                    llm = _get_llm(state["metadata"]["llm_model"])
                judgment = judge_with_llm(llm, build_evidence(d, issue, description))
            except Exception:
                unverified += 1  # logged and counted; does not block the run
                continue
            if judgment.verdict == "reject":
                rejections.append((d["decision_id"], f"{label}: {judgment.reason}"))

    verdict_value = "reject" if rejections else "accept"
    if rejections:
        reasoning = "Rejected -- " + " | ".join(f"{i}: {text}" for i, text in rejections)
    elif decisions:
        reasoning = f"Accepted all {len(decisions)} decision(s)."
    else:
        reasoning = "Nothing to review: this stage made no decisions."
    if unverified:
        reasoning += f" {unverified} decision(s) could not be judged by the LLM and passed unverified."

    halted = verdict_value == "reject" and attempt >= critic["max_rounds"]
    halt_reason = None
    if halted:
        halt_reason = (f"{agent} was rejected on attempt {attempt} of {critic['max_rounds']}; the run "
                       f"stops so a rejected transformation never reaches the next stage. "
                       f"Last reason: {reasoning}")
        print(f"[Critic] HALT: {halt_reason}")

    verdict = CriticVerdict(
        verdict_id=str(uuid.uuid4()),
        agent=agent,
        attempt=attempt,
        decision_ids_reviewed=[d["decision_id"] for d in decisions],
        rejected_decision_ids=[i for i, _ in rejections if i != "(stage)"],
        verdict=verdict_value,
        reasoning=reasoning,
        distribution_ok=not rejections,
        model_score_delta=None,
        confidence=round((len(decisions) - unverified) / len(decisions), 3) if decisions else 1.0,
        unverified_decisions=unverified,
    )

    errors = []
    if unverified:
        errors.append(ErrorRecord(
            agent="critic", error_type="critic_llm_unverified",
            message=f"{unverified} of {len(decisions)} {agent} decision(s) passed unverified (LLM output unusable)",
            timestamp=_now(), recoverable=True))

    return {
        "critic": CriticState(
            verdicts=critic["verdicts"] + [verdict],  # known issue 1: carry history forward
            current_verdict=verdict,
            total_rounds=critic["total_rounds"] + 1,
            max_rounds=critic["max_rounds"],
            rounds_per_agent=rounds_per_agent,
            halted=halted,
            halt_reason=halt_reason,
        ),
        "visualization_events": [VizEvent(
            event_id=str(uuid.uuid4()), agent="critic", event_type="confidence_update",
            payload={"agent": agent, "verdict": verdict_value,
                     "confidence": verdict["confidence"], "reasoning": reasoning},
            timestamp=_now())],
        "errors": errors,
    }
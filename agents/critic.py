"""
agents/critic.py -- the validation gate that runs after EVERY preprocessing stage.

Environment: LOCAL (Mac).

For each decision the stage made that has not been reviewed yet:
  1. Hard checks (pure Python, no LLM): missing before/after evidence, or more
     than MAX_ROW_LOSS of the rows removed in one step -> automatic reject.
  2. Cleaning decisions: LLM judgment on the before/after evidence -> accept or reject.
     - CRITIC_MODEL set (e.g. "critic-ft"): the QLoRA fine-tuned Critic, prompted in
       the exact plain-JSON format it was trained on. Retrieval stays OFF (measured: it
       hurt the fine-tuned model).
     - CRITIC_MODEL empty: the prompted baseline (tool calling), optionally with
       retrieved labelled cases (CRITIC_RETRIEVAL=true).
     Other stages: deterministic invariants only (no measured LLM judgment for them).
  3. If the LLM output is unusable, the decision passes as UNVERIFIED and is counted.

One rejected decision rejects the whole stage. main.critic_router then redoes the
stage, or -- once the stage has used max_rounds attempts -- the Critic sets
halted=True and the graph goes straight to END: a rejected transformation never
reaches the next stage.

Planned, NOT built: a Docker sandbox per decision and a downstream model-score check.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

import config
from agents.critic_policy import POLICY_TEXT
from agents.evidence import build_evidence
from agents.llm_factory import make_chat_model
from agents.registries import (
    CLEANING_REGISTRY, ENCODING_REGISTRY, FEATURE_SELECTION_REGISTRY, IMBALANCE_REGISTRY,
    SCALING_REGISTRY,
)
from state.schema import AgentState, CriticState, CriticVerdict, ErrorRecord, VizEvent

MAX_ROW_LOSS = 0.30   # removing >30% of rows in ONE step is never a routine fix
ROW_LOSS_EXEMPT_ISSUES = {"duplicate_rows"}  # removing rows IS the fix for duplicates

# Which registry describes each stage's strategies.
STAGE_REGISTRIES = {
    "cleaner": CLEANING_REGISTRY,
    "encoder": ENCODING_REGISTRY,
    "scaler": SCALING_REGISTRY,
    "imbalance_handler": IMBALANCE_REGISTRY,
    "feature_selector": FEATURE_SELECTION_REGISTRY,
}
# The LLM judges only stages with a written policy, a labelled dataset and a MEASURED
# accuracy: cleaning. The other stages are gated by deterministic invariants instead,
# so no unmeasured LLM judgment can stop (or wave through) a run.
LLM_JUDGED_STAGES = {"cleaner"}


def invariant_check(decision: dict) -> Optional[str]:
    """Deterministic gate for non-cleaning stages. Returns a reject reason or None."""
    stats = decision.get("stats") or {}
    before, after = stats.get("before") or {}, stats.get("after") or {}
    if (stats.get("rows_after") or 0) < (stats.get("rows_before") or 0):
        return "this stage must never remove rows"
    if after.get("present") and (after.get("missing") or 0) > (before.get("missing") or 0):
        return (f"introduced {(after.get('missing') or 0) - (before.get('missing') or 0)} "
                f"new missing value(s)")
    return None


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


# The policy text comes from agents/critic_policy.py, the same rules that label the
# evaluation dataset, so the Critic is told exactly what it is graded on.
SYSTEM_PROMPT = f"""You are the Critic: the validation gate of a data-preprocessing pipeline.
You review ONE transformation that has already been applied. You receive the data-quality issue it
targeted, the strategy used, what changed ("change"), and statistics of the affected column
BEFORE and AFTER.

{POLICY_TEXT}

Return verdict ("accept" or "reject") and reason: one sentence citing the rule number and the numbers you used."""

# Appended only when retrieved cases are present, so the no-retrieval prompt is
# byte-for-byte the one the baselines were measured with.
RETRIEVAL_NOTE = """similar_cases are past transformations reviewed under the same policy, each with its
correct verdict and the rule that decided it. Use them to apply the policy consistently,
but decide on THIS transformation's own numbers."""

# EXACTLY the instruction the fine-tuned model was trained with (notebooks/critic-qlora.ipynb).
# A fine-tuned model is only as good as the match between its training and serving format.
JSON_FORMAT_INSTRUCTION = ('\n\nRespond with ONLY a JSON object and nothing else: '
                           '{"reason": "<one sentence citing the rule number and the numbers>", '
                           '"verdict": "accept" or "reject"}')

_VERDICT = re.compile(r'"verdict"\s*:\s*"([^"]*)"')
_REASON = re.compile(r'"reason"\s*:\s*"([^"]*)"')

_RETRIEVER = None  # cached: building the store embeds 300 cases, so do it once per process


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_judgment(text: str) -> DecisionJudgment:
    """Plain-JSON-text answer -> DecisionJudgment. Same rule as the Kaggle evaluation:
    find the "verdict" field by pattern, so a stray word around the JSON doesn't matter.
    Raises ValueError when there is no usable verdict (counted as malformed)."""
    match = _VERDICT.search(text or "")
    if not match:
        raise ValueError(f"no verdict in model output: {(text or '')[:200]!r}")
    reason = _REASON.search(text)
    return DecisionJudgment(verdict=match.group(1), reason=reason.group(1) if reason else "")


class JsonTextJudge:
    """The fine-tuned Critic. It was trained to answer in plain JSON text (no tool
    calling), so it is served the same way: same instruction appended, same parser."""

    def __init__(self, chat_model):
        self.chat_model = chat_model

    def invoke(self, messages: list[tuple[str, str]]) -> DecisionJudgment:
        messages = list(messages)
        role, content = messages[-1]
        messages[-1] = (role, content + JSON_FORMAT_INSTRUCTION)
        return parse_judgment(self.chat_model.invoke(messages).content)


def make_critic_llm(model: str):
    """The real Critic LLM. Separate from _get_llm so tests can check it directly
    (conftest replaces _get_llm with a fake in every test)."""
    if config.CRITIC_MODEL:
        if config.LLM_PROVIDER.strip().lower() != "ollama":
            raise ValueError("CRITIC_MODEL is an Ollama model name; it needs LLM_PROVIDER=ollama")
        return JsonTextJudge(make_chat_model(model, ollama_model=config.CRITIC_MODEL))
    return make_chat_model(model).with_structured_output(DecisionJudgment, method="function_calling")


def _get_llm(model: str):
    """Isolated so tests monkeypatch exactly this (all tests run with no model)."""
    return make_critic_llm(model)


def _get_retriever():
    """None unless CRITIC_RETRIEVAL=true. Isolated so tests can monkeypatch it."""
    global _RETRIEVER
    if not config.CRITIC_RETRIEVAL:
        return None
    if _RETRIEVER is None:
        from vector_store.retrieval import HybridRetriever, default_embedder, load_cases
        _RETRIEVER = HybridRetriever(load_cases(), default_embedder())
    return _RETRIEVER


def build_critic_messages(evidence: dict, patterns: Optional[list[str]] = None) -> list[tuple[str, str]]:
    payload: dict = {"transformation": evidence}
    system = SYSTEM_PROMPT
    if patterns:
        payload["similar_cases"] = patterns
        system = f"{SYSTEM_PROMPT}\n\n{RETRIEVAL_NOTE}"
    return [("system", system), ("human", json.dumps(payload, indent=2, default=str))]


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

    # The redo cap is PER STAGE, not one global counter.
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
    errors: list[ErrorRecord] = []

    retriever = None
    # Retrieval only for the PROMPTED Critic: it measurably hurt the fine-tuned one.
    if decisions and agent in LLM_JUDGED_STAGES and not config.CRITIC_MODEL:
        try:
            retriever = _get_retriever()
        except Exception as exc:
            # Retrieval is an enhancement: if the store can't be built, judge without it.
            errors.append(ErrorRecord(agent="critic", error_type="retrieval_unavailable",
                                      message=str(exc), timestamp=_now(), recoverable=True))

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
            if rule is None and agent not in LLM_JUDGED_STAGES:
                rule = invariant_check(d)
            if rule:
                rejections.append((d["decision_id"], f"{label}: {rule}"))
                continue
            if agent not in LLM_JUDGED_STAGES:
                continue  # passed the deterministic gate; no LLM judgment for this stage
            description = registry.get(d["action"], {}).get("description", "")
            evidence = build_evidence(d, issue, description)
            try:
                patterns = retriever.retrieve(evidence) if retriever is not None else None
                if llm is None:
                    llm = _get_llm(state["metadata"]["llm_model"])
                judgment = judge_with_llm(llm, evidence, patterns)
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

    if unverified:
        errors.append(ErrorRecord(
            agent="critic", error_type="critic_llm_unverified",
            message=f"{unverified} of {len(decisions)} {agent} decision(s) passed unverified (LLM output unusable)",
            timestamp=_now(), recoverable=True))

    return {
        "critic": CriticState(
            verdicts=critic["verdicts"] + [verdict],  # carry history forward explicitly
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
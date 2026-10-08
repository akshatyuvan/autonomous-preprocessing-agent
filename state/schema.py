"""
state/schema.py  —  The single source of truth for what every agent can read/write.

WHY NESTED, NOT FLAT?
---------------------
A flat dict looks tempting at first:
    state = {"missing_pct": 0.3, "issues": [...], "cleaning_log": [...], ...}

But as your graph grows you get:
- Name collisions:  which agent wrote "confidence"? The Critic or the Analyst?
- Unclear ownership: can the Cleaning agent write to "profile_summary"? Should it?
- No IDE help:      state["profile_summary"] fails silently at runtime, not at type-check

Nested TypedDicts give you:
- One sub-dict per agent → clear ownership, easy to reason about
- When you add a new agent later, you add one new key — nothing else breaks

LANGGRAPH STATE BASICS (learn this once, use it everywhere):
- LangGraph passes the full `AgentState` dict into every node function.
- Each node returns a PARTIAL dict — only the keys it changed.
- LangGraph merges partial returns back into the full state automatically.
- `Annotated[list, operator.add]` means "append to this list" instead of replace.
  IMPORTANT: this only works on TOP-LEVEL keys (errors, visualization_events).
  Inside a nested sub-dict it is ignored: returning "cleaner": {...} replaces the
  whole sub-dict, so agents must carry their own history forward explicitly.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, Optional
from typing_extensions import TypedDict


# ---------------------------------------------------------
# SUB-SCHEMAS  (one per agent domain)
# ---------------------------------------------------------

class DataIssue(TypedDict):
    """
    A single data quality problem found by the Profiler.
    Structured so the Cleaning agent can act on it directly.
    """
    issue_id: str                   # e.g. "issue_001"
    column: str                     # which column has the problem
    issue_type: Literal[            # controlled vocab prevents typos
        "missing_values",
        "type_mismatch",
        "outlier",
        "duplicate_rows",
        "inconsistent_category",
        "unknown"
    ]
    severity: Literal["low", "medium", "high"]
    affected_rows: int              # how many rows are impacted
    detail: str                     # human-readable description
    suggested_fix: Optional[str]    # Profiler's hint, Cleaner can override


class ProfilerState(TypedDict):
    """
    Output produced by the Profiler agent.
    Written once; read by Cleaner and ChromaDB retrieval.
    """
    run_complete: bool
    row_count: int
    column_count: int
    issues: list[DataIssue]         # list of all detected problems
    profile_summary: str            # natural language summary for LLM context
    column_stats: dict[str, Any]    # {col: {dtype, null_pct, unique_count, ...}}


class CleaningDecision(TypedDict):
    """
    One cleaning action chosen and applied by the Cleaning agent.
    The Critic evaluates its stats (before/after), not its justification.
    """
    decision_id: str                # "clean_r2_003" = round 2, 3rd issue: unique across redos
    issue_id: str                   # links back to DataIssue.issue_id
    column: str                     # "__all_columns__" for row-level issues (duplicates)
    # A CLEANING_REGISTRY key. Deliberately a plain str, NOT a Literal: with a
    # Literal, every new technique would also need a schema edit, so "one registry
    # entry, no other code changes" would be false. The registry IS the allowed
    # vocabulary, and it is enforced at runtime by agents/dispatch.py.
    action: str
    parameters: dict[str, Any]      # e.g. {"fill_value": 0, "placeholder_tokens": [...]}
    justification: str              # the Cleaner LLM's reason (logged, not shown to the Critic)
    status: Literal[
        "proposed",
        "applied",
        "rejected",
        "redo_requested"
    ]
    redo_count: int                 # how many earlier Cleaner rounds were rejected
    chosen_by: Literal["llm", "fallback"]  # fallback = LLM down, malformed, or unfitting choice
    stats: dict[str, Any]           # strategy stats + column "before"/"after" summaries
    error: Optional[str]            # why the LLM's choice was NOT used, if it wasn't


class CleanerState(TypedDict):
    """
    Output produced by the Cleaning agent across (potentially multiple) rounds.
    `decisions` is a PLAIN list: nested reducers are ignored (see module docstring),
    so cleaner_node carries the history forward itself: old decisions + new ones.
    """
    current_round: int
    decisions: list[CleaningDecision]
    dataset_snapshot_path: Optional[str]   # parquet of the latest cleaned dataset


class CriticVerdict(TypedDict):
    """
    The Critic's verdict on ONE attempt of ONE stage.
    This is the signal that drives the conditional edge in LangGraph.
    """
    verdict_id: str
    agent: str                      # which stage was reviewed, e.g. "cleaner"
    attempt: int                    # that stage's attempt number (1 = first try)
    decision_ids_reviewed: list[str]
    rejected_decision_ids: list[str]
    verdict: Literal["accept", "reject"]
    reasoning: str                  # fed back to the stage on a redo
    distribution_ok: bool
    model_score_delta: Optional[float]  # planned: quick downstream model check (not built)
    confidence: float               # share of decisions actually judged (not unverified)
    unverified_decisions: int       # LLM output unusable -> let through, counted, logged


class APIResponse(TypedDict):
    """
    Structured output returned by the FastAPI endpoint.
    Built at the end of the pipeline from existing state.
    No agent writes to this — it's assembled once at the end.
    """
    session_id: str
    dataset_name: str
    cleaned_data_path: str
    agent_trace: list[dict]        # summary of every decision made
    confidence_scores: dict        # per-column confidence after cleaning
    insights: list[Insight]
    total_rounds: int
    errors: list[ErrorRecord]


class CriticState(TypedDict):
    """
    Accumulated output of the Critic across all stages and rounds.
    `verdicts` is a PLAIN list (nested reducers are ignored): critic_node
    appends to it explicitly.
    """
    verdicts: list[CriticVerdict]
    current_verdict: Optional[CriticVerdict]  # latest one, for routing
    total_rounds: int               # total reviews across ALL stages (for traces)
    max_rounds: int                 # max attempts PER STAGE before the run halts
    rounds_per_agent: dict[str, int]  # attempts reviewed per stage -- the cap is per stage
    halted: bool                    # True -> critic_router sends the graph to END
    halt_reason: Optional[str]


class Insight(TypedDict):
    """One finding from the Analyst, with an explicit trustworthiness label."""
    insight_id: str
    finding: str
    trustworthiness: Literal["high", "medium", "low"]
    reason_for_rating: str          # e.g. "3 of 5 values in this column were imputed"


class AnalystState(TypedDict):
    """Output of the Analyst agent, only populated after Critic accepts."""
    run_complete: bool
    insights: list[Insight]
    analysis_summary: str
    retrieved_sources: list[dict]      # what was fetched from knowledge base
    rag_collection: str                # which ChromaDB collection was queried

class EncodingDecision(TypedDict):
    decision_id: str
    column: str
    method: Literal["one_hot", "ordinal", "target_encoding", "no_action"]
    justification: str
    status: Literal["proposed", "applied", "rejected", "redo_requested"]
    redo_count: int

class EncodingState(TypedDict):
    run_complete: bool
    decisions: Annotated[list[EncodingDecision], operator.add]
    dataset_snapshot_path: Optional[str]


class ScalingDecision(TypedDict):
    decision_id: str
    column: str
    method: Literal["standard", "minmax", "robust", "no_action"]
    justification: str
    status: Literal["proposed", "applied", "rejected", "redo_requested"]
    redo_count: int

class ScalingState(TypedDict):
    run_complete: bool
    decisions: Annotated[list[ScalingDecision], operator.add]
    dataset_snapshot_path: Optional[str]


class ImbalanceDecision(TypedDict):
    decision_id: str
    technique: Literal["smote", "smote_nc", "borderline_smote", "adasyn", "smote_tomek", "class_weights", "no_action"]
    justification: str
    class_distribution_before: dict[str, int]
    class_distribution_after: dict[str, int]
    status: Literal["proposed", "applied", "rejected", "redo_requested"]
    redo_count: int

class ImbalanceState(TypedDict):
    run_complete: bool
    user_opted_in: bool          # always check this before running
    decisions: Annotated[list[ImbalanceDecision], operator.add]
    dataset_snapshot_path: Optional[str]


class FeatureSelectionDecision(TypedDict):
    decision_id: str
    column: str
    action: Literal["keep", "drop_low_variance", "drop_correlated", "drop_low_mutual_info"]
    justification: str
    status: Literal["proposed", "applied", "rejected", "redo_requested"]
    redo_count: int

class FeatureSelectorState(TypedDict):
    run_complete: bool
    decisions: Annotated[list[FeatureSelectionDecision], operator.add]
    dataset_snapshot_path: Optional[str]

# ---------------------------------------------------------
# TOP-LEVEL STATE  (what LangGraph sees)
# ---------------------------------------------------------

class AgentState(TypedDict):
    """
    The single state object threaded through the entire graph.

    Design rules:
    1. `input` and analyst are top-level exit points -- easy to find.
    2. Each agent owns exactly one sub-dict. No agent writes to another's.
    3. `metadata` holds config/session info.
    4. `errors` uses operator.add so any node can append without overwriting.
    5. `visualization_events` is the bus for the frontend. Agents push events;
       the viz layer reads them. Clean separation of concerns.
    """

    # Entry point
    input: InputState

    # Per-agent namespaces
    profiler: ProfilerState
    cleaner: CleanerState
    encoder: EncodingState
    scaler: ScalingState
    imbalance_handler: ImbalanceState
    feature_selector: FeatureSelectorState
    critic: CriticState
    analyst: AnalystState

    # Cross-cutting concerns
    metadata: MetadataState
    errors: Annotated[list[ErrorRecord], operator.add]

    # Frontend event bus (visualization layer reads this)
    visualization_events: Annotated[list[VizEvent], operator.add]
    api_response: Optional[APIResponse]

class RequestedSteps(TypedDict):
    """
    User's selection of which pipeline steps to run.
    Defaults applied in make_initial_state() if not provided.
    """
    cleaning: bool
    encoding: bool
    scaling: bool
    imbalance_handling: bool       # always defaults to False — opt-in only
    feature_selection: bool
    datetime_engineering: bool


class InputState(TypedDict):
    """
    Provided at graph invocation time. Never mutated after that.
    """
    dataset_path: str               # path to the raw messy CSV / parquet
    dataset_name: str               # human label for logs/UI
    target_column: Optional[str]    # for downstream eval, can be None
    requested_steps: RequestedSteps


class MetadataState(TypedDict):
    """
    Session-level config. Set once at invocation; agents only update
    current_active_agent and pipeline_steps_run.
    """
    session_id: str
    started_at: str                 # ISO timestamp
    max_critic_rounds: int          # cap for the redo loop (default 3)
    llm_model: str                  # e.g. "gpt-4o-mini" or fine-tuned model path
    chroma_collection: str          # which ChromaDB collection to query
    rag_collection: str             # Collection 2 (analyst RAG)
    current_active_agent: Optional[str]   # which agent is running right now
    pipeline_steps_run: list[str]          # actual execution order (for trace + dependency notes)


class ErrorRecord(TypedDict):
    """Structured error -- any node can append one of these to state['errors']."""
    agent: str                      # which agent raised this
    error_type: str                 # e.g. "llm_fallback_used", "dataset_load_error"
    message: str
    timestamp: str
    recoverable: bool               # can the graph continue, or must it halt?


class VizEvent(TypedDict):
    """
    Events pushed by agents for the visualization layer to consume.
    Agents don't know about the UI -- they just emit structured events.

    cell_status values map to UI colors:
      "untouched"  -> grey
      "flagged"    -> amber
      "cleaning"   -> blue flash
      "rejected"   -> red flash -> back to amber
      "verified"   -> green
    """
    event_id: str
    agent: Literal["profiler", "cleaner", "critic", "analyst", "encoder", "scaler", "imbalance_handler", "feature_selector"]
    event_type: Literal[
        "cell_status_change",
        "critic_reasoning_chunk",   # streamed text for side panel
        "confidence_update",        # drives the arc widget
        "round_complete"
    ]
    payload: dict[str, Any]         # {col, row, status} or {text_chunk} etc.
    timestamp: str


# ---------------------------------------------------------
# INITIALIZER  -- creates a valid empty state
# ---------------------------------------------------------

def make_initial_state(
    dataset_path: str,
    dataset_name: str,
    session_id: str = "",
    target_column: Optional[str] = None,
    llm_model: str = "gpt-4o-mini",
    max_critic_rounds: int = 3,
    chroma_collection: str = "data_quality_patterns",
    requested_steps: Optional[RequestedSteps] = None,
) -> AgentState:
    """
    Returns a fully-initialized AgentState with sensible defaults.
    Call this before invoking the graph -- never build state by hand in tests.
    """
    import uuid
    from datetime import datetime, timezone

    # Default: everything on except imbalance handling (opt-in only)
    if requested_steps is None:
        requested_steps = RequestedSteps(
            cleaning=True,
            encoding=True,
            scaling=True,
            imbalance_handling=False,
            feature_selection=True,
            datetime_engineering=True,
        )

    return AgentState(
        input=InputState(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            target_column=target_column,
            requested_steps=requested_steps,
        ),
        profiler=ProfilerState(
            run_complete=False,
            row_count=0,
            column_count=0,
            issues=[],
            profile_summary="",
            column_stats={},
        ),
        cleaner=CleanerState(
            current_round=0,
            decisions=[],
            dataset_snapshot_path=None
        ),
        encoder=EncodingState(
            run_complete=False,
            decisions=[],
            dataset_snapshot_path=None
        ),
        scaler=ScalingState(
            run_complete=False,
            decisions=[],
            dataset_snapshot_path=None
        ),
        imbalance_handler=ImbalanceState(
            run_complete=False,
            user_opted_in=requested_steps["imbalance_handling"],
            decisions=[],
            dataset_snapshot_path=None,
        ),
        feature_selector=FeatureSelectorState(
            run_complete=False,
            decisions=[],
            dataset_snapshot_path=None
        ),
        critic=CriticState(
            verdicts=[],
            current_verdict=None,
            total_rounds=0,
            max_rounds=max_critic_rounds,
            rounds_per_agent={},
            halted=False,
            halt_reason=None,
        ),
        analyst=AnalystState(
            run_complete=False,
            insights=[],
            analysis_summary="",
            retrieved_sources=[],
            rag_collection="data_science_knowledge_base",
        ),
        metadata=MetadataState(
            session_id=session_id or str(uuid.uuid4()),
            started_at=datetime.now(timezone.utc).isoformat(),
            max_critic_rounds=max_critic_rounds,
            llm_model=llm_model,
            chroma_collection=chroma_collection,
            rag_collection="data_science_knowledge_base",
            current_active_agent=None,
            pipeline_steps_run=[],
        ),
        errors=[],
        visualization_events=[],
        api_response=None,
    )
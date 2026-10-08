"""
api/main.py -- thin synchronous HTTP API over the pipeline.

Environment: LOCAL (Mac).
  Run:  uvicorn api.main:app --reload     then open http://127.0.0.1:8000/docs

"Thin" and "synchronous" on purpose: one request runs the whole graph and returns
when it finishes. With a local 3B model that can take minutes; a production version
would queue the job and return a job id instead (README: Planned, not built).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import config
from main import run_pipeline

app = FastAPI(title="Autonomous Preprocessing Agent", version="0.1.0")


class RequestedSteps(BaseModel):
    cleaning: bool = True
    encoding: bool = True
    scaling: bool = True
    imbalance_handling: bool = False  # opt-in, and needs target_column
    feature_selection: bool = True
    datetime_engineering: bool = False


class RunRequest(BaseModel):
    dataset_path: str = Field(description="CSV or parquet file inside data/raw/")
    dataset_name: str = "api_run"
    target_column: Optional[str] = None
    requested_steps: RequestedSteps = Field(default_factory=RequestedSteps)


class DecisionSummary(BaseModel):
    stage: str
    decision_id: str
    column: str
    action: str
    chosen_by: str


class VerdictSummary(BaseModel):
    agent: str
    attempt: int
    verdict: str
    reasoning: str


class RunResponse(BaseModel):
    session_id: str
    steps_run: list[str]
    completed: bool           # every requested stage passed its gate
    halted: bool              # the Critic stopped the run
    halt_reason: Optional[str]
    output_path: Optional[str]
    decisions: list[DecisionSummary]
    verdicts: list[VerdictSummary]
    errors: list[dict]


STAGES = ["cleaner", "encoder", "scaler", "imbalance_handler", "feature_selector"]


def _safe_path(raw: str) -> Path:
    """Only files inside data/raw/ may be read. Without this check, a caller could
    make the server read ANY file it can access, e.g. '../../.env' (path traversal)."""
    base = Path(config.RAW_DIR).resolve()  # read at call time so tests can redirect it
    path = Path(raw).resolve()             # resolve() collapses '..' before the check
    if not path.is_relative_to(base):
        raise HTTPException(status_code=400, detail="dataset_path must be inside data/raw/")
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"no such file: {raw}")
    return path


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# A plain `def` (not `async def`): FastAPI runs it in a worker thread, so the
# long, blocking pipeline call doesn't freeze the server's event loop.
@app.post("/pipeline/run", response_model=RunResponse)
def run(request: RunRequest) -> RunResponse:
    path = _safe_path(request.dataset_path)
    state = run_pipeline(
        dataset_path=str(path),
        dataset_name=request.dataset_name,
        target_column=request.target_column,
        requested_steps=request.requested_steps.model_dump(),
    )
    steps = state["metadata"]["pipeline_steps_run"]
    output = next((state[s]["dataset_snapshot_path"] for s in reversed(steps)
                   if state[s]["dataset_snapshot_path"]), None)
    return RunResponse(
        session_id=state["metadata"]["session_id"],
        steps_run=steps,
        completed=state["analyst"]["run_complete"],
        halted=state["critic"]["halted"],
        halt_reason=state["critic"]["halt_reason"],
        output_path=output,
        decisions=[DecisionSummary(stage=s, decision_id=d["decision_id"], column=d["column"],
                                   action=d["action"], chosen_by=d["chosen_by"])
                   for s in STAGES for d in state[s]["decisions"]],
        verdicts=[VerdictSummary(agent=v["agent"], attempt=v["attempt"], verdict=v["verdict"],
                                 reasoning=v["reasoning"]) for v in state["critic"]["verdicts"]],
        errors=[dict(e) for e in state["errors"]],
    )
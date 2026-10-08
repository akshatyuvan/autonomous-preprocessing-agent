"""
agents/scaler.py -- Scaling stage: puts numeric features on comparable scales.
All logic lives in agents/stage_runner.py; this file only DESCRIBES the stage.
"""
from agents.registries import SCALING_REGISTRY
from agents.stage_runner import StageSpec, run_registry_stage, scalable_columns
from state.schema import AgentState

SPEC = StageSpec(
    name="scaler",
    id_prefix="scale",
    registry=SCALING_REGISTRY,
    targets=scalable_columns,
    purpose="Put each numeric feature on a scale suited to its distribution.",
)


def scaler_node(state: AgentState) -> dict:
    return run_registry_stage(state, SPEC)
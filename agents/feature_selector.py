"""
agents/feature_selector.py -- Feature selection stage: removes uninformative features.
All logic lives in agents/stage_runner.py; this file only DESCRIBES the stage.
"""
from agents.registries import FEATURE_SELECTION_REGISTRY
from agents.stage_runner import StageSpec, numeric_columns, run_registry_stage
from state.schema import AgentState

SPEC = StageSpec(
    name="feature_selector",
    id_prefix="feat",
    registry=FEATURE_SELECTION_REGISTRY,
    targets=numeric_columns,
    purpose="Keep each feature unless it is constant, redundant, or unrelated to the target.",
)


def feature_selector_node(state: AgentState) -> dict:
    return run_registry_stage(state, SPEC)
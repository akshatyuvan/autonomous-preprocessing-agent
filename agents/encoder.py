"""
agents/encoder.py -- Encoding stage: turns text/categorical columns into numbers.
All logic lives in agents/stage_runner.py; this file only DESCRIBES the stage.
"""
from agents.registries import ENCODING_REGISTRY
from agents.stage_runner import StageSpec, run_registry_stage, text_columns
from state.schema import AgentState

SPEC = StageSpec(
    name="encoder",
    id_prefix="enc",
    registry=ENCODING_REGISTRY,
    targets=text_columns,
    purpose="Turn each text/categorical column into numbers a model can use.",
)


def encoder_node(state: AgentState) -> dict:
    return run_registry_stage(state, SPEC)
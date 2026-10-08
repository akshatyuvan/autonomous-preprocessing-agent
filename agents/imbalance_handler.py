"""
agents/imbalance_handler.py -- Class-imbalance stage (opt-in, needs a target column).
All logic lives in agents/stage_runner.py; this file only DESCRIBES the stage.
"""
from agents.registries import IMBALANCE_REGISTRY
from agents.stage_runner import StageSpec, imbalanced_target, run_registry_stage
from state.schema import AgentState

SPEC = StageSpec(
    name="imbalance_handler",
    id_prefix="imb",
    registry=IMBALANCE_REGISTRY,
    targets=imbalanced_target,
    purpose="Correct class imbalance in the target column without distorting the data.",
)


def imbalance_handler_node(state: AgentState) -> dict:
    return run_registry_stage(state, SPEC)
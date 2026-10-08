"""
main.py  --  Builds and runs the LangGraph agent graph.

GRAPH FLOW:
  START -> profiler -> orchestrator_router -> {cleaner | encoder | scaler |
                                                imbalance_handler |
                                                feature_selector | analyst | END}
  every preprocessing agent -> critic -> critic_router:
      accept                         -> orchestrator_proxy (picks the NEXT requested step)
      reject, attempts left          -> the SAME agent again (redo)
      reject, attempts used up       -> END (halted: a rejected transformation never
                                        reaches the next stage -- resume bullet 2)

The critic is ONE shared node; the routing functions only READ state.
The decision to halt is made inside critic_node (it knows the attempt count),
so the router stays a trivial, testable lookup.
"""

from langgraph.graph import END, START, StateGraph

from agents.analyst import analyst_node
from agents.cleaner import cleaner_node
from agents.critic import critic_node
from agents.encoder import encoder_node
from agents.feature_selector import feature_selector_node
from agents.imbalance_handler import imbalance_handler_node
from agents.profiler import profiler_node
from agents.scaler import scaler_node
from state.schema import AgentState, make_initial_state

# Shared by the profiler edge and the orchestrator_proxy edge: same decision, same targets.
STAGE_ROUTES = {
    "go_to_cleaner": "cleaner",
    "go_to_encoder": "encoder",
    "go_to_scaler": "scaler",
    "go_to_imbalance_handler": "imbalance_handler",
    "go_to_feature_selector": "feature_selector",
    "go_to_analyst": "analyst",
    "halt": END,
}


def orchestrator_router(state: AgentState) -> str:
    """
    Picks the next requested step that hasn't run yet. Called right after the
    Profiler (first step) and after every Critic accept (next step).
    """
    # No profile means no dataset: stop instead of running every stage on nothing.
    if not state["profiler"]["run_complete"]:
        return "halt"

    steps = state["input"]["requested_steps"]
    pipeline_steps_run = state["metadata"]["pipeline_steps_run"]

    # Dependency rule: encoding/scaling/feature_selection need cleaning first
    needs_cleaning = steps["encoding"] or steps["scaling"] or steps["feature_selection"]
    if needs_cleaning and not steps["cleaning"] and "cleaner" not in pipeline_steps_run:
        return "go_to_cleaner"

    if steps["cleaning"] and "cleaner" not in pipeline_steps_run:
        return "go_to_cleaner"
    if steps["encoding"] and "encoder" not in pipeline_steps_run:
        return "go_to_encoder"
    if steps["scaling"] and "scaler" not in pipeline_steps_run:
        return "go_to_scaler"
    if steps["imbalance_handling"] and "imbalance_handler" not in pipeline_steps_run:
        return "go_to_imbalance_handler"
    if steps["feature_selection"] and "feature_selector" not in pipeline_steps_run:
        return "go_to_feature_selector"

    return "go_to_analyst"  # all requested steps done


def critic_router(state: AgentState) -> str:
    """halted -> END; accept -> orchestrator; reject -> redo the stage that was reviewed."""
    critic = state["critic"]
    if critic["halted"]:
        return "halt"
    verdict = critic["current_verdict"]
    if verdict is None or verdict["verdict"] == "accept":
        return "go_to_orchestrator"
    # The verdict names the stage it reviewed, so the redo target can't drift.
    return f"redo_{verdict['agent']}"


def build_graph():
    """Constructs the agent graph and returns it compiled, ready to invoke."""
    graph = StateGraph(AgentState)

    graph.add_node("profiler", profiler_node)
    graph.add_node("cleaner", cleaner_node)
    graph.add_node("encoder", encoder_node)
    graph.add_node("scaler", scaler_node)
    graph.add_node("imbalance_handler", imbalance_handler_node)
    graph.add_node("feature_selector", feature_selector_node)
    graph.add_node("critic", critic_node)
    graph.add_node("analyst", analyst_node)
    # Conditional edges need a NODE to start from, so "back to the orchestrator"
    # is a no-op hop that re-runs the same routing function.
    graph.add_node("orchestrator_proxy", lambda state: {})

    graph.add_edge(START, "profiler")
    graph.add_conditional_edges("profiler", orchestrator_router, STAGE_ROUTES)

    # Every preprocessing agent hands off to the SAME shared critic: the gate.
    for agent_name in ["cleaner", "encoder", "scaler", "imbalance_handler", "feature_selector"]:
        graph.add_edge(agent_name, "critic")

    graph.add_conditional_edges(
        "critic",
        critic_router,
        {
            "go_to_orchestrator": "orchestrator_proxy",
            "redo_cleaner": "cleaner",
            "redo_encoder": "encoder",
            "redo_scaler": "scaler",
            "redo_imbalance_handler": "imbalance_handler",
            "redo_feature_selector": "feature_selector",
            "halt": END,
        },
    )
    graph.add_conditional_edges("orchestrator_proxy", orchestrator_router, STAGE_ROUTES)

    graph.add_edge("analyst", END)
    return graph.compile()


def run_pipeline(
    dataset_path: str,
    dataset_name: str = "my_dataset",
    target_column: str | None = None,
    requested_steps: dict | None = None,
) -> AgentState:
    graph = build_graph()

    initial_state = make_initial_state(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        target_column=target_column,
        requested_steps=requested_steps,
    )

    print(f"\n{'='*60}")
    print(f"Starting pipeline for: {dataset_name}")
    print(f"Session: {initial_state['metadata']['session_id']}")
    print(f"{'='*60}\n")

    final_state = graph.invoke(initial_state, {"recursion_limit": 50})

    print(f"\n{'='*60}")
    if final_state["critic"]["halted"]:
        print(f"RUN HALTED: {final_state['critic']['halt_reason']}")
    elif not final_state["analyst"]["run_complete"]:
        print("Run stopped early (see errors).")
    else:
        print("Pipeline complete.")
    print(f"Steps run: {final_state['metadata']['pipeline_steps_run']}")
    print(f"Viz events emitted: {len(final_state['visualization_events'])}")
    print(f"Errors recorded: {len(final_state['errors'])}")
    print(f"{'='*60}\n")

    return final_state


if __name__ == "__main__":
    result = run_pipeline(
        dataset_path="data/raw/sample.csv",
        dataset_name="smoke_test",
    )
    print("Critic attempts per stage:", result["critic"]["rounds_per_agent"])
    for v in result["critic"]["verdicts"]:
        print(f"- {v['agent']} attempt {v['attempt']}: {v['verdict']} -- {v['reasoning'][:300]}")
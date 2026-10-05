"""
tests/test_graph.py — graph structure and routing tests.

Environment: LOCAL (Mac). LLM mocked via tests/conftest.py.
"""
import pandas as pd
import pytest

from main import build_graph, run_pipeline


@pytest.fixture
def csv_path(tmp_path):
    # A real, clean file: from Day 2 the Profiler actually loads the dataset,
    # so "fake.csv" would now (correctly) produce a load error.
    path = tmp_path / "tiny.csv"
    pd.DataFrame({
        "age": [25.0, 30.0, 35.0, 40.0],
        "city": ["Pune", "Mumbai", "Delhi", "Pune"],
    }).to_csv(path, index=False)
    return str(path)


def test_graph_builds_without_error():
    assert build_graph() is not None


def test_graph_runs_end_to_end(csv_path):
    result = run_pipeline(dataset_path=csv_path, dataset_name="test_run")
    assert result["profiler"]["run_complete"] is True
    assert result["analyst"]["run_complete"] is True


def test_critic_runs_once_per_step_without_looping(csv_path):
    # The Critic is ONE shared node reviewing every pipeline step, so with the
    # always-accept placeholder it runs exactly once per step: no redo loops.
    # (The old assertion `total_rounds == 1` predates the extra steps.)
    result = run_pipeline(csv_path, "loop_test")
    steps = result["metadata"]["pipeline_steps_run"]
    assert len(steps) == len(set(steps))
    assert result["critic"]["total_rounds"] == len(steps)


def test_viz_events_are_accumulated(csv_path):
    result = run_pipeline(csv_path, "viz_test")
    assert len(result["visualization_events"]) >= 4


def test_errors_list_is_empty_on_clean_run(csv_path):
    result = run_pipeline(csv_path, "error_test")
    assert result["errors"] == []
"""
tests/test_stages.py -- the four registry-driven stages and their gate.
LLM always faked. Environment: LOCAL (Mac).
"""
import numpy as np
import pandas as pd
import pytest

from agents import critic as critic_mod
from agents import stage_runner
from agents.critic import critic_node
from agents.dispatch import StrategyNotApplicableError, apply_strategy
from agents.encoder import encoder_node
from agents.feature_selector import feature_selector_node
from agents.registries import (
    ENCODING_REGISTRY, FEATURE_SELECTION_REGISTRY, IMBALANCE_REGISTRY, SCALING_REGISTRY,
)
from agents.scaler import scaler_node
from agents.stage_runner import REQUIREMENTS, StageChoice, current_dataset_path
from main import run_pipeline
from state.schema import make_initial_state
from tests.test_cleaner import _write_messy_csv
from tests.test_critic import _decision


def _state_after(tmp_path, df, target=None):
    """A state where the Cleaner already ran and left `df` as its snapshot."""
    path = tmp_path / "prev.parquet"
    df.to_parquet(path, index=False)
    state = make_initial_state(dataset_path=str(tmp_path / "raw.csv"), dataset_name="t",
                               target_column=target)
    state["cleaner"]["dataset_snapshot_path"] = str(path)
    state["metadata"]["pipeline_steps_run"] = ["cleaner"]
    return state


class _FixedStageLLM:
    def __init__(self, strategy):
        self.strategy = strategy

    def invoke(self, messages):
        return StageChoice(strategy=self.strategy, justification="test")


def test_every_stage_registry_entry_is_complete():
    for registry in (ENCODING_REGISTRY, SCALING_REGISTRY, IMBALANCE_REGISTRY,
                     FEATURE_SELECTION_REGISTRY):
        assert "no_action" in registry
        for key, entry in registry.items():
            assert callable(entry["apply"]), key
            assert entry["applies_when"] and entry["description"], key
            assert set(entry.get("requires", [])) <= set(REQUIREMENTS), key


def test_scaler_standardises_numeric_columns_but_not_binary_ones(tmp_path):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "flag": [0, 1, 0, 1, 0]})
    out = scaler_node(_state_after(tmp_path, df))
    snap = pd.read_parquet(out["scaler"]["dataset_snapshot_path"])
    assert snap["x"].mean() == pytest.approx(0.0, abs=1e-9)
    assert snap["x"].std(ddof=0) == pytest.approx(1.0)
    assert snap["flag"].tolist() == [0, 1, 0, 1, 0]
    assert [d["action"] for d in out["scaler"]["decisions"]] == ["standard"]


def test_encoder_uses_ordinal_for_ordered_values_and_one_hot_otherwise(tmp_path):
    df = pd.DataFrame({"city": ["Pune", "Delhi", "Pune", "Delhi"],
                       "size": ["small", "large", "medium", "small"]})
    out = encoder_node(_state_after(tmp_path, df))
    snap = pd.read_parquet(out["encoder"]["dataset_snapshot_path"])
    assert [d["action"] for d in out["encoder"]["decisions"]] == ["one_hot", "ordinal"]
    assert "city" not in snap.columns and "city_Pune" in snap.columns
    assert snap["size"].tolist() == [0, 2, 1, 0]


def test_feature_selector_drops_only_when_the_rule_holds(tmp_path, monkeypatch):
    monkeypatch.setattr(stage_runner, "_get_llm", lambda *a, **k: _FixedStageLLM("drop_low_variance"))
    df = pd.DataFrame({"c": [1.0] * 6, "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    out = feature_selector_node(_state_after(tmp_path, df))
    snap = pd.read_parquet(out["feature_selector"]["dataset_snapshot_path"])
    assert list(snap.columns) == ["x"]  # the constant column went; the varied one stayed
    constant, varied = out["feature_selector"]["decisions"]
    assert constant["action"] == "drop_low_variance" and constant["chosen_by"] == "llm"
    assert varied["action"] == "no_action" and varied["chosen_by"] == "fallback"
    assert varied["error"].startswith("not_applicable")


def test_class_weights_are_balanced_weights():
    df = pd.DataFrame({"a": range(100), "label": ["yes"] * 90 + ["no"] * 10})
    _, stats = apply_strategy(IMBALANCE_REGISTRY, "class_weights", df, "label", {"target": "label"})
    assert stats["class_weights"] == {"no": 5.0, "yes": 0.5556}


def test_smote_balances_the_classes():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(100, 2)), columns=["a", "b"])
    df["label"] = ["yes"] * 90 + ["no"] * 10
    out, stats = apply_strategy(IMBALANCE_REGISTRY, "smote", df, "label", {"target": "label"})
    assert stats["class_counts_after"] == {"no": 90, "yes": 90}
    assert list(out.columns) == ["a", "b", "label"]


def test_target_encoding_refuses_without_a_target():
    df = pd.DataFrame({"city": ["Pune", "Delhi"]})
    with pytest.raises(StrategyNotApplicableError, match="target"):
        apply_strategy(ENCODING_REGISTRY, "target_encoding", df, "city", {"target": None})


def test_each_stage_reads_the_previous_stages_snapshot():
    state = make_initial_state(dataset_path="raw.csv", dataset_name="t")
    assert current_dataset_path(state, "cleaner") == "raw.csv"
    state["cleaner"]["dataset_snapshot_path"] = "c.parquet"
    state["encoder"]["dataset_snapshot_path"] = "e.parquet"
    state["metadata"]["pipeline_steps_run"] = ["cleaner", "encoder"]
    assert current_dataset_path(state, "scaler") == "e.parquet"
    assert current_dataset_path(state, "encoder") == "c.parquet"  # a redo restarts from its input


class _CountingLLM:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        raise AssertionError("non-cleaning stages must not call the Critic LLM")


def test_critic_gates_other_stages_deterministically(monkeypatch):
    llm = _CountingLLM()
    monkeypatch.setattr(critic_mod, "_get_llm", lambda *a, **k: llm)
    state = make_initial_state(dataset_path="raw.csv", dataset_name="t")
    state["metadata"]["current_active_agent"] = "scaler"

    bad = _decision("scale_r1_001")
    bad["stats"]["before"] = {"present": True, "missing": 0}
    bad["stats"]["after"] = {"present": True, "missing": 3}
    state["scaler"].update(decisions=[bad], dataset_snapshot_path="s.parquet", current_round=1)
    out = critic_node(state)
    assert out["critic"]["current_verdict"]["verdict"] == "reject"
    assert "missing" in out["critic"]["current_verdict"]["reasoning"]

    state["critic"] = out["critic"]
    state["scaler"]["decisions"] = [bad, _decision("scale_r2_001")]
    out = critic_node(state)
    assert out["critic"]["current_verdict"]["verdict"] == "accept"
    assert llm.calls == 0


def test_full_pipeline_runs_every_requested_stage(tmp_path):
    csv = tmp_path / "messy.csv"
    _write_messy_csv(csv)
    result = run_pipeline(str(csv), "all_stages")
    assert result["metadata"]["pipeline_steps_run"] == [
        "cleaner", "encoder", "scaler", "feature_selector"]
    assert result["analyst"]["run_complete"] is True
    assert result["scaler"]["decisions"]
    assert all(d["chosen_by"] == "llm" for d in result["encoder"]["decisions"])
"""
tests/test_critic.py -- the validation gate. LLM always faked. Environment: LOCAL (Mac).
"""
from agents import critic as critic_mod
from agents.critic import DecisionJudgment, critic_node
from state.schema import make_initial_state


class _JudgeLLM:
    def __init__(self, verdict="accept", reason="ok"):
        self.verdict, self.reason, self.calls = verdict, reason, 0

    def invoke(self, messages):
        self.calls += 1
        return DecisionJudgment(verdict=self.verdict, reason=self.reason)


class _BrokenJudge:
    def invoke(self, messages):
        raise ValueError("model returned text that is not valid JSON")


def _use_llm(monkeypatch, llm):
    monkeypatch.setattr(critic_mod, "_get_llm", lambda *args, **kwargs: llm)


def _decision(decision_id, rows_before=10, rows_after=10):
    return {"decision_id": decision_id, "issue_id": "issue_001", "column": "age",
            "action": "impute_median", "parameters": {}, "justification": "x",
            "status": "applied", "redo_count": 0, "chosen_by": "llm", "error": None,
            "stats": {"strategy": "impute_median", "column": "age",
                      "rows_before": rows_before, "rows_after": rows_after,
                      "before": {"missing": 2}, "after": {"missing": 0}}}


def _state(decisions, issue_type="missing_values", snapshot="snap.parquet"):
    state = make_initial_state(dataset_path="unused.csv", dataset_name="t")
    state["metadata"]["current_active_agent"] = "cleaner"
    state["cleaner"] = {"current_round": 1, "decisions": decisions,
                        "dataset_snapshot_path": snapshot}
    state["profiler"]["issues"] = [{
        "issue_id": "issue_001", "column": "age", "issue_type": issue_type,
        "severity": "high", "affected_rows": 2, "detail": "d", "suggested_fix": None}]
    return state


def test_stage_without_decisions_is_accepted_without_llm(monkeypatch):
    llm = _JudgeLLM("reject")
    _use_llm(monkeypatch, llm)
    state = make_initial_state(dataset_path="unused.csv", dataset_name="t")
    state["metadata"]["current_active_agent"] = "encoder"
    out = critic_node(state)
    assert out["critic"]["current_verdict"]["verdict"] == "accept"
    assert llm.calls == 0


def test_llm_rejection_rejects_the_stage(monkeypatch):
    _use_llm(monkeypatch, _JudgeLLM("reject", "column is too skewed for this"))
    out = critic_node(_state([_decision("clean_r1_001")]))
    verdict = out["critic"]["current_verdict"]
    assert verdict["verdict"] == "reject"
    assert verdict["agent"] == "cleaner"
    assert verdict["rejected_decision_ids"] == ["clean_r1_001"]
    assert "too skewed" in verdict["reasoning"]
    assert out["critic"]["halted"] is False  # attempt 1 of 3


def test_unusable_llm_output_passes_unverified_and_is_counted(monkeypatch):
    _use_llm(monkeypatch, _BrokenJudge())
    out = critic_node(_state([_decision("clean_r1_001")]))
    verdict = out["critic"]["current_verdict"]
    assert verdict["verdict"] == "accept"
    assert verdict["unverified_decisions"] == 1
    assert verdict["confidence"] == 0.0
    assert len(out["errors"]) == 1


def test_hard_row_loss_limit_rejects_without_llm(monkeypatch):
    llm = _JudgeLLM("accept")
    _use_llm(monkeypatch, llm)
    out = critic_node(_state([_decision("clean_r1_001", rows_before=100, rows_after=50)]))
    assert out["critic"]["current_verdict"]["verdict"] == "reject"
    assert "50%" in out["critic"]["current_verdict"]["reasoning"]
    assert llm.calls == 0


def test_duplicate_removal_is_exempt_from_row_loss_limit(monkeypatch):
    llm = _JudgeLLM("accept")
    _use_llm(monkeypatch, llm)
    state = _state([_decision("clean_r1_001", rows_before=100, rows_after=50)],
                   issue_type="duplicate_rows")
    out = critic_node(state)
    assert out["critic"]["current_verdict"]["verdict"] == "accept"
    assert llm.calls == 1


def test_missing_snapshot_rejects_the_stage(monkeypatch):
    llm = _JudgeLLM("accept")
    _use_llm(monkeypatch, llm)
    out = critic_node(_state([_decision("clean_r1_001")], snapshot=None))
    assert out["critic"]["current_verdict"]["verdict"] == "reject"
    assert "snapshot" in out["critic"]["current_verdict"]["reasoning"]
    assert llm.calls == 0


def test_third_rejection_of_a_stage_halts_the_run(monkeypatch):
    _use_llm(monkeypatch, _JudgeLLM("reject", "bad"))
    state = _state([])
    for attempt in (1, 2, 3):
        state["cleaner"]["decisions"] = [_decision(f"clean_r{attempt}_001")]
        out = critic_node(state)
        state["critic"] = out["critic"]
        assert out["critic"]["halted"] is (attempt == 3)
    assert "cleaner" in out["critic"]["halt_reason"]
    assert out["critic"]["rounds_per_agent"] == {"cleaner": 3}


def test_redo_cap_is_per_stage_not_global(monkeypatch):
    # Known issue 2: with a GLOBAL counter, 2 cleaner rejections + 1 encoder review
    # would hit the cap of 3 even though no single stage used up its attempts.
    _use_llm(monkeypatch, _JudgeLLM("reject", "bad"))
    state = _state([])
    for attempt in (1, 2):
        state["cleaner"]["decisions"] = [_decision(f"clean_r{attempt}_001")]
        state["critic"] = critic_node(state)["critic"]
    state["metadata"]["current_active_agent"] = "encoder"
    out = critic_node(state)
    assert out["critic"]["rounds_per_agent"] == {"cleaner": 2, "encoder": 1}
    assert out["critic"]["total_rounds"] == 3
    assert out["critic"]["halted"] is False


def test_only_unreviewed_decisions_are_judged(monkeypatch):
    _use_llm(monkeypatch, _JudgeLLM("accept"))
    state = _state([_decision("clean_r1_001")])
    state["critic"] = critic_node(state)["critic"]
    state["cleaner"]["decisions"] = [_decision("clean_r1_001"), _decision("clean_r2_001")]
    out = critic_node(state)
    assert out["critic"]["current_verdict"]["decision_ids_reviewed"] == ["clean_r2_001"]
    assert len(out["critic"]["verdicts"]) == 2  # history carried forward


def test_verdict_formatting_is_repaired():
    judgment = DecisionJudgment(verdict=" Rejected. ", reason=None)
    assert judgment.verdict == "reject"
    assert judgment.reason == ""
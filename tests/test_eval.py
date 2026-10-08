"""
tests/test_eval.py -- evaluation metrics and the eval loop. No model needed.
Environment: LOCAL (Mac).
"""
from agents.critic import DecisionJudgment
from evaluation.metrics import critic_metrics
from evaluation.run_critic_eval import run


def _row(label, prediction, borderline=False):
    return {"label": label, "prediction": prediction, "borderline": borderline}


def test_metrics_on_hand_computed_rows():
    rows = [
        _row("accept", "accept"),                   # true negative
        _row("accept", "reject", borderline=True),  # false positive
        _row("reject", "reject"),                   # true positive
        _row("reject", None),                       # malformed -> effectively accept -> false negative
        _row("reject", "accept", borderline=True),  # false negative
        _row("accept", None),                       # malformed -> effectively accept -> "lucky" correct
    ]
    m = critic_metrics(rows)
    assert m["malformed"] == 2 and m["malformed_rate"] == 0.333
    assert m["accuracy"] == 0.5           # 3 of 6 correct, malformed treated as accept
    assert m["accuracy_strict"] == 0.333  # 2 of 6, malformed counted as wrong
    assert m["reject_precision"] == 0.5
    assert m["reject_recall"] == 0.333
    assert m["reject_f1"] == 0.4
    assert m["accuracy_clear"] == 0.75
    assert m["accuracy_borderline"] == 0.0
    assert m["majority_baseline"] == {"label": "accept", "accuracy": 0.5}
    assert m["confusion"]["actual_reject"] == {"pred_accept": 1, "pred_reject": 1, "malformed": 1}


def test_metrics_when_nothing_is_rejected():
    m = critic_metrics([_row("reject", "accept"), _row("accept", "accept")])
    assert m["reject_precision"] is None
    assert m["reject_recall"] == 0.0
    assert m["reject_f1"] is None


def test_run_records_malformed_output_without_crashing():
    examples = [
        {"id": "e1", "scenario": "s", "strategy": "no_action", "borderline": False,
         "label": "reject", "evidence": {}},
        {"id": "e2", "scenario": "s", "strategy": "drop_duplicates", "borderline": False,
         "label": "accept", "evidence": {}},
    ]
    answers = iter([DecisionJudgment(verdict="reject", reason="rule 1"), None])

    def fake_judge(llm, evidence):
        answer = next(answers)
        if answer is None:
            raise ValueError("no usable judgment")
        return answer

    rows = run(examples, llm=None, judge=fake_judge)
    assert rows[0]["prediction"] == "reject"
    assert rows[1]["prediction"] is None
    assert "ValueError" in rows[1]["error"]
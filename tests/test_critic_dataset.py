"""
tests/test_critic_dataset.py -- review policy + dataset generator. Environment: LOCAL (Mac).
"""
import json

import pytest

from agents.critic_policy import label_from_evidence
from evaluation.critic_dataset import build_dataset, split


def _ev(strategy, before=None, change=None):
    return {"strategy": strategy, "before": before or {}, "change": change or {}}


def _label(evidence):
    return label_from_evidence(evidence)[0]


@pytest.fixture(scope="module")
def small_dataset():
    # Built once for the module: generation runs the real detectors, so it isn't free.
    return build_dataset(n=24, seed=7)[0]


def test_mean_imputation_rule_and_borderline_band():
    label, _, borderline = label_from_evidence(
        _ev("impute_mean", {"rows": 100, "missing": 5, "skew": 1.2}))
    assert label == "reject" and borderline is True
    label, _, borderline = label_from_evidence(
        _ev("impute_mean", {"rows": 100, "missing": 5, "skew": 0.2}))
    assert label == "accept" and borderline is False


def test_loss_rules_use_the_ten_percent_threshold():
    assert _label(_ev("drop_rows_missing", {"rows": 100, "missing": 20},
                      {"rows_before": 100, "rows_after": 80})) == "reject"
    assert _label(_ev("cast_numeric", {"rows": 100, "missing": 0},
                      {"coerced_to_missing": 3})) == "accept"
    assert _label(_ev("cap_iqr", {"rows": 100, "missing": 0},
                      {"values_capped": 12})) == "reject"


def test_fixed_rules():
    assert _label(_ev("no_action")) == "reject"
    assert _label(_ev("drop_duplicates")) == "accept"
    assert _label(_ev("drop_column", {"rows": 100, "missing": 70})) == "accept"
    assert _label(_ev("drop_column", {"rows": 100, "missing": 10})) == "reject"
    assert _label(_ev("impute_constant", {"rows": 100, "missing": 5, "min": 10.0, "max": 90.0},
                      {"fill_value": 0.0})) == "reject"


def test_unknown_strategy_has_no_silent_default():
    with pytest.raises(ValueError, match="no rule"):
        label_from_evidence(_ev("impute_knn", {"rows": 10, "missing": 1}))


def test_dataset_is_deterministic(small_dataset):
    again, _ = build_dataset(n=24, seed=7)
    assert json.dumps(small_dataset, default=str) == json.dumps(again, default=str)
    assert len(small_dataset) == 24
    assert {e["label"] for e in small_dataset} <= {"accept", "reject"}


def test_label_is_recoverable_from_the_saved_evidence(small_dataset):
    # After a JSON round trip (exactly what's saved to disk), the evidence alone must
    # still produce the stored label: nothing the label depends on is hidden from the model.
    for example in small_dataset:
        evidence = json.loads(json.dumps(example["evidence"], default=str))
        assert label_from_evidence(evidence)[0] == example["label"], example["id"]


def test_split_is_disjoint_and_sized(small_dataset):
    train, test = split(small_dataset, seed=7)
    assert len(test) == 6 and len(train) == 18
    assert not {e["id"] for e in train} & {e["id"] for e in test}
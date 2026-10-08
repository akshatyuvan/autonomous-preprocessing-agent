"""
tests/test_mlflow_logging.py -- metric flattening + idempotent logging.
Uses a throwaway SQLite store in tmp_path. Environment: LOCAL (Mac).
"""
import json

from evaluation.log_to_mlflow import flatten_metrics, log_results


def test_flatten_metrics_keeps_only_numbers():
    metrics = {
        "accuracy": 0.59, "reject_precision": None, "n": 100,
        "majority_baseline": {"label": "accept", "accuracy": 0.58},
        "accuracy_by_rule": {"rule 1": {"n": 13, "accuracy": 1.0}},
        "confusion": {"actual_accept": {"pred_accept": 31}},
    }
    assert flatten_metrics(metrics) == {
        "accuracy": 0.59, "n": 100.0, "majority_baseline_accuracy": 0.58, "accuracy_rule_1": 1.0}


def test_logging_the_same_file_twice_creates_one_run(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "x.json").write_text(json.dumps({
        "condition": "x", "provider": "ollama", "model": "m", "retrieval": False,
        "test_file": "t", "metrics": {"accuracy": 0.5}}))
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    assert log_results(results, uri, tmp_path / "artifacts") == 1
    assert log_results(results, uri, tmp_path / "artifacts") == 0
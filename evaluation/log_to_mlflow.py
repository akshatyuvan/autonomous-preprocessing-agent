"""
evaluation/log_to_mlflow.py -- record every saved evaluation run in MLflow.

Environment: LOCAL (Mac).
  Log:   python -m evaluation.log_to_mlflow
  View:  mlflow ui --backend-store-uri sqlite:///mlflow.db     -> http://127.0.0.1:5000

The JSON files in evaluation/results/ stay the source of truth (committed to git);
MLflow is the comparison UI on top. Re-running is safe: a results file whose exact
content is already logged (same SHA-256) is skipped.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mlflow

RESULTS_DIR = Path("evaluation/results")
EXPERIMENT = "critic-evaluation"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_ARTIFACT_DIR = Path("mlartifacts")


def flatten_metrics(metrics: dict) -> dict[str, float]:
    """MLflow metrics must be plain numbers: keep numeric top-level values, flatten
    the majority baseline and per-rule accuracy, and skip None."""
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[key] = float(value)
    baseline = (metrics.get("majority_baseline") or {}).get("accuracy")
    if isinstance(baseline, (int, float)):
        flat["majority_baseline_accuracy"] = float(baseline)
    for rule, info in (metrics.get("accuracy_by_rule") or {}).items():
        if info.get("accuracy") is not None:
            flat[f"accuracy_{rule.replace(' ', '_')}"] = float(info["accuracy"])
    return flat


def log_results(results_dir: Path = RESULTS_DIR, tracking_uri: str = DEFAULT_TRACKING_URI,
                artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> int:
    """Log each results JSON as one MLflow run. Returns how many NEW runs were logged."""
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        # Explicit artifact location, so artifacts never land in a surprise folder.
        mlflow.create_experiment(EXPERIMENT, artifact_location=Path(artifact_dir).resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)

    logged = 0
    for path in sorted(Path(results_dir).glob("*.json")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        if len(mlflow.search_runs(filter_string=f"tags.results_sha = '{digest}'")):
            print(f"skip   {path.name} (already logged)")
            continue
        data = json.loads(path.read_text())
        with mlflow.start_run(run_name=data.get("condition", path.stem)):
            mlflow.set_tags({"results_file": path.name, "results_sha": digest})
            mlflow.log_params({k: str(data.get(k))
                               for k in ("condition", "provider", "model", "retrieval", "test_file")})
            mlflow.log_metrics(flatten_metrics(data["metrics"]))
            mlflow.log_artifact(str(path))  # the full per-example predictions travel with the run
        print(f"logged {path.name}")
        logged += 1
    return logged


if __name__ == "__main__":
    print(f"{log_results()} new run(s) logged to experiment '{EXPERIMENT}'.")
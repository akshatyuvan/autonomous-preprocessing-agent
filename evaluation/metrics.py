"""
evaluation/metrics.py -- Critic evaluation metrics, as pure functions (unit-tested).

Each row: {"label": "accept"|"reject", "prediction": "accept"|"reject"|None, "borderline": bool}
prediction None = malformed (unparseable output, or no tool call).

Two views of malformed output:
  - "effective" metrics treat it as ACCEPT, because that is what the pipeline does
    (critic_node lets unjudged decisions through, unverified);
  - accuracy_strict counts it as WRONG, so a lucky "accept" can't hide a bad parse.
"Reject" is the positive class: catching bad transformations is the gate's job,
and reject PRECISION is what decides whether a noisy gate halts good runs.
"""
from __future__ import annotations

from typing import Any, Callable, Optional


def _ratio(part: float, whole: float) -> Optional[float]:
    return round(part / whole, 3) if whole else None


def critic_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    raw = [(r["label"], r["prediction"]) for r in rows]
    effective = [(label, pred or "accept") for label, pred in raw]

    tp = sum(l == "reject" and p == "reject" for l, p in effective)
    fp = sum(l == "accept" and p == "reject" for l, p in effective)
    fn = sum(l == "reject" and p == "accept" for l, p in effective)
    tn = sum(l == "accept" and p == "accept" for l, p in effective)

    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else None)

    def accuracy_where(keep: Callable[[dict], bool]) -> Optional[float]:
        pairs = [pair for pair, row in zip(effective, rows) if keep(row)]
        return _ratio(sum(l == p for l, p in pairs), len(pairs))

    n_accept = sum(l == "accept" for l, _ in raw)
    malformed = sum(p is None for _, p in raw)

    return {
        "n": n,
        "malformed": malformed,
        "malformed_rate": _ratio(malformed, n),
        "accuracy": _ratio(tp + tn, n),
        "accuracy_strict": _ratio(sum(l == p for l, p in raw), n),
        "reject_precision": None if precision is None else round(precision, 3),
        "reject_recall": None if recall is None else round(recall, 3),
        "reject_f1": None if f1 is None else round(f1, 3),
        "n_borderline": sum(bool(r["borderline"]) for r in rows),
        "accuracy_clear": accuracy_where(lambda r: not r["borderline"]),
        "accuracy_borderline": accuracy_where(lambda r: bool(r["borderline"])),
        # The number any model must beat: always predict the most common label.
        "majority_baseline": {
            "label": "accept" if n_accept >= n - n_accept else "reject",
            "accuracy": _ratio(max(n_accept, n - n_accept), n),
        },
        # Raw predictions, with malformed kept as its own column.
        "confusion": {
            f"actual_{actual}": {
                "pred_accept": sum(l == actual and p == "accept" for l, p in raw),
                "pred_reject": sum(l == actual and p == "reject" for l, p in raw),
                "malformed": sum(l == actual and p is None for l, p in raw),
            }
            for actual in ("accept", "reject")
        },
    }
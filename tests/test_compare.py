"""
tests/test_compare.py -- McNemar's exact test, checked by hand. Environment: LOCAL (Mac).
"""
import pytest

from evaluation.compare import mcnemar


def _rows(results):
    # results: list of (id, correct?) -> rows whose label/prediction match iff correct
    return [{"id": i, "label": "reject", "prediction": "reject" if ok else "accept"}
            for i, ok in results]


def test_balanced_disagreements_give_no_evidence():
    a = _rows([(1, True), (2, True), (3, False), (4, False)])
    b = _rows([(1, True), (2, False), (3, True), (4, True)])
    # a_only=1, b_only=2, n=3: P(split at least this lopsided) = 2*(C(3,0)+C(3,1))/8 = 1.0
    result = mcnemar(a, b)
    assert (result["a_only_correct"], result["b_only_correct"]) == (1, 2)
    assert result["p_value"] == 1.0


def test_one_sided_disagreements_are_significant():
    a = _rows([(i, False) for i in range(6)])
    b = _rows([(i, True) for i in range(6)])
    # a_only=0, b_only=6: p = 2 * C(6,0) / 2**6 = 0.03125
    assert mcnemar(a, b)["p_value"] == pytest.approx(0.0312, abs=1e-4)


def test_different_examples_are_refused():
    with pytest.raises(ValueError, match="same examples"):
        mcnemar(_rows([(1, True)]), _rows([(2, True)]))
"""
agents/critic_policy.py -- the Critic's review policy, in ONE place.

Used twice:
  - agents/critic.py puts POLICY_TEXT into the Critic LLM's prompt;
  - evaluation/critic_dataset.py labels examples with label_from_evidence().
So the prompted Critic is told exactly the rules its test labels follow. Without
this, the eval would punish the model for not guessing an unstated policy.

Thresholds are conventions, not laws of nature. What matters is that they are
explicit, applied identically everywhere, and that cases near them are reported
separately ("borderline").
"""
from __future__ import annotations

from typing import Any

SKEW_LIMIT = 1.0             # mean imputation rejected above this |skew|
MAX_CHANGE_SHARE = 0.10      # removing/losing/capping more than 10% in one step -> reject
MOSTLY_MISSING_SHARE = 0.50  # above this, imputing invents most of the column
SKEW_BAND = (0.7, 1.3)       # |skew| inside this band = borderline for mean imputation
SHARE_BAND = (0.05, 0.15)    # affected share inside this band = borderline for loss rules

ALWAYS_ACCEPT = {"drop_duplicates", "standardize_category", "flag_outlier"}
IMPUTERS = {"impute_mean", "impute_median", "impute_mode", "impute_constant"}
ROW_REMOVERS = {"drop_rows_missing", "drop_outlier_rows"}
CONVERTERS = {"cast_numeric", "parse_datetime"}

POLICY_TEXT = f"""Review policy (apply these rules exactly):
1. no_action is rejected: it leaves the data-quality issue unfixed.
2. drop_duplicates, standardize_category and flag_outlier are accepted.
3. drop_column is accepted only when more than {MOSTLY_MISSING_SHARE:.0%} of the column's values are
   missing (before.missing / before.rows); otherwise it is rejected.
4. Any imputation is rejected when more than {MOSTLY_MISSING_SHARE:.0%} of the values are missing.
5. impute_mean is rejected when |before.skew| > {SKEW_LIMIT}; otherwise it is accepted.
6. impute_constant is rejected when the fill value lies outside [before.min, before.max] of a numeric column.
7. Removing rows (drop_rows_missing, drop_outlier_rows: rows_before - rows_after), losing values in a
   conversion (cast_numeric, parse_datetime: coerced_to_missing) or capping values (cap_iqr: values_capped)
   is rejected when it affects more than {MAX_CHANGE_SHARE:.0%} of the rows or non-missing values;
   otherwise it is accepted.
8. impute_median and impute_mode are accepted unless rule 4 applies."""


def _share(part: float, whole: float) -> float:
    return part / whole if whole else 0.0


def _in_band(value: float, band: tuple[float, float]) -> bool:
    return band[0] <= value <= band[1]


def label_from_evidence(evidence: dict[str, Any]) -> tuple[str, str, bool]:
    """
    Apply POLICY_TEXT to the evidence the Critic sees.
    Returns (label, the rule that decided it, borderline?).

    It reads ONLY the evidence dict, never hidden generator parameters, so the
    label is always derivable from exactly what the model is shown: the task is
    well-posed, and a wrong answer is the model's fault, not missing information.
    """
    strategy = evidence["strategy"]
    before = evidence.get("before") or {}
    change = evidence.get("change") or {}
    rows = before.get("rows") or 0
    missing = before.get("missing") or 0
    missing_share = _share(missing, rows)

    if strategy == "no_action":
        return "reject", "rule 1: no_action leaves the issue unfixed", False
    if strategy in ALWAYS_ACCEPT:
        return "accept", f"rule 2: {strategy} is a lossless fix", False

    if strategy == "drop_column":
        if missing_share > MOSTLY_MISSING_SHARE:
            return "accept", f"rule 3: {missing_share:.0%} missing, dropping is justified", False
        return "reject", f"rule 3: only {missing_share:.0%} missing, dropping loses real data", False

    if strategy in IMPUTERS:
        if missing_share > MOSTLY_MISSING_SHARE:
            return "reject", f"rule 4: {missing_share:.0%} missing, imputation invents most values", False
        if strategy == "impute_mean":
            skew = abs(before.get("skew") or 0.0)
            borderline = _in_band(skew, SKEW_BAND)
            if skew > SKEW_LIMIT:
                return "reject", f"rule 5: |skew| {skew:.2f} > {SKEW_LIMIT}", borderline
            return "accept", f"rule 5: |skew| {skew:.2f} <= {SKEW_LIMIT}", borderline
        if strategy == "impute_constant":
            fill, low, high = change.get("fill_value"), before.get("min"), before.get("max")
            if isinstance(fill, (int, float)) and low is not None and high is not None \
                    and not low <= fill <= high:
                return "reject", f"rule 6: fill {fill} outside [{low}, {high}]", False
        return "accept", f"rule 8: {strategy} with {missing_share:.0%} missing", False

    if strategy in ROW_REMOVERS:
        rows_before = change.get("rows_before", 0)
        share = _share(rows_before - change.get("rows_after", 0), rows_before)
    elif strategy in CONVERTERS:
        attempted = rows - missing - change.get("placeholders_blanked", 0)
        share = _share(change.get("coerced_to_missing", 0), attempted)
    elif strategy == "cap_iqr":
        share = _share(change.get("values_capped", 0), rows - missing)
    else:
        # A new strategy with no policy rule yet: say so rather than guess.
        raise ValueError(f"critic_policy has no rule for strategy '{strategy}'")

    borderline = _in_band(share, SHARE_BAND)
    if share > MAX_CHANGE_SHARE:
        return "reject", f"rule 7: {strategy} affected {share:.1%} > {MAX_CHANGE_SHARE:.0%}", borderline
    return "accept", f"rule 7: {strategy} affected {share:.1%} <= {MAX_CHANGE_SHARE:.0%}", borderline
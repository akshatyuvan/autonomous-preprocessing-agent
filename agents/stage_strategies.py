"""
agents/stage_strategies.py -- technique implementations for the encoder, scaler,
imbalance_handler and feature_selector stages.

Same contract as agents/cleaning_strategies.py:
    fn(df, column, params) -> (new_df, stats)
input DataFrame never modified, StrategyNotApplicableError when the data doesn't fit.
params always carries {"target": <target column name or None>}.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from agents.dispatch import StrategyNotApplicableError

Params = dict[str, Any]
Result = tuple[pd.DataFrame, dict[str, Any]]

MAX_ONE_HOT_CATEGORIES = 10   # above this, one-hot creates too many columns
LOW_VARIANCE_THRESHOLD = 1e-6  # "almost constant"
CORRELATION_LIMIT = 0.9        # |r| above this = redundant feature
LOW_MUTUAL_INFO = 0.01         # below this = no measurable link to the target
# Ordinal encoding needs a KNOWN order; we only claim one when the values match a
# vocabulary whose order is unambiguous. Anything else is treated as unordered.
ORDERED_VOCABULARIES = [
    ["low", "medium", "high"],
    ["small", "medium", "large"],
    ["basic", "premium", "enterprise"],
    ["poor", "fair", "good", "excellent"],
]


# ---------------------------------------------------------------- helpers
def _is_text(s: pd.Series) -> bool:
    return s.dtype == object or isinstance(s.dtype, pd.StringDtype)


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _normalise(s: pd.Series) -> pd.Series:
    return s.map(lambda v: v.strip().casefold() if isinstance(v, str) else v)


def _require_numeric(s: pd.Series, name: str) -> None:
    if not _is_numeric(s):
        raise StrategyNotApplicableError(f"{name} needs a numeric column, got dtype '{s.dtype}'")


def _target(df: pd.DataFrame, params: Params, name: str) -> str:
    target = params.get("target")
    if not target or target not in df.columns:
        raise StrategyNotApplicableError(f"{name} needs a target column, and none is set")
    return target


def ordered_vocabulary(s: pd.Series) -> Optional[list[str]]:
    """The known order this column's values follow, or None."""
    values = set(_normalise(s.dropna()))
    if len(values) < 2:
        return None
    for vocab in ORDERED_VOCABULARIES:
        if values <= set(vocab):
            return vocab
    return None


# ---------------------------------------------------------------- encoding
def one_hot(df: pd.DataFrame, column: str, params: Params) -> Result:
    s = df[column]
    if s.nunique(dropna=True) > MAX_ONE_HOT_CATEGORIES:
        raise StrategyNotApplicableError(f"one_hot: {s.nunique()} categories is too many columns")
    # One 0/1 column per category; a missing value becomes a row of zeros.
    dummies = pd.get_dummies(s, prefix=column, dtype=int)
    out = pd.concat([df.drop(columns=[column]), dummies], axis=1)
    return out, {"columns_added": [str(c) for c in dummies.columns]}


def ordinal(df: pd.DataFrame, column: str, params: Params) -> Result:
    vocab = ordered_vocabulary(df[column])
    if vocab is None:
        raise StrategyNotApplicableError("ordinal: values don't follow a known order")
    out = df.copy()
    # low/medium/high -> 0/1/2: the integers keep the RANK, which one-hot would throw away.
    out[column] = _normalise(df[column]).map({v: i for i, v in enumerate(vocab)})
    return out, {"order": vocab}


def target_encoding(df: pd.DataFrame, column: str, params: Params) -> Result:
    target = _target(df, params, "target_encoding")
    if not _is_numeric(df[target]):
        raise StrategyNotApplicableError("target_encoding needs a numeric target")
    means = df[target].groupby(df[column]).mean()
    out = df.copy()
    # LEAKAGE WARNING: fitted on the FULL dataset. In a real modelling pipeline this
    # must be fitted inside cross-validation folds, or the target leaks into the feature.
    out[column] = df[column].map(means).astype("float64")
    return out, {"categories_encoded": int(means.size)}


# ---------------------------------------------------------------- scaling
def _scale(df: pd.DataFrame, column: str, make_scaler: Callable, name: str) -> Result:
    _require_numeric(df[column], name)
    values = df[column].astype("float64")  # float first: scaled values aren't integers
    mask = values.notna()
    if mask.sum() < 2:
        raise StrategyNotApplicableError(f"{name} needs at least 2 non-missing values")
    scaled = values.copy()
    # Fit on the non-missing values only; missing stays missing (not this stage's job).
    scaled[mask] = make_scaler().fit_transform(values[mask].to_frame()).ravel()
    out = df.copy()
    out[column] = scaled
    return out, {"values_scaled": int(mask.sum())}


def standard_scale(df: pd.DataFrame, column: str, params: Params) -> Result:
    return _scale(df, column, StandardScaler, "standard")    # mean 0, std 1


def minmax_scale(df: pd.DataFrame, column: str, params: Params) -> Result:
    return _scale(df, column, MinMaxScaler, "minmax")        # range [0, 1]


def robust_scale(df: pd.DataFrame, column: str, params: Params) -> Result:
    return _scale(df, column, RobustScaler, "robust")        # median 0, IQR 1: outlier-proof


# ---------------------------------------------------------------- imbalance
def _class_counts(y: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in y.value_counts().sort_index().items()}


def class_weights(df: pd.DataFrame, column: str, params: Params) -> Result:
    y = df[column].dropna()
    counts = y.value_counts().sort_index()
    if len(counts) < 2:
        raise StrategyNotApplicableError("class_weights needs at least 2 classes")
    n, k = len(y), len(counts)
    # "balanced" weights: n / (k * class_count). No rows change; the downstream
    # model's loss uses these so minority-class errors cost more.
    weights = {str(c): round(n / (k * cnt), 4) for c, cnt in counts.items()}
    return df.copy(), {"class_counts": _class_counts(y), "class_weights": weights}


def _resample(df: pd.DataFrame, column: str, name: str, make_sampler: Callable) -> Result:
    try:
        import imblearn  # noqa: F401  (only checking it's installed)
    except ImportError as exc:
        raise StrategyNotApplicableError(f"{name} needs imbalanced-learn installed") from exc
    if df[column].isna().any():
        raise StrategyNotApplicableError(f"{name}: target has missing values")
    X, y = df.drop(columns=[column]), df[column]
    try:
        X_res, y_res = make_sampler(X).fit_resample(X, y)
    except (ValueError, TypeError) as exc:
        # e.g. text features for plain SMOTE, or too few minority rows for k neighbours
        raise StrategyNotApplicableError(f"{name} cannot resample this data: {exc}") from exc
    out = pd.DataFrame(X_res, columns=X.columns)
    out[column] = list(y_res)
    return out[list(df.columns)], {"class_counts_before": _class_counts(y),
                                   "class_counts_after": _class_counts(out[column])}


def smote(df: pd.DataFrame, column: str, params: Params) -> Result:
    def make(X):
        from imblearn.over_sampling import SMOTE
        return SMOTE(random_state=0)
    return _resample(df, column, "smote", make)


def smote_nc(df: pd.DataFrame, column: str, params: Params) -> Result:
    def make(X):
        from imblearn.over_sampling import SMOTENC
        categorical = [i for i, c in enumerate(X.columns) if _is_text(X[c])]
        return SMOTENC(categorical_features=categorical, random_state=0)
    return _resample(df, column, "smote_nc", make)


def borderline_smote(df: pd.DataFrame, column: str, params: Params) -> Result:
    def make(X):
        from imblearn.over_sampling import BorderlineSMOTE
        return BorderlineSMOTE(random_state=0)
    return _resample(df, column, "borderline_smote", make)


def adasyn(df: pd.DataFrame, column: str, params: Params) -> Result:
    def make(X):
        from imblearn.over_sampling import ADASYN
        return ADASYN(random_state=0)
    return _resample(df, column, "adasyn", make)


def smote_tomek(df: pd.DataFrame, column: str, params: Params) -> Result:
    def make(X):
        from imblearn.combine import SMOTETomek
        return SMOTETomek(random_state=0)
    return _resample(df, column, "smote_tomek", make)


# ---------------------------------------------------------------- feature selection
# Each "drop" strategy CHECKS its own rule and refuses when it doesn't hold, so even a
# wrong LLM choice can never delete a feature that doesn't meet the criterion.
def drop_low_variance(df: pd.DataFrame, column: str, params: Params) -> Result:
    _require_numeric(df[column], "drop_low_variance")
    variance = float(df[column].var())
    if not variance < LOW_VARIANCE_THRESHOLD:
        raise StrategyNotApplicableError(f"drop_low_variance: variance {variance:.4g} is not near zero")
    return df.drop(columns=[column]), {"variance": variance}


def drop_correlated(df: pd.DataFrame, column: str, params: Params) -> Result:
    _require_numeric(df[column], "drop_correlated")
    target = params.get("target")
    others = [c for c in df.columns if c not in (column, target) and _is_numeric(df[c])]
    if not others:
        raise StrategyNotApplicableError("drop_correlated: no other numeric columns")
    corr = df[others].corrwith(df[column]).abs().dropna()
    if corr.empty or corr.max() <= CORRELATION_LIMIT:
        raise StrategyNotApplicableError("drop_correlated: no feature is correlated above the limit")
    partner = str(corr.idxmax())
    return df.drop(columns=[column]), {"correlated_with": partner,
                                       "abs_correlation": round(float(corr.max()), 4)}


def drop_low_mutual_info(df: pd.DataFrame, column: str, params: Params) -> Result:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    target = _target(df, params, "drop_low_mutual_info")
    _require_numeric(df[column], "drop_low_mutual_info")
    data = df[[column, target]].dropna()
    if len(data) < 10:
        raise StrategyNotApplicableError("drop_low_mutual_info needs at least 10 complete rows")
    y = data[target]
    is_classification = not _is_numeric(y) or y.nunique() <= 20
    estimate = mutual_info_classif if is_classification else mutual_info_regression
    mi = float(estimate(data[[column]].astype("float64"), y, random_state=0)[0])
    if mi >= LOW_MUTUAL_INFO:
        raise StrategyNotApplicableError(f"drop_low_mutual_info: MI {mi:.4f} is not low")
    return df.drop(columns=[column]), {"mutual_information": round(mi, 4)}
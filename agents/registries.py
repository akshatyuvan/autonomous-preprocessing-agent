"""
agents/registries.py -- technique registries, one per preprocessing stage.

Each entry is ONE technique:
  name / applies_when / description  -> written for the LLM (it never sees "apply")
  apply                              -> the function that executes it
  requires (stage registries)        -> preconditions checked in Python before the LLM
                                        sees the option (names = agents/stage_runner.REQUIREMENTS)
  handles  (cleaning registry)       -> which Profiler issue types it fixes

DESIGN RULE: agents never hard-code "if numeric, use X". They filter a registry and
let the LLM choose from what fits. Adding a technique = ONE entry here; no stage, graph
or prompt code changes. (A technique with a brand-new KIND of precondition also needs
one predicate added to REQUIREMENTS.)

ORDER MATTERS: when the LLM's choice is unusable, the fallback takes the first entry
that fits, so each registry lists its most conservative applicable option first.
"""
from agents import cleaning_strategies as cs
from agents import stage_strategies as ss

NO_ACTION = {
    "name": "No action",
    "applies_when": "The column is already in good shape for this step",
    "description": "Leaves the data unchanged.",
    "requires": [],
    "needs_column": False,
    "apply": cs.no_action,
}

# ---------------------------------------------------------
# IMBALANCE HANDLING REGISTRY  (dataset-level: the "column" is the target)
# ---------------------------------------------------------

IMBALANCE_REGISTRY = {
    "class_weights": {
        "name": "Class Weights",
        "applies_when": "Imbalance present but synthetic samples are risky (small dataset, sensitive domain)",
        "requires": [],
        "description": "No resampling: reweights the loss so minority-class errors cost more.",
        "apply": ss.class_weights,
    },
    "smote": {
        "name": "SMOTE",
        "applies_when": "Binary or multiclass imbalance, all features numeric",
        "requires": ["all_numeric_features"],
        "description": "Generates synthetic minority samples by interpolating between neighbours.",
        "apply": ss.smote,
    },
    "smote_nc": {
        "name": "SMOTE-NC",
        "applies_when": "Imbalance AND the dataset has both categorical and continuous features",
        "requires": ["mixed_feature_types"],
        "description": "SMOTE variant for mixed categorical + continuous data.",
        "apply": ss.smote_nc,
    },
    "borderline_smote": {
        "name": "Borderline-SMOTE",
        "applies_when": "Imbalance with many minority samples near the decision boundary",
        "requires": ["all_numeric_features"],
        "description": "Generates synthetic samples near the class boundary rather than uniformly.",
        "apply": ss.borderline_smote,
    },
    "adasyn": {
        "name": "ADASYN",
        "applies_when": "Imbalance with non-uniform minority class density",
        "requires": ["all_numeric_features"],
        "description": "Adaptively generates more synthetic samples in harder-to-learn regions.",
        "apply": ss.adasyn,
    },
    "smote_tomek": {
        "name": "SMOTE + Tomek Links",
        "applies_when": "Imbalance combined with noisy or overlapping class boundaries",
        "requires": ["all_numeric_features"],
        "description": "SMOTE oversampling followed by Tomek-link cleaning of ambiguous samples.",
        "apply": ss.smote_tomek,
    },
    "no_action": NO_ACTION,
}

# ---------------------------------------------------------
# ENCODING REGISTRY
# ---------------------------------------------------------

ENCODING_REGISTRY = {
    "ordinal": {
        "name": "Ordinal Encoding",
        "applies_when": "Categorical column with a natural order (e.g. low/medium/high)",
        "requires": ["categorical", "has_order"],
        "description": "Maps categories to integers preserving rank order.",
        "apply": ss.ordinal,
    },
    "one_hot": {
        "name": "One-Hot Encoding",
        "applies_when": "Low-cardinality categorical column (few unique values), no inherent order",
        "requires": ["categorical", "low_cardinality"],
        "description": "Creates a binary column per category.",
        "apply": ss.one_hot,
    },
    "target_encoding": {
        "name": "Target Encoding",
        "applies_when": "High-cardinality categorical column where one-hot would create too many columns",
        "requires": ["categorical", "high_cardinality", "target_column_present"],
        "description": "Replaces each category with the mean target value for that category.",
        "apply": ss.target_encoding,
    },
    "no_action": NO_ACTION,
}

# ---------------------------------------------------------
# SCALING REGISTRY
# ---------------------------------------------------------

SCALING_REGISTRY = {
    "standard": {
        "name": "StandardScaler",
        "applies_when": "Feature is roughly normally distributed, no extreme outliers",
        "requires": ["numeric"],
        "description": "Centers to mean 0, scales to unit variance.",
        "apply": ss.standard_scale,
    },
    "minmax": {
        "name": "MinMaxScaler",
        "applies_when": "Feature needs to be bounded in a fixed range (e.g. for neural nets)",
        "requires": ["numeric"],
        "description": "Scales the feature to the range [0, 1].",
        "apply": ss.minmax_scale,
    },
    "robust": {
        "name": "RobustScaler",
        "applies_when": "Feature has significant outliers",
        "requires": ["numeric"],
        "description": "Uses median and IQR instead of mean/std, so outliers don't dominate.",
        "apply": ss.robust_scale,
    },
    "no_action": {
        **NO_ACTION,
        "applies_when": "Downstream model is tree-based (Random Forest, XGBoost): scaling has no effect",
    },
}

# ---------------------------------------------------------
# FEATURE SELECTION REGISTRY  ("keep" first: dropping is never the default)
# ---------------------------------------------------------

FEATURE_SELECTION_REGISTRY = {
    "no_action": {**NO_ACTION, "name": "Keep", "applies_when": "The feature carries useful information"},
    "drop_low_variance": {
        "name": "Variance Threshold",
        "applies_when": "Feature has near-zero variance (almost constant across all rows)",
        "requires": ["numeric"],
        "description": "Drops a feature that barely varies; refuses if variance is not near zero.",
        "apply": ss.drop_low_variance,
    },
    "drop_correlated": {
        "name": "Correlation Dropping",
        "applies_when": "Feature is highly correlated (|r| > 0.9) with another retained feature",
        "requires": ["numeric"],
        "description": "Drops a redundant feature; refuses if no correlation exceeds 0.9.",
        "apply": ss.drop_correlated,
    },
    "drop_low_mutual_info": {
        "name": "Mutual Information Ranking",
        "applies_when": "Feature has very low mutual information with the target column",
        "requires": ["numeric", "target_column_present"],
        "description": "Drops a feature unrelated to the target; refuses if MI is not low.",
        "apply": ss.drop_low_mutual_info,
    },
}

# ---------------------------------------------------------
# CLEANING REGISTRY  (issue-driven; used by agents/cleaner.py)
# ---------------------------------------------------------

CLEANING_REGISTRY = {
    "drop_duplicates": {
        "name": "Drop duplicate rows",
        "handles": ["duplicate_rows"],
        "applies_when": "Fully identical rows exist and each row should represent one real record",
        "description": "Keeps the first occurrence of each identical row and removes the rest.",
        "needs_column": False,
        "apply": cs.drop_duplicates,
    },
    "impute_median": {
        "name": "Median imputation",
        "handles": ["missing_values"],
        "applies_when": "Numeric column that is skewed or has outliers, which would pull the mean",
        "description": "Fills missing values with the column median.",
        "apply": cs.impute_median,
    },
    "impute_mean": {
        "name": "Mean imputation",
        "handles": ["missing_values"],
        "applies_when": "Numeric column, few missing values, roughly symmetric distribution without strong outliers",
        "description": "Fills missing values with the column mean.",
        "apply": cs.impute_mean,
    },
    "impute_mode": {
        "name": "Mode imputation",
        "handles": ["missing_values"],
        "applies_when": "Categorical or text column with a clearly dominant value",
        "description": "Fills missing values with the most frequent value.",
        "apply": cs.impute_mode,
    },
    "impute_constant": {
        "name": "Constant fill",
        "handles": ["missing_values"],
        "applies_when": "Missingness is meaningful (e.g. fill 'Unknown') or a known domain default exists; requires params.fill_value",
        "description": "Fills missing values with a fixed value supplied in params.",
        "apply": cs.impute_constant,
    },
    "drop_rows_missing": {
        "name": "Drop rows with missing values",
        "handles": ["missing_values"],
        "applies_when": "Very few rows are affected and removing them will not bias the dataset",
        "description": "Removes the rows where this column is missing.",
        "apply": cs.drop_rows_missing,
    },
    "drop_column": {
        "name": "Drop column",
        "handles": ["missing_values"],
        "applies_when": "Most of the column is missing, so imputation would invent most of its values",
        "description": "Removes the column entirely.",
        "apply": cs.drop_column,
    },
    "cast_numeric": {
        "name": "Cast to numeric",
        "handles": ["type_mismatch"],
        "applies_when": "Numbers are stored as text",
        "description": "Converts text to numbers; unparseable values become missing and are counted.",
        "apply": cs.cast_numeric,
    },
    "parse_datetime": {
        "name": "Parse dates",
        "handles": ["type_mismatch"],
        "applies_when": "Dates are stored as text",
        "description": "Converts text to datetimes; unparseable values become missing and are counted.",
        "apply": cs.parse_datetime,
    },
    "standardize_category": {
        "name": "Standardize category spelling",
        "handles": ["inconsistent_category"],
        "applies_when": "The same category appears with different case or extra spaces",
        "description": "Maps every variant to the most frequent spelling of that category.",
        "apply": cs.standardize_category,
    },
    "cap_iqr": {
        "name": "Cap at IQR fences",
        "handles": ["outlier"],
        "applies_when": "Extreme values are plausible but would distort means or scaling; keeps every row",
        "description": "Clips values to [Q1 - 1.5*IQR, Q3 + 1.5*IQR].",
        "apply": cs.cap_iqr,
    },
    "drop_outlier_rows": {
        "name": "Drop outlier rows",
        "handles": ["outlier"],
        "applies_when": "Extremes are clearly data errors and the dataset is large enough to lose rows",
        "description": "Removes rows whose value lies outside the IQR fences.",
        "apply": cs.drop_outlier_rows,
    },
    "flag_outlier": {
        "name": "Flag outliers",
        "handles": ["outlier"],
        "applies_when": "Unclear whether extremes are real; keep values and let a model decide",
        "description": "Adds a boolean <column>_is_outlier column and changes no values.",
        "apply": cs.flag_outlier,
    },
    "no_action": {
        "name": "No action",
        "handles": ["missing_values", "type_mismatch", "outlier", "duplicate_rows",
                    "inconsistent_category", "unknown"],
        "applies_when": "The issue is negligible or every fix would destroy real signal",
        "description": "Leaves the data unchanged.",
        "needs_column": False,
        "apply": cs.no_action,
    },
}
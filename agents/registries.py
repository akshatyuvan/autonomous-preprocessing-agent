"""
agents/registries.py

Technique registries — one per preprocessing step.
Each entry describes ONE available technique: when it applies,
what it needs, and a human-readable description the LLM can reason over.

DESIGN RULE: Agents read from these registries to decide what to propose.
Agents NEVER hardcode "if column is numeric, use mean imputation" logic
directly in agent code. Instead they ask: "given this registry and this
column's characteristics, which entry fits best?"

WHY THIS MATTERS: Adding a new technique later (e.g. a new SMOTE variant
published next year) means adding ONE dict entry here. Zero changes to
agents/imbalance_handler.py itself. This is what answers the interview
question "what happens when a better technique gets published?"
"""
from agents import cleaning_strategies as cs
# ---------------------------------------------------------
# IMBALANCE HANDLING REGISTRY
# ---------------------------------------------------------

IMBALANCE_REGISTRY = {
    "smote": {
        "name": "SMOTE",
        "applies_when": "Binary or multiclass imbalance, all features numeric",
        "requires": ["all_numeric_features"],
        "description": "Synthetic Minority Oversampling — generates synthetic minority samples via interpolation between neighbors.",
    },
    "smote_nc": {
        "name": "SMOTE-NC",
        "applies_when": "Imbalance present AND dataset has both categorical and continuous features",
        "requires": ["mixed_feature_types"],
        "description": "SMOTE variant for mixed categorical+continuous data.",
    },
    "borderline_smote": {
        "name": "Borderline-SMOTE",
        "applies_when": "Imbalance with many minority samples near the decision boundary",
        "requires": ["all_numeric_features"],
        "description": "Focuses synthetic sample generation near the class boundary rather than uniformly.",
    },
    "adasyn": {
        "name": "ADASYN",
        "applies_when": "Imbalance with non-uniform minority class density",
        "requires": ["all_numeric_features"],
        "description": "Adaptively generates more synthetic samples in harder-to-learn regions.",
    },
    "smote_tomek": {
        "name": "SMOTE + Tomek Links",
        "applies_when": "Imbalance combined with noisy/overlapping class boundaries",
        "requires": ["all_numeric_features"],
        "description": "SMOTE oversampling followed by Tomek link cleaning to remove ambiguous samples.",
    },
    "class_weights": {
        "name": "Class Weights",
        "applies_when": "Imbalance present but synthetic sample generation is risky (small dataset, sensitive domain)",
        "requires": [],
        "description": "No resampling — reweights the loss function so the model penalizes minority-class errors more.",
    },
}


# ---------------------------------------------------------
# ENCODING REGISTRY
# ---------------------------------------------------------

ENCODING_REGISTRY = {
    "one_hot": {
        "name": "One-Hot Encoding",
        "applies_when": "Low-cardinality categorical column (few unique values), no inherent order",
        "requires": ["categorical", "low_cardinality"],
        "description": "Creates a binary column per category.",
    },
    "ordinal": {
        "name": "Ordinal Encoding",
        "applies_when": "Categorical column with a natural order (e.g. low/medium/high)",
        "requires": ["categorical", "has_order"],
        "description": "Maps categories to integers preserving rank order.",
    },
    "target_encoding": {
        "name": "Target Encoding",
        "applies_when": "High-cardinality categorical column where one-hot would create too many columns",
        "requires": ["categorical", "high_cardinality", "target_column_present"],
        "description": "Replaces each category with the mean target value for that category.",
    },
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
    },
    "minmax": {
        "name": "MinMaxScaler",
        "applies_when": "Feature needs to be bounded in a fixed range (e.g. for neural nets)",
        "requires": ["numeric"],
        "description": "Scales feature to a fixed range, typically [0, 1].",
    },
    "robust": {
        "name": "RobustScaler",
        "applies_when": "Feature has significant outliers",
        "requires": ["numeric"],
        "description": "Uses median and IQR instead of mean/std — robust to outliers.",
    },
    "no_action": {
        "name": "No Scaling",
        "applies_when": "Downstream model is tree-based (Random Forest, XGBoost) — scaling not needed",
        "requires": [],
        "description": "Tree-based models split on raw thresholds, scaling has no effect.",
    },
}


# ---------------------------------------------------------
# FEATURE SELECTION REGISTRY
# ---------------------------------------------------------

FEATURE_SELECTION_REGISTRY = {
    "drop_low_variance": {
        "name": "Variance Threshold",
        "applies_when": "Feature has near-zero variance (almost constant across all rows)",
        "requires": ["numeric"],
        "description": "Drops features that carry almost no information because they barely vary.",
    },
    "drop_correlated": {
        "name": "Correlation Dropping",
        "applies_when": "Feature is highly correlated (>0.9) with another retained feature",
        "requires": ["numeric"],
        "description": "Drops redundant features that duplicate information already captured.",
    },
    "drop_low_mutual_info": {
        "name": "Mutual Information Ranking",
        "applies_when": "Feature has very low mutual information with the target column",
        "requires": ["target_column_present"],
        "description": "Drops features that show little statistical relationship with the prediction target.",
    },
}


# ---------------------------------------------------------
# CLEANING REGISTRY
# ---------------------------------------------------------
# Unlike the four registries above, each entry here also stores the function
# that APPLIES the technique ("apply"). That is what makes resume bullet 1
# literally true: a new technique = one entry here, and agents/dispatch.py,
# the Cleaner node and the graph do not change.
#
# "handles" lists the DataIssue.issue_type values the technique can fix. The
# Cleaner shows the LLM only the options whose "handles" match the issue.
# "applies_when" and "description" are written FOR the LLM; it never sees "apply".

CLEANING_REGISTRY = {
    "drop_duplicates": {
        "name": "Drop duplicate rows",
        "handles": ["duplicate_rows"],
        "applies_when": "Fully identical rows exist and each row should represent one real record",
        "description": "Keeps the first occurrence of each identical row and removes the rest.",
        "needs_column": False,
        "apply": cs.drop_duplicates,
    },
    "impute_mean": {
        "name": "Mean imputation",
        "handles": ["missing_values"],
        "applies_when": "Numeric column, few missing values, roughly symmetric distribution without strong outliers",
        "description": "Fills missing values with the column mean.",
        "apply": cs.impute_mean,
    },
    "impute_median": {
        "name": "Median imputation",
        "handles": ["missing_values"],
        "applies_when": "Numeric column that is skewed or has outliers, which would pull the mean",
        "description": "Fills missing values with the column median.",
        "apply": cs.impute_median,
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
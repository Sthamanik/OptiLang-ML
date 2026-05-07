"""
preprocess.py
Loads raw executions.csv, applies all feature engineering decisions,
and saves the feature matrix + metadata + fitted transformers to disk.

Outputs
-------
data/executions_features_raw.csv  — suggestion-level feature matrix (pre-scaled)
data/executions_meta.csv          — identifiers + labels kept separate
models/ohe_node_type.joblib       — fitted OneHotEncoder for node_type_at_line
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "executions.csv")

_DROPS = (
    # Zero variance across all rows
    ["grade", "uses_lists", "uses_dicts", "uses_recursion", "uses_exceptions", "inside_try"]
    # Exact duplicates / near-redundant
    + ["is_inside_loop", "total_time_ms_at_line", "min_time_ms_at_line", "max_time_ms_at_line"]
    # Target leakage
    + ["score", "correctness_score", "efficiency_complexity_score", "quality_score", "maintainability_score"]
    # Program-level noise (repeat per program, not per suggestion)
    + [
        "token_count", "ast_node_count", "source_lines", "execution_time_ms",
        "total_lines_executed", "unique_lines_profiled", "peak_memory_bytes",
        "total_suggestions", "suggestion_density", "complexity_confidence",
        "function_count", "loop_count", "if_count", "try_count",
        "assignment_count", "call_count", "binary_op_count",
        "high_severity_count", "medium_severity_count", "low_severity_count",
        "static_suggestion_count", "hybrid_suggestion_count", "dynamic_suggestion_count",
        "count_unused_vars", "count_dead_code", "count_constant_folding",
        "count_early_return", "count_loop_invariant", "count_string_concat_loop",
        "count_nested_loops", "count_hot_loop", "count_repeated_computation",
        "count_expensive_calls",
    ]
    # Non-generalizable identifiers
    + ["line_number", "nearest_function_name", "same_line_suggestion_count", "co_occurring_patterns"]
    # Circular / derived from target
    + ["detector_family", "score_dimension", "impact_score", "complexity_class",
       "severity", "severity_ordinal", "pattern"]
)
ALL_DROPS = list(set(_DROPS))

FEATURE_COLS = [
    # Structural
    "loop_depth", "branch_depth", "function_depth",
    "inside_function", "inside_loop", "inside_branch",
    "max_recursion_depth", "complexity_ordinal", "relative_line_position",
    # Dynamic / runtime
    "execution_count_at_line", "avg_time_ms_at_line", "line_dominance",
    "line_execution_rank", "line_time_rank", "memory_bytes_at_line",
    # Function-level dynamic
    "function_call_count", "function_total_time_ms", "function_avg_time_ms",
    # Categorical (will be OHE'd)
    "node_type_at_line",
]

LOG1P_COLS = [
    "execution_count_at_line", "avg_time_ms_at_line",
    "function_call_count", "function_total_time_ms", "function_avg_time_ms",
]

META_COLS = [
    "program_id", "execution_id", "suggestion_id",
    "pattern", "severity", "severity_ordinal",
    "impact_score", "complexity_class", "complexity_ordinal",
]


def preprocess(data_path: str = DATA_PATH) -> None:
    df = pd.read_csv(data_path)

    df = df.drop(columns=[c for c in ALL_DROPS if c in df.columns], errors="ignore")

    for c in LOG1P_COLS:
        df[c] = np.log1p(df[c])

    for c in ["inside_function", "inside_loop", "inside_branch"]:
        if c in df.columns:
            df[c] = df[c].astype(int)

    df["was_executed"] = (df["execution_count_at_line"] > 0).astype(int)

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    node_encoded = ohe.fit_transform(df[["node_type_at_line"]])
    node_df = pd.DataFrame(
        node_encoded,
        columns=[f"node_{c}" for c in ohe.categories_[0]],
        index=df.index,
    )
    df = pd.concat([df.drop(columns=["node_type_at_line"]), node_df], axis=1)

    numeric_feature_cols = [
        c for c in df.columns
        if c not in META_COLS and c in df.columns
    ]

    df[numeric_feature_cols].to_csv(
        os.path.join(DATA_DIR, "executions_features_raw.csv"), index=False
    )

    raw = pd.read_csv(data_path)  
    meta_present = [c for c in META_COLS if c in raw.columns]
    raw[meta_present].to_csv(os.path.join(DATA_DIR, "executions_meta.csv"), index=False)

    joblib.dump(ohe, os.path.join(MODELS_DIR, "ohe_node_type.joblib"))

    print(f"Features : {df[numeric_feature_cols].shape}")
    print(f"Meta     : {len(meta_present)} columns saved")
    print("Done.")


if __name__ == "__main__":
    preprocess()

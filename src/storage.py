"""CSV storage helpers for the OptiLang ML dataset."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List

ML_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ML_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
EXECUTIONS_CSV = DATA_DIR / "executions.csv"


EXECUTION_FIELDNAMES: List[str] = [
    # --- Identity / grouping ---
    "program_id",
    "execution_id",
    "suggestion_id",
    "source_path",
    "source_hash",
    # --- Core suggestion features ---
    "line_number",
    "pattern",
    "severity",
    "severity_ordinal",
    "detector_family",
    "score_dimension",
    "impact_score",
    # --- Program shape ---
    "token_count",
    "ast_node_count",
    "function_count",
    "loop_count",
    "if_count",
    "try_count",
    "assignment_count",
    "call_count",
    "binary_op_count",
    "uses_lists",
    "uses_dicts",
    "uses_recursion",
    "uses_exceptions",
    # --- Structural (AST) ---
    "node_type_at_line",
    "inside_function",
    "inside_loop",
    "inside_branch",
    "inside_try",
    "loop_depth",
    "branch_depth",
    "function_depth",
    "nearest_function_name",
    "is_inside_loop",
    "relative_line_position",
    "co_occurring_patterns",
    "same_line_suggestion_count",
    # --- Dynamic — line level ---
    "execution_count_at_line",
    "avg_time_ms_at_line",
    "total_time_ms_at_line",
    "min_time_ms_at_line",
    "max_time_ms_at_line",
    "memory_vars_at_line",
    "memory_bytes_at_line",
    "line_dominance",
    "line_execution_rank",
    "line_time_rank",
    # --- Dynamic — function level ---
    "function_call_count",
    "function_total_time_ms",
    "function_avg_time_ms",
    "max_recursion_depth",
    # --- Program-level context ---
    "source_lines",
    "complexity_class",
    "complexity_ordinal",
    "complexity_confidence",
    "execution_time_ms",
    "total_lines_executed",
    "unique_lines_profiled",
    "peak_memory_bytes",
    "total_suggestions",
    "suggestion_density",
    # --- Scoring context ---
    "score",
    "grade",
    "correctness_score",
    "efficiency_complexity_score",
    "quality_score",
    "maintainability_score",
    # --- Relationship/count context ---
    "high_severity_count",
    "medium_severity_count",
    "low_severity_count",
    "static_suggestion_count",
    "hybrid_suggestion_count",
    "dynamic_suggestion_count",
    "count_unused_vars",
    "count_dead_code",
    "count_constant_folding",
    "count_early_return",
    "count_loop_invariant",
    "count_string_concat_loop",
    "count_nested_loops",
    "count_hot_loop",
    "count_repeated_computation",
    "count_expensive_calls",
]


def ensure_data_dirs() -> None:
    """Create the ML data directory if it does not already exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def append_executions(rows: Iterable[Dict[str, object]]) -> int:
    """Append suggestion-level rows to executions.csv.

    Writes the header only when the file is new or empty.
    Returns the number of rows written.
    """
    ensure_data_dirs()
    write_header = not EXECUTIONS_CSV.exists() or EXECUTIONS_CSV.stat().st_size == 0
    count = 0
    with EXECUTIONS_CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=EXECUTION_FIELDNAMES,
            extrasaction="ignore",
        )
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def write_executions(rows: Iterable[Dict[str, object]]) -> int:
    """Replace executions.csv with suggestion-level rows.

    This is the default dataset-generation behavior so repeated runner calls do
    not silently duplicate the dataset.
    """
    ensure_data_dirs()
    count = 0
    with EXECUTIONS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=EXECUTION_FIELDNAMES,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def read_executions() -> List[Dict[str, str]]:
    """Read all rows from executions.csv.

    Returns an empty list if the file does not exist yet.
    """
    if not EXECUTIONS_CSV.exists():
        return []
    with EXECUTIONS_CSV.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def reset_executions() -> None:
    """Delete executions.csv — use when you want a clean dataset rebuild."""
    if EXECUTIONS_CSV.exists():
        EXECUTIONS_CSV.unlink()

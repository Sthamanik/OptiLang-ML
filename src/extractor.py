"""Feature extraction from OptiLang pipeline result objects."""

from __future__ import annotations

import dataclasses
import logging
from typing import Dict, Iterable, List, Optional, Set

from optilang.ast_nodes import (
    ASTNode,
    AssignmentNode,
    AugmentedAssignmentNode,
    BinaryOpNode,
    DictNode,
    ForNode,
    FunctionCallNode,
    FunctionDefNode,
    IfNode,
    ListNode,
    ProgramNode,
    TryNode,
    WhileNode,
)
from optilang.lexer import tokenize
from optilang.models import ExecutionResult, OptimizationReport
from optilang.scoring import ScoreReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _source_line_count(source: str) -> int:
    return len(source.splitlines()) if source else 0


def _profiling_time_ms(result: ExecutionResult) -> float:
    if result.profiling is not None:
        return result.profiling.total_execution_time_ms
    logger.warning("profiling unavailable — falling back to result.execution_time")
    return result.execution_time * 1000.0


def _identity_fields(
    metadata_row: Optional[Dict[str, str]],
    execution_id: Optional[str],
    pattern: str,
    line: int,
) -> Dict[str, str]:
    metadata_row = metadata_row or {}
    resolved_execution_id = execution_id or str(metadata_row.get("execution_id", ""))
    program_id = str(metadata_row.get("program_id", resolved_execution_id))
    source_path = str(metadata_row.get("source_path", ""))
    source_hash = str(metadata_row.get("source_hash", ""))
    suggestion_id = (
        f"{resolved_execution_id}:{pattern}:{line}"
        if resolved_execution_id
        else f"{program_id}:{pattern}:{line}"
    )
    return {
        "program_id": program_id,
        "execution_id": resolved_execution_id,
        "suggestion_id": suggestion_id,
        "source_path": source_path,
        "source_hash": source_hash,
    }


def _walk_children(node: ASTNode) -> Iterable[ASTNode]:
    """Yield direct AST children, including tuple-packed branch nodes."""
    for field in dataclasses.fields(node):
        value = getattr(node, field.name)
        if isinstance(value, ASTNode):
            yield value
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, ASTNode):
                    yield item
                elif isinstance(item, tuple):
                    for element in item:
                        if isinstance(element, ASTNode):
                            yield element
        elif isinstance(value, tuple):
            for element in value:
                if isinstance(element, ASTNode):
                    yield element


def _walk(node: ASTNode) -> Iterable[ASTNode]:
    """Depth-first AST walk."""
    yield node
    for child in _walk_children(node):
        yield from _walk(child)


def _token_count(source: str) -> int:
    try:
        return len(tokenize(source))
    except Exception:
        logger.debug("token counting failed", exc_info=True)
        return 0


def _severity_ordinal(severity: str) -> int:
    return {"low": 1, "medium": 2, "high": 3}.get(severity, 0)


_STATIC_PATTERNS: Set[str] = {
    "unused_vars",
    "dead_code",
    "constant_folding",
    "early_return",
}

_HYBRID_PATTERNS: Set[str] = {
    "loop_invariant",
    "string_concat_loop",
    "nested_loops",
}

_DYNAMIC_PATTERNS: Set[str] = {
    "hot_loop",
    "repeated_computation",
    "expensive_calls",
}

_EFFICIENCY_PATTERNS: Set[str] = {
    "hot_loop",
    "loop_invariant",
    "repeated_computation",
    "expensive_calls",
}

_QUALITY_PATTERNS: Set[str] = {
    "dead_code",
    "string_concat_loop",
}

_MAINTAINABILITY_PATTERNS: Set[str] = {
    "unused_vars",
    "early_return",
    "nested_loops",
    "constant_folding",
}

_KNOWN_PATTERNS: List[str] = [
    "unused_vars",
    "dead_code",
    "constant_folding",
    "early_return",
    "loop_invariant",
    "string_concat_loop",
    "nested_loops",
    "hot_loop",
    "repeated_computation",
    "expensive_calls",
]


def _detector_family(pattern: str) -> str:
    if pattern in _STATIC_PATTERNS:
        return "static"
    if pattern in _HYBRID_PATTERNS:
        return "hybrid"
    if pattern in _DYNAMIC_PATTERNS:
        return "dynamic"
    return "unknown"


def _score_dimension(pattern: str) -> str:
    if pattern in _EFFICIENCY_PATTERNS:
        return "efficiency"
    if pattern in _QUALITY_PATTERNS:
        return "quality"
    if pattern in _MAINTAINABILITY_PATTERNS:
        return "maintainability"
    return "unknown"


def _program_shape(source: str, ast: Optional[ProgramNode]) -> Dict[str, object]:
    shape: Dict[str, object] = {
        "token_count": _token_count(source),
        "ast_node_count": 0,
        "function_count": 0,
        "loop_count": 0,
        "if_count": 0,
        "try_count": 0,
        "assignment_count": 0,
        "call_count": 0,
        "binary_op_count": 0,
        "uses_lists": False,
        "uses_dicts": False,
        "uses_recursion": False,
        "uses_exceptions": False,
    }
    if ast is None:
        return shape

    nodes = list(_walk(ast))
    function_names = {
        node.name.name for node in nodes if isinstance(node, FunctionDefNode)
    }
    called_names = {
        node.function.name for node in nodes if isinstance(node, FunctionCallNode)
    }

    shape.update(
        {
            "ast_node_count": len(nodes),
            "function_count": sum(isinstance(n, FunctionDefNode) for n in nodes),
            "loop_count": sum(isinstance(n, (ForNode, WhileNode)) for n in nodes),
            "if_count": sum(isinstance(n, IfNode) for n in nodes),
            "try_count": sum(isinstance(n, TryNode) for n in nodes),
            "assignment_count": sum(
                isinstance(n, (AssignmentNode, AugmentedAssignmentNode)) for n in nodes
            ),
            "call_count": sum(isinstance(n, FunctionCallNode) for n in nodes),
            "binary_op_count": sum(isinstance(n, BinaryOpNode) for n in nodes),
            "uses_lists": any(isinstance(n, ListNode) for n in nodes),
            "uses_dicts": any(isinstance(n, DictNode) for n in nodes),
            "uses_recursion": bool(function_names & called_names),
            "uses_exceptions": any(isinstance(n, TryNode) for n in nodes),
        }
    )
    return shape


def _line_context(ast: Optional[ProgramNode]) -> Dict[int, Dict[str, object]]:
    """
    Map source lines to their structural context.

    The deepest context wins for nested nodes on the same source line, which
    gives suggestion rows the most specific available location signal.
    """
    context: Dict[int, Dict[str, object]] = {}
    if ast is None:
        return context

    statement_node_names = {
        "AssignmentNode",
        "AugmentedAssignmentNode",
        "ForNode",
        "WhileNode",
        "IfNode",
        "TryNode",
        "FunctionDefNode",
        "ReturnNode",
        "BreakNode",
        "ContinueNode",
        "PassNode",
    }

    def update(node: ASTNode, state: Dict[str, object]) -> None:
        line = getattr(node, "line", None)
        if isinstance(line, int):
            previous = context.get(line)
            current_score = (
                int(state["loop_depth"])
                + int(state["branch_depth"])
                + int(state["function_depth"])
                + int(bool(state["inside_try"]))
            )
            current_priority = 1 if type(node).__name__ in statement_node_names else 0
            previous_score = int(previous["_specificity"]) if previous else -1
            previous_priority = int(previous["_priority"]) if previous else -1
            if (current_score, current_priority) >= (
                previous_score,
                previous_priority,
            ):
                context[line] = {
                    "node_type_at_line": type(node).__name__,
                    "inside_function": bool(state["inside_function"]),
                    "inside_loop": bool(state["inside_loop"]),
                    "inside_branch": bool(state["inside_branch"]),
                    "inside_try": bool(state["inside_try"]),
                    "loop_depth": int(state["loop_depth"]),
                    "branch_depth": int(state["branch_depth"]),
                    "function_depth": int(state["function_depth"]),
                    "nearest_function_name": state["nearest_function_name"],
                    "_specificity": current_score,
                    "_priority": current_priority,
                }

    def visit(node: ASTNode, state: Dict[str, object]) -> None:
        update(node, state)

        if isinstance(node, FunctionDefNode):
            next_state = {
                **state,
                "inside_function": True,
                "function_depth": int(state["function_depth"]) + 1,
                "nearest_function_name": node.name.name,
            }
        elif isinstance(node, (ForNode, WhileNode)):
            next_state = {
                **state,
                "inside_loop": True,
                "loop_depth": int(state["loop_depth"]) + 1,
            }
        elif isinstance(node, IfNode):
            next_state = {
                **state,
                "inside_branch": True,
                "branch_depth": int(state["branch_depth"]) + 1,
            }
        elif isinstance(node, TryNode):
            next_state = {
                **state,
                "inside_try": True,
            }
        else:
            next_state = state

        for child in _walk_children(node):
            visit(child, next_state)

    visit(
        ast,
        {
            "inside_function": False,
            "inside_loop": False,
            "inside_branch": False,
            "inside_try": False,
            "loop_depth": 0,
            "branch_depth": 0,
            "function_depth": 0,
            "nearest_function_name": "",
        },
    )
    for value in context.values():
        value.pop("_specificity", None)
        value.pop("_priority", None)
    return context


_COMPLEXITY_ORDINAL: Dict[str, int] = {
    "O(1)": 1,
    "O(log n)": 2,
    "O(n)": 3,
    "O(n log n)": 4,
    "O(n^2)": 5,
    "O(n²)": 5,  # unicode superscript variant from scorer
    "O(n^k)": 5,  # generic polynomial — treat as n^2 tier
    "O(n^3) or worse": 6,
    "O(n^3)": 6,
    "O(n³)": 6,  # unicode superscript variant from scorer
    "O(2^n)": 7,
}


def _complexity_ordinal(complexity_class: str) -> int:
    return _COMPLEXITY_ORDINAL.get(complexity_class, 0)


def _line_ranks(profiling) -> Dict[int, Dict[str, int]]:
    if profiling is None:
        return {}

    execution_ranked = sorted(
        profiling.line_stats.values(),
        key=lambda stats: stats.execution_count,
        reverse=True,
    )
    time_ranked = sorted(
        profiling.line_stats.values(),
        key=lambda stats: stats.total_time_ms,
        reverse=True,
    )
    ranks: Dict[int, Dict[str, int]] = {}
    for rank, stats in enumerate(execution_ranked, 1):
        ranks.setdefault(stats.line_number, {})["line_execution_rank"] = rank
    for rank, stats in enumerate(time_ranked, 1):
        ranks.setdefault(stats.line_number, {})["line_time_rank"] = rank
    return ranks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract(
    source: str,
    result: ExecutionResult,
    report: Optional[OptimizationReport],
    score: ScoreReport,
    metadata_row: Optional[Dict[str, str]] = None,
    execution_id: Optional[str] = None,
    ast: Optional[ProgramNode] = None,
) -> List[Dict[str, object]]:
    """
    Convert one pipeline run into flat suggestion rows for executions.csv.

    Skips runs with errors — errored programs carry no useful ML signal.
    Returns empty list when errors exist or no suggestions are found.

    Parameters
    ----------
    source          Raw source string of the executed program.
    result          ExecutionResult from interpreter pipeline.
    report          OptimizationReport from analyzer. May be None.
    score           ScoreReport from scoring stage.
    metadata_row    Optional identity/traceability fields for the source program.
    execution_id    Stable id for this exact source execution/version.
    ast             Parsed AST root. Used for loop depth resolution. May be None.
    """
    if result.errors:
        logger.debug("skipping errored execution")
        return []

    raw_suggestions = list(report.suggestions) if report is not None else []

    # Deduplicate by (pattern, line) — the constant_folding detector can
    # visit the same AST node multiple times via different walk paths,
    # producing identical rows that would corrupt training data.
    seen_keys: set = set()
    suggestions = []
    for s in raw_suggestions:
        key = (s.pattern, s.line)
        if key not in seen_keys:
            seen_keys.add(key)
            suggestions.append(s)

    if not suggestions:
        return []

    source_lines = _source_line_count(source)
    execution_time_ms = _profiling_time_ms(result)
    total_suggestions = len(suggestions)
    co_occurring = "|".join(sorted({s.pattern for s in suggestions}))
    line_context = _line_context(ast)
    program_shape = _program_shape(source, ast)

    # --- Program-level profiling ---
    profiling = result.profiling
    total_lines_executed = profiling.total_lines_executed if profiling else 0
    peak_memory_bytes = profiling.peak_memory_bytes if profiling else 0
    unique_lines_profiled = len(profiling.line_stats) if profiling else 0
    complexity_confidence = (
        round(profiling.complexity_confidence, 3) if profiling else 0.0
    )
    ranks = _line_ranks(profiling)

    severity_counts = {"low": 0, "medium": 0, "high": 0}
    family_counts = {"static": 0, "hybrid": 0, "dynamic": 0}
    pattern_counts = {pattern: 0 for pattern in _KNOWN_PATTERNS}
    same_line_counts: Dict[int, int] = {}
    for suggestion in suggestions:
        severity_counts[suggestion.severity] = (
            severity_counts.get(suggestion.severity, 0) + 1
        )
        family = _detector_family(suggestion.pattern)
        if family in family_counts:
            family_counts[family] += 1
        if suggestion.pattern in pattern_counts:
            pattern_counts[suggestion.pattern] += 1
        same_line_counts[suggestion.line] = same_line_counts.get(suggestion.line, 0) + 1

    program_context: Dict[str, object] = {
        # Program-level context
        "source_lines": source_lines,
        "complexity_class": score.complexity_class,
        "complexity_ordinal": _complexity_ordinal(score.complexity_class),
        "complexity_confidence": complexity_confidence,
        "execution_time_ms": round(execution_time_ms, 3),
        "total_lines_executed": total_lines_executed,
        "unique_lines_profiled": unique_lines_profiled,
        "peak_memory_bytes": peak_memory_bytes,
        "total_suggestions": total_suggestions,
        "suggestion_density": (
            round(total_suggestions / source_lines, 6) if source_lines > 0 else 0.0
        ),
        "score": round(score.score, 2),
        "grade": score.grade,
        "correctness_score": round(score.dimensions.correctness, 2),
        "efficiency_complexity_score": round(score.dimensions.efficiency_complexity, 2),
        "quality_score": round(score.dimensions.quality, 2),
        "maintainability_score": round(score.dimensions.maintainability, 2),
        "high_severity_count": severity_counts.get("high", 0),
        "medium_severity_count": severity_counts.get("medium", 0),
        "low_severity_count": severity_counts.get("low", 0),
        "static_suggestion_count": family_counts["static"],
        "hybrid_suggestion_count": family_counts["hybrid"],
        "dynamic_suggestion_count": family_counts["dynamic"],
        **program_shape,
        **{f"count_{pattern}": pattern_counts[pattern] for pattern in _KNOWN_PATTERNS},
    }

    rows: List[Dict[str, object]] = []
    for suggestion in suggestions:
        context = line_context.get(
            suggestion.line,
            {
                "node_type_at_line": "",
                "inside_function": False,
                "inside_loop": False,
                "inside_branch": False,
                "inside_try": False,
                "loop_depth": 0,
                "branch_depth": 0,
                "function_depth": 0,
                "nearest_function_name": "",
            },
        )

        # --- Line-level profiling features ---
        line_stat = profiling.line_stats.get(suggestion.line) if profiling else None
        execution_count_at_line = line_stat.execution_count if line_stat else 0
        avg_time_ms_at_line = round(line_stat.avg_time_ms, 3) if line_stat else 0.0
        total_time_ms_at_line = round(line_stat.total_time_ms, 3) if line_stat else 0.0
        min_time_ms_at_line = (
            round(line_stat.min_time_ms, 3)
            if line_stat and line_stat.min_time_ms != float("inf")
            else 0.0
        )
        max_time_ms_at_line = round(line_stat.max_time_ms, 3) if line_stat else 0.0
        memory_vars_at_line = line_stat.memory_vars if line_stat else 0
        memory_bytes_at_line = line_stat.memory_bytes if line_stat else 0
        line_dominance = (
            round(execution_count_at_line / total_lines_executed, 6)
            if total_lines_executed > 0
            else 0.0
        )
        line_rank = ranks.get(suggestion.line, {})

        # --- Relative position (0–1 normalized) ---
        relative_line_position = (
            round(suggestion.line / source_lines, 4) if source_lines > 0 else 0.0
        )

        # --- Function-level profiling features ---
        function_call_count = 0
        function_total_time_ms = 0.0
        function_avg_time_ms = 0.0
        max_recursion_depth = 0
        nearest_function_name = str(context.get("nearest_function_name", ""))
        if profiling and nearest_function_name in profiling.function_stats:
            function_stat = profiling.function_stats[nearest_function_name]
            function_call_count = function_stat.call_count
            function_total_time_ms = round(function_stat.total_time_ms, 3)
            function_avg_time_ms = round(function_stat.avg_time_ms, 3)
            max_recursion_depth = function_stat.max_recursion_depth

        rows.append(
            {
                **_identity_fields(
                    metadata_row=metadata_row,
                    execution_id=execution_id,
                    pattern=suggestion.pattern,
                    line=suggestion.line,
                ),
                **program_context,
                # Core suggestion identity
                "line_number": suggestion.line,
                "pattern": suggestion.pattern,
                "severity": suggestion.severity,
                "severity_ordinal": _severity_ordinal(suggestion.severity),
                "detector_family": _detector_family(suggestion.pattern),
                "score_dimension": _score_dimension(suggestion.pattern),
                "impact_score": suggestion.impact_score,
                # Structural (AST)
                "node_type_at_line": context["node_type_at_line"],
                "inside_function": context["inside_function"],
                "inside_loop": context["inside_loop"],
                "inside_branch": context["inside_branch"],
                "inside_try": context["inside_try"],
                "loop_depth": context["loop_depth"],
                "branch_depth": context["branch_depth"],
                "function_depth": context["function_depth"],
                "nearest_function_name": nearest_function_name,
                "is_inside_loop": context["inside_loop"],
                "relative_line_position": relative_line_position,
                "co_occurring_patterns": co_occurring,
                "same_line_suggestion_count": same_line_counts.get(suggestion.line, 0),
                # Dynamic — line level
                "execution_count_at_line": execution_count_at_line,
                "avg_time_ms_at_line": avg_time_ms_at_line,
                "total_time_ms_at_line": total_time_ms_at_line,
                "min_time_ms_at_line": min_time_ms_at_line,
                "max_time_ms_at_line": max_time_ms_at_line,
                "memory_vars_at_line": memory_vars_at_line,
                "memory_bytes_at_line": memory_bytes_at_line,
                "line_dominance": line_dominance,
                "line_execution_rank": line_rank.get("line_execution_rank", 0),
                "line_time_rank": line_rank.get("line_time_rank", 0),
                # Dynamic — function level
                "function_call_count": function_call_count,
                "function_total_time_ms": function_total_time_ms,
                "function_avg_time_ms": function_avg_time_ms,
                "max_recursion_depth": max_recursion_depth,
            }
        )

    return rows

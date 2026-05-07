"""Batch runner for OptiLang ML fixtures."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from optilang.executor import execute
from optilang.lexer import tokenize
from optilang.models import OptimizationReport
from optilang.optimizer import analyze
from optilang.parser import parse
from optilang.scoring import ScoreReport, calculate_score

from .extractor import extract
from .storage import (
    EXECUTIONS_CSV,
    RAW_DIR,
    append_executions,
    write_executions,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _source_line_count(source: str) -> int:
    return len(source.splitlines()) if source else 0


def _collect_sources(raw_dir: Path) -> List[Path]:
    """Recursively collect all .py files under raw_dir."""
    return sorted(raw_dir.rglob("*.py"))


def _program_id(source_path: Path, raw_dir: Path = RAW_DIR) -> str:
    """Return a stable corpus-relative program identifier."""
    try:
        return source_path.resolve().relative_to(raw_dir.resolve()).as_posix()
    except ValueError:
        return source_path.resolve().as_posix()


def _source_hash(source: str) -> str:
    """Short stable content hash used to distinguish execution versions."""
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def run_one(
    source_path: Path,
    timeout_seconds: float = 5.0,
    raw_dir: Path = RAW_DIR,
) -> List[Dict[str, object]]:
    """Execute one source file and return flat suggestion rows."""

    source = source_path.read_text(encoding="utf-8")
    program_id = _program_id(source_path, raw_dir)
    source_hash = _source_hash(source)
    execution_id = f"{program_id}@{source_hash}"

    result = execute(source, timeout_seconds=timeout_seconds)
    ast = None
    report: Optional[OptimizationReport] = None

    if not result.errors:
        try:
            ast = parse(tokenize(source))
            report = analyze(ast, result.profiling, result.symbol_table)
        except Exception as exc:
            result.errors.append(str(exc))

    score: ScoreReport = calculate_score(
        profiling_data=result.profiling.to_dict() if result.profiling else None,
        optimizer_report=report,
        source_lines=_source_line_count(source),
        errors=result.errors,
    )

    return extract(
        source=source,
        result=result,
        report=report,
        score=score,
        metadata_row={
            "program_id": program_id,
            "source_path": source_path.as_posix(),
            "source_hash": source_hash,
        },
        execution_id=execution_id,
        ast=ast,
    )


def run_all(
    raw_dir: Path = RAW_DIR,
    limit: Optional[int] = None,
    skip_pathological: bool = False,
    timeout_seconds: float = 5.0,
) -> List[Dict[str, object]]:
    """Run all source files and return flat suggestion rows."""

    sources = _collect_sources(raw_dir)

    if skip_pathological:
        sources = [s for s in sources if "pathological" not in s.parts]

    if limit is not None:
        sources = sources[:limit]

    all_rows: List[Dict[str, object]] = []
    failed: List[str] = []

    for i, source_path in enumerate(sources, 1):
        try:
            rows = run_one(
                source_path,
                timeout_seconds=timeout_seconds,
                raw_dir=raw_dir,
            )
            all_rows.extend(rows)
            print(f"[{i}/{len(sources)}] OK   {source_path.name}  →  {len(rows)} rows")
        except Exception as exc:
            failed.append(source_path.name)
            print(f"[{i}/{len(sources)}] FAIL {source_path.name}  →  {exc}")

    if failed:
        print(f"\nFailed ({len(failed)}): {', '.join(failed)}")

    return all_rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run OptiLang ML fixtures.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help="Directory containing raw .py program files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N programs.",
    )
    parser.add_argument(
        "--skip-pathological",
        action="store_true",
        help="Skip programs inside the pathological/ subdirectory.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Per-program execution timeout in seconds.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append rows to executions.csv instead of replacing the dataset.",
    )
    args = parser.parse_args(argv)

    rows = run_all(
        raw_dir=args.raw_dir,
        limit=args.limit,
        skip_pathological=args.skip_pathological,
        timeout_seconds=args.timeout,
    )

    if args.append:
        count = append_executions(rows)
        action = "Appended"
    else:
        count = write_executions(rows)
        action = "Wrote"

    print(f"\n{action} {count} suggestion rows → {EXECUTIONS_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

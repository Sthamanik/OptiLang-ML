# Contributing to OptiLang ML Extension

Thank you for your interest in contributing! This document covers everything you need to get started — from setting up your local environment to submitting a pull request.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Project Structure Overview](#project-structure-overview)
4. [Development Setup](#development-setup)
5. [Running the Pipeline Locally](#running-the-pipeline-locally)
6. [How to Contribute](#how-to-contribute)
7. [Pull Request Guidelines](#pull-request-guidelines)
8. [Coding Standards](#coding-standards)
9. [Testing](#testing)
10. [Important Rules & Gotchas](#important-rules--gotchas)
11. [Reporting Bugs](#reporting-bugs)
12. [Feature Requests](#feature-requests)

---

## Code of Conduct

Be respectful. Be constructive. Focus on the work. We welcome contributors of all experience levels.

---

## Getting Started

Before contributing, please:

- Read this document fully
- Browse [open issues](../../issues) to avoid duplicate work
- Comment on an issue before starting work on it — this prevents multiple people working on the same thing
- For significant changes (new pipeline stages, model architecture changes, new feature groups), open a discussion issue first

---

## Project Structure Overview

```text
Optilang-ML/
├── data/          # Generated datasets — git-ignored, never commit these
├── models/        # Trained artifacts — git-ignored, never commit these
├── notebooks/     # EDA and training notebooks (run in order: 01 → 04)
├── plots/         # Saved visualisations from notebooks
└── src/
    ├── runner.py       # Corpus execution + raw row collection
    ├── extractor.py    # Feature extraction from profiling results
    ├── storage.py      # CSV read/write helpers
    ├── preprocess.py   # Feature engineering + transformer fitting
    ├── train.py        # Clustering → classification → rank prediction
    ├── predict.py      # Inference class + CLI
    └── pipeline.py     # High-level orchestrator
```

Key things to know:
- `data/` and `models/` are fully git-ignored. Use `data/.gitkeep` and `models/.gitkeep` to preserve the directory structure.
- Pipeline stages are sequential and stateful — artifacts from one stage feed directly into the next. Never mix artifacts from different training runs.
- The holdout set (`data/holdout_programs.csv`) must never be touched during training. This is enforced in `train.py` but please respect it in notebooks too.

---

## Development Setup

**Requirements:**
- Python 3.9+
- Git

**Steps:**

```bash
# 1. Fork and clone the repository
git clone https://github.com/<your-username>/optilang-ml.git
cd optilang-ml

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify the directory structure is intact
ls data/       # should contain .gitkeep
ls models/     # should contain .gitkeep
```

If there is no `requirements.txt` yet and you are adding one as part of your contribution, include all transitive dependencies and pin major versions.

---

## Running the Pipeline Locally

Run these steps in order when setting up from scratch or after collecting new data:

```bash
# Step 1 — Collect raw execution data (use --limit for a quick smoke test)
python3 -m optilang.ml.src.runner --limit 20

# Step 2 — Feature engineering
python3 -m optilang.ml.src.preprocess

# Step 3 — Train all models
python3 -m optilang.ml.src.train

# Step 4 — Run inference
python3 -m optilang.ml.src.predict --input-csv data/executions.csv --output-csv predictions.csv
```

> **Never** run steps out of order or mix artifacts from different runs. If you change anything in `preprocess.py`, you must re-run all subsequent steps.

---

## How to Contribute

### Bug Fixes

1. Open an issue describing the bug (or pick an existing one)
2. Create a branch: `git checkout -b fix/<short-description>`
3. Make your changes, add a test if applicable
4. Submit a pull request referencing the issue

### New Features

1. Open a discussion issue first for anything non-trivial
2. Create a branch: `git checkout -b feat/<short-description>`
3. Implement the feature with appropriate tests and documentation updates
4. Submit a pull request

### Notebooks

Notebook contributions (new analysis, improved clustering sweeps, better evaluation metrics) are welcome. Keep notebooks clean and runnable top-to-bottom. Strip output before committing:

```bash
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebooks/*.ipynb
```

### Documentation

Documentation improvements — including this file — are always welcome. Open a PR directly for typo fixes and small clarifications; open an issue first for structural changes.

---

## Pull Request Guidelines

- **One concern per PR.** Don't bundle a bug fix with a refactor and a new feature.
- **Reference the related issue** in the PR description (`Closes #42`).
- **Describe what changed and why** — not just what the diff shows.
- **Keep PRs small and reviewable.** Large PRs take longer to review and are more likely to introduce conflicts.
- **Do not commit generated files.** No `data/*.csv`, `models/*.joblib`, `models/*.pkl`, `*.png` plots, or notebook outputs.
- **Update documentation** if your change affects the public API, CLI interface, pipeline behaviour, or feature columns.

---

## Coding Standards

- Follow [PEP 8](https://peps.python.org/pep-0008/) for all Python code
- Use type hints on all function signatures
- Write docstrings for all public functions and classes (Google style preferred)
- Keep functions focused — one responsibility per function
- Prefer explicit over implicit; avoid magic numbers and unnamed constants

```python
# Good
LOG_TRANSFORM_COLS = [
    "execution_count_at_line",
    "avg_time_ms_at_line",
    "function_call_count",
]

def apply_log_transforms(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Apply log1p transform to the specified numeric columns.

    Args:
        df: Input feature DataFrame.
        columns: Column names to transform.

    Returns:
        DataFrame with transformed columns.
    """
    result = df.copy()
    for col in columns:
        result[col] = np.log1p(result[col])
    return result
```

---

## Testing

There is currently no formal test suite. If you are adding a test suite, please open an issue first to align on the testing framework (pytest is preferred).

For now, the expected validation approach is:

- Run the full pipeline end-to-end with `--limit 20` and confirm it completes without errors
- For `predict.py` changes, confirm predictions include all expected output columns
- For `preprocess.py` changes, confirm feature column counts and shapes are consistent downstream

If you add tests, place them in a `tests/` directory and ensure they can be run with:

```bash
pytest tests/
```

---

## Important Rules & Gotchas

These are easy to get wrong. Please read before touching the pipeline:

**Holdout discipline.** `holdout_programs.csv` lists programs excluded from all training phases. Never include holdout programs in clustering, classification, or rank prediction. This is checked in `train.py` — do not remove that check.

**Artifact consistency.** The scaler, PCA, OHE, and classifiers are fitted as a unit. If you retrain any single artifact in isolation, the rest will silently produce incorrect results. Always run the full `train.py`.

**Re-run order after data changes.** Any change to `runner.py` or `extractor.py` means you must re-run `preprocess.py` → `train.py`. Any change to `preprocess.py` means you must re-run `train.py`. Do not skip steps.

**Runner default behaviour.** `runner.py` replaces `executions.csv` by default. Use `--append` only when intentionally merging independent collection runs.

**Score proxy vs. runtime speedup.** `predicted_priority_rank` is based on `expected_score_improvement_pct`, a proxy derived from OptiLang's scoring rules — not measured runtime improvement. Do not present it as a runtime speedup metric in documentation or comments.

**Cluster reference fallback.** If `models/clustering_medoids.joblib` is missing, `OptiLangPredictor` falls back to deriving cluster reference vectors from `executions_clustered.csv`. This is a degraded mode. If you see this warning, re-run `train.py`.

---

## Reporting Bugs

Open an issue and include:

- A minimal reproduction: the exact command you ran and the input data (or a `--limit N` reproduction)
- The full error message and traceback
- Your Python version (`python3 --version`) and OS
- Which pipeline stage failed (runner / preprocess / train / predict)

---

## Feature Requests

Open an issue with the label `enhancement`. Describe:

- The problem you're trying to solve
- How your proposed feature addresses it
- Any alternative approaches you considered

Planned future work (good places to contribute):
- Auto-fix and re-execution support for measuring true runtime speedup
- Expanding the corpus beyond TheAlgorithms/Python
- Persisting true k-medoids centroids to eliminate the cluster reference fallback
- A formal test suite

---

## Questions?

Open an issue with the label `question`. We'll do our best to respond promptly.
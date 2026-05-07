# OptiLang ML Extension

This package is a self-contained machine-learning extension for OptiLang. It builds an end-to-end pipeline from raw Python source programs to actionable, prioritised optimization predictions:

```
Raw Programs → Execution & Extraction → Feature Engineering
    → Clustering → Classification → Rank Prediction → Inference
```

Models are trained on a corpus of 1 000 real-world Python programs sourced from [TheAlgorithms/Python](https://github.com/TheAlgorithms/Python). The trained artifacts are used at inference time to cluster incoming suggestion rows, classify their optimization pattern, and rank which clusters deserve the most attention per program.

---

## Table of Contents

1. [Directory Layout](#directory-layout)
2. [Dataset & Corpus](#dataset--corpus)
3. [ML Pipeline Phases](#ml-pipeline-phases)
   - [Phase 1 — Raw Program Corpus](#phase-1--raw-program-corpus)
   - [Phase 2/3 — Runner, Extractor & Storage](#phase-23--runner-extractor--storage)
   - [Phase 4 — Preprocessing & Feature Engineering](#phase-4--preprocessing--feature-engineering)
   - [Phase 5 — Training (Clustering → Classification → Prediction)](#phase-5--training-clustering--classification--prediction)
   - [Phase 6 — Inference](#phase-6--inference)
4. [Source Modules](#source-modules)
5. [Notebooks](#notebooks)
6. [Trained Model Artifacts](#trained-model-artifacts)
7. [Python API](#python-api)
8. [CLI Reference](#cli-reference)
9. [Development Notes](#development-notes)

---

## Directory Layout

```text
Optilang-ML/
├── README.md
├── __init__.py
├── .gitignore
│
├── data/                                  # Generated datasets (git-ignored)
│   ├── executions.csv                     # Raw: one row per suggestion per execution
│   ├── executions_meta.csv                # Metadata: identifiers + labels
│   ├── executions_features_raw.csv        # Feature matrix (pre-scaled)
│   ├── executions_clean.csv               # Cleaned subset used by notebooks
│   ├── executions_clustered.csv           # Metadata + learned strategy cluster
│   ├── holdout_indices.csv                # Row indices held out during training
│   └── holdout_programs.csv               # Program IDs held out during training
│
├── models/                                # Trained artifacts (git-ignored)
│   ├── clustering_scaler.joblib           # StandardScaler fitted on features
│   ├── clustering_pca.joblib              # PCA transform (dimensionality reduction)
│   ├── ohe_node_type.joblib               # OneHotEncoder for node_type_at_line
│   ├── le_pattern.joblib                  # LabelEncoder for optimization pattern
│   ├── le_severity.joblib                 # LabelEncoder for severity
│   ├── cluster_names_k6.json             # Human-readable names for the 6 clusters
│   ├── classifier_xgboost_k6.pkl         # XGBoost pattern classifier (k=6)
│   └── best_randomforest_rank_predictor.joblib  # Random Forest rank predictor
│
├── notebooks/
│   ├── 01_eda.ipynb                       # Exploratory data analysis & metadata export
│   ├── 02_clustering.ipynb                # Strategy clustering (k-medoids sweep)
│   ├── 03_classification.ipynb            # Pattern classification benchmarks
│   └── 04_prediction.ipynb               # Rank prediction & evaluation
│
├── plots/                                 # Saved visualisations from notebooks
│   ├── cluster_pca_2d.png
│   ├── cluster_feature_profile.png
│   ├── cluster_pattern_heatmap.png
│   ├── confusion_matrix_patterns.png
│   ├── feature_importance_pattern.png
│   ├── prediction_rank_residuals.png
│   └── ...  (17 total)
│
└── src/
    ├── runner.py       # Execute corpus programs + collect raw rows
    ├── extractor.py    # Feature extraction from profiling results
    ├── storage.py      # CSV persistence helpers
    ├── preprocess.py   # Feature engineering & transformer fitting
    ├── train.py        # Clustering, classification, and rank prediction training
    ├── predict.py      # OptiLangPredictor inference class + CLI
    └── pipeline.py     # OptiLangMLPipeline high-level orchestrator
```

---

## Dataset & Corpus

| Item | Value |
|------|-------|
| Corpus source | [TheAlgorithms/Python](https://github.com/TheAlgorithms/Python) |
| Total programs | 1 000 |
| Execution meta rows | 690 |
| Successful executions | 684 |
| Pathological / error executions | 6 |
| Suggestion rows | 1 410 |

Raw programs are stored under `data/raw/`, preserving the original category folder structure and filenames. There are **no** manifest, strategy label, or family label files in the raw path — all labels are derived during preprocessing and training.

---

## ML Pipeline Phases

### Phase 1 — Raw Program Corpus

Raw programs live under:

```text
optilang/ml/data/raw/
```

The directory mirrors the TheAlgorithms/Python repository tree. Each file is a standalone Python algorithm or problem solution.

---

### Phase 2/3 — Runner, Extractor & Storage

The runner passes each source file through the OptiLang execution engine, collects profiling data and optimization suggestions via `extractor.py`, then persists all rows to `executions.csv` via `storage.py`.

**Run the runner:**

```bash
python3 -m optilang.ml.src.runner
```

> The runner **replaces** `executions.csv` by default to prevent duplicate rows across repeated runs. Use `--append` only when intentionally merging independent collection runs.

**Useful options:**

```bash
# Limit to 20 programs for a quick smoke test
python3 -m optilang.ml.src.runner --limit 20

# Skip programs known to hang or crash
python3 -m optilang.ml.src.runner --skip-pathological

# Reduce per-program timeout (default: 5 s)
python3 -m optilang.ml.src.runner --timeout 3

# Append to existing CSV instead of replacing it
python3 -m optilang.ml.src.runner --append
```

**Output files:**

```text
optilang/ml/data/
├── executions.csv                   # one row per suggestion per execution
```

**Stable identity fields in `executions.csv`** allow downstream notebooks to aggregate suggestions back to program level:

| Field | Description |
|-------|-------------|
| `program_id` | Corpus-relative source path |
| `execution_id` | `program_id` + short source hash |
| `suggestion_id` | Stable id for a suggestion within one execution |
| `source_path` | Absolute path for traceability |
| `source_hash` | SHA hash for debugging / change detection |

---

### Phase 4 — Preprocessing & Feature Engineering

Preprocessing loads `executions.csv`, drops high-noise / zero-variance / leaky columns, applies log-transforms, one-hot encodes the AST node type, and saves the feature matrix and fitted transformers.

```bash
python3 -m optilang.ml.src.preprocess
```

**Feature columns used for modelling:**

| Category | Features |
|----------|----------|
| Structural | `loop_depth`, `branch_depth`, `function_depth`, `inside_function`, `inside_loop`, `inside_branch`, `max_recursion_depth`, `complexity_ordinal`, `relative_line_position` |
| Runtime | `execution_count_at_line`, `avg_time_ms_at_line`, `line_dominance`, `line_execution_rank`, `line_time_rank`, `memory_bytes_at_line` |
| Function-level | `function_call_count`, `function_total_time_ms`, `function_avg_time_ms` |
| Categorical | `node_type_at_line` (one-hot encoded → `node_*` columns) |

Log₁₊ transforms are applied to: `execution_count_at_line`, `avg_time_ms_at_line`, `function_call_count`, `function_total_time_ms`, `function_avg_time_ms`.

**Outputs:**

| File | Description |
|------|-------------|
| `data/executions_features_raw.csv` | Suggestion-level feature matrix (pre-scaled) |
| `data/executions_meta.csv` | Identifiers + labels kept separate |
| `models/ohe_node_type.joblib` | Fitted `OneHotEncoder` for `node_type_at_line` |

---

### Phase 5 — Training (Clustering → Classification → Prediction)

Training runs three sequential stages, each building on the previous.

```bash
python3 -m optilang.ml.src.train
```

#### Stage 1: Strategy Clustering

- Algorithm: **k-medoids** with a metric sweep over multiple `k` values (see `02_clustering.ipynb`)
- Final model: **k = 6** clusters, each with a human-readable name in `models/cluster_names_k6.json`
- Artifacts saved: `clustering_scaler.joblib`, `clustering_pca.joblib`, `executions_clustered.csv`, `holdout_programs.csv`

#### Stage 2: Pattern Classification

- Model: **XGBoost** classifier
- Target: `pattern` (one of the 10 OptiLang optimization detectors)
- Input: scaled feature matrix + cluster assignment
- Artifact saved: `classifier_xgboost_k6.pkl`

#### Stage 3: Rank Prediction

- Model: **Random Forest** regressor
- Target: `expected_score_improvement_pct` — the expected percentage score recovery if one cluster is fixed for a given program
- This is a **score proxy** derived from OptiLang's scorer rules, *not* measured runtime speedup. True runtime improvement requires a future auto-fix and re-run stage.
- Artifact saved: `best_randomforest_rank_predictor.joblib`

> **Note on the prediction target:** `04_prediction.ipynb` does **not** predict the heuristic `impact_score`. It predicts `expected_score_improvement_pct`, which is a more actionable, scorer-grounded signal.

---

### Phase 6 — Inference

Run predictions on a new execution CSV or directly on a Python source file/directory:

```bash
# Score an existing executions CSV
python3 -m optilang.ml.src.predict --input-csv path/to/executions.csv --output-csv predictions.csv

# Execute a Python file first, then predict
python3 -m optilang.ml.src.predict --source path/to/my_program.py --output-csv predictions.csv

# Execute an entire directory of Python files
python3 -m optilang.ml.src.predict --source path/to/programs/ --output-csv predictions.csv
```

Each output row includes:

| Column | Description |
|--------|-------------|
| `program_id` | Source program identifier |
| `suggestion_id` | Suggestion identifier |
| `cluster` | Assigned strategy cluster ID |
| `cluster_name` | Human-readable cluster name |
| `predicted_pattern` | Predicted optimization pattern label |
| `pattern_confidence` | Classifier probability for the predicted pattern |
| `predicted_priority_rank` | Predicted rank (lower = higher priority) |
| `cluster_suggestion_count` | Number of suggestions in this cluster for the program |
| `cluster_share` | Fraction of program's suggestions in this cluster |

---

## Source Modules

| Module | Responsibility |
|--------|---------------|
| `runner.py` | Iterates the raw corpus, calls OptiLang execution + `extractor.py` per file, returns raw rows |
| `extractor.py` | Converts `ExecutionResult` + `OptimizerReport` into flat suggestion-level feature rows |
| `storage.py` | Typed CSV read/write helpers (`write_executions`, `append_executions`, `read_executions`, `reset_executions`) |
| `preprocess.py` | Feature engineering, column dropping, log-transforms, OHE fitting; saves feature matrix + metadata |
| `train.py` | Orchestrates clustering → classification → rank prediction; saves all model artifacts |
| `predict.py` | `OptiLangPredictor` class for inference; accepts CSV or `pd.DataFrame`; also exposes a CLI |
| `pipeline.py` | `OptiLangMLPipeline` high-level orchestrator; wraps all stages into a single API surface |

---

## Notebooks

Run notebooks in order. Each notebook expects artifacts from the previous step.

| Notebook | Purpose |
|----------|---------|
| `01_eda.ipynb` | Exploratory analysis of raw executions; exports `executions_meta.csv` |
| `02_clustering.ipynb` | k-medoids metric sweep; produces cluster assignments and plots |
| `03_classification.ipynb` | XGBoost / baseline pattern classification benchmarks; confusion matrices |
| `04_prediction.ipynb` | Random Forest rank prediction; residual analysis; holdout evaluation |

> All generated plots are saved to `plots/` automatically by the notebooks.

---

## Trained Model Artifacts

| Artifact | Type | Description |
|----------|------|-------------|
| `clustering_scaler.joblib` | `StandardScaler` | Feature scaler fitted on training suggestions |
| `clustering_pca.joblib` | `PCA` | Dimensionality reduction before k-medoids |
| `ohe_node_type.joblib` | `OneHotEncoder` | Encodes `node_type_at_line` AST node category |
| `le_pattern.joblib` | `LabelEncoder` | Encodes the 10 optimization pattern labels |
| `le_severity.joblib` | `LabelEncoder` | Encodes suggestion severity levels |
| `cluster_names_k6.json` | JSON | Maps cluster IDs 0–5 to human-readable strategy names |
| `classifier_xgboost_k6.pkl` | XGBoost | Predicts optimization pattern given features + cluster |
| `best_randomforest_rank_predictor.joblib` | Random Forest | Predicts program-cluster priority rank |

> All model artifacts are **git-ignored**. They must be generated locally by running the training pipeline, or obtained from a shared artefact store.

---

## Python API

### `OptiLangMLPipeline` — High-Level Orchestrator

```python
from optilang.ml import OptiLangMLPipeline

# Step-by-step
pipeline = OptiLangMLPipeline()
pipeline.collect_data(limit=50)   # run programs, extract features, write executions.csv
pipeline.preprocess()              # feature engineering, save feature matrix & transformers
pipeline.train()                   # clustering → classification → rank prediction
results = pipeline.predict(input_data="data/executions.csv", output_csv="predictions.csv")

# Or run everything in one call
results = OptiLangMLPipeline.full_lifecycle(
    raw_dir="optilang/ml/data/raw/",
    output_predictions="predictions.csv",
)
```

### `OptiLangPredictor` — Low-Level Inference

```python
from optilang.ml import OptiLangPredictor
import pandas as pd

predictor = OptiLangPredictor()  # loads all model artifacts automatically

# From a CSV file
results = predictor.predict_csv("data/executions.csv", output_csv="predictions.csv")

# From a DataFrame
raw_rows = pd.read_csv("data/executions.csv")
results = predictor.predict_rows(raw_rows)

print(results[["program_id", "cluster_name", "predicted_pattern", "predicted_priority_rank"]])
```

### Individual Training Stages

```python
from optilang.ml import run_preprocess, run_clustering, run_classification, run_prediction

run_preprocess()

import pandas as pd
X_raw = pd.read_csv("data/executions_features_raw.csv")
meta  = pd.read_csv("data/executions_meta.csv")

labels, k, holdout_idx, holdout_programs = run_clustering(X_raw, meta)
run_classification(X_raw, meta, holdout_programs, k)
run_prediction(meta, X_raw)
```

---

## CLI Reference

### Runner

```bash
python3 -m optilang.ml.src.runner [OPTIONS]

Options:
  --limit N             Process only the first N programs
  --skip-pathological   Skip programs that are known to hang or error
  --timeout SECONDS     Per-program execution timeout (default: 5)
  --append              Append to existing executions.csv instead of replacing
```

### Predictor

```bash
python3 -m optilang.ml.src.predict [OPTIONS]

Input (mutually exclusive):
  --input-csv PATH      Executions-style CSV to score (default: data/executions.csv)
  --source PATH         Python file or directory to execute before prediction

Options:
  --output-csv PATH     Destination for prediction rows (default: data/predictions.csv)
  --models-dir PATH     Path to model artifacts directory (default: models/)
  --data-dir PATH       Path to data directory (default: data/)
  --timeout SECONDS     Per-program timeout when using --source (default: 5)
  --limit N             Limit number of source programs when using --source
  --skip-pathological   Skip pathological programs when using --source
  --top N               Number of preview rows to print (default: 10)
```

---

## Development Notes

- **No raw data or models are committed.** The `.gitignore` excludes all `data/*.csv`, `models/`, `*.joblib`, and `*.pkl` files. Use `data/.gitkeep` and `models/.gitkeep` to preserve the directory structure in git.
- **Holdout discipline.** `holdout_programs.csv` lists the program IDs excluded from all training phases. Never include holdout programs in clustering, classification, or rank prediction training.
- **Re-training.** After collecting new data, always re-run preprocessing → training in order. Mixing artifacts from different training runs will produce silent mis-alignments between the scaler, PCA, and classifiers.
- **Cluster reference fallback.** If `models/clustering_medoids.joblib` is missing, `OptiLangPredictor` will fall back to deriving cluster reference vectors from `executions_clustered.csv`. Re-run `train.py` to persist true medoids and eliminate this warning.
- **Score proxy vs. runtime speedup.** `predicted_priority_rank` reflects `expected_score_improvement_pct`, a proxy built from OptiLang's scoring rules. It is **not** a measured runtime improvement. Auto-fix and re-execution support is planned for a future release.

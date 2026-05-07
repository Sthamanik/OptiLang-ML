"""
Unified orchestrator for the OptiLang ML lifecycle.
Integrates extractor.py (via runner), storage.py, preprocess.py, train.py, and predict.py
into a single, production-ready interface.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from .preprocess import preprocess as _preprocess
from .train import run_clustering, run_classification, run_prediction
from .predict import OptiLangPredictor
from .runner import run_all
from .storage import write_executions, append_executions, DATA_DIR as STORAGE_DATA_DIR

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_MODELS_DIR = BASE_DIR / "models"
DEFAULT_RAW_DIR = STORAGE_DATA_DIR / "raw"


class OptiLangMLPipeline:
    """End-to-end orchestrator: data extraction → feature engineering → training → inference."""

    def __init__(
        self,
        data_dir: Union[Path, str] = DEFAULT_DATA_DIR,
        models_dir: Union[Path, str] = DEFAULT_MODELS_DIR,
        raw_dir: Optional[Union[Path, str]] = None,
        executions_csv: Optional[Union[Path, str]] = None,
    ) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.models_dir = Path(models_dir).resolve()
        self.raw_dir = Path(raw_dir).resolve() if raw_dir else DEFAULT_RAW_DIR
        self.executions_csv = Path(executions_csv).resolve() if executions_csv else self.data_dir / "executions.csv"

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def collect_data(self, append: bool = False, limit: Optional[int] = None, timeout: float = 5.0) -> int:
        """
        Run source programs, extract features via extractor.py, and persist to storage.
        runner.run_all() internally calls extractor.extract() for each program.
        """
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Raw source directory not found: {self.raw_dir}")

        rows = run_all(raw_dir=self.raw_dir, limit=limit, timeout_seconds=timeout)
        if not rows:
            raise ValueError("No execution rows were generated. Check source files and timeouts.")

        count = append_executions(rows) if append else write_executions(rows)
        print(f"[Collect] Wrote {count} rows to {self.executions_csv}")
        return count

    def preprocess(self) -> None:
        """Engineer features from executions.csv and save intermediate artifacts."""
        if not self.executions_csv.exists():
            raise FileNotFoundError(f"Missing {self.executions_csv}. Run .collect_data() first.")
        _preprocess(data_path=str(self.executions_csv))
        print(f"[Preprocess] Features & metadata saved to {self.data_dir} and {self.models_dir}.")

 
    def train(self) -> dict:
        """Run the full training pipeline: clustering → classification → rank prediction."""
        features_path = self.data_dir / "executions_features_raw.csv"
        meta_path = self.data_dir / "executions_meta.csv"

        if not features_path.exists() or not meta_path.exists():
            raise FileNotFoundError("Missing preprocessed files. Run .preprocess() first.")

        X_raw = pd.read_csv(features_path)
        meta = pd.read_csv(meta_path).reset_index(drop=True)

        print("\n── Phase 1: Clustering ──────────────────────────")
        final_labels, final_k, idx_holdout, holdout_prgs = run_clustering(X_raw, meta)
        meta["cluster"] = final_labels
        meta.to_csv(self.data_dir / "executions_clustered.csv", index=False)
        pd.DataFrame({"program_id": sorted(holdout_prgs)}).to_csv(
            self.data_dir / "holdout_programs.csv", index=False
        )

        print("\n── Phase 2: Classification ──────────────────────")
        X_with_cluster = X_raw.copy()
        X_with_cluster["cluster"] = final_labels
        run_classification(X_with_cluster, meta, {str(p) for p in holdout_prgs}, final_k)

        print("\n── Phase 3: Rank Prediction ─────────────────────")
        best_reg = run_prediction(meta, X_raw)

        print("\n[Train] All models saved to models/")
        return {"k": final_k, "holdout_programs": holdout_prgs, "best_rank_predictor": best_reg}

    def predict(
        self,
        input_data: Union[Path, str, pd.DataFrame],
        output_csv: Optional[Union[Path, str]] = None,
    ) -> pd.DataFrame:
        """Run inference using trained models on new suggestion rows."""
        predictor = OptiLangPredictor(models_dir=self.models_dir, data_dir=self.data_dir)

        if isinstance(input_data, pd.DataFrame):
            result = predictor.predict_rows(input_data)
        elif isinstance(input_data, (str, Path)):
            result = predictor.predict_csv(input_data)
        else:
            raise TypeError("input_data must be a pandas DataFrame or path to a CSV file.")

        if output_csv:
            out_path = Path(output_csv)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(out_path, index=False)
            print(f"[Predict] Results saved to {out_path}")

        return result

    @classmethod
    def full_lifecycle(
        cls,
        raw_dir: Optional[Union[Path, str]] = None,
        output_predictions: Optional[Union[Path, str]] = None,
        data_dir: Union[Path, str] = DEFAULT_DATA_DIR,
        models_dir: Union[Path, str] = DEFAULT_MODELS_DIR,
        append_data: bool = False,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Execute: collect_data → preprocess → train → predict in one call."""
        pipeline = cls(data_dir=data_dir, models_dir=models_dir, raw_dir=raw_dir)
        pipeline.collect_data(append=append_data, limit=limit)
        pipeline.preprocess()
        pipeline.train()
        return pipeline.predict(
            input_data=pipeline.executions_csv,
            output_csv=output_predictions,
        )
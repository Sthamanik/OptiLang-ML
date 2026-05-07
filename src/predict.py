"""
predict.py
Runs the trained OptiLang ML pipeline on new suggestion rows.

Inputs can be an executions-style CSV produced by runner.py, or a Python source
file/directory that will first be executed and extracted into suggestion rows.

Outputs one row per suggestion with:
  - assigned cluster id/name
  - predicted optimization pattern
  - classifier confidence when available
  - predicted program-cluster priority rank
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Optional, Sequence

import joblib
import numpy as np
import pandas as pd

from .preprocess import FEATURE_COLS, LOG1P_COLS, META_COLS
from .runner import run_all, run_one
from .storage import RAW_DIR


ML_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ML_DIR / "data"
MODELS_DIR = ML_DIR / "models"

DEFAULT_INPUT_CSV = DATA_DIR / "executions.csv"
DEFAULT_OUTPUT_CSV = DATA_DIR / "predictions.csv"


class OptiLangPredictor:
    """Inference wrapper for the artifacts saved by preprocess.py and train.py."""

    def __init__(self, models_dir: Path | str = MODELS_DIR, data_dir: Path | str = DATA_DIR):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)

        self.ohe = self._load_optional("ohe_node_type.joblib")
        self.scaler = self._load_required("clustering_scaler.joblib")
        self.pca = self._load_required("clustering_pca.joblib")
        self.pattern_encoder = self._load_optional("le_pattern.joblib")
        self.classifier = self._load_required(self._latest("classifier_*_k*.pkl"))
        self.rank_predictor = self._load_required(self._latest("best_*_rank_predictor.joblib"))

        self.cluster_labels, self.medoid_vectors = self._load_cluster_reference()
        self.cluster_names = self._load_cluster_names()

    def predict_rows(self, raw_rows: pd.DataFrame) -> pd.DataFrame:
        if raw_rows.empty:
            raise ValueError("No suggestion rows were provided.")

        X_raw = self._preprocess_rows(raw_rows)
        clusters = self._assign_clusters(X_raw)
        pattern_labels, pattern_confidence = self._classify_patterns(X_raw, clusters)
        cluster_predictions = self._predict_cluster_priority(raw_rows, X_raw, clusters)

        result = self._base_output(raw_rows)
        result["cluster"] = clusters
        result["cluster_name"] = result["cluster"].map(self.cluster_names).fillna("")
        result["predicted_pattern"] = pattern_labels
        result["pattern_confidence"] = pattern_confidence

        result = result.merge(
            cluster_predictions,
            on=["program_id", "cluster"],
            how="left",
        )
        result = result.sort_values(
            ["program_id", "predicted_priority_rank", "cluster", "suggestion_id"],
            na_position="last",
        ).reset_index(drop=True)
        return result

    def predict_csv(self, input_csv: Path | str, output_csv: Optional[Path | str] = None) -> pd.DataFrame:
        raw_rows = pd.read_csv(input_csv)
        result = self.predict_rows(raw_rows)
        if output_csv is not None:
            Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(output_csv, index=False)
        return result

    def _load_required(self, name_or_path: str | Path):
        path = self.models_dir / name_or_path if isinstance(name_or_path, str) else name_or_path
        if not path.exists():
            raise FileNotFoundError(f"Missing model artifact: {path}")
        return joblib.load(path)

    def _load_optional(self, name: str):
        path = self.models_dir / name
        return joblib.load(path) if path.exists() else None

    def _latest(self, pattern: str) -> Path:
        matches = sorted(self.models_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if not matches:
            raise FileNotFoundError(f"No model artifact matched {self.models_dir / pattern}")
        return matches[0]

    def _load_cluster_reference(self) -> tuple[np.ndarray, np.ndarray]:
        medoid_path = self.models_dir / "clustering_medoids.joblib"
        if medoid_path.exists():
            bundle = joblib.load(medoid_path)
            if isinstance(bundle, dict):
                return (
                    np.asarray(bundle.get("cluster_labels"), dtype=int),
                    np.asarray(bundle["medoid_vectors"], dtype=float),
                )
            return np.arange(len(bundle), dtype=int), np.asarray(bundle, dtype=float)

        clustered_path = self.data_dir / "executions_clustered.csv"
        features_path = self.data_dir / "executions_features_raw.csv"
        if clustered_path.exists() and features_path.exists():
            warnings.warn(
                "models/clustering_medoids.joblib is missing; deriving cluster reference "
                "vectors from executions_clustered.csv. Re-run train.py to persist true medoids.",
                RuntimeWarning,
            )
            clustered = pd.read_csv(clustered_path)
            X_train = pd.read_csv(features_path)
            if len(clustered) != len(X_train) or "cluster" not in clustered.columns:
                raise ValueError("Cannot derive cluster references from mismatched clustered/features data.")
            X_train = self._align_frame(X_train, self._feature_names(self.scaler), fill_value=0.0)
            X_pca = self.pca.transform(self.scaler.transform(X_train))
            ref = pd.DataFrame(X_pca)
            ref["cluster"] = clustered["cluster"].astype(int).values
            labels = np.asarray(sorted(ref["cluster"].unique()), dtype=int)
            vectors = ref.groupby("cluster").mean().loc[labels].to_numpy(dtype=float)
            return labels, vectors

        raise FileNotFoundError(
            "Missing models/clustering_medoids.joblib. Re-run optilang.ml.src.train "
            "or keep executions_clustered.csv and executions_features_raw.csv available."
        )

    def _load_cluster_names(self) -> dict[int, str]:
        matches = sorted(self.models_dir.glob("cluster_names_k*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not matches:
            return {}
        with matches[0].open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        return {int(k): str(v) for k, v in loaded.items()}

    def _preprocess_rows(self, raw_rows: pd.DataFrame) -> pd.DataFrame:
        df = raw_rows.copy()

        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = "Unknown" if col == "node_type_at_line" else 0

        for col in LOG1P_COLS:
            df[col] = np.log1p(pd.to_numeric(df[col], errors="coerce").fillna(0.0))

        for col in ["inside_function", "inside_loop", "inside_branch"]:
            df[col] = df[col].fillna(False).astype(int)

        df["was_executed"] = (pd.to_numeric(df["execution_count_at_line"], errors="coerce").fillna(0.0) > 0).astype(int)

        node_df = self._encode_node_type(df["node_type_at_line"])
        df = pd.concat([df.drop(columns=["node_type_at_line"]), node_df], axis=1)

        feature_names = self._feature_names(self.scaler)
        return self._align_frame(df, feature_names, fill_value=0.0)

    def _encode_node_type(self, values: pd.Series) -> pd.DataFrame:
        expected_node_cols = [
            col for col in self._feature_names(self.scaler)
            if col.startswith("node_")
        ]
        if not expected_node_cols:
            return pd.DataFrame(index=values.index)

        if self.ohe is not None and hasattr(self.ohe, "categories_"):
            encoded = self.ohe.transform(values.fillna("Unknown").astype(str).to_frame())
            cols = [f"node_{category}" for category in self.ohe.categories_[0]]
            node_df = pd.DataFrame(encoded, columns=cols, index=values.index)
        else:
            node_df = pd.DataFrame(index=values.index)
            clean_values = values.fillna("Unknown").astype(str)
            for col in expected_node_cols:
                node_df[col] = (clean_values == col.removeprefix("node_")).astype(float)

        return self._align_frame(node_df, expected_node_cols, fill_value=0.0)

    def _assign_clusters(self, X_raw: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X_raw)
        X_pca = self.pca.transform(X_scaled)
        distances = np.linalg.norm(
            X_pca[:, np.newaxis, :] - self.medoid_vectors[np.newaxis, :, :],
            axis=2,
        )
        return self.cluster_labels[np.argmin(distances, axis=1)].astype(int)

    def _classify_patterns(self, X_raw: pd.DataFrame, clusters: np.ndarray) -> tuple[list[str], np.ndarray]:
        clf_input = X_raw.copy()
        clf_input["cluster"] = clusters
        clf_input = self._align_frame(clf_input, self._feature_names(self.classifier), fill_value=0.0)

        encoded = self.classifier.predict(clf_input)
        if self.pattern_encoder is not None:
            labels = self.pattern_encoder.inverse_transform(encoded.astype(int)).tolist()
        else:
            labels = [str(value) for value in encoded]

        if hasattr(self.classifier, "predict_proba"):
            confidence = np.max(self.classifier.predict_proba(clf_input), axis=1)
        else:
            confidence = np.full(len(clf_input), np.nan)
        return labels, confidence

    def _predict_cluster_priority(
        self,
        raw_rows: pd.DataFrame,
        X_raw: pd.DataFrame,
        clusters: np.ndarray,
    ) -> pd.DataFrame:
        meta = self._base_output(raw_rows)
        meta["cluster"] = clusters

        scaled = pd.DataFrame(
            self.scaler.transform(X_raw),
            columns=[f"scaled_{col}" for col in X_raw.columns],
            index=X_raw.index,
        )
        row_level = pd.concat([meta[["program_id", "suggestion_id", "cluster"]], scaled], axis=1)

        program_counts = (
            row_level.groupby("program_id")["suggestion_id"]
            .count()
            .rename("program_suggestion_count")
        )
        cluster_frame = (
            row_level.groupby(["program_id", "cluster"])
            .agg(cluster_suggestion_count=("suggestion_id", "count"))
            .reset_index()
            .merge(program_counts.reset_index(), on="program_id", how="left")
        )
        cluster_frame["cluster_share"] = (
            cluster_frame["cluster_suggestion_count"] / cluster_frame["program_suggestion_count"]
        )

        scaled_means = row_level.groupby(["program_id", "cluster"])[scaled.columns].mean().reset_index()
        cluster_frame = cluster_frame.merge(scaled_means, on=["program_id", "cluster"], how="left")

        pred_input = self._align_frame(cluster_frame, self._feature_names(self.rank_predictor), fill_value=0.0)
        cluster_frame["predicted_priority_rank"] = self.rank_predictor.predict(pred_input)
        cluster_frame["predicted_priority_rank"] = cluster_frame["predicted_priority_rank"].round(3)
        return cluster_frame[
            [
                "program_id",
                "cluster",
                "cluster_suggestion_count",
                "cluster_share",
                "program_suggestion_count",
                "predicted_priority_rank",
            ]
        ]

    def _base_output(self, raw_rows: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(index=raw_rows.index)
        for col in META_COLS:
            if col in raw_rows.columns:
                output[col] = raw_rows[col]

        if "program_id" not in output:
            output["program_id"] = "input_program"
        output["program_id"] = output["program_id"].fillna("input_program").astype(str)

        if "suggestion_id" not in output:
            output["suggestion_id"] = [f"{pid}:suggestion:{i}" for i, pid in enumerate(output["program_id"])]
        output["suggestion_id"] = output["suggestion_id"].fillna("").astype(str)
        missing_ids = output["suggestion_id"].eq("")
        output.loc[missing_ids, "suggestion_id"] = [
            f"{pid}:suggestion:{i}" for i, pid in zip(output.index[missing_ids], output.loc[missing_ids, "program_id"])
        ]

        for col in ["pattern", "severity", "impact_score", "complexity_class"]:
            if col in raw_rows.columns and col not in output:
                output[col] = raw_rows[col]
        return output

    @staticmethod
    def _feature_names(model) -> list[str]:
        if hasattr(model, "feature_names_in_"):
            return [str(col) for col in model.feature_names_in_]
        raise AttributeError(f"{type(model).__name__} does not expose feature_names_in_.")

    @staticmethod
    def _align_frame(df: pd.DataFrame, columns: Sequence[str], fill_value: float) -> pd.DataFrame:
        aligned = df.copy()
        for col in columns:
            if col not in aligned.columns:
                aligned[col] = fill_value
        aligned = aligned.loc[:, list(columns)]
        for col in columns:
            aligned[col] = pd.to_numeric(aligned[col], errors="coerce").fillna(fill_value)
        return aligned


def _load_source_rows(source: Path, timeout: float, limit: Optional[int], skip_pathological: bool) -> pd.DataFrame:
    if source.is_file():
        rows = run_one(source, timeout_seconds=timeout, raw_dir=RAW_DIR)
    elif source.is_dir():
        rows = run_all(
            raw_dir=source,
            limit=limit,
            skip_pathological=skip_pathological,
            timeout_seconds=timeout,
        )
    else:
        raise FileNotFoundError(f"Source path does not exist: {source}")
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run trained OptiLang ML predictions.")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Executions-style CSV to score.",
    )
    input_group.add_argument(
        "--source",
        type=Path,
        help="Python file or directory to execute before prediction.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Where to write prediction rows.",
    )
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-pathological", action="store_true")
    parser.add_argument("--top", type=int, default=10, help="Rows to print after writing output.")
    args = parser.parse_args(argv)

    predictor = OptiLangPredictor(models_dir=args.models_dir, data_dir=args.data_dir)

    if args.source is not None:
        raw_rows = _load_source_rows(args.source, args.timeout, args.limit, args.skip_pathological)
        result = predictor.predict_rows(raw_rows)
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output_csv, index=False)
    else:
        result = predictor.predict_csv(args.input_csv, args.output_csv)

    preview_cols = [
        col for col in [
            "program_id",
            "suggestion_id",
            "cluster",
            "cluster_name",
            "predicted_pattern",
            "pattern_confidence",
            "predicted_priority_rank",
        ]
        if col in result.columns
    ]
    print(f"Wrote {len(result)} prediction rows -> {args.output_csv}")
    if args.top > 0 and preview_cols:
        print(result[preview_cols].head(args.top).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

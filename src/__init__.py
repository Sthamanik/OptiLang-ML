"""Implementation modules for the OptiLang ML extension."""

from .extractor import extract
from .runner import run_one, run_all
from .storage import (
    EXECUTIONS_CSV,
    DATA_DIR as STORAGE_DATA_DIR,
    append_executions,
    read_executions,
    reset_executions,
    write_executions,
)

from .preprocess import preprocess as run_preprocess
from .train import run_clustering, run_classification, run_prediction, main as train_main
from .predict import OptiLangPredictor
from .pipeline import OptiLangMLPipeline

__all__ = [
    # Data Collection & Storage
    "run_one",
    "run_all",
    "extract",
    "write_executions",
    "append_executions",
    "read_executions",
    "reset_executions",
    "EXECUTIONS_CSV",
    "STORAGE_DATA_DIR",
    # ML Lifecycle
    "run_preprocess",
    "run_clustering",
    "run_classification",
    "run_prediction",
    "train_main",
    "OptiLangPredictor",
    # Pipeline
    "OptiLangMLPipeline",
]
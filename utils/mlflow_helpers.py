"""
MLflow tracking helpers for logging parameters, metrics, artifacts, and registered models.
Provides explicit manual logging wrappers to prevent duplication and ensure blueprint compliance.
"""

import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
from typing import Dict, Any, Optional
import numpy as np
import mlflow
import mlflow.pytorch
import torch
from utils.logging import get_logger

logger = get_logger("MLflowHelpers")

def setup_experiment(
    experiment_name: str = "AML-CNN-BASELINE",
    tracking_uri: Optional[str] = "./mlruns"
) -> str:
    """Configures MLflow tracking URI and initializes or sets the active experiment.

    Args:
        experiment_name: Name of the MLflow experiment.
        tracking_uri: Path or URI for MLflow tracking store.

    Returns:
        The experiment ID as a string.
    """
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    if tracking_uri:
        if os.path.isabs(tracking_uri) and "://" not in tracking_uri:
            from pathlib import Path
            tracking_uri = Path(tracking_uri).as_uri()
        mlflow.set_tracking_uri(tracking_uri)

    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        exp_id = mlflow.create_experiment(experiment_name)
    else:
        exp_id = exp.experiment_id

    mlflow.set_experiment(experiment_name)
    return exp_id

def log_params(params: Dict[str, Any]) -> None:
    """Logs a dictionary of parameters to the currently active MLflow run.

    Args:
        params: Flat or nested configuration dictionary.
    """
    # Flatten nested dictionaries for MLflow parameter logging
    flat_params = {}
    def _flatten(d: Dict[str, Any], prefix: str = "") -> None:
        for k, v in d.items():
            full_key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            if isinstance(v, dict):
                _flatten(v, prefix=full_key)
            else:
                flat_params[full_key] = str(v)
    
    _flatten(params)
    for k, v in flat_params.items():
        mlflow.log_param(k[:100], v[:500])  # Enforce MLflow length constraints (100 chars for Windows FileStore MAX_PATH safety)

def log_epoch_metrics(metrics: Dict[str, float], step: int) -> None:
    """Logs numerical metrics for a given training epoch step.

    Args:
        metrics: Dictionary of metric names and float values.
        step: Epoch or iteration step integer.
    """
    for k, v in metrics.items():
        if v is not None:
            mlflow.log_metric(k, float(v), step=step)

def log_artifact(local_file_path: str, artifact_path: Optional[str] = None) -> None:
    """Logs a local file artifact (e.g., plot image or CSV report) to MLflow.

    Args:
        local_file_path: Absolute or relative path to local file.
        artifact_path: Optional subdirectory path within MLflow artifact store.
    """
    if os.path.exists(local_file_path):
        mlflow.log_artifact(local_file_path, artifact_path)
    else:
        logger.warning(f"Artifact file not found: {local_file_path}. Skipping artifact log.")

def log_and_register_model(
    model: torch.nn.Module,
    artifact_path: str = "model",
    registered_model_name: Optional[str] = "FraudCNN"
) -> None:
    """Logs PyTorch model artifact and registers it in the MLflow model registry.

    Args:
        model: PyTorch model instance.
        artifact_path: Subdirectory within MLflow artifact store.
        registered_model_name: Name under which model is registered.
    """
    # Log PyTorch model cleanly without torch.export trace issues
    example_input = np.random.randn(1, 3, 32, 32).astype(np.float32)

    def _log_attempt(reg_name):
        try:
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=artifact_path,
                registered_model_name=reg_name,
                input_example=example_input
            )
        except Exception:
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=artifact_path,
                registered_model_name=reg_name,
                serialization_format="pickle"
            )

    try:
        _log_attempt(registered_model_name)
    except Exception as e:
        if registered_model_name is not None:
            logger.warning(f"Could not register model '{registered_model_name}' to Model Registry ({e}). Falling back to artifact-only logging.")
            _log_attempt(None)
        else:
            raise

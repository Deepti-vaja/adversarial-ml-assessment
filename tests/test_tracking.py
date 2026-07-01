"""
Automated verification suite for the Experiment Tracking subsystem.
Proves implementation correctness across experiment setup, parameter flattening, step metrics,
nested run hierarchies, artifact attachment, and end-to-end PyTorch model registration & reloading.
"""

import os
import shutil
import pytest
import numpy as np
import torch
import torch.nn as nn
import mlflow
from utils.mlflow_helpers import (
    setup_experiment,
    log_params,
    log_epoch_metrics,
    log_artifact,
    log_and_register_model
)

class DummyCNN(nn.Module):
    """Simple linear model for tracking verification."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, 3)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

@pytest.fixture
def tracking_env(tmp_path):
    """Provides an isolated temporary MLflow tracking directory."""
    uri = (tmp_path / "mlruns").as_uri()
    mlflow.set_tracking_uri(uri)
    yield uri
    # Clean up active runs if any remained open
    while mlflow.active_run():
        mlflow.end_run()

def test_setup_experiment(tracking_env):
    """Verifies experiment setup and idempotency."""
    exp_name = "AML-TEST-EXP"
    exp_id_1 = setup_experiment(exp_name, tracking_uri=tracking_env)
    assert exp_id_1 is not None, "Experiment ID should not be None."
    
    exp = mlflow.get_experiment_by_name(exp_name)
    assert exp is not None
    assert exp.experiment_id == exp_id_1

    # Check idempotency
    exp_id_2 = setup_experiment(exp_name, tracking_uri=tracking_env)
    assert exp_id_1 == exp_id_2, "Repeated setup should return identical experiment ID."

def test_log_params(tracking_env):
    """Verifies nested dictionary flattening and length constraint truncation."""
    setup_experiment("AML-TEST-PARAMS", tracking_uri=tracking_env)
    
    long_val = "x" * 600
    long_key = "k" * 300
    params = {
        "optimizer": {
            "lr": 0.001,
            "type": "adamw"
        },
        long_key: long_val
    }

    with mlflow.start_run() as run:
        log_params(params)
        run_id = run.info.run_id

    client = mlflow.tracking.MlflowClient()
    logged_data = client.get_run(run_id).data.params
    assert logged_data["optimizer.lr"] == "0.001"
    assert logged_data["optimizer.type"] == "adamw"
    
    truncated_key = long_key[:100]
    assert truncated_key in logged_data
    assert len(logged_data[truncated_key]) == 500

def test_log_epoch_metrics(tracking_env):
    """Verifies step-indexed metric tracking and None filtering."""
    setup_experiment("AML-TEST-METRICS", tracking_uri=tracking_env)
    
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        log_epoch_metrics({"train_loss": 0.5, "val_loss": None}, step=1)
        log_epoch_metrics({"train_loss": 0.3, "val_loss": 0.4}, step=2)

    client = mlflow.tracking.MlflowClient()
    train_history = client.get_metric_history(run_id, "train_loss")
    assert len(train_history) == 2
    assert train_history[0].step == 1 and abs(train_history[0].value - 0.5) < 1e-6
    assert train_history[1].step == 2 and abs(train_history[1].value - 0.3) < 1e-6

    val_history = client.get_metric_history(run_id, "val_loss")
    assert len(val_history) == 1
    assert val_history[0].step == 2 and abs(val_history[0].value - 0.4) < 1e-6

def test_nested_runs(tracking_env):
    """Verifies child runs contain parent run hierarchy linking."""
    setup_experiment("AML-TEST-NESTED", tracking_uri=tracking_env)

    with mlflow.start_run(run_name="parent_run") as parent_run:
        parent_id = parent_run.info.run_id
        with mlflow.start_run(run_name="child_run", nested=True) as child_run:
            child_id = child_run.info.run_id
            log_params({"attack": "fgsm"})

    client = mlflow.tracking.MlflowClient()
    child_tags = client.get_run(child_id).data.tags
    assert child_tags.get("mlflow.parentRunId") == parent_id

def test_log_artifact_and_model(tracking_env, tmp_path):
    """Verifies local artifact logging and warning on missing artifact."""
    setup_experiment("AML-TEST-ARTIFACT", tracking_uri=tracking_env)

    # Create temporary file
    report_file = tmp_path / "test_report.txt"
    report_file.write_text("Test evaluation report")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        log_artifact(str(report_file), "reports")
        # Log missing file (should emit warning and not raise error)
        log_artifact(str(tmp_path / "non_existent.png"), "plots")

    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id, "reports")
    assert any("test_report.txt" in a.path for a in artifacts)

def test_log_and_reload_model(tracking_env):
    """End-to-End verification proving logged model can be reloaded with prediction parity."""
    setup_experiment("AML-TEST-MODEL", tracking_uri=tracking_env)

    model = DummyCNN()
    model.eval()
    dummy_input = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        original_preds = model(dummy_input).numpy()

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        log_and_register_model(model, artifact_path="model", registered_model_name="TestCNN")

    # Construct artifact URI for reloading
    model_uri = f"runs:/{run_id}/model"
    reloaded_model = mlflow.pytorch.load_model(model_uri)
    reloaded_model.eval()

    with torch.no_grad():
        reloaded_preds = reloaded_model(dummy_input).numpy()

    assert np.allclose(original_preds, reloaded_preds, atol=1e-6), "Reloaded model predictions diverged from original model."

"""
Orchestration script for training the baseline FraudCNN model.
Runs data loading, supervised training loop with CosineAnnealingLR, validation tracking,
test set evaluation against the >=75% clean accuracy gate, MLflow logging, and checkpointing.
"""

import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import argparse
import sys
from typing import Dict, Any
import torch
import torch.nn as nn
import mlflow

from data.cifar10_loader import get_dataloaders, load_config
from models.fraud_cnn import build_model
from training.trainer import Trainer
from utils.seed import set_seed
from utils.logging import get_logger
from utils.timing import Timer
from utils.metrics import save_confusion_matrix_figure
from utils.mlflow_helpers import (
    setup_experiment,
    log_params,
    log_epoch_metrics,
    log_artifact,
    log_and_register_model
)

def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Train Baseline FraudCNN Model")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/baseline.yaml",
        help="Path to baseline configuration YAML"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a 1-epoch dry run on a small subset for testing pipeline integrity"
    )
    return parser.parse_args()

def run_baseline_training(args: argparse.Namespace) -> float:
    """Main execution workflow for baseline training.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Final clean test set accuracy as a float.
    """
    logger = get_logger("TrainBaseline", log_file="./reports/train_baseline.log")
    logger.info(f"Starting baseline training pipeline with config: {args.config}")

    # 1. Load config and apply CLI overrides
    config = load_config(args.config)
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.lr is not None:
        config.setdefault("training", {})["learning_rate"] = args.lr
    if args.batch_size is not None:
        config.setdefault("data", {})["batch_size"] = args.batch_size
        config.setdefault("training", {})["batch_size"] = args.batch_size

    if args.dry_run:
        logger.info("DRY RUN ENABLED: Limiting training to 1 epoch for verification.")
        config.setdefault("training", {})["epochs"] = 1

    seed = config.get("seed", 42)
    set_seed(seed)

    # 2. Setup DataLoaders
    logger.info("Initializing CIFAR-10 proxy dataset and DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # If dry run, limit dataloaders or batches
    if args.dry_run:
        # Create tiny subset loaders for rapid execution
        train_subset = torch.utils.data.Subset(train_loader.dataset, list(range(min(128, len(train_loader.dataset)))))
        val_subset = torch.utils.data.Subset(val_loader.dataset, list(range(min(64, len(val_loader.dataset)))))
        test_subset = torch.utils.data.Subset(test_loader.dataset, list(range(min(64, len(test_loader.dataset)))))
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=32, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_subset, batch_size=32, shuffle=False)

    # 3. Build Model & Optimization Suite
    logger.info("Instantiating FraudCNN architecture...")
    model = build_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Computation device selected: {device}")

    train_cfg = config.get("training", {})
    epochs = train_cfg.get("epochs", 30)
    lr = train_cfg.get("learning_rate", 0.001)
    weight_decay = train_cfg.get("weight_decay", 0.0001)
    optimizer_name = train_cfg.get("optimizer", "AdamW")

    if optimizer_name.upper() == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    trainer = Trainer(
        model=model,
        device=device,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        config=config,
        logger=logger
    )

    # 4. Setup MLflow Tracking
    setup_experiment(experiment_name="AML-CNN-BASELINE")
    
    checkpoint_dir = train_cfg.get("checkpoint_dir", "./models/checkpoints")
    checkpoint_name = train_cfg.get("checkpoint_name", "fraud_cnn_baseline.pth")
    best_ckpt_path = os.path.join(checkpoint_dir, checkpoint_name)

    timer = Timer()
    timer.start()

    best_val_acc = 0.0

    with mlflow.start_run(run_name="fraud_cnn_baseline") as run:
        logger.info(f"MLflow Run ID started: {run.info.run_id}")
        log_params(config)

        # 5. Training & Validation Loop
        for epoch in range(1, epochs + 1):
            train_metrics = trainer.train_epoch(train_loader, epoch)
            val_metrics = trainer.evaluate(val_loader)

            epoch_log = {
                "train_loss": train_metrics["train_loss"],
                "train_acc": train_metrics["train_acc"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"]
            }
            log_epoch_metrics(epoch_log, step=epoch)

            logger.info(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {train_metrics['train_loss']:.4f} | Train Acc: {train_metrics['train_acc']*100:.2f}% | "
                f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']*100:.2f}%"
            )

            # Save best checkpoint
            if val_metrics["accuracy"] >= best_val_acc or epoch == 1:
                best_val_acc = val_metrics["accuracy"]
                trainer.save_checkpoint(
                    filepath=best_ckpt_path,
                    epoch=epoch,
                    metrics={"val_acc": best_val_acc, "val_loss": val_metrics["loss"]}
                )

        elapsed_hours = timer.stop() / 3600.0
        mlflow.log_metric("training_overhead_hours", elapsed_hours)
        logger.info(f"Training completed in {elapsed_hours*60:.2f} minutes ({elapsed_hours:.4f} hours).")

        # 6. Final Evaluation on Clean Test Split
        logger.info("Loading best checkpoint for final clean test set evaluation...")
        if os.path.exists(best_ckpt_path):
            trainer.load_checkpoint(best_ckpt_path, load_optimizer=False)

        test_results = trainer.evaluate(test_loader, return_confusion_matrix=True)
        clean_test_acc = test_results["accuracy"]
        test_loss = test_results["loss"]

        logger.info(f"Final Clean Test Loss: {test_loss:.4f} | Clean Test Accuracy: {clean_test_acc*100:.2f}%")
        mlflow.log_metric("clean_test_accuracy", clean_test_acc)
        mlflow.log_metric("clean_test_loss", test_loss)

        # 7. Save and Log Confusion Matrix Artifact
        cm = test_results.get("confusion_matrix")
        if cm is not None:
            cm_fig_path = "./reports/confusion_matrix_baseline.png"
            save_confusion_matrix_figure(cm, cm_fig_path, title="Baseline FraudCNN Confusion Matrix")
            log_artifact(cm_fig_path)
            # Also log generated CSV version
            cm_csv_path = cm_fig_path.replace(".png", ".csv")
            log_artifact(cm_csv_path)

        # 8. Check Acceptance Criterion Gate (>= 75% clean accuracy)
        if not args.dry_run:
            if clean_test_acc >= 0.75:
                logger.info(
                    f"[SUCCESS] Clean test accuracy ({clean_test_acc*100:.2f}%) "
                    f"meets or exceeds the >=75.0% acceptance threshold."
                )
            else:
                limitation_msg = (
                    f"[LIMITATION] Clean test accuracy ({clean_test_acc*100:.2f}%) did not satisfy the >=75.0% "
                    f"acceptance threshold. Potential architectural refinements for subsequent iteration include: "
                    f"adding residual skip connections, increasing filter progression depth, or using cyclic learning rates."
                )
                logger.warning(limitation_msg)

        # 9. Register Model Artifact in MLflow
        logger.info("Logging and registering FraudCNN model artifact in MLflow...")
        try:
            log_and_register_model(trainer.model, artifact_path="model", registered_model_name="FraudCNN")
        except Exception as e:
            logger.warning(f"Could not register model to MLflow model registry (local file store limitation or offline): {e}")
            # Fallback log model weights artifact
            mlflow.log_artifact(best_ckpt_path, "checkpoints")

    return clean_test_acc

if __name__ == "__main__":
    args = parse_args()
    try:
        run_baseline_training(args)
    except Exception as e:
        sys.stderr.write(f"Fatal error in baseline training: {str(e)}\n")
        sys.exit(1)

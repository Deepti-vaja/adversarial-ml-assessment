"""
Adversarial Training defense module for Task D1.
Fine-tunes the baseline FraudCNN model using ART's AdversarialTrainer with PGD-generated examples
for >=10 epochs, outputs robust checkpoint, and logs run parameters and metrics to MLflow AML-DEFENSE.
"""

import os
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import mlflow

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescentPyTorch
from art.defences.trainer import AdversarialTrainer

from data.cifar10_loader import load_config, get_dataloaders
from models.fraud_cnn import build_model
from training.trainer import Trainer
from utils.seed import set_seed
from utils.logging import get_logger
from utils.timing import timer
from utils.mlflow_helpers import setup_experiment, log_params

logger = get_logger("AdversarialTraining")


def extract_arrays_from_loader(loader: torch.utils.data.DataLoader, max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts numpy arrays (x, y) from a PyTorch DataLoader."""
    x_list, y_list = [], []
    total = 0
    for images, targets in loader:
        x_list.append(images.numpy())
        y_list.append(targets.numpy())
        total += len(targets)
        if max_samples is not None and total >= max_samples:
            break
    x_arr = np.concatenate(x_list, axis=0)
    y_arr = np.concatenate(y_list, axis=0)
    if max_samples is not None:
        return x_arr[:max_samples], y_arr[:max_samples]
    return x_arr, y_arr


def run_adversarial_training(
    defense_config_path: str = "./configs/defenses.yaml",
    max_samples: Optional[int] = None,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, PyTorchClassifier, Dict[str, Any]]:
    """Executes adversarial fine-tuning of the baseline FraudCNN architecture using ART AdversarialTrainer.

    Args:
        defense_config_path: Path to configuration YAML defining adversarial training hyperparameters.
        max_samples: Optional cap on dataset size for rapid testing/verification.
        device: Target computation device (CUDA or CPU).

    Returns:
        Tuple of (fine-tuned PyTorch model, wrapped ART PyTorchClassifier, training summary dictionary).
    """
    config = load_config(defense_config_path)
    seed = config.get("seed", 42)
    set_seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config_path = config.get("model", {}).get("config_path", "./configs/baseline.yaml")
    baseline_ckpt_path = config.get("model", {}).get("baseline_checkpoint_path", "./models/checkpoints/fraud_cnn_baseline.pth")
    output_ckpt_path = config.get("model", {}).get("adv_trained_checkpoint_path", "./models/checkpoints/fraud_cnn_adv_trained.pth")

    adv_cfg = config.get("adversarial_training", {})
    epochs = adv_cfg.get("epochs", 10)
    batch_size = adv_cfg.get("batch_size", 64)
    lr = adv_cfg.get("learning_rate", 0.0005)

    atk_cfg = adv_cfg.get("attack", {})
    eps = atk_cfg.get("eps", 0.05)
    eps_step = atk_cfg.get("eps_step", 0.01)
    max_iter = atk_cfg.get("max_iter", 7)

    logger.info(f"Loading baseline config from {model_config_path} and model weights from {baseline_ckpt_path}...")
    base_cfg = load_config(model_config_path)
    model = build_model(base_cfg)
    model.to(device)

    # Cleanly restore baseline checkpoint
    dummy_opt = torch.optim.AdamW(model.parameters(), lr=lr)
    trainer_helper = Trainer(model=model, device=device, optimizer=dummy_opt, criterion=nn.CrossEntropyLoss())
    if os.path.exists(baseline_ckpt_path):
        trainer_helper.load_checkpoint(baseline_ckpt_path, load_optimizer=False)
    else:
        logger.warning(f"Baseline checkpoint {baseline_ckpt_path} not found. Starting fine-tuning from initialized weights.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    device_type = "gpu" if device.type == "cuda" else "cpu"
    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=3,
        clip_values=(0.0, 1.0),
        device_type=device_type
    )

    pgd_attack = ProjectedGradientDescentPyTorch(
        estimator=classifier,
        norm=np.inf,
        eps=float(eps),
        eps_step=float(eps_step),
        max_iter=int(max_iter),
        targeted=False,
        batch_size=batch_size
    )

    adv_trainer = AdversarialTrainer(
        classifier=classifier,
        attacks=pgd_attack,
        ratio=0.5
    )

    logger.info("Loading training and validation datasets...")
    train_loader, val_loader, _ = get_dataloaders(base_cfg)
    x_train, y_train = extract_arrays_from_loader(train_loader, max_samples=max_samples)
    x_val, y_val = extract_arrays_from_loader(val_loader, max_samples=max_samples)

    logger.info(f"Starting Adversarial Fine-Tuning for {epochs} epochs on {len(x_train)} samples...")
    setup_experiment("AML-DEFENSE")

    with mlflow.start_run(run_name="Adversarial_Training_PGD"):
        log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "pgd_eps": eps,
            "pgd_eps_step": eps_step,
            "pgd_max_iter": max_iter,
            "defense_type": "AdversarialTraining"
        })

        with timer() as t:
            adv_trainer.fit(x_train, y_train, nb_epochs=epochs, batch_size=batch_size)
        training_hours = t.elapsed_hours
        logger.info(f"Adversarial fine-tuning completed in {t.elapsed_seconds:.2f} seconds ({training_hours:.4f} hours).")

        # Evaluate clean and robust validation accuracy
        clean_preds = classifier.predict(x_val, batch_size=batch_size)
        clean_val_acc = float(np.mean(np.argmax(clean_preds, axis=1) == y_val))

        x_val_adv = pgd_attack.generate(x=x_val)
        adv_preds = classifier.predict(x_val_adv, batch_size=batch_size)
        robust_val_acc = float(np.mean(np.argmax(adv_preds, axis=1) == y_val))

        logger.info(f"Validation Clean Accuracy: {clean_val_acc*100:.2f}% | Validation PGD Robust Accuracy: {robust_val_acc*100:.2f}%")

        mlflow.log_metric("clean_val_acc", clean_val_acc)
        mlflow.log_metric("robust_val_acc", robust_val_acc)
        mlflow.log_metric("training_overhead_hours", training_hours)

        # Save robust model checkpoint
        metrics = {
            "clean_val_acc": clean_val_acc,
            "robust_val_acc": robust_val_acc,
            "training_hours": training_hours
        }
        trainer_helper.save_checkpoint(
            filepath=output_ckpt_path,
            epoch=epochs,
            metrics=metrics,
            model_config=base_cfg.get("model", {})
        )

    summary = {
        "clean_val_acc": clean_val_acc,
        "robust_val_acc": robust_val_acc,
        "training_hours": training_hours,
        "checkpoint_path": output_ckpt_path
    }

    return model, classifier, summary

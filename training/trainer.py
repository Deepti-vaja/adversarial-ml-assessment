"""
Modular training engine for supervised baseline models and downstream adversarial training.
Provides clean encapsulation of training epochs, evaluation loops, and checkpointing.
"""

import os
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.metrics import compute_accuracy, compute_confusion_matrix
from utils.logging import get_logger

class Trainer:
    """
    Production-ready modular training engine encapsulating optimization loops,
    validation evaluation, and model checkpointing.
    """
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[Any] = None
    ) -> None:
        """
        Args:
            model: PyTorch model module to train.
            device: Target computation device (CUDA or CPU).
            optimizer: PyTorch optimizer instance (e.g., Adam or AdamW).
            criterion: Loss function (e.g., CrossEntropyLoss).
            scheduler: Optional learning rate scheduler (e.g., CosineAnnealingLR).
            config: Configuration dictionary storing training hyperparameters.
            logger: Structured logger instance.
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.config = config if config is not None else {}
        self.logger = logger if logger is not None else get_logger("Trainer")

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Executes a single supervised training epoch over the training dataset.

        Args:
            train_loader: DataLoader providing training image-target batches.
            epoch: Current epoch index (1-based for logging).

        Returns:
            Dictionary containing average 'train_loss' and 'train_acc'.
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            preds = torch.argmax(logits, dim=1)
            total_correct += int((preds == targets).sum().item())
            total_samples += batch_size

        if self.scheduler is not None:
            self.scheduler.step()

        mean_loss = total_loss / max(total_samples, 1)
        mean_acc = total_correct / max(total_samples, 1)

        return {
            "train_loss": mean_loss,
            "train_acc": mean_acc
        }

    def evaluate(
        self,
        eval_loader: DataLoader,
        return_confusion_matrix: bool = False
    ) -> Dict[str, Any]:
        """Evaluates the model on a validation or test dataset without backpropagation.

        Args:
            eval_loader: DataLoader providing validation/test batches.
            return_confusion_matrix: Whether to compute and return confusion matrix array.

        Returns:
            Dictionary containing 'loss', 'accuracy', and optionally 'confusion_matrix'.
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets in eval_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                logits = self.model(images)
                loss = self.criterion(logits, targets)

                total_loss += loss.item() * images.size(0)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

        if len(all_preds) > 0:
            cat_preds = torch.cat(all_preds, dim=0)
            cat_targets = torch.cat(all_targets, dim=0)
            total_samples = cat_targets.size(0)
            mean_loss = total_loss / max(total_samples, 1)
            acc = compute_accuracy(cat_preds, cat_targets)
        else:
            mean_loss = 0.0
            acc = 0.0
            cat_preds = torch.empty(0, dtype=torch.long)
            cat_targets = torch.empty(0, dtype=torch.long)

        results: Dict[str, Any] = {
            "loss": float(mean_loss),
            "accuracy": float(acc)
        }

        if return_confusion_matrix:
            num_classes = self.config.get("model", {}).get("num_classes", 3)
            cm = compute_confusion_matrix(cat_preds, cat_targets, num_classes=num_classes)
            results["confusion_matrix"] = cm

        return results

    def save_checkpoint(
        self,
        filepath: str,
        epoch: int,
        metrics: Dict[str, float],
        model_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Saves complete model checkpoint including optimizer states and config metadata.

        Args:
            filepath: Absolute or relative target file path for `.pth`.
            epoch: Last completed training epoch.
            metrics: Evaluation metrics dictionary at checkpoint time.
            model_config: Model initialization configuration for downstream ART loading.

        Returns:
            Path to saved `.pth` checkpoint.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "model_config": model_config if model_config is not None else self.config.get("model", {})
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint successfully saved to: {filepath}")
        return filepath

    def load_checkpoint(self, filepath: str, load_optimizer: bool = True) -> Dict[str, Any]:
        """Restores model and optionally optimizer/scheduler state from `.pth` checkpoint.

        Args:
            filepath: Path to checkpoint `.pth` file.
            load_optimizer: If True, restores optimizer and scheduler state dicts.

        Returns:
            Loaded checkpoint dictionary containing metadata.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.logger.info(f"Loaded checkpoint from {filepath} (epoch {checkpoint.get('epoch', 'unknown')})")
        return checkpoint

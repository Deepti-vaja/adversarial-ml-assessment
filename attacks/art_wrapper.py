"""
Standardized ART (Adversarial Robustness Toolbox) PyTorchClassifier wrapping utility.
Ensures uniform wrapping of PyTorch models with correct input shape, class count, and clip bounds
for white-box evasion evaluation and subsequent defense tasks.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from art.estimators.classification import PyTorchClassifier

from data.cifar10_loader import load_config
from models.fraud_cnn import build_model
from training.trainer import Trainer

def get_art_classifier(
    model: nn.Module,
    criterion: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    input_shape: Tuple[int, ...] = (3, 32, 32),
    nb_classes: int = 3,
    clip_values: Tuple[float, float] = (0.0, 1.0),
    device: Optional[torch.device] = None
) -> PyTorchClassifier:
    """Wraps a PyTorch nn.Module in an ART PyTorchClassifier.

    Args:
        model: PyTorch classification neural network.
        criterion: Loss function used during training/gradient evaluation.
        optimizer: Optional optimizer (required only if training via ART).
        input_shape: Tuple specifying input dimensions C, H, W.
        nb_classes: Number of target classes.
        clip_values: Tuple of minimum and maximum valid pixel intensities.
        device: Computation device (cuda or cpu).

    Returns:
        Configured ART PyTorchClassifier instance ready for white-box attack generation.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    device_type = "gpu" if device.type == "cuda" else "cpu"

    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=input_shape,
        nb_classes=nb_classes,
        clip_values=clip_values,
        device_type=device_type
    )

    return classifier

def load_wrapped_model(
    model_config_path: str = "./configs/baseline.yaml",
    checkpoint_path: str = "./models/checkpoints/fraud_cnn_baseline.pth",
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, PyTorchClassifier]:
    """Loads baseline FraudCNN architecture from checkpoint and wraps it in ART PyTorchClassifier.

    Args:
        model_config_path: Path to model configuration YAML.
        checkpoint_path: Path to trained PyTorch weights checkpoint (.pth).
        device: Computation device.

    Returns:
        Tuple of (raw PyTorch model, ART PyTorchClassifier).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(model_config_path)
    model = build_model(config)
    model.to(device)

    # Use Trainer to cleanly restore checkpoint weights
    dummy_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = Trainer(model=model, device=device, optimizer=dummy_optimizer, criterion=nn.CrossEntropyLoss())
    trainer.load_checkpoint(checkpoint_path, load_optimizer=False)

    model.eval()

    classifier = get_art_classifier(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        input_shape=(3, 32, 32),
        nb_classes=config.get("model", {}).get("num_classes", 3),
        clip_values=(0.0, 1.0),
        device=device
    )

    return model, classifier

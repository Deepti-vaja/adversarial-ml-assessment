"""
Metrics computation utility for evaluation and MLflow artifact reporting.
Includes accuracy calculation, confusion matrix array generation, and confusion matrix figure plotting.
"""

import os
from typing import Union, Tuple, Dict, Any, List
import numpy as np
import torch

# Try importing matplotlib and seaborn for confusion matrix plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from data.mapping import SUPERCLASS_NAMES

def compute_accuracy(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray]
) -> float:
    """Computes classification accuracy given predictions and target labels.

    Args:
        predictions: Class predictions or logits of shape (N,) or (N, C).
        targets: True class indices of shape (N,).

    Returns:
        Accuracy as a float between 0.0 and 1.0.
    """
    if isinstance(predictions, torch.Tensor):
        preds = predictions.detach().cpu().numpy()
    else:
        preds = np.asarray(predictions)

    if isinstance(targets, torch.Tensor):
        targs = targets.detach().cpu().numpy()
    else:
        targs = np.asarray(targets)

    if preds.ndim == 2:
        preds = np.argmax(preds, axis=1)

    if len(preds) == 0:
        return 0.0

    return float(np.mean(preds == targs))

def compute_confusion_matrix(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    num_classes: int = 3
) -> np.ndarray:
    """Computes confusion matrix where row i represents true class and column j represents predicted class.

    Args:
        predictions: Class predictions or logits of shape (N,) or (N, C).
        targets: True class indices of shape (N,).
        num_classes: Number of classification classes (default 3 for proxy taxonomy).

    Returns:
        Integer numpy array of shape (num_classes, num_classes).
    """
    if isinstance(predictions, torch.Tensor):
        preds = predictions.detach().cpu().numpy()
    else:
        preds = np.asarray(predictions)

    if isinstance(targets, torch.Tensor):
        targs = targets.detach().cpu().numpy()
    else:
        targs = np.asarray(targets)

    if preds.ndim == 2:
        preds = np.argmax(preds, axis=1)

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(targs, preds):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[int(t), int(p)] += 1
    return cm

def save_confusion_matrix_figure(
    cm: np.ndarray,
    output_path: str,
    class_names: List[str] = SUPERCLASS_NAMES,
    title: str = "Confusion Matrix"
) -> str:
    """Plots and saves confusion matrix as a PNG figure.

    Args:
        cm: Confusion matrix numpy array of shape (C, C).
        output_path: Destination file path for PNG image.
        class_names: List of class display names.
        title: Figure title.

    Returns:
        Path to the saved PNG image.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if not HAS_MATPLOTLIB:
        # Fallback if matplotlib is not installed: save as CSV instead
        csv_path = output_path.replace(".png", ".csv")
        np.savetxt(csv_path, cm, delimiter=",", fmt="%d", header=",".join(class_names))
        return csv_path

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=ax
    )
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    # Also save CSV alongside PNG
    csv_path = output_path.replace(".png", ".csv")
    np.savetxt(csv_path, cm, delimiter=",", fmt="%d", header=",".join(class_names))

    return output_path

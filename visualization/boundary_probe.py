"""
Linear Interpolation Boundary Probe module for Boundary Analysis (Task C1).
Measures margin and local linearity of the decision boundary by interpolating between
20 clean-adversarial pairs in pixel space, plotting softmax confidences, and identifying crossing points.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


def run_linear_interpolation_probe(
    model: nn.Module,
    x_clean: Union[torch.Tensor, np.ndarray],
    x_adv: Union[torch.Tensor, np.ndarray],
    num_steps: int = 51,
    device: Optional[torch.device] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Executes linear interpolation between pairs of clean and adversarial samples.
    Computes softmax confidence trajectories along alpha in [0, 1] clipped to [0, 1] pixel range.

    Args:
        model: Trained PyTorch classifier.
        x_clean: Clean image tensor or array of shape (N, C, H, W).
        x_adv: Adversarial image tensor or array of shape (N, C, H, W).
        num_steps: Number of interpolation steps from alpha=0 to alpha=1.
        device: PyTorch evaluation device.

    Returns:
        Tuple of (alphas, confidences, crossing_alphas, clean_preds).
        - alphas: Array of shape (num_steps,) with interpolation parameters.
        - confidences: Array of shape (N, num_steps, num_classes) with softmax confidences.
        - crossing_alphas: Array of shape (N,) indicating the alpha where class prediction flips (-1.0 if no flip).
        - clean_preds: Array of shape (N,) indicating the class predicted at alpha=0.
    """
    if device is None:
        device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")

    was_training = model.training
    model.eval()

    if isinstance(x_clean, np.ndarray):
        x_clean = torch.from_numpy(x_clean)
    if isinstance(x_adv, np.ndarray):
        x_adv = torch.from_numpy(x_adv)

    n = len(x_clean)
    alphas = np.linspace(0.0, 1.0, num_steps, dtype=np.float32)
    confidences = np.zeros((n, num_steps, getattr(model, "num_classes", 3)), dtype=np.float32)

    with torch.no_grad():
        for step_idx, alpha in enumerate(alphas):
            # Interpolation in pixel space with strict clipping to [0.0, 1.0] valid range
            x_interp = torch.clamp((1.0 - alpha) * x_clean + alpha * x_adv, 0.0, 1.0).to(device)
            logits = model(x_interp)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            confidences[:, step_idx, :] = probs

    if was_training:
        model.train()

    clean_preds = np.argmax(confidences[:, 0, :], axis=1)
    crossing_alphas = np.full(n, -1.0, dtype=np.float32)

    for idx in range(n):
        orig_cls = clean_preds[idx]
        step_preds = np.argmax(confidences[idx], axis=1)
        # Find first step where predicted label flips or confidence in orig_cls drops below 0.5
        flips = np.where((step_preds != orig_cls) | (confidences[idx, :, orig_cls] < 0.5))[0]
        if len(flips) > 0:
            crossing_alphas[idx] = alphas[flips[0]]

    return alphas, confidences, crossing_alphas, clean_preds


def plot_boundary_probe(
    alphas: np.ndarray,
    confidences: np.ndarray,
    crossing_alphas: np.ndarray,
    clean_preds: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: Optional[str] = "./reports/boundary_probe.png",
    max_plots: int = 20
) -> plt.Figure:
    """
    Renders confidence vs alpha curves for up to 20 clean-adversarial interpolation pairs.

    Args:
        alphas: Interpolation steps (num_steps,).
        confidences: Softmax trajectories (N, num_steps, num_classes).
        crossing_alphas: Identified boundary crossing alphas (N,).
        clean_preds: Initial class predictions at alpha=0 (N,).
        class_names: Superclass names.
        output_path: Path to save generated plot figure.
        max_plots: Maximum number of sample subplots to display.

    Returns:
        Matplotlib Figure object.
    """
    if class_names is None:
        class_names = ["Genuine", "Tampered", "Forged"]

    n = min(len(confidences), max_plots)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.5 * rows))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    colors = ["#2b5c8f", "#d95f02", "#7570b3"]

    for idx in range(n):
        ax = axes[idx]
        for cls_idx, cls_name in enumerate(class_names):
            ax.plot(
                alphas, confidences[idx, :, cls_idx],
                color=colors[cls_idx], label=cls_name if idx == 0 else "", linewidth=2.0
            )

        cross_a = crossing_alphas[idx]
        if cross_a >= 0.0:
            ax.axvline(x=cross_a, color="red", linestyle="--", alpha=0.8, label="Crossing" if idx == 0 else "")
            ax.scatter([cross_a], [0.5], color="red", s=30, zorder=5)

        ax.set_title(f"Pair {idx+1} (Init: {class_names[clean_preds[idx]]})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Alpha (α)", fontsize=8)
        ax.set_ylabel("Confidence", fontsize=8)
        ax.set_ylim([-0.05, 1.05])
        ax.grid(True, linestyle=":", alpha=0.5)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Linear Interpolation Boundary Probe (Softmax Confidence Trajectories)", fontsize=14, fontweight="bold")
    if n > 0:
        fig.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def evaluate_boundary_probe_pairs(
    model: nn.Module,
    x_clean: Union[torch.Tensor, np.ndarray],
    x_adv: Union[torch.Tensor, np.ndarray],
    class_names: Optional[List[str]] = None,
    output_path: Optional[str] = "./reports/boundary_probe.png",
    num_steps: int = 51
) -> Dict[str, Union[float, np.ndarray]]:
    """
    High-level orchestrator for Task C1 Linear Interpolation Probe.

    Returns:
        Dictionary containing probe metrics and summary results.
    """
    alphas, confs, cross_a, c_preds = run_linear_interpolation_probe(
        model, x_clean[:20], x_adv[:20], num_steps=num_steps
    )
    fig = plot_boundary_probe(alphas, confs, cross_a, c_preds, class_names=class_names, output_path=output_path)

    valid_crossings = cross_a[cross_a >= 0.0]
    mean_crossing = float(np.mean(valid_crossings)) if len(valid_crossings) > 0 else -1.0

    return {
        "alphas": alphas,
        "confidences": confs,
        "crossing_alphas": cross_a,
        "mean_crossing_alpha": mean_crossing,
        "crossing_rate": float(len(valid_crossings)) / float(len(cross_a)) if len(cross_a) > 0 else 0.0,
        "plot_path": output_path
    }

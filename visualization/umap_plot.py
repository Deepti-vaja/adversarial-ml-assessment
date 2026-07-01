"""
2D Feature Projection visualization module for Boundary Analysis (Task C1).
Projects penultimate layer activations of clean and adversarial test samples into 2D using UMAP (or t-SNE fallback).
Renders side-by-side comparison plots colored by true class and predicted class.
"""

import os
from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    from sklearn.manifold import TSNE


def compute_2d_projection(
    clean_features: np.ndarray,
    adv_features: np.ndarray,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Computes 2D embedding projections for clean and adversarial feature vectors.
    Note on leakage documentation: To visualize the exact feature space distortion under attack
    in a unified coordinate space, the 2D projector is fit on the combined clean and adversarial
    activation vectors.

    Args:
        clean_features: Array of shape (N, D) representing clean penultimate activations.
        adv_features: Array of shape (N, D) representing adversarial penultimate activations.
        random_state: Seed for reproducible 2D projection.

    Returns:
        Tuple of (clean_2d, adv_2d, method_name).
    """
    combined = np.vstack([clean_features, adv_features])

    if HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=random_state)
        method_name = "UMAP"
    else:
        # Fallback to t-SNE if UMAP is not installed in the environment
        reducer = TSNE(n_components=2, random_state=random_state, init="pca", learning_rate="auto")
        method_name = "t-SNE"

    embedded = reducer.fit_transform(combined)
    n = len(clean_features)
    clean_2d = embedded[:n]
    adv_2d = embedded[n:]
    return clean_2d, adv_2d, method_name


def plot_feature_projections(
    clean_2d: np.ndarray,
    adv_2d: np.ndarray,
    true_labels: np.ndarray,
    clean_preds: np.ndarray,
    adv_preds: np.ndarray,
    class_names: Optional[List[str]] = None,
    method_name: str = "t-SNE",
    output_path: Optional[str] = "./reports/umap_tsne_plot.png"
) -> plt.Figure:
    """
    Renders side-by-side 2D projection plots comparing clean vs adversarial feature distributions,
    colored by true class and predicted class.

    Args:
        clean_2d: 2D projection of clean features (N, 2).
        adv_2d: 2D projection of adversarial features (N, 2).
        true_labels: Ground truth super-class labels (N,).
        clean_preds: Model predictions on clean samples (N,).
        adv_preds: Model predictions on adversarial samples (N,).
        class_names: List of class names (default: Genuine, Tampered, Forged).
        method_name: Name of projection method used ('UMAP' or 't-SNE').
        output_path: Optional filesystem path to save the generated figure.

    Returns:
        Matplotlib Figure object.
    """
    if class_names is None:
        class_names = ["Genuine", "Tampered", "Forged"]

    colors = ["#2b5c8f", "#d95f02", "#7570b3"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot (0, 0): Clean features colored by True Class
    for cls_idx, cls_name in enumerate(class_names):
        mask = (true_labels == cls_idx)
        axes[0, 0].scatter(
            clean_2d[mask, 0], clean_2d[mask, 1],
            c=colors[cls_idx % len(colors)], label=cls_name, alpha=0.7, edgecolors="none", s=25
        )
    axes[0, 0].set_title(f"Clean Features ({method_name}) - Colored by True Class", fontsize=12, fontweight="bold")
    axes[0, 0].legend(loc="best")
    axes[0, 0].grid(True, linestyle="--", alpha=0.4)

    # Plot (0, 1): Adversarial features colored by True Class
    for cls_idx, cls_name in enumerate(class_names):
        mask = (true_labels == cls_idx)
        axes[0, 1].scatter(
            adv_2d[mask, 0], adv_2d[mask, 1],
            c=colors[cls_idx % len(colors)], label=cls_name, alpha=0.7, edgecolors="none", s=25
        )
    axes[0, 1].set_title(f"Adversarial Features ({method_name}) - Colored by True Class", fontsize=12, fontweight="bold")
    axes[0, 1].legend(loc="best")
    axes[0, 1].grid(True, linestyle="--", alpha=0.4)

    # Plot (1, 0): Clean features colored by Predicted Class
    for cls_idx, cls_name in enumerate(class_names):
        mask = (clean_preds == cls_idx)
        axes[1, 0].scatter(
            clean_2d[mask, 0], clean_2d[mask, 1],
            c=colors[cls_idx % len(colors)], label=f"Pred: {cls_name}", alpha=0.7, edgecolors="none", s=25
        )
    axes[1, 0].set_title(f"Clean Features ({method_name}) - Colored by Predicted Class", fontsize=12, fontweight="bold")
    axes[1, 0].legend(loc="best")
    axes[1, 0].grid(True, linestyle="--", alpha=0.4)

    # Plot (1, 1): Adversarial features colored by Predicted Class
    for cls_idx, cls_name in enumerate(class_names):
        mask = (adv_preds == cls_idx)
        axes[1, 1].scatter(
            adv_2d[mask, 0], adv_2d[mask, 1],
            c=colors[cls_idx % len(colors)], label=f"Pred: {cls_name}", alpha=0.7, edgecolors="none", s=25
        )
    axes[1, 1].set_title(f"Adversarial Features ({method_name}) - Colored by Predicted Class", fontsize=12, fontweight="bold")
    axes[1, 1].legend(loc="best")
    axes[1, 1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def generate_umap_or_tsne_plot(
    clean_features: np.ndarray,
    adv_features: np.ndarray,
    true_labels: np.ndarray,
    clean_preds: np.ndarray,
    adv_preds: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: Optional[str] = "./reports/umap_tsne_plot.png",
    random_state: int = 42
) -> plt.Figure:
    """
    High-level orchestrator function to compute 2D projections and generate side-by-side plots.

    Returns:
        Matplotlib Figure object.
    """
    clean_2d, adv_2d, method_name = compute_2d_projection(clean_features, adv_features, random_state=random_state)
    fig = plot_feature_projections(
        clean_2d, adv_2d, true_labels, clean_preds, adv_preds,
        class_names=class_names, method_name=method_name, output_path=output_path
    )
    return fig

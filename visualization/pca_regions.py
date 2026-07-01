"""
PCA Decision Region visualization module for Boundary Analysis (Task C1).
Reduces penultimate layer activations to 2 Principal Components and renders decision regions
with overlaid clean and adversarial points.
"""

import os
from typing import List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import torch.nn as nn


def compute_pca_2d(
    clean_features: np.ndarray,
    adv_features: np.ndarray
) -> Tuple[PCA, np.ndarray, np.ndarray]:
    """
    Fits PCA (2 components) on clean penultimate layer activations and transforms both clean and adversarial features.
    Documented Methodology: PCA is fit strictly in penultimate activation space (e.g. 128-dimensional representation),
    not pixel space.

    Args:
        clean_features: Penultimate layer features of clean samples (N, D).
        adv_features: Penultimate layer features of adversarial samples (N, D).

    Returns:
        Tuple of (fitted_pca, clean_2d, adv_2d).
    """
    pca = PCA(n_components=2, random_state=42)
    clean_2d = pca.fit_transform(clean_features)
    adv_2d = pca.transform(adv_features)
    return pca, clean_2d, adv_2d


def predict_from_pca_grid(
    grid_2d: np.ndarray,
    pca: PCA,
    model: Optional[nn.Module] = None,
    clean_2d: Optional[np.ndarray] = None,
    clean_preds: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Predicts superclass labels for 2D grid coordinates in PCA subspace.
    Documented Methodology:
    To render decision boundaries in 2D PCA space corresponding to the high-dimensional classifier,
    points in 2D PCA space are mapped back to D-dimensional activation space via inverse PCA projection
    (pca.inverse_transform), and evaluated using the model's final linear classification head.
    If model is not provided, a lightweight nearest-centroid / logistic decision rule is fit in 2D.
    """
    if model is not None:
        device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")
        reconstructed_d = pca.inverse_transform(grid_2d)
        tensor_d = torch.from_numpy(reconstructed_d.astype(np.float32)).to(device)

        # Locate final linear layer
        final_linear = None
        if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
            final_linear = model.classifier[-1]
        else:
            linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
            if linears:
                final_linear = linears[-1]

        if final_linear is not None:
            with torch.no_grad():
                logits = final_linear(tensor_d)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
            return preds

    # Fallback if model/linear head unavailable: train a lightweight logistic classifier in 2D
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=42, max_iter=200)
    clf.fit(clean_2d, clean_preds)
    return clf.predict(grid_2d)


def plot_pca_decision_regions(
    pca: PCA,
    clean_2d: np.ndarray,
    adv_2d: np.ndarray,
    clean_preds: np.ndarray,
    adv_preds: np.ndarray,
    model: Optional[nn.Module] = None,
    class_names: Optional[List[str]] = None,
    output_path: Optional[str] = "./reports/pca_decision_regions.png",
    grid_resolution: int = 200
) -> plt.Figure:
    """
    Renders 2D PCA decision regions with clean and adversarial point overlays.

    Args:
        pca: Fitted PCA object.
        clean_2d: Projected clean features (N, 2).
        adv_2d: Projected adversarial features (N, 2).
        clean_preds: Predicted labels for clean samples (N,).
        adv_preds: Predicted labels for adversarial samples (N,).
        model: Optional PyTorch model to evaluate reconstructed grid points.
        class_names: Names of superclasses.
        output_path: Optional path to save output plot.
        grid_resolution: Meshgrid points along each axis.

    Returns:
        Matplotlib Figure object.
    """
    if class_names is None:
        class_names = ["Genuine", "Tampered", "Forged"]

    x_min = min(clean_2d[:, 0].min(), adv_2d[:, 0].min()) - 1.0
    x_max = max(clean_2d[:, 0].max(), adv_2d[:, 0].max()) + 1.0
    y_min = min(clean_2d[:, 1].min(), adv_2d[:, 1].min()) - 1.0
    y_max = max(clean_2d[:, 1].max(), adv_2d[:, 1].max()) + 1.0

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_from_pca_grid(grid_points, pca, model=model, clean_2d=clean_2d, clean_preds=clean_preds)
    Z = Z.reshape(xx.shape)

    colors = ["#2b5c8f", "#d95f02", "#7570b3"]
    cmap_bg = plt.matplotlib.colors.ListedColormap(["#cce5ff", "#ffe5cc", "#e5e0ff"])

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_bg)
    ax.contour(xx, yy, Z, colors="k", linewidths=0.5, alpha=0.5)

    for cls_idx, cls_name in enumerate(class_names):
        mask_clean = (clean_preds == cls_idx)
        ax.scatter(
            clean_2d[mask_clean, 0], clean_2d[mask_clean, 1],
            c=colors[cls_idx], label=f"Clean ({cls_name})", marker="o", edgecolors="k", s=40, alpha=0.8
        )
        mask_adv = (adv_preds == cls_idx)
        ax.scatter(
            adv_2d[mask_adv, 0], adv_2d[mask_adv, 1],
            c=colors[cls_idx], label=f"Adv ({cls_name})", marker="^", edgecolors="red", s=50, alpha=0.9
        )

    ax.set_title("PCA Decision Regions (Penultimate Activation Subspace)", fontsize=13, fontweight="bold")
    ax.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def generate_pca_decision_plot(
    clean_features: np.ndarray,
    adv_features: np.ndarray,
    clean_preds: np.ndarray,
    adv_preds: np.ndarray,
    model: Optional[nn.Module] = None,
    class_names: Optional[List[str]] = None,
    output_path: Optional[str] = "./reports/pca_decision_regions.png"
) -> plt.Figure:
    """
    Orchestrates PCA projection and renders decision region plot.
    """
    pca, clean_2d, adv_2d = compute_pca_2d(clean_features, adv_features)
    fig = plot_pca_decision_regions(
        pca, clean_2d, adv_2d, clean_preds, adv_preds,
        model=model, class_names=class_names, output_path=output_path
    )
    return fig

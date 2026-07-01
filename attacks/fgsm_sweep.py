"""
Fast Gradient Sign Method (FGSM) epsilon sweep evaluation.
Evaluates model robustness across perturbation bounds eps in [0.01, 0.05, 0.1, 0.2].
Generates accuracy vs. epsilon curves and reports perturbation metrics.
"""

import os
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

from utils.metrics import compute_accuracy

# Try importing matplotlib for curve plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def evaluate_fgsm_sweep(
    classifier: PyTorchClassifier,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epsilons: List[float] = [0.01, 0.05, 0.1, 0.2],
    batch_size: int = 64
) -> Dict[str, Any]:
    """Runs FGSM epsilon sweep on provided test data.

    Args:
        classifier: Wrapped ART PyTorchClassifier.
        x_test: Input test images array of shape (N, 3, 32, 32).
        y_test: True class labels array of shape (N,).
        epsilons: List of epsilon bounds to evaluate.
        batch_size: Batch size for attack generation and prediction.

    Returns:
        Dictionary containing sweep metrics, clean accuracy, per-epsilon accuracy, ASR, and perturbation norms.
    """
    # First compute clean predictions
    clean_preds = classifier.predict(x_test, batch_size=batch_size)
    clean_pred_labels = np.argmax(clean_preds, axis=1)
    clean_acc = float(np.mean(clean_pred_labels == y_test))

    results = {
        "clean_accuracy": clean_acc,
        "epsilons": epsilons,
        "adversarial_accuracy": {},
        "attack_success_rate": {},
        "mean_linf_norm": {},
        "mean_l2_norm": {}
    }

    # Sweep over epsilon values
    for eps in epsilons:
        attack = FastGradientMethod(
            estimator=classifier,
            eps=float(eps),
            eps_step=float(eps),
            targeted=False,
            norm=np.inf,
            batch_size=batch_size
        )

        x_adv = attack.generate(x=x_test)
        adv_preds = classifier.predict(x_adv, batch_size=batch_size)
        adv_pred_labels = np.argmax(adv_preds, axis=1)

        adv_acc = float(np.mean(adv_pred_labels == y_test))

        # Attack success rate (ASR) on initially correctly classified samples
        correct_mask = (clean_pred_labels == y_test)
        if np.any(correct_mask):
            asr = float(np.mean(adv_pred_labels[correct_mask] != clean_pred_labels[correct_mask]))
        else:
            asr = 0.0

        # Perturbation norms
        diff = (x_adv - x_test).reshape(len(x_test), -1)
        linf_norm = float(np.mean(np.max(np.abs(diff), axis=1)))
        l2_norm = float(np.mean(np.linalg.norm(diff, axis=1)))

        results["adversarial_accuracy"][eps] = adv_acc
        results["attack_success_rate"][eps] = asr
        results["mean_linf_norm"][eps] = linf_norm
        results["mean_l2_norm"][eps] = l2_norm

    return results

def plot_fgsm_sweep(
    results: Dict[str, Any],
    output_path: str = "./reports/fgsm_epsilon_sweep.png"
) -> Tuple[str, str]:
    """Plots and saves FGSM accuracy vs. epsilon curve as PNG and CSV.

    Args:
        results: Results dictionary returned by evaluate_fgsm_sweep.
        output_path: Destination PNG figure path.

    Returns:
        Tuple of (png_path, csv_path).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    csv_path = output_path.replace(".png", ".csv")

    eps_list = [0.0] + list(results["epsilons"])
    acc_list = [results["clean_accuracy"]] + [results["adversarial_accuracy"][e] for e in results["epsilons"]]
    asr_list = [0.0] + [results["attack_success_rate"][e] for e in results["epsilons"]]

    # Save CSV
    data = np.column_stack((eps_list, acc_list, asr_list))
    np.savetxt(csv_path, data, delimiter=",", header="epsilon,accuracy,attack_success_rate", comments="", fmt="%.6f")

    if HAS_MATPLOTLIB:
        fig, ax1 = plt.subplots(figsize=(7, 5))

        color = "tab:blue"
        ax1.set_xlabel("Epsilon (L-inf Perturbation Bound)")
        ax1.set_ylabel("Classification Accuracy", color=color)
        ax1.plot(eps_list, [a * 100 for a in acc_list], marker="o", color=color, linewidth=2, label="Accuracy (%)")
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.set_ylim([0, 105])
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2 = ax1.twinx()
        color = "tab:red"
        ax2.set_ylabel("Attack Success Rate (ASR %)", color=color)
        ax2.plot(eps_list, [a * 100 for a in asr_list], marker="s", color=color, linewidth=2, linestyle="--", label="ASR (%)")
        ax2.tick_params(axis="y", labelcolor=color)
        ax2.set_ylim([0, 105])

        plt.title("FGSM Epsilon Sweep: Accuracy vs. Perturbation Bound")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    return output_path, csv_path

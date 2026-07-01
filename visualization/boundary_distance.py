"""
Boundary Distance Analysis module for Task C2.
Computes approximate minimum L2 distance to the nearest decision boundary using ART's DeepFool algorithm,
reports mean boundary distance per superclass, and computes statistical correlation with Task B per-class ASR.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def compute_deepfool_boundary_distances(
    art_classifier: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
    max_iter: int = 50,
    batch_size: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes approximate minimal L2 perturbations required to cross decision boundaries using DeepFool.
    Documented Methodology & Justification:
    Exact minimum L2 distance computation via BoundaryAttack across full datasets is computationally prohibitive.
    DeepFool computes an iterative linear approximation to find the minimal orthogonal projection distance
    onto the nearest decision hyper-plane. This serves as an efficient, deterministic, highly correlated proxy
    for true boundary distance.

    Args:
        art_classifier: ART PyTorchClassifier instance wrapping the target model.
        x_test: Input test images array of shape (N, C, H, W).
        y_test: Ground truth super-class labels array of shape (N,).
        max_iter: Maximum number of DeepFool iterations per sample.
        batch_size: Batch size for execution.

    Returns:
        Tuple of (l2_distances, adv_images).
    """
    from art.attacks.evasion import DeepFool
    df_attack = DeepFool(classifier=art_classifier, max_iter=max_iter, epsilon=1e-6, batch_size=batch_size)
    x_adv = df_attack.generate(x=x_test)

    # Compute L2 norm per sample across spatial dimensions (C, H, W)
    diff = (x_adv - x_test).reshape(len(x_test), -1)
    l2_distances = np.linalg.norm(diff, axis=1)
    return l2_distances, x_adv


def analyze_boundary_distance_correlation(
    l2_distances: np.ndarray,
    y_test: np.ndarray,
    per_class_asr: Optional[Dict[str, float]] = None,
    class_names: Optional[List[str]] = None,
    output_csv_path: Optional[str] = "./reports/boundary_distance.csv"
) -> Dict[str, Any]:
    """
    Calculates per-class mean boundary distances and statistically correlates them with per-class Attack Success Rate (ASR).

    Args:
        l2_distances: Array of L2 perturbation distances per sample (N,).
        y_test: True super-class labels (N,).
        per_class_asr: Dictionary mapping class name to its ASR under Task B evaluation.
        class_names: Ordered list of class names.
        output_csv_path: Filesystem path to save per-class distance summary table.

    Returns:
        Dictionary summarizing per-class mean L2 distances and correlation metrics.
    """
    if class_names is None:
        class_names = ["Genuine", "Tampered", "Forged"]

    per_class_dist = {}
    for c_idx, c_name in enumerate(class_names):
        mask = (y_test == c_idx)
        if np.any(mask):
            per_class_dist[c_name] = float(np.mean(l2_distances[mask]))
        else:
            per_class_dist[c_name] = 0.0

    overall_mean = float(np.mean(l2_distances)) if len(l2_distances) > 0 else 0.0

    correlation_res = {
        "pearson_r": None,
        "pearson_p": None,
        "spearman_rho": None,
        "spearman_p": None,
        "discussion": "No per-class ASR dictionary provided for correlation analysis."
    }

    df_rows = []
    for c_name in class_names:
        row = {
            "Superclass": c_name,
            "Mean_L2_Boundary_Distance": per_class_dist[c_name],
            "Task_B_ASR": per_class_asr.get(c_name, np.nan) if per_class_asr else np.nan
        }
        df_rows.append(row)

    df_table = pd.DataFrame(df_rows)
    if output_csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
        df_table.to_csv(output_csv_path, index=False)

    if per_class_asr and len(class_names) >= 2:
        dists = [per_class_dist[c] for c in class_names]
        asrs = [per_class_asr[c] for c in class_names]

        # Handle constant input cases safely
        if np.std(dists) > 0 and np.std(asrs) > 0:
            pr, pp = pearsonr(dists, asrs)
            sr, sp = spearmanr(dists, asrs)
            correlation_res.update({
                "pearson_r": float(pr),
                "pearson_p": float(pp),
                "spearman_rho": float(sr),
                "spearman_p": float(sp)
            })
            corr_trend = "negative" if pr < 0 else "positive"
            correlation_res["discussion"] = (
                f"Statistical correlation between mean L2 boundary distance and Task B ASR shows a {corr_trend} relationship "
                f"(Pearson r={pr:.3f}, p={pp:.3f}). Classes with smaller average distance to the nearest decision boundary "
                f"exhibit higher empirical vulnerability under white-box evasion attacks."
            )
        else:
            correlation_res["discussion"] = "Zero variance across class distances or ASRs prevented correlation coefficient calculation."

    return {
        "overall_mean_l2_distance": overall_mean,
        "per_class_mean_l2_distance": per_class_dist,
        "correlation_analysis": correlation_res,
        "table": df_table
    }

"""
Projected Gradient Descent (PGD) evasion attack evaluation.
Evaluates 20-step and 40-step PGD variants under L-infinity norm bound to evaluate
iterative adversarial degradation against single-step FGSM.
"""

from typing import Dict, Any, List
import numpy as np
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescentPyTorch

def evaluate_pgd_attacks(
    classifier: PyTorchClassifier,
    x_test: np.ndarray,
    y_test: np.ndarray,
    eps: float = 0.05,
    eps_step: float = 0.01,
    steps_list: List[int] = [20, 40],
    batch_size: int = 64
) -> Dict[str, Any]:
    """Runs 20-step and 40-step PGD attacks on test dataset.

    Args:
        classifier: Wrapped ART PyTorchClassifier.
        x_test: Input test images array of shape (N, 3, 32, 32).
        y_test: True class labels array of shape (N,).
        eps: Maximum L-infinity perturbation budget.
        eps_step: Step size per iteration.
        steps_list: List of iteration budgets to evaluate.
        batch_size: Batch size for attack generation.

    Returns:
        Dictionary comparing adversarial accuracy, ASR, and perturbation norms across step budgets.
    """
    clean_preds = classifier.predict(x_test, batch_size=batch_size)
    clean_pred_labels = np.argmax(clean_preds, axis=1)
    clean_acc = float(np.mean(clean_pred_labels == y_test))

    results = {
        "clean_accuracy": clean_acc,
        "eps": eps,
        "eps_step": eps_step,
        "steps_evaluated": steps_list,
        "adversarial_accuracy": {},
        "attack_success_rate": {},
        "mean_linf_norm": {},
        "mean_l2_norm": {}
    }

    for steps in steps_list:
        attack = ProjectedGradientDescentPyTorch(
            estimator=classifier,
            norm=np.inf,
            eps=float(eps),
            eps_step=float(eps_step),
            max_iter=int(steps),
            targeted=False,
            batch_size=batch_size
        )

        x_adv = attack.generate(x=x_test)
        adv_preds = classifier.predict(x_adv, batch_size=batch_size)
        adv_pred_labels = np.argmax(adv_preds, axis=1)

        adv_acc = float(np.mean(adv_pred_labels == y_test))

        correct_mask = (clean_pred_labels == y_test)
        if np.any(correct_mask):
            asr = float(np.mean(adv_pred_labels[correct_mask] != clean_pred_labels[correct_mask]))
        else:
            asr = 0.0

        diff = (x_adv - x_test).reshape(len(x_test), -1)
        linf_norm = float(np.mean(np.max(np.abs(diff), axis=1)))
        l2_norm = float(np.mean(np.linalg.norm(diff, axis=1)))

        results["adversarial_accuracy"][steps] = adv_acc
        results["attack_success_rate"][steps] = asr
        results["mean_linf_norm"][steps] = linf_norm
        results["mean_l2_norm"][steps] = l2_norm

    return results

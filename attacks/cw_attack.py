"""
Carlini & Wagner (C&W) L2 evasion attack evaluation.
Evaluates targeted/untargeted L2 norm minimization attack on a stratified subset
of 200 samples per class (total 600 samples) to measure minimum distortion needed for misclassification.
"""

from typing import Dict, Any, Tuple
import numpy as np
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import CarliniL2Method

def sample_stratified_subset(
    x_test: np.ndarray,
    y_test: np.ndarray,
    samples_per_class: int = 200,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Selects a stratified subset of images with equal representation per class.

    Args:
        x_test: Input images array.
        y_test: Target labels array.
        samples_per_class: Desired number of samples per class.
        seed: Random seed for reproducible sampling.

    Returns:
        Tuple of (stratified_x, stratified_y).
    """
    rng = np.random.RandomState(seed)
    unique_classes = np.unique(y_test)

    selected_indices = []
    for c in unique_classes:
        class_indices = np.where(y_test == c)[0]
        n_take = min(samples_per_class, len(class_indices))
        if n_take > 0:
            chosen = rng.choice(class_indices, size=n_take, replace=False)
            selected_indices.extend(chosen)

    selected_indices = np.array(selected_indices)
    rng.shuffle(selected_indices)

    return x_test[selected_indices], y_test[selected_indices]

def evaluate_cw_attack(
    classifier: PyTorchClassifier,
    x_test: np.ndarray,
    y_test: np.ndarray,
    confidence: float = 0.0,
    learning_rate: float = 0.01,
    max_iter: int = 50,
    binary_search_steps: int = 3,
    batch_size: int = 32
) -> Dict[str, Any]:
    """Runs Carlini & Wagner L2 attack on provided dataset.

    Args:
        classifier: Wrapped ART PyTorchClassifier.
        x_test: Input test images array of shape (N, 3, 32, 32).
        y_test: True class labels array of shape (N,).
        confidence: Confidence parameter kappa.
        learning_rate: Optimizer step size for L2 minimization.
        max_iter: Maximum optimization iterations per search step.
        binary_search_steps: Number of binary search steps for constant c.
        batch_size: Batch size for attack optimization.

    Returns:
        Dictionary reporting adversarial accuracy, ASR, and mean L2 distortion.
    """
    clean_preds = classifier.predict(x_test, batch_size=batch_size)
    clean_pred_labels = np.argmax(clean_preds, axis=1)
    clean_acc = float(np.mean(clean_pred_labels == y_test))

    attack = CarliniL2Method(
        classifier=classifier,
        confidence=float(confidence),
        learning_rate=float(learning_rate),
        max_iter=int(max_iter),
        binary_search_steps=int(binary_search_steps),
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

    results = {
        "clean_accuracy": clean_acc,
        "adversarial_accuracy": adv_acc,
        "attack_success_rate": asr,
        "mean_linf_norm": linf_norm,
        "mean_l2_norm": l2_norm,
        "samples_evaluated": len(x_test)
    }

    return results

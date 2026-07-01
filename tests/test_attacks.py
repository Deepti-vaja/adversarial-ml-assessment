"""
Automated unit verification suite for Task B Adversarial Evaluation subsystem.
Tests ART PyTorchClassifier wrapping, FGSM sweep execution, PGD multi-step evaluation,
and C&W L2 attack optimization.
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
from art.estimators.classification import PyTorchClassifier

from models.fraud_cnn import FraudCNN
from attacks.art_wrapper import get_art_classifier
from attacks.fgsm_sweep import evaluate_fgsm_sweep
from attacks.pgd_attack import evaluate_pgd_attacks
from attacks.cw_attack import sample_stratified_subset, evaluate_cw_attack

class TestAdversarialEvaluation(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = FraudCNN(in_channels=3, num_classes=3)
        self.classifier = get_art_classifier(
            model=self.model,
            criterion=nn.CrossEntropyLoss(),
            input_shape=(3, 32, 32),
            nb_classes=3,
            clip_values=(0.0, 1.0),
            device=self.device
        )
        # Dummy test data (18 images, balanced across 3 classes)
        self.x_test = np.random.rand(18, 3, 32, 32).astype(np.float32)
        self.y_test = np.array([0]*6 + [1]*6 + [2]*6, dtype=np.int64)

    def test_art_wrapper_creation(self):
        self.assertIsInstance(self.classifier, PyTorchClassifier)
        self.assertFalse(self.model.training)
        self.assertEqual(float(self.classifier.clip_values[0]), 0.0)
        self.assertEqual(float(self.classifier.clip_values[1]), 1.0)
        self.assertEqual(self.classifier.nb_classes, 3)

    def test_fgsm_attack_sweep(self):
        res = evaluate_fgsm_sweep(self.classifier, self.x_test, self.y_test, epsilons=[0.05], batch_size=8)
        self.assertIn("clean_accuracy", res)
        self.assertIn(0.05, res["adversarial_accuracy"])
        self.assertIn(0.05, res["mean_linf_norm"])
        # Verify L-infinity perturbation norm bounded by eps + tolerance
        self.assertLessEqual(res["mean_linf_norm"][0.05], 0.05 + 1e-4)

    def test_pgd_attack_evaluation(self):
        res = evaluate_pgd_attacks(self.classifier, self.x_test, self.y_test, eps=0.05, eps_step=0.02, steps_list=[2], batch_size=8)
        self.assertIn(2, res["adversarial_accuracy"])
        self.assertIn(2, res["attack_success_rate"])
        self.assertLessEqual(res["mean_linf_norm"][2], 0.05 + 1e-4)

    def test_cw_stratified_sampling_and_attack(self):
        x_sub, y_sub = sample_stratified_subset(self.x_test, self.y_test, samples_per_class=2, seed=42)
        self.assertEqual(len(x_sub), 6)
        unique, counts = np.unique(y_sub, return_counts=True)
        self.assertTrue(np.all(counts == 2))

        res = evaluate_cw_attack(self.classifier, x_sub, y_sub, max_iter=2, binary_search_steps=1, batch_size=6)
        self.assertIn("mean_l2_norm", res)
        self.assertIn("attack_success_rate", res)
        self.assertGreaterEqual(res["mean_l2_norm"], 0.0)

if __name__ == "__main__":
    unittest.main()

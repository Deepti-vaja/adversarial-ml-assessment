"""
Automated unit and integration verification suite for the Task D Defense subsystem.
Verifies Feature Squeezing preprocessing wrapper, Adversarial Training execution and checkpointing,
and unified defense evaluation table generation adhering to exact blueprint schema.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from art.estimators.classification import PyTorchClassifier

from models.fraud_cnn import FraudCNN
from defenses.feature_squeezing import get_feature_squeezed_classifier
from defenses.adversarial_training import run_adversarial_training
from defenses.evaluate_defenses import run_defense_evaluation


class TestDefensesSubsystem(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.test_dir = tempfile.mkdtemp()
        self.model = FraudCNN(in_channels=3, num_classes=3)
        self.x_test = np.random.rand(12, 3, 32, 32).astype(np.float32)
        self.y_test = np.array([0]*4 + [1]*4 + [2]*4, dtype=np.int64)

        # Create temporary baseline config and checkpoint for fast testing
        self.config_path = os.path.join(self.test_dir, "test_defenses.yaml").replace("\\", "/")
        self.baseline_config_path = os.path.join(self.test_dir, "test_baseline.yaml").replace("\\", "/")
        self.baseline_ckpt_path = os.path.join(self.test_dir, "test_baseline.pth").replace("\\", "/")
        self.adv_ckpt_path = os.path.join(self.test_dir, "test_adv_trained.pth").replace("\\", "/")
        self.output_csv_path = os.path.join(self.test_dir, "test_defense_comparison.csv").replace("\\", "/")

        with open(self.baseline_config_path, "w") as f:
            f.write(f"""
seed: 42
model:
  in_channels: 3
  num_classes: 3
data:
  dataset_dir: "./data/cifar10"
  val_ratio: 0.1
  batch_size: 4
  num_workers: 0
""")

        with open(self.config_path, "w") as f:
            f.write(f"""
seed: 42
model:
  config_path: "{self.baseline_config_path}"
  baseline_checkpoint_path: "{self.baseline_ckpt_path}"
  adv_trained_checkpoint_path: "{self.adv_ckpt_path}"
adversarial_training:
  epochs: 1
  batch_size: 4
  learning_rate: 0.001
  attack:
    eps: 0.05
    eps_step: 0.02
    max_iter: 2
feature_squeezing:
  bit_depth: 5
  clip_values: [0.0, 1.0]
evaluation:
  batch_size: 4
  output_csv_path: "{self.output_csv_path}"
""")

        # Save dummy baseline checkpoint
        torch.save({
            "epoch": 1,
            "state_dict": self.model.state_dict(),
            "metrics": {"clean_val_acc": 0.8},
            "model_config": {"in_channels": 3, "num_classes": 3}
        }, self.baseline_ckpt_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_feature_squeezing_classifier(self):
        classifier = get_feature_squeezed_classifier(
            self.model,
            bit_depth=5,
            window_size=3,
            clip_values=(0.0, 1.0),
            device=self.device
        )
        self.assertIsInstance(classifier, PyTorchClassifier)
        self.assertEqual(len(classifier.preprocessing_defences), 2)

        preds = classifier.predict(self.x_test, batch_size=4)
        self.assertEqual(preds.shape, (12, 3))
        # Ensure outputs are valid probabilities / logits
        self.assertFalse(np.isnan(preds).any())

    def test_adversarial_training_execution(self):
        model, classifier, summary = run_adversarial_training(
            defense_config_path=self.config_path,
            max_samples=8,
            device=self.device
        )
        self.assertIsInstance(model, nn.Module)
        self.assertIsInstance(classifier, PyTorchClassifier)
        self.assertIn("robust_val_acc", summary)
        self.assertIn("training_hours", summary)
        self.assertTrue(os.path.exists(self.adv_ckpt_path))

    def test_evaluate_defenses_table_generation(self):
        df = run_defense_evaluation(
            config_path=self.config_path,
            max_samples=12,
            device=self.device
        )
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(os.path.exists(self.output_csv_path))

        expected_cols = [
            "Model Variant",
            "Clean Accuracy",
            "FGSM Accuracy (eps=0.05)",
            "PGD-40 Accuracy",
            "C&W Success Rate",
            "Training Overhead (hours)"
        ]
        self.assertListEqual(list(df.columns), expected_cols)
        self.assertEqual(len(df), 3)
        variants = df["Model Variant"].tolist()
        self.assertIn("Baseline (Clean Training)", variants)
        self.assertIn("Adversarial Training (PGD)", variants)
        self.assertIn("Feature Squeezing (Median + Bit-Depth)", variants)


if __name__ == "__main__":
    unittest.main()

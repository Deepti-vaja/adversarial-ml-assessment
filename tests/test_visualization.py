"""
Automated unit verification suite for Milestone 4 (Task C) Boundary Analysis subsystem.
Tests penultimate feature extraction, UMAP/t-SNE projection plotting, PCA decision region rendering,
linear interpolation boundary probing, and DeepFool boundary distance correlation analysis.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from art.estimators.classification import PyTorchClassifier

from attacks.art_wrapper import get_art_classifier
from models.fraud_cnn import FraudCNN
from visualization.feature_extractor import FeatureExtractor, extract_features_from_tensor, extract_features_from_loader
from visualization.umap_plot import generate_umap_or_tsne_plot
from visualization.pca_regions import generate_pca_decision_plot
from visualization.boundary_probe import evaluate_boundary_probe_pairs
from visualization.boundary_distance import compute_deepfool_boundary_distances, analyze_boundary_distance_correlation


class TestBoundaryAnalysis(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.device = torch.device("cpu")
        self.model = FraudCNN(in_channels=3, num_classes=3)
        self.model.eval()

        # Dummy dataset (18 images, balanced across 3 classes)
        self.x_test = np.random.rand(18, 3, 32, 32).astype(np.float32)
        self.y_test = np.array([0]*6 + [1]*6 + [2]*6, dtype=np.int64)

        dataset = TensorDataset(torch.from_numpy(self.x_test), torch.from_numpy(self.y_test))
        self.loader = DataLoader(dataset, batch_size=6, shuffle=False)

        self.art_classifier = get_art_classifier(
            model=self.model,
            criterion=nn.CrossEntropyLoss(),
            input_shape=(3, 32, 32),
            nb_classes=3,
            clip_values=(0.0, 1.0),
            device=self.device
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_feature_extractor_tensor(self):
        feats = extract_features_from_tensor(self.model, self.x_test, batch_size=8, device=self.device)
        self.assertIsInstance(feats, np.ndarray)
        self.assertEqual(feats.shape, (18, 128))

    def test_feature_extractor_loader(self):
        feats, labels, preds = extract_features_from_loader(self.model, self.loader, device=self.device)
        self.assertEqual(feats.shape, (18, 128))
        self.assertEqual(labels.shape, (18,))
        self.assertEqual(preds.shape, (18,))
        np.testing.assert_array_equal(labels, self.y_test)

    def test_umap_or_tsne_plot_generation(self):
        clean_feats = np.random.randn(18, 128).astype(np.float32)
        adv_feats = clean_feats + 0.1 * np.random.randn(18, 128).astype(np.float32)
        clean_preds = self.y_test
        adv_preds = (self.y_test + 1) % 3

        plot_path = os.path.join(self.test_dir, "umap_test.png")
        fig = generate_umap_or_tsne_plot(
            clean_feats, adv_feats, self.y_test, clean_preds, adv_preds,
            output_path=plot_path, random_state=42
        )
        self.assertIsNotNone(fig)
        self.assertTrue(os.path.exists(plot_path))
        self.assertGreater(os.path.getsize(plot_path), 0)

    def test_pca_decision_plot_generation(self):
        clean_feats = np.random.randn(18, 128).astype(np.float32)
        adv_feats = clean_feats + 0.2 * np.random.randn(18, 128).astype(np.float32)
        clean_preds = self.y_test
        adv_preds = (self.y_test + 1) % 3

        plot_path = os.path.join(self.test_dir, "pca_test.png")
        fig = generate_pca_decision_plot(
            clean_feats, adv_feats, clean_preds, adv_preds,
            model=self.model, output_path=plot_path
        )
        self.assertIsNotNone(fig)
        self.assertTrue(os.path.exists(plot_path))
        self.assertGreater(os.path.getsize(plot_path), 0)

    def test_boundary_probe_evaluation(self):
        x_adv = np.clip(self.x_test + 0.1 * np.random.randn(*self.x_test.shape).astype(np.float32), 0.0, 1.0)
        plot_path = os.path.join(self.test_dir, "probe_test.png")
        res = evaluate_boundary_probe_pairs(
            self.model, self.x_test[:10], x_adv[:10],
            output_path=plot_path, num_steps=11
        )
        self.assertEqual(res["confidences"].shape, (10, 11, 3))
        self.assertEqual(res["crossing_alphas"].shape, (10,))
        self.assertIn("crossing_rate", res)
        self.assertTrue(os.path.exists(plot_path))

    def test_boundary_distance_and_correlation(self):
        l2_dists, x_adv = compute_deepfool_boundary_distances(
            self.art_classifier, self.x_test[:12], self.y_test[:12], max_iter=5, batch_size=6
        )
        self.assertEqual(l2_dists.shape, (12,))
        self.assertTrue(np.all(l2_dists >= 0.0))

        csv_path = os.path.join(self.test_dir, "dist_test.csv")
        dummy_asr = {"Genuine": 0.3, "Tampered": 0.8, "Forged": 0.5}
        res = analyze_boundary_distance_correlation(
            l2_dists, self.y_test[:12], per_class_asr=dummy_asr, output_csv_path=csv_path
        )
        self.assertIn("overall_mean_l2_distance", res)
        self.assertIn("correlation_analysis", res)
        self.assertTrue(os.path.exists(csv_path))


if __name__ == "__main__":
    unittest.main()

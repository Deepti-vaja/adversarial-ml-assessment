"""
Unit tests for FraudCNN model architecture and Trainer engine.
"""

import os
import shutil
import tempfile
import unittest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from models.fraud_cnn import FraudCNN, build_model
from training.trainer import Trainer

class TestFraudCNNAndTrainer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.device = torch.device("cpu")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_fraud_cnn_initialization_and_forward(self):
        model = FraudCNN(in_channels=3, num_classes=3)
        self.assertIsInstance(model, nn.Module)

        # Check block output channels
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (2, 3))

    def test_build_model_factory(self):
        config = {
            "model": {
                "in_channels": 3,
                "num_classes": 3,
                "dropout_conv": 0.2,
                "dropout_fc": 0.4
            }
        }
        model = build_model(config)
        self.assertIsInstance(model, FraudCNN)

    def test_trainer_train_epoch_and_evaluate(self):
        model = FraudCNN(in_channels=3, num_classes=3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(model=model, device=self.device, optimizer=optimizer, criterion=criterion)

        # Create dummy DataLoader
        images = torch.randn(8, 3, 32, 32)
        targets = torch.randint(0, 3, (8,))
        dataset = TensorDataset(images, targets)
        loader = DataLoader(dataset, batch_size=4)

        train_metrics = trainer.train_epoch(loader, epoch=1)
        self.assertIn("train_loss", train_metrics)
        self.assertIn("train_acc", train_metrics)

        eval_metrics = trainer.evaluate(loader, return_confusion_matrix=True)
        self.assertIn("loss", eval_metrics)
        self.assertIn("accuracy", eval_metrics)
        self.assertIn("confusion_matrix", eval_metrics)
        self.assertEqual(eval_metrics["confusion_matrix"].shape, (3, 3))

    def test_checkpoint_save_and_load(self):
        model = FraudCNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        trainer = Trainer(model=model, device=self.device, optimizer=optimizer, criterion=nn.CrossEntropyLoss())

        ckpt_path = os.path.join(self.temp_dir, "test_ckpt.pth")
        trainer.save_checkpoint(ckpt_path, epoch=1, metrics={"val_acc": 0.85})
        self.assertTrue(os.path.exists(ckpt_path))

        new_model = FraudCNN()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        new_trainer = Trainer(model=new_model, device=self.device, optimizer=new_optimizer, criterion=nn.CrossEntropyLoss())
        loaded = new_trainer.load_checkpoint(ckpt_path)
        self.assertEqual(loaded["epoch"], 1)
        self.assertEqual(loaded["metrics"]["val_acc"], 0.85)

if __name__ == "__main__":
    unittest.main()

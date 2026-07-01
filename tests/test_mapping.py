"""
Unit tests for the CIFAR-10 mapping, split generation, custom dataset,
and dataloader factory.
"""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
import torchvision

from data.mapping import (
    CIFAR10_CLASSES,
    CLASS_MAPPING,
    SUPERCLASS_NAMES,
    INDEX_MAPPING,
    map_index_to_superclass,
    get_superclass_name
)
from data.splits import get_train_val_indices
from data.cifar10_loader import CIFAR10SuperClassDataset, get_dataloaders
from utils.seed import set_seed

class TestMappingAndLoader(unittest.TestCase):
    """Test suite verifying mapping correctness, splits reproducibility, and loader logic."""

    def test_class_lists_and_bounds(self) -> None:
        """Verify the predefined constants cover all items and matches specifications."""
        self.assertEqual(len(CIFAR10_CLASSES), 10)
        self.assertEqual(len(SUPERCLASS_NAMES), 3)
        self.assertEqual(len(CLASS_MAPPING), 10)
        self.assertEqual(len(INDEX_MAPPING), 10)

    def test_superclass_mapping_accuracy(self) -> None:
        """Ensure specific classes map to their designated super-classes exactly as required."""
        # Genuine: airplane (0), automobile (1), ship (8), truck (9)
        self.assertEqual(CLASS_MAPPING["airplane"], 0)
        self.assertEqual(CLASS_MAPPING["automobile"], 0)
        self.assertEqual(CLASS_MAPPING["ship"], 0)
        self.assertEqual(CLASS_MAPPING["truck"], 0)

        # Tampered: bird (2), cat (3), deer (4), dog (5)
        self.assertEqual(CLASS_MAPPING["bird"], 1)
        self.assertEqual(CLASS_MAPPING["cat"], 1)
        self.assertEqual(CLASS_MAPPING["deer"], 1)
        self.assertEqual(CLASS_MAPPING["dog"], 1)

        # Forged: frog (6), horse (7)
        self.assertEqual(CLASS_MAPPING["frog"], 2)
        self.assertEqual(CLASS_MAPPING["horse"], 2)

    def test_mapping_functions(self) -> None:
        """Verify label mapping helper functions behavior and exception raising."""
        # Correct index maps
        self.assertEqual(map_index_to_superclass(0), 0)  # airplane -> Genuine
        self.assertEqual(map_index_to_superclass(3), 1)  # cat -> Tampered
        self.assertEqual(map_index_to_superclass(7), 2)  # horse -> Forged

        # Invalid index error raising
        with self.assertRaises(ValueError):
            map_index_to_superclass(-1)
        with self.assertRaises(ValueError):
            map_index_to_superclass(10)

        # Name retrieval
        self.assertEqual(get_superclass_name(0), "Genuine")
        self.assertEqual(get_superclass_name(1), "Tampered")
        self.assertEqual(get_superclass_name(2), "Forged")

        with self.assertRaises(ValueError):
            get_superclass_name(-1)
        with self.assertRaises(ValueError):
            get_superclass_name(3)

    def test_reproducible_splits(self) -> None:
        """Ensure split generator returns reproducible, disjoint training/validation indices."""
        num_samples = 100
        val_ratio = 0.2
        seed = 1337

        train_idx_1, val_idx_1 = get_train_val_indices(num_samples, val_ratio, seed)
        train_idx_2, val_idx_2 = get_train_val_indices(num_samples, val_ratio, seed)

        # Test reproducibility
        self.assertEqual(train_idx_1, train_idx_2)
        self.assertEqual(val_idx_1, val_idx_2)

        # Test sizes
        self.assertEqual(len(val_idx_1), 20)
        self.assertEqual(len(train_idx_1), 80)

        # Test disjointness and coverage
        union = set(train_idx_1).union(set(val_idx_1))
        intersection = set(train_idx_1).intersection(set(val_idx_1))
        
        self.assertEqual(len(union), num_samples)
        self.assertEqual(len(intersection), 0)

        # Test seed variation alters indices
        train_idx_diff, _ = get_train_val_indices(num_samples, val_ratio, seed + 1)
        self.assertNotEqual(train_idx_1, train_idx_diff)

    @patch("torchvision.datasets.CIFAR10")
    def test_custom_dataset_wrapping(self, mock_cifar10) -> None:
        """Test custom Dataset wrapping, label mapping, and subsetting."""
        # 1. Create a dummy dataset instance
        mock_dataset = mock_cifar10.return_value
        mock_dataset.__len__.return_value = 10
        
        # Setup mock returns: dummy 32x32 image and original label = index (0-9)
        mock_data = [
            (Image.new("RGB", (32, 32)), i) for i in range(10)
        ]
        mock_dataset.__getitem__.side_effect = lambda idx: mock_data[idx]

        # 2. Check full mapping without indices subsetting
        dataset = CIFAR10SuperClassDataset(mock_dataset, indices=None, transform=None)
        self.assertEqual(len(dataset), 10)
        
        # airplane (0) -> Genuine (0)
        img, label = dataset[0]
        self.assertEqual(label, 0)
        
        # cat (3) -> Tampered (1)
        img, label = dataset[3]
        self.assertEqual(label, 1)

        # horse (7) -> Forged (2)
        img, label = dataset[7]
        self.assertEqual(label, 2)

        # 3. Check subset with validation indices
        subset_indices = [1, 3, 7]  # automobile (0), cat (1), horse (2)
        dataset_sub = CIFAR10SuperClassDataset(mock_dataset, indices=subset_indices, transform=None)
        self.assertEqual(len(dataset_sub), 3)
        
        _, label0 = dataset_sub[0]  # original 1 -> mapped 0
        _, label1 = dataset_sub[1]  # original 3 -> mapped 1
        _, label2 = dataset_sub[2]  # original 7 -> mapped 2
        
        self.assertEqual(label0, 0)
        self.assertEqual(label1, 1)
        self.assertEqual(label2, 2)

    @patch("torchvision.datasets.CIFAR10")
    def test_dataloaders_generation(self, mock_cifar10) -> None:
        """Verify the dataloaders return correct batches, tensor shapes, and values range."""
        from unittest.mock import MagicMock
        
        # Mock training set (100 samples)
        mock_train = MagicMock()
        mock_train.__len__.return_value = 100
        mock_train_data = [
            (Image.new("RGB", (32, 32)), i % 10) for i in range(100)
        ]
        mock_train.__getitem__.side_effect = lambda idx: mock_train_data[idx]
        
        # Mock test set (50 samples)
        mock_test = MagicMock()
        mock_test.__len__.return_value = 50
        mock_test_data = [
            (Image.new("RGB", (32, 32)), i % 10) for i in range(50)
        ]
        mock_test.__getitem__.side_effect = lambda idx: mock_test_data[idx]

        # Use side_effect to distinguish train=True/False construction
        def mock_init(root, train=True, download=True, transform=None):
            return mock_train if train else mock_test

        mock_cifar10.side_effect = mock_init

        # Define configurations
        config = {
            "seed": 42,
            "data": {
                "dataset_dir": "./mock_data",
                "val_ratio": 0.2,
                "batch_size": 16,
                "num_workers": 0,
                "pin_memory": False,
                "augmentation": {
                    "random_crop": {"size": 32, "padding": 2},
                    "random_horizontal_flip": {"p": 0.5},
                    "color_jitter": {"brightness": 0.1, "contrast": 0.1, "saturation": 0.1, "hue": 0.05}
                }
            }
        }

        # Generate dataloaders
        train_loader, val_loader, test_loader = get_dataloaders(config)

        # 1. Verify dataloader lengths matching splits
        # 100 samples train_raw -> 80 train, 20 val. Batch size 16.
        # Test: 50 samples.
        self.assertEqual(len(train_loader.dataset), 80)
        self.assertEqual(len(val_loader.dataset), 20)
        self.assertEqual(len(test_loader.dataset), 50)

        # 2. Check shapes of yielded batches
        images, labels = next(iter(train_loader))
        self.assertEqual(images.shape, (16, 3, 32, 32))
        self.assertEqual(labels.shape, (16,))
        
        # Verify labels are strictly [0, 1, 2]
        self.assertTrue(torch.all((labels >= 0) & (labels <= 2)))
        
        # Verify images range is strictly [0.0, 1.0]
        self.assertTrue(torch.all(images >= 0.0))
        self.assertTrue(torch.all(images <= 1.0))

if __name__ == "__main__":
    unittest.main()

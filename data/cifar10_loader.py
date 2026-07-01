"""
This module handles downloading the CIFAR-10 dataset, mapping it to the 3-class
super-class taxonomy, applying data augmentations, and creating PyTorch DataLoaders.
"""
import os
import yaml
from typing import Union, Dict, Any, Tuple, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

from data.mapping import map_index_to_superclass
from data.splits import get_train_val_indices
from utils.seed import set_seed

class CIFAR10SuperClassDataset(Dataset):
    """
    Custom PyTorch Dataset that wraps a torchvision CIFAR-10 dataset,
    filters/remaps labels to the 3-class taxonomy, and applies split-specific transforms.
    """
    def __init__(
        self,
        cifar10_dataset: torchvision.datasets.CIFAR10,
        indices: Optional[List[int]] = None,
        transform: Optional[transforms.Compose] = None
    ) -> None:
        """
        Args:
            cifar10_dataset: Underling CIFAR-10 dataset instance (with transform=None).
            indices: List of subset indices to use. If None, uses all samples.
            transform: Transformations to apply to the images.
        """
        self.dataset = cifar10_dataset
        self.indices = indices if indices is not None else list(range(len(cifar10_dataset)))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        original_idx = self.indices[idx]
        image, label = self.dataset[original_idx]

        # Remap original label (0-9) to superclass index (0-2)
        mapped_label = map_index_to_superclass(label)

        if self.transform is not None:
            image = self.transform(image)

        return image, mapped_label

def load_config(config_path_or_dict: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Helper to parse yaml config or return dictionary config.

    Args:
        config_path_or_dict: Path to YAML config file or config dictionary.

    Returns:
        Config dictionary.
    """
    if isinstance(config_path_or_dict, str):
        if not os.path.exists(config_path_or_dict):
            raise FileNotFoundError(f"Config file not found: {config_path_or_dict}")
        with open(config_path_or_dict, "r") as f:
            return yaml.safe_load(f)
    return config_path_or_dict

def get_dataloaders(
    config_path_or_dict: Union[str, Dict[str, Any]]
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Constructs train, validation, and test DataLoaders using the provided configuration.

    Args:
        config_path_or_dict: Config file path or dictionary.

    Returns:
        A tuple of (train_loader, val_loader, test_loader).
    """
    # Load config and set seeds
    config = load_config(config_path_or_dict)
    seed = config.get("seed", 42)
    set_seed(seed)

    data_cfg = config.get("data", {})
    dataset_dir = data_cfg.get("dataset_dir", "./data/cifar10")
    val_ratio = data_cfg.get("val_ratio", 0.1)
    batch_size = data_cfg.get("batch_size", 64)
    num_workers = data_cfg.get("num_workers", 0)
    pin_memory = data_cfg.get("pin_memory", True)

    # 1. Setup Train / Val Transform (with Augmentations for train)
    aug_cfg = data_cfg.get("augmentation", {})
    
    # Training transforms: crop, flip, color jitter, to tensor
    crop_cfg = aug_cfg.get("random_crop", {"size": 32, "padding": 4})
    flip_cfg = aug_cfg.get("random_horizontal_flip", {"p": 0.5})
    jitter_cfg = aug_cfg.get("color_jitter", {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1})

    train_transform = transforms.Compose([
        transforms.RandomCrop(size=crop_cfg["size"], padding=crop_cfg["padding"]),
        transforms.RandomHorizontalFlip(p=flip_cfg["p"]),
        transforms.ColorJitter(
            brightness=jitter_cfg["brightness"],
            contrast=jitter_cfg["contrast"],
            saturation=jitter_cfg["saturation"],
            hue=jitter_cfg["hue"]
        ),
        transforms.ToTensor()  # Norms automatically to [0.0, 1.0]
    ])

    # Validation and Test transforms: ONLY ToTensor (keep raw [0.0, 1.0] range)
    eval_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 2. Download and Load CIFAR-10 splits
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Load raw torchvision datasets (without transform on underlying object)
    cifar10_train_raw = torchvision.datasets.CIFAR10(
        root=dataset_dir, train=True, download=True, transform=None
    )
    cifar10_test_raw = torchvision.datasets.CIFAR10(
        root=dataset_dir, train=False, download=True, transform=None
    )

    # 3. Create reproducible Train/Val split indices
    train_indices, val_indices = get_train_val_indices(
        num_samples=len(cifar10_train_raw),
        val_ratio=val_ratio,
        seed=seed
    )

    # 4. Wrap in custom Dataset classes
    train_dataset = CIFAR10SuperClassDataset(
        cifar10_dataset=cifar10_train_raw,
        indices=train_indices,
        transform=train_transform
    )
    val_dataset = CIFAR10SuperClassDataset(
        cifar10_dataset=cifar10_train_raw,
        indices=val_indices,
        transform=eval_transform
    )
    test_dataset = CIFAR10SuperClassDataset(
        cifar10_dataset=cifar10_test_raw,
        indices=None,
        transform=eval_transform
    )

    # 5. Create PyTorch DataLoaders
    # Generator for reproducible DataLoader shuffling (if shuffle=True)
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=g
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader

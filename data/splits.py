"""
This module handles reproducible splitting of datasets into train and validation sets
by generating indices using a seeded local random generator.
"""
from typing import Tuple, List
import numpy as np

def get_train_val_indices(
    num_samples: int,
    val_ratio: float,
    seed: int
) -> Tuple[List[int], List[int]]:
    """Generate reproducible train and validation indices.

    Args:
        num_samples: Total number of samples in the training set (e.g. 50000).
        val_ratio: Fraction of samples to allocate to the validation set.
        seed: Random seed for shuffling.

    Returns:
        A tuple of (train_indices, val_indices) as lists of integers.
    """
    if val_ratio < 0.0 or val_ratio >= 1.0:
        raise ValueError(f"val_ratio must be in range [0.0, 1.0), got {val_ratio}")

    # Use default_rng with the given seed for local determinism
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)

    val_size = int(np.floor(val_ratio * num_samples))
    val_indices = indices[:val_size].tolist()
    train_indices = indices[val_size:].tolist()

    return train_indices, val_indices

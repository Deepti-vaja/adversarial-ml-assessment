"""
This module defines the mapping from original CIFAR-10 classes to the 
3-class document fraud proxy taxonomy (Genuine, Tampered, Forged).
"""

# The 10 original CIFAR-10 classes in their canonical index order
CIFAR10_CLASSES = [
    "airplane",      # 0
    "automobile",    # 1
    "bird",          # 2
    "cat",           # 3
    "deer",          # 4
    "dog",           # 5
    "frog",          # 6
    "horse",         # 7
    "ship",          # 8
    "truck"          # 9
]

# Specific mapping from CIFAR-10 class names to super-class integers
# 0 -> Genuine, 1 -> Tampered, 2 -> Forged
CLASS_MAPPING = {
    "airplane": 0,
    "automobile": 0,
    "ship": 0,
    "truck": 0,

    "bird": 1,
    "cat": 1,
    "deer": 1,
    "dog": 1,

    "frog": 2,
    "horse": 2,
}

# The names of the target super-classes corresponding to indices [0, 1, 2]
SUPERCLASS_NAMES = ["Genuine", "Tampered", "Forged"]

# Precomputed index-to-index mapping for fast translation at runtime
INDEX_MAPPING = {
    i: CLASS_MAPPING[name] for i, name in enumerate(CIFAR10_CLASSES)
}

def map_index_to_superclass(cifar10_label_idx: int) -> int:
    """Remap a CIFAR-10 index label (0-9) to the super-class index (0-2).

    Args:
        cifar10_label_idx: Original integer label in range [0, 9]

    Returns:
        Remapped super-class label in range [0, 2]
    """
    if cifar10_label_idx not in INDEX_MAPPING:
        raise ValueError(
            f"Invalid CIFAR-10 class index: {cifar10_label_idx}. Expected 0-9."
        )
    return INDEX_MAPPING[cifar10_label_idx]

def get_superclass_name(superclass_idx: int) -> str:
    """Retrieve the human-readable name of the superclass index.

    Args:
        superclass_idx: Super-class label in range [0, 2]

    Returns:
        The corresponding string name ("Genuine", "Tampered", "Forged")
    """
    if superclass_idx < 0 or superclass_idx >= len(SUPERCLASS_NAMES):
        raise ValueError(
            f"Invalid super-class index: {superclass_idx}. Expected 0, 1, or 2."
        )
    return SUPERCLASS_NAMES[superclass_idx]

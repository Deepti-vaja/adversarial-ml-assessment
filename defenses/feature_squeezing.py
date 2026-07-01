"""
Inference-time Feature Squeezing defense module for Task D1.
Implements preprocessing defenses combining median spatial smoothing and bit-depth reduction
via ART's FeatureSqueezing and SpatialSmoothing preprocessor wrappers.
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from art.estimators.classification import PyTorchClassifier
from art.defences.preprocessor import Preprocessor, FeatureSqueezing, SpatialSmoothing

from utils.logging import get_logger

logger = get_logger("FeatureSqueezingDefense")


class CompositePreprocessor(Preprocessor):
    """Chains spatial median smoothing and bit-depth feature squeezing in a single ART Preprocessor object."""
    def __init__(
        self,
        window_size: int = 3,
        bit_depth: int = 5,
        clip_values: Tuple[float, float] = (0.0, 1.0)
    ):
        super().__init__(is_fitted=True, apply_fit=False, apply_predict=True, clip_values=clip_values)
        self.smoother = SpatialSmoothing(window_size=window_size, channels_first=True, clip_values=clip_values) if window_size > 1 else None
        self.squeezer = FeatureSqueezing(clip_values=clip_values, bit_depth=bit_depth) if bit_depth < 8 else None

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        res_x, res_y = x, y
        if self.smoother is not None:
            res_x, res_y = self.smoother(res_x, res_y)
        if self.squeezer is not None:
            res_x, res_y = self.squeezer(res_x, res_y)
        return res_x, res_y

    def estimate_gradient(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return grad


def get_feature_squeezed_classifier(
    model: nn.Module,
    bit_depth: int = 5,
    window_size: int = 3,
    clip_values: Tuple[float, float] = (0.0, 1.0),
    device: Optional[torch.device] = None
) -> PyTorchClassifier:
    """Wraps a PyTorch neural network in an ART PyTorchClassifier equipped with Feature Squeezing preprocessors.

    Justification of Baseline Implementation Decisions:
        - Median Spatial Smoothing (window_size=3): Local median smoothing reduces high-frequency noise introduced
          by adversarial perturbations while preserving macro edge features of scanned documents.
        - Bit-Depth Reduction (bit_depth=5): Quantizes continuous pixel values into 2^5 = 32 discrete buckets per channel,
          eliminating adversarial gradient sensitivity and small Lp perturbations below the quantization step size.

    Args:
        model: PyTorch classification model.
        bit_depth: Target bit depth per color channel (e.g., 5 reduces to 32 discrete levels).
        window_size: Kernel window size for local spatial median smoothing.
        clip_values: Minimum and maximum valid pixel intensities.
        device: Target computation device (CUDA or CPU).

    Returns:
        Configured ART PyTorchClassifier instance executing preprocessing defenses before forward evaluation.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    device_type = "gpu" if device.type == "cuda" else "cpu"

    # Initialize composite preprocessing defense
    preprocessors = [CompositePreprocessor(window_size=window_size, bit_depth=bit_depth, clip_values=clip_values)]

    logger.info(
        f"Initialized Feature Squeezing classifier with bit_depth={bit_depth} and window_size={window_size}."
    )

    classifier = PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        input_shape=(3, 32, 32),
        nb_classes=3,
        clip_values=clip_values,
        preprocessing_defences=preprocessors,
        device_type=device_type
    )

    return classifier

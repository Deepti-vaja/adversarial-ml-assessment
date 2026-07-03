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
from art.defences.preprocessor import FeatureSqueezing, SpatialSmoothing
from art.defences.preprocessor.preprocessor import PreprocessorPyTorch

from utils.logging import get_logger

logger = get_logger("FeatureSqueezingDefense")


class PyTorchSpatialSmoothing(PreprocessorPyTorch):
    """PyTorch wrapper for ART SpatialSmoothing with Straight-Through Estimator autograd support."""
    def __init__(self, window_size: int = 3, clip_values: Tuple[float, float] = (0.0, 1.0)):
        super().__init__(device_type="gpu" if torch.cuda.is_available() else "cpu", is_fitted=True, apply_fit=False, apply_predict=True)
        self.smoother = SpatialSmoothing(window_size=window_size, channels_first=True, clip_values=clip_values)

    def forward(self, x: torch.Tensor, y: Optional[Any] = None) -> Tuple[torch.Tensor, Optional[Any]]:
        res_x = x.detach().cpu().numpy()
        res_y = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
        res_x, res_y = self.smoother(res_x, res_y)
        x_out_detached = torch.tensor(res_x.astype(np.float32), device=x.device)
        x_out = x + (x_out_detached - x).detach()
        y_out = torch.tensor(res_y, device=y.device) if isinstance(y, torch.Tensor) and res_y is not None else res_y
        return x_out, y_out

    def estimate_gradient(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return grad


class PyTorchFeatureSqueezing(PreprocessorPyTorch):
    """PyTorch wrapper for ART FeatureSqueezing with Straight-Through Estimator autograd support."""
    def __init__(self, bit_depth: int = 5, clip_values: Tuple[float, float] = (0.0, 1.0)):
        super().__init__(device_type="gpu" if torch.cuda.is_available() else "cpu", is_fitted=True, apply_fit=False, apply_predict=True)
        self.squeezer = FeatureSqueezing(clip_values=clip_values, bit_depth=bit_depth)

    def forward(self, x: torch.Tensor, y: Optional[Any] = None) -> Tuple[torch.Tensor, Optional[Any]]:
        res_x = x.detach().cpu().numpy()
        res_y = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
        res_x, res_y = self.squeezer(res_x, res_y)
        x_out_detached = torch.tensor(res_x.astype(np.float32), device=x.device)
        x_out = x + (x_out_detached - x).detach()
        y_out = torch.tensor(res_y, device=y.device) if isinstance(y, torch.Tensor) and res_y is not None else res_y
        return x_out, y_out

    def estimate_gradient(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return grad


class CompositePreprocessor(list, PreprocessorPyTorch):
    """Chains spatial median smoothing and bit-depth feature squeezing in a single ART Preprocessor object."""
    def __init__(
        self,
        window_size: int = 3,
        bit_depth: int = 5,
        clip_values: Tuple[float, float] = (0.0, 1.0)
    ):
        list.__init__(self)
        PreprocessorPyTorch.__init__(self, device_type="gpu" if torch.cuda.is_available() else "cpu", is_fitted=True, apply_fit=False, apply_predict=True)
        self.clip_values = clip_values
        self.smoother = SpatialSmoothing(window_size=window_size, channels_first=True, clip_values=clip_values) if window_size > 1 else None
        self.squeezer = FeatureSqueezing(clip_values=clip_values, bit_depth=bit_depth) if bit_depth < 8 else None
        if self.smoother is not None:
            self.append(self.smoother)
        if self.squeezer is not None:
            self.append(self.squeezer)

    def forward(self, x: torch.Tensor, y: Optional[Any] = None) -> Tuple[torch.Tensor, Optional[Any]]:
        res_x = x.detach().cpu().numpy()
        res_y = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
        if self.smoother is not None:
            res_x, res_y = self.smoother(res_x, res_y)
        if self.squeezer is not None:
            res_x, res_y = self.squeezer(res_x, res_y)
        x_out_detached = torch.tensor(res_x.astype(np.float32), device=x.device)
        x_out = x + (x_out_detached - x).detach()
        y_out = torch.tensor(res_y, device=y.device) if isinstance(y, torch.Tensor) and res_y is not None else res_y
        return x_out, y_out

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        res_x, res_y = x, y
        if self.smoother is not None:
            res_x, res_y = self.smoother(res_x, res_y)
        if self.squeezer is not None:
            res_x, res_y = self.squeezer(res_x, res_y)
        return res_x.astype(x.dtype), res_y

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
    preprocessors = []
    if window_size > 1:
        preprocessors.append(PyTorchSpatialSmoothing(window_size=window_size, clip_values=clip_values))
    if bit_depth < 8:
        preprocessors.append(PyTorchFeatureSqueezing(bit_depth=bit_depth, clip_values=clip_values))

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

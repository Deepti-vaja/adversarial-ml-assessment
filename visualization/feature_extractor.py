"""
Feature extraction module for Boundary Analysis (Task C).
Extracts penultimate layer feature activations from FraudCNN or generic PyTorch classification models
using PyTorch forward hooks.
"""

from typing import Tuple, Optional, Union, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class FeatureExtractor:
    """
    Extracts penultimate layer features from a PyTorch neural network model.
    Registers a forward hook on the target layer (defaulting to the input of the final classifier layer)
    and collects activation representations during evaluation passes.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None) -> None:
        """
        Args:
            model: PyTorch neural network module.
            target_layer: Specific module to hook. If None, automatically locates the
                          penultimate layer (e.g. input to model.classifier[-1] or last nn.Linear).
        """
        self.model = model
        self.features: List[torch.Tensor] = []
        self.hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None

        if target_layer is None:
            self.target_layer = self._find_penultimate_layer(model)
        else:
            self.target_layer = target_layer

    def _find_penultimate_layer(self, model: nn.Module) -> nn.Module:
        """Locates the layer immediately preceding or feeding into the final linear classifier."""
        if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
            # For FraudCNN: classifier[-1] is the final nn.Linear(128, num_classes).
            # Hooking classifier[-1] pre-hook or hooking classifier[-2] forward hook gives the 128-d vector.
            if len(model.classifier) >= 2:
                return model.classifier[-2]
            return model.classifier[0]
        
        # Fallback: locate the last nn.Linear layer and find its predecessor if sequential,
        # or hook the last linear layer's pre-hook. By default, find all linear layers:
        linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        if len(linear_layers) >= 2:
            return linear_layers[-2]
        elif len(linear_layers) == 1:
            return linear_layers[0]
        raise ValueError("Could not automatically locate penultimate layer in model.")

    def _hook_fn(self, module: nn.Module, input_tensor: Union[torch.Tensor, Tuple[torch.Tensor, ...]], output_tensor: torch.Tensor) -> None:
        """Forward hook callback to capture layer activations."""
        # If hooking the predecessor (like Dropout/ReLU before final Linear), output_tensor is the feature vector.
        if isinstance(output_tensor, torch.Tensor):
            feat = output_tensor.detach().cpu()
        elif isinstance(input_tensor, tuple) and len(input_tensor) > 0 and isinstance(input_tensor[0], torch.Tensor):
            feat = input_tensor[0].detach().cpu()
        else:
            raise RuntimeError("Captured hook activation is not a valid tensor.")
        
        if feat.dim() > 2:
            feat = torch.flatten(feat, start_dim=1)
        self.features.append(feat)

    def register(self) -> None:
        """Registers the forward hook on the target layer."""
        if self.hook_handle is None:
            self.hook_handle = self.target_layer.register_forward_hook(self._hook_fn)

    def remove(self) -> None:
        """Removes the forward hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def clear(self) -> None:
        """Clears accumulated features."""
        self.features.clear()

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()


def extract_features_from_tensor(
    model: nn.Module,
    x: Union[torch.Tensor, np.ndarray],
    batch_size: int = 64,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """
    Extracts penultimate features for a given tensor or numpy array of input images.

    Args:
        model: Trained PyTorch model.
        x: Input image tensor or numpy array of shape (N, C, H, W).
        batch_size: Batch size for inference evaluation.
        device: PyTorch device (CPU/GPU). If None, infers from model parameter.

    Returns:
        Numpy array of extracted features with shape (N, feature_dim).
    """
    if device is None:
        device = next(model.parameters()).device

    was_training = model.training
    model.eval()

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    extractor = FeatureExtractor(model)
    extractor.register()

    extracted_list = []
    try:
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                batch = x[i:i + batch_size].to(device)
                extractor.clear()
                _ = model(batch)
                if len(extractor.features) > 0:
                    extracted_list.append(torch.cat(extractor.features, dim=0).numpy())
    finally:
        extractor.remove()
        if was_training:
            model.train()

    if not extracted_list:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(extracted_list, axis=0)


def extract_features_from_loader(
    model: nn.Module,
    dataloader: DataLoader,
    max_samples: Optional[int] = None,
    device: Optional[torch.device] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts penultimate features, true labels, and predicted labels from a DataLoader.

    Args:
        model: Trained PyTorch model.
        dataloader: PyTorch DataLoader providing (images, labels).
        max_samples: Optional maximum number of samples to process.
        device: PyTorch device.

    Returns:
        Tuple of (features_array, labels_array, predictions_array).
    """
    if device is None:
        device = next(model.parameters()).device

    was_training = model.training
    model.eval()

    extractor = FeatureExtractor(model)
    extractor.register()

    feat_list, label_list, pred_list = [], [], []
    total_processed = 0

    try:
        with torch.no_grad():
            for images, targets in dataloader:
                batch_images = images.to(device)
                extractor.clear()
                logits = model(batch_images)
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                if len(extractor.features) > 0:
                    batch_feats = torch.cat(extractor.features, dim=0).numpy()
                else:
                    batch_feats = np.zeros((len(images), 1), dtype=np.float32)

                feat_list.append(batch_feats)
                label_list.append(targets.numpy())
                pred_list.append(preds)

                total_processed += len(targets)
                if max_samples is not None and total_processed >= max_samples:
                    break
    finally:
        extractor.remove()
        if was_training:
            model.train()

    features = np.concatenate(feat_list, axis=0)
    labels = np.concatenate(label_list, axis=0)
    predictions = np.concatenate(pred_list, axis=0)

    if max_samples is not None:
        features = features[:max_samples]
        labels = labels[:max_samples]
        predictions = predictions[:max_samples]

    return features, labels, predictions

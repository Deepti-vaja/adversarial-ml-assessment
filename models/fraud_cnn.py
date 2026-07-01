"""
FraudCNN architecture module for document fraud detection proxy dataset (CIFAR-10 remapped).
Implements a from-scratch CNN with four convolutional blocks following a 32 -> 64 -> 128 -> 256 channel progression.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn

class FraudCNN(nn.Module):
    """
    FraudCNN: Production-ready baseline convolutional neural network designed for
    classifying proxy scanned document images into Genuine (0), Tampered (1), or Forged (2).

    Architecture Design & Justification:
    - Block 1 (3 -> 32 channels): Extracts low-level spatial features (edges, corners, grain patterns).
    - Block 2 (32 -> 64 channels): Combines edges into localized texture descriptors.
    - Block 3 (64 -> 128 channels): Captures higher-level structural anomalies and document components.
    - Block 4 (128 -> 256 channels): Compresses complex semantic patterns before global pooling.
    - Batch Normalization stabilizes gradient flow across deep layers.
    - Spatial Dropout (p=0.3) inside convolutional blocks and FC Dropout (p=0.5) mitigate overfitting.
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        dropout_conv: float = 0.3,
        dropout_fc: float = 0.5
    ) -> None:
        """
        Args:
            in_channels: Number of input image channels (3 for RGB).
            num_classes: Number of target classification super-classes (3).
            dropout_conv: Dropout probability applied within spatial feature blocks.
            dropout_fc: Dropout probability applied before classifier head.
        """
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Block 1: Initial channel progression to 32 filters
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output spatial size: 16x16
            nn.Dropout2d(p=dropout_conv)
        )

        # Block 2: Channel progression 32 -> 64 filters
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output spatial size: 8x8
            nn.Dropout2d(p=dropout_conv)
        )

        # Block 3: Channel progression 64 -> 128 filters
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output spatial size: 4x4
            nn.Dropout2d(p=dropout_conv)
        )

        # Block 4: Channel progression 128 -> 256 filters
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),           # Global average pooling -> 1x1
            nn.Dropout2d(p=dropout_conv)
        )

        # Fully Connected Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),                           # Shape: (B, 256)
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),
            nn.Linear(128, num_classes)             # Output shape: (B, num_classes logits)
        )

        # Custom initialization from scratch (no ImageNet pretraining)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initializes weights using Kaiming Normal initialization for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the 4 convolutional blocks and classifier head.

        Args:
            x: Input image tensor of shape (B, 3, H, W).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        logits = self.classifier(x)
        return logits

def build_model(config: Optional[Dict[str, Any]] = None) -> FraudCNN:
    """Factory function to instantiate FraudCNN from a configuration dictionary.

    Args:
        config: Configuration dictionary (optional).

    Returns:
        Instantiated FraudCNN PyTorch module.
    """
    if config is None:
        config = {}

    model_cfg = config.get("model", {})
    in_channels = model_cfg.get("in_channels", 3)
    num_classes = model_cfg.get("num_classes", 3)
    dropout_conv = model_cfg.get("dropout_conv", 0.3)
    dropout_fc = model_cfg.get("dropout_fc", 0.5)

    return FraudCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout_conv=dropout_conv,
        dropout_fc=dropout_fc
    )

"""
models/vit_2d.py - 2D Vision Transformer for Alzheimer's detection.
"""

import torch
import torch.nn as nn
from transformers import ViTModel
import logging

logger = logging.getLogger(__name__)

class ViT2D(nn.Module):
    def __init__(
        self,
        num_labels: int,
        freeze_layers: bool = True,
        input_size: int = 224,
        patch_size: int = 16,
        dropout_rate: float = 0.1
    ):
        super().__init__()

        # Load pretrained ViT
        self.vit = ViTModel.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            add_pooling_layer=False
        )
        hidden_size = self.vit.config.hidden_size

        # Medical image preprocessing
        self.preprocess = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_labels)
        )

        if freeze_layers:
            self._freeze_layers()

        # Initialize weights
        self._init_weights()

        # Log model statistics
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

    def _init_weights(self):
        """Initialize new weights."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _freeze_layers(self):
        """Freeze pretrained layers."""
        for param in self.vit.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Preprocess grayscale to RGB-like
        x = self.preprocess(x)

        # Normalize
        x = (x - x.mean(dim=[2, 3], keepdim=True)) / (x.std(dim=[2, 3], keepdim=True) + 1e-6)

        # Pass through ViT
        outputs = self.vit(pixel_values=x, return_dict=True)

        # Get CLS token and classify
        x = outputs.last_hidden_state[:, 0]
        x = self.classifier(x)

        return x
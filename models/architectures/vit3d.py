"""
Memory-efficient 3D Vision Transformer for Alzheimer's detection.
"""

import torch
import torch.nn as nn
from transformers import ViTModel
import logging
import numpy as np
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class ViT3D(nn.Module):
    def __init__(
            self,
            num_labels: int,
            freeze_layers: bool = True,
            input_size: int = 224,
            patch_size: int = 16,
            dropout_rate: float = 0.1
    ):
        super().__init__()

        self.input_size = input_size
        self.patch_size = patch_size
        self.num_patches = (input_size // patch_size) ** 3

        logger.info(f"Initializing ViT3D with:")
        logger.info(f"- Input size: {input_size}x{input_size}x{input_size}")
        logger.info(f"- Patch size: {patch_size}x{patch_size}x{patch_size}")
        logger.info(f"- Number of patches: {self.num_patches}")
        logger.info(f"- Number of classes: {num_labels}")

        # Load pre-trained ViT
        self.vit = ViTModel.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            add_pooling_layer=False
        )
        hidden_size = self.vit.config.hidden_size

        # Efficient preprocessing
        self.preprocess = nn.Sequential(
            nn.Conv2d(1, 3, 3, padding=1, bias=False),  # Simpler conv
            nn.BatchNorm2d(3),  # BatchNorm uses less memory
            nn.ReLU()  # ReLU is more memory efficient than GELU
        )

        # Memory-efficient slice selection
        self.slice_selection = nn.Sequential(
            nn.Conv3d(1, 8, 1, bias=False),  # 1x1x1 conv uses less memory
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 3, 1, bias=False),
            nn.Softmax(dim=2)
        )

        # Feature fusion (simplified)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Classification head
        self.classifier = nn.Linear(hidden_size, num_labels)

        self._init_weights()
        if freeze_layers:
            self._freeze_layers()

        # Log model statistics
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _freeze_layers(self):
        """Freeze pretrained layers."""
        self.vit.eval()  # Set to eval mode to save memory
        for param in self.vit.parameters():
            param.requires_grad = False

    @torch.no_grad()  # Memory optimization
    def _get_central_slices(self, x: torch.Tensor) -> tuple:
        """Get central slices efficiently."""
        B, C, D, H, W = x.shape

        # Get central indices
        d_mid = D // 2
        h_mid = H // 2
        w_mid = W // 2

        # Extract central slices directly
        axial = x[:, :, d_mid].clone()     # [B, C, H, W]
        sagittal = x[:, :, :, h_mid].clone()   # [B, C, D, W]
        coronal = x[:, :, :, :, w_mid].clone() # [B, C, D, H]

        return axial, sagittal.transpose(2, 3), coronal.transpose(2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Memory-efficient forward pass."""
        # Get central slices (more memory efficient than attention)
        axial, sagittal, coronal = self._get_central_slices(x)

        # Process views in sequence to save memory
        view_features = []
        for view in [axial, sagittal, coronal]:
            # Preprocess
            view = self.preprocess(view)

            # Basic normalization
            view = F.normalize(view.flatten(2), dim=-1).reshape_as(view)

            # Get features
            with torch.no_grad():  # Don't store gradients for ViT if frozen
                outputs = self.vit(pixel_values=view, return_dict=True)
            features = outputs.last_hidden_state[:, 0]
            view_features.append(features)

            # Clear cache after each view
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

        # Combine features
        x = torch.cat(view_features, dim=1)
        x = self.fusion(x)
        x = self.classifier(x)

        return x


def create_model(config: dict) -> nn.Module:
    """Create a memory-efficient ViT3D model."""
    model = ViT3D(
        num_labels=config['model']['num_labels'],
        freeze_layers=config['model']['freeze_layers'],
        input_size=config['model']['input_size'],
        patch_size=config['model']['patch_size'],
        dropout_rate=config['model'].get('dropout_rate', 0.1)
    )
    return model
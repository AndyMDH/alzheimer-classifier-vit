"""
3D Vision Transformer model for Alzheimer's detection with fixed initialization.
"""

import torch
import torch.nn as nn
from transformers import ViTModel
import logging
import numpy as np
import math
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class LayerNormWithFixedInit(nn.LayerNorm):
    """Custom LayerNorm with fixed initialization."""
    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

class ViT3D(nn.Module):
    """3D Vision Transformer optimized for medical image analysis."""

    def __init__(
            self,
            num_labels: int,
            freeze_layers: bool = True,
            input_size: int = 224,
            patch_size: int = 16,
            dropout_rate: float = 0.1
    ):
        super().__init__()

        # Validate input parameters
        assert input_size % patch_size == 0, "Input size must be divisible by patch size"
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_patches = (input_size // patch_size) ** 3

        # Log initialization parameters
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
        hidden_size = self.vit.config.hidden_size  # 768

        # Medical image specific preprocessing
        self.preprocess = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, padding=3),
            nn.InstanceNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.InstanceNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.InstanceNorm2d(3)
        )

        # Slice selection module with attention
        self.slice_attention = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.InstanceNorm3d(16),
            nn.GELU(),
            nn.Conv3d(16, 8, kernel_size=3, padding=1),
            nn.InstanceNorm3d(8),
            nn.GELU(),
            nn.Conv3d(8, 3, kernel_size=1),
            nn.Softmax(dim=2)
        )

        # Feature enhancement module
        self.feature_enhance = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            LayerNormWithFixedInit(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        # View fusion module
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 2),
            LayerNormWithFixedInit(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, hidden_size),
            LayerNormWithFixedInit(hidden_size)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_labels)
        )

        # Initialize weights explicitly
        self._init_weights()

        # Freeze layers selectively
        if freeze_layers:
            self._freeze_layers()

        # Log model statistics
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

    def _init_weights(self):
        """Initialize weights explicitly."""
        def init_module(m):
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, LayerNormWithFixedInit):
                m.reset_parameters()

        self.apply(init_module)

    def _freeze_layers(self):
        """Selective freezing for better transfer learning."""
        # Freeze early layers
        for param in self.vit.embeddings.parameters():
            param.requires_grad = False

        # Only train the last 4 transformer layers
        for layer in self.vit.encoder.layer[:-4]:
            for param in layer.parameters():
                param.requires_grad = False

    def _get_attention_weighted_slices(self, x: torch.Tensor) -> tuple:
        """Extract attention-weighted slices from volume."""
        B, C, D, H, W = x.shape

        # Generate attention weights for each direction
        attention_weights = self.slice_attention(x)

        # Extract weighted slices for each view
        d_center, h_center, w_center = D//2, H//2, W//2
        span = 3  # Consider slices around center

        # Compute weighted averages around central slices
        axial_region = x[:, :, d_center-span:d_center+span+1]
        sagittal_region = x[:, :, :, h_center-span:h_center+span+1]
        coronal_region = x[:, :, :, :, w_center-span:w_center+span+1]

        axial = (axial_region * attention_weights[:, 0:1, d_center-span:d_center+span+1]).sum(dim=2)
        sagittal = (sagittal_region * attention_weights[:, 1:2, :, h_center-span:h_center+span+1]).sum(dim=3)
        coronal = (coronal_region * attention_weights[:, 2:3, :, :, w_center-span:w_center+span+1]).sum(dim=4)

        return axial, sagittal, coronal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with enhanced medical image processing."""
        # Extract attention-weighted slices
        axial, sagittal, coronal = self._get_attention_weighted_slices(x)

        # Process each view
        view_features = []
        for view in [axial, sagittal, coronal]:
            # Medical image specific preprocessing
            view = self.preprocess(view)

            # Normalize to match pretrained model's distribution
            view = (view - view.mean(dim=[2, 3], keepdim=True)) / (view.std(dim=[2, 3], keepdim=True) + 1e-6)

            # Pass through ViT
            outputs = self.vit(pixel_values=view, return_dict=True)

            # Enhance features
            features = outputs.last_hidden_state[:, 0]
            enhanced = self.feature_enhance(features) + features
            view_features.append(enhanced)

        # Combine view features
        combined = torch.cat(view_features, dim=1)
        fused = self.fusion(combined)

        # Final classification
        output = self.classifier(fused)

        return output


def create_model(config: dict) -> nn.Module:
    """Create a ViT3D model from config."""
    model = ViT3D(
        num_labels=config['model']['num_labels'],
        freeze_layers=config['model']['freeze_layers'],
        input_size=config['model']['input_size'],
        patch_size=config['model']['patch_size'],
        dropout_rate=config['model'].get('dropout_rate', 0.1)
    )
    return model
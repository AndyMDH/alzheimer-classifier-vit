"""
3D Vision Transformer model for Alzheimer's detection with transfer learning.
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
import logging
import numpy as np
import math

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

        # Input validation
        assert patch_size in [8, 16, 32], "Patch size must be 8, 16, or 32"
        assert input_size % patch_size == 0, "Input size must be divisible by patch size"

        self.input_size = input_size
        self.patch_size = patch_size
        self.num_patches = (input_size // patch_size) ** 3

        logger.info(f"Initializing ViT3D with:")
        logger.info(f"- Input size: {input_size}x{input_size}x{input_size}")
        logger.info(f"- Patch size: {patch_size}x{patch_size}x{patch_size}")
        logger.info(f"- Number of patches: {self.num_patches}")
        logger.info(f"- Number of classes: {num_labels}")

        # Create base configuration
        self.config = ViTConfig.from_pretrained('google/vit-base-patch16-224-in21k')
        self.config.patch_size = patch_size
        self.config.image_size = input_size
        self.config.num_labels = num_labels
        self.config.num_patches = self.num_patches

        # Create patch embedding
        self.patch_embed = nn.Conv3d(
            1,  # input channels
            self.config.hidden_size,
            kernel_size=(patch_size, patch_size, patch_size),
            stride=(patch_size, patch_size, patch_size)
        )

        # Create layernorm
        self.norm_pre = nn.LayerNorm(self.config.hidden_size)
        self.norm_post = nn.LayerNorm(self.config.hidden_size)

        # Create transformer encoder
        self.transformer = ViTModel(self.config)

        # Create CLS token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.config.hidden_size))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.config.hidden_size)
        )

        # Dropout and classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.config.hidden_size, num_labels)
        )

        # Initialize weights
        self._init_weights()

        # Freeze layers if specified
        if freeze_layers:
            self._freeze_pretrained_layers()

        # Log model statistics
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

    def _init_weights(self):
        """Initialize weights."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Initialize patch embeddings
        nn.init.kaiming_normal_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

        # Initialize fc layers
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _freeze_pretrained_layers(self):
        """Freeze pretrained layers."""
        for param in self.transformer.parameters():
            param.requires_grad = False

        # Unfreeze task-specific layers
        for param in self.patch_embed.parameters():
            param.requires_grad = True
        self.cls_token.requires_grad = True
        self.pos_embed.requires_grad = True
        for param in self.fc.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        # Input shape: [B, C, D, H, W]
        B = x.shape[0]

        # Patch embedding: [B, C, D, H, W] -> [B, hidden_size, D', H', W']
        x = self.patch_embed(x)

        # Flatten patches: [B, hidden_size, D', H', W'] -> [B, num_patches, hidden_size]
        x = x.flatten(2).transpose(1, 2)

        # Apply pre-norm
        x = self.norm_pre(x)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add position embeddings
        x = x + self.pos_embed
        x = self.dropout(x)

        # Apply transformer (using the transformer's encoder directly)
        encoder_outputs = self.transformer.encoder(x)
        x = encoder_outputs[0] if isinstance(encoder_outputs, tuple) else encoder_outputs

        # Use CLS token for classification
        x = x[:, 0]

        # Apply final norm and classification
        x = self.norm_post(x)
        x = self.dropout(x)
        x = self.fc(x)

        return x


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


__all__ = ['create_model', 'ViT3D']
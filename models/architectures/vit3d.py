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
        self.num_patches = (input_size // patch_size) ** 3  # 14^3 = 2744 patches

        logger.info(f"Initializing ViT3D with:")
        logger.info(f"- Input size: {input_size}x{input_size}x{input_size}")
        logger.info(f"- Patch size: {patch_size}x{patch_size}x{patch_size}")
        logger.info(f"- Number of patches: {self.num_patches}")
        logger.info(f"- Number of classes: {num_labels}")

        # Load pre-trained ViT
        self.vit = ViTModel.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            add_pooling_layer=False,
            ignore_mismatched_sizes=True
        )
        hidden_size = self.vit.config.hidden_size  # 768

        # Create 3D patch embedding
        self.patch_embed = nn.Sequential(
            # Conv3d to extract patches
            nn.Conv3d(
                in_channels=1,
                out_channels=hidden_size,
                kernel_size=patch_size,
                stride=patch_size
            ),
            # Reshape to [B, num_patches, hidden_size]
            Rearrange('b c d h w -> b (d h w) c'),
        )

        # Create layernorm
        self.pre_norm = nn.LayerNorm(hidden_size)
        self.post_norm = nn.LayerNorm(hidden_size)

        # Create CLS token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, hidden_size))

        # Dropout and classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_labels)
        )

        # Initialize weights
        self._init_weights()

        # Use transformers layers directly
        self.encoder = self.vit.encoder
        self.layernorm = self.vit.layernorm

        # Freeze pre-trained layers if specified
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

        # Initialize conv layer
        nn.init.trunc_normal_(self.patch_embed[0].weight, std=0.02)
        if self.patch_embed[0].bias is not None:
            nn.init.zeros_(self.patch_embed[0].bias)

        # Initialize fc layers
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _freeze_pretrained_layers(self):
        """Freeze pretrained transformer layers."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.layernorm.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        B = x.shape[0]  # Batch size

        # Extract and embed patches: [B, 1, 224, 224, 224] -> [B, 2744, 768]
        x = self.patch_embed(x)

        # Add CLS token: [B, 2744, 768] -> [B, 2745, 768]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add position embeddings and normalize
        x = x + self.pos_embed
        x = self.pre_norm(x)
        x = self.dropout(x)

        # Pass through transformer encoder directly
        encoder_outputs = self.encoder(x)
        sequence_output = encoder_outputs[0]

        # Apply final layernorm (from ViT)
        sequence_output = self.layernorm(sequence_output)

        # Take CLS token output and classify
        x = sequence_output[:, 0]
        x = self.post_norm(x)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class Rearrange(nn.Module):
    def __init__(self, pattern):
        super().__init__()
        self.pattern = pattern

    def forward(self, x):
        return x.permute(0, 2, 3, 4, 1).reshape(x.size(0), -1, x.size(1))


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
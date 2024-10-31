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

        # Initialize transformer
        self.transformer = ViTModel(self.config)

        # Create CLS token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.config.hidden_size))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.config.hidden_size)
        )

        # Create learnable temperature parameter for attention
        self.temp = nn.Parameter(torch.ones([]) * 0.07)

        # Layer normalization and dropout
        self.norm = nn.LayerNorm(self.config.hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

        # Classification head
        self.fc = nn.Linear(self.config.hidden_size, num_labels)

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
        """Initialize weights with truncated normal distribution."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights_)

    def _init_weights_(self, m):
        """Initialize network weights."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _freeze_pretrained_layers(self):
        """Freeze pretrained layers while keeping new layers trainable."""
        for param in self.transformer.parameters():
            param.requires_grad = False

        # Unfreeze task-specific layers
        for param in self.patch_embed.parameters():
            param.requires_grad = True
        self.cls_token.requires_grad = True
        self.pos_embed.requires_grad = True
        self.fc.requires_grad = True
        self.norm.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        # Input shape: [B, C, D, H, W]
        B = x.shape[0]

        # Patch embedding: [B, C, D, H, W] -> [B, hidden_size, D', H', W']
        x = self.patch_embed(x)

        # Flatten patches: [B, hidden_size, D', H', W'] -> [B, num_patches, hidden_size]
        D = x.size(2)
        H = x.size(3)
        W = x.size(4)
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # [B, D', H', W', hidden_size]
        x = x.view(B, D * H * W, -1)  # [B, num_patches, hidden_size]

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add position embeddings
        x = x + self.pos_embed
        x = self.dropout(x)

        # Apply transformer
        x = self.transformer(inputs_embeds=x, return_dict=True).last_hidden_state

        # Use CLS token for classification
        x = x[:, 0]  # Take CLS token
        x = self.norm(x)
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
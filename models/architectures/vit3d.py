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
        self.patch_embed = nn.Sequential(
            nn.Conv3d(
                1,  # input channels
                self.config.hidden_size,
                kernel_size=(patch_size, patch_size, patch_size),
                stride=(patch_size, patch_size, patch_size)
            ),
            nn.LayerNorm([self.config.hidden_size])
        )

        # Initialize ViT
        self.vit = ViTModel(self.config)

        # Replace patch embeddings with our 3D version
        self.vit.embeddings.patch_embeddings = self.patch_embed

        # Adjust position embeddings for 3D
        self.vit.embeddings.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.config.hidden_size)
        )

        # Initialize position embeddings
        self._init_3d_pos_embeddings()

        # Create classifier
        self.layer_norm = nn.LayerNorm(self.config.hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
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

    def _init_3d_pos_embeddings(self):
        """Initialize position embeddings for 3D data."""
        patch_dim = int(round(self.num_patches ** (1 / 3)))
        position_embedding = torch.zeros(1, self.num_patches + 1, self.config.hidden_size)

        # Create positional encodings for each dimension
        for pos in range(self.num_patches):
            # Convert linear position to 3D coordinates
            z = pos // (patch_dim * patch_dim)
            y = (pos % (patch_dim * patch_dim)) // patch_dim
            x = pos % patch_dim

            for i in range(self.config.hidden_size // 3):
                div_term = np.exp(i * -math.log(10000.0) / (self.config.hidden_size // 3))
                pos_x = x * div_term
                pos_y = y * div_term
                pos_z = z * div_term

                position_embedding[0, pos + 1, i * 3] = math.sin(pos_x)
                position_embedding[0, pos + 1, i * 3 + 1] = math.sin(pos_y)
                position_embedding[0, pos + 1, i * 3 + 2] = math.sin(pos_z)

        self.vit.embeddings.position_embeddings.data.copy_(position_embedding)

    def _init_weights(self):
        """Initialize model weights."""
        # Initialize patch embeddings
        nn.init.normal_(self.patch_embed[0].weight, std=0.02)
        if self.patch_embed[0].bias is not None:
            nn.init.zeros_(self.patch_embed[0].bias)

        # Initialize classifier
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _freeze_pretrained_layers(self):
        """Freeze pretrained layers."""
        for param in self.vit.parameters():
            param.requires_grad = False

        # Unfreeze task-specific layers
        for param in self.patch_embed.parameters():
            param.requires_grad = True
        self.vit.embeddings.position_embeddings.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
        self.layer_norm.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        B, C, D, H, W = x.shape
        assert C == 1, f"Expected 1 channel, got {C}"
        assert D == H == W == self.input_size, \
            f"Expected {self.input_size}x{self.input_size}x{self.input_size} input, got {D}x{H}x{W}"

        # Extract patches
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        # Add classification token
        cls_token = self.vit.embeddings.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Add position embeddings
        x = x + self.vit.embeddings.position_embeddings

        # Apply ViT encoder
        outputs = self.vit(inputs_embeds=x)
        x = outputs.last_hidden_state[:, 0]

        # Apply classification head
        x = self.layer_norm(x)
        x = self.dropout(x)
        logits = self.classifier(x)

        return logits


def create_vit_3d(
        num_labels: int,
        freeze_layers: bool = True,
        input_size: int = 224,
        patch_size: int = 16,
        dropout_rate: float = 0.1
) -> nn.Module:
    """Create ViT3D model."""
    try:
        model = ViT3D(
            num_labels=num_labels,
            freeze_layers=freeze_layers,
            input_size=input_size,
            patch_size=patch_size,
            dropout_rate=dropout_rate
        )
        logger.info("Created ViT3D model successfully")
        return model
    except Exception as e:
        logger.error(f"Error creating ViT3D model: {str(e)}")
        raise


__all__ = ['create_vit_3d', 'ViT3D']
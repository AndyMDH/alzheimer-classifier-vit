"""
3D Vision Transformer model for Alzheimer's detection.
"""

import torch
import torch.nn as nn
from transformers import ViTModel


class ViT3D(nn.Module):
    def __init__(self, num_labels: int, freeze_layers: bool = True):
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

        # Modify the patch embedding layer for 3D input
        in_channels = 1  # Assuming single-channel 3D volumes
        patch_size = self.vit.config.patch_size
        self.vit.embeddings.patch_embeddings = nn.Conv3d(
            in_channels, self.vit.config.hidden_size,
            kernel_size=(patch_size, patch_size, patch_size),
            stride=(patch_size, patch_size, patch_size)
        )

        # Adjust position embeddings for 3D
        num_patches = (224 // patch_size) ** 3  # Assuming 224x224x224 input
        self.vit.embeddings.position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches + 1, self.vit.config.hidden_size)
        )

        if freeze_layers:
            for param in self.vit.parameters():
                param.requires_grad = False

        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        logits = self.classifier(outputs.last_hidden_state[:, 0])
        return logits


def create_vit_3d(num_labels: int, freeze_layers: bool = True) -> nn.Module:
    """Create a 3D Vision Transformer model with transfer learning."""
    return ViT3D(num_labels, freeze_layers)
"""
3D Vision Transformer model for Alzheimer's detection.
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class ViT3D(nn.Module):
    def __init__(self, num_labels: int, freeze_layers: bool = True, input_size: int = 224, patch_size: int = 16):
        super().__init__()
        assert patch_size in [8, 16, 32], "Patch size must be 8, 16, or 32"
        assert input_size % patch_size == 0, "Input size must be divisible by patch size"

        # Create a custom configuration
        config = ViTConfig.from_pretrained('google/vit-base-patch16-224-in21k')
        config.patch_size = patch_size
        config.image_size = input_size

        # Initialize the ViT model with the custom config
        self.vit = ViTModel(config)

        # Modify the patch embedding layer for 3D input
        in_channels = 1  # Assuming single-channel 3D volumes
        self.vit.embeddings.patch_embeddings = nn.Conv3d(
            in_channels, config.hidden_size,
            kernel_size=(patch_size, patch_size, patch_size),
            stride=(patch_size, patch_size, patch_size)
        )

        # Adjust position embeddings for 3D
        num_patches = (input_size // patch_size) ** 3
        self.vit.embeddings.position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.hidden_size)
        )

        if freeze_layers:
            for param in self.vit.parameters():
                param.requires_grad = False
            # Unfreeze the modified layers
            for param in self.vit.embeddings.patch_embeddings.parameters():
                param.requires_grad = True
            self.vit.embeddings.position_embeddings.requires_grad = True

        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, pixel_values):
        B, C, D, H, W = pixel_values.shape
        assert C == 1, f"Expected 1 channel, got {C}"
        assert D == H == W, f"Expected cubic input, got {D}x{H}x{W}"

        outputs = self.vit(pixel_values)
        logits = self.classifier(outputs.last_hidden_state[:, 0])
        return logits

def create_vit_3d(num_labels: int, freeze_layers: bool = True, input_size: int = 224, patch_size: int = 16) -> nn.Module:
    """Create a 3D Vision Transformer model with transfer learning."""
    return ViT3D(num_labels, freeze_layers, input_size, patch_size)

# Make sure to export the create_vit_3d function
__all__ = ['create_vit_3d']
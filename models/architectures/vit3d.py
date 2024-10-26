"""
3D Vision Transformer model for Alzheimer's detection with transfer learning.
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
import logging
from typing import Dict, Any
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class ViT3D(nn.Module):
    """3D Vision Transformer with transfer learning from 2D ViT."""
    
    def __init__(
        self,
        num_labels: int,
        freeze_layers: bool = True,
        input_size: int = 224,
        patch_size: int = 16,
        dropout_rate: float = 0.1
    ):
        """
        Initialize the 3D Vision Transformer.

        Args:
            num_labels: Number of output classes
            freeze_layers: Whether to freeze the pretrained layers
            input_size: Input image size (assumed cubic)
            patch_size: Size of patches (assumed cubic)
            dropout_rate: Dropout rate for the classifier
        """
        super().__init__()
        assert patch_size in [8, 16, 32], "Patch size must be 8, 16, or 32"
        assert input_size % patch_size == 0, "Input size must be divisible by patch size"
        
        # Log model configuration
        logger.info(f"Initializing 3D ViT with:")
        logger.info(f"- Input size: {input_size}x{input_size}x{input_size}")
        logger.info(f"- Patch size: {patch_size}x{patch_size}x{patch_size}")
        logger.info(f"- Number of classes: {num_labels}")
        logger.info(f"- Freeze layers: {freeze_layers}")

        # Create a custom configuration
        self.config = ViTConfig.from_pretrained('google/vit-base-patch16-224-in21k')
        self.config.patch_size = patch_size
        self.config.image_size = input_size
        self.config.num_labels = num_labels
        
        # Initialize the ViT model with the custom config
        self.vit = ViTModel(self.config)

        # Modify the patch embedding layer for 3D input
        in_channels = 1  # Single-channel 3D volumes
        self.vit.embeddings.patch_embeddings = nn.Conv3d(
            in_channels,
            self.config.hidden_size,
            kernel_size=(patch_size, patch_size, patch_size),
            stride=(patch_size, patch_size, patch_size)
        )

        # Adjust position embeddings for 3D
        num_patches = (input_size // patch_size) ** 3
        self.vit.embeddings.position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches + 1, self.config.hidden_size)
        )

        # Initialize new parameters
        nn.init.trunc_normal_(
            self.vit.embeddings.position_embeddings,
            std=self.config.initializer_range
        )
        self.vit.embeddings.patch_embeddings.apply(self._init_weights)

        # Freeze/unfreeze layers based on configuration
        if freeze_layers:
            self._freeze_pretrained_layers()
            
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.config.hidden_size, num_labels)
        )
        
        # Initialize classifier
        self.classifier.apply(self._init_weights)
        
        # Log model statistics
        num_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {num_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            nn.init.trunc_normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _freeze_pretrained_layers(self):
        """Freeze pretrained layers while keeping new layers trainable."""
        for param in self.vit.parameters():
            param.requires_grad = False
            
        # Unfreeze the modified layers
        for param in self.vit.embeddings.patch_embeddings.parameters():
            param.requires_grad = True
        self.vit.embeddings.position_embeddings.requires_grad = True

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            pixel_values: Input tensor of shape (batch_size, channels, depth, height, width)

        Returns:
            Tensor of shape (batch_size, num_labels)
        """
        B, C, D, H, W = pixel_values.shape
        assert C == 1, f"Expected 1 channel, got {C}"
        assert D == H == W, f"Expected cubic input, got {D}x{H}x{W}"

        # Get ViT features
        outputs = self.vit(pixel_values)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        
        # Apply classification head
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

    def get_attention_maps(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Get attention maps for visualization.
        
        Args:
            pixel_values: Input tensor
            
        Returns:
            Attention weights from last layer
        """
        outputs = self.vit(pixel_values, output_attentions=True)
        return outputs.attentions[-1]  # Last layer attention

def create_model(config: Dict[str, Any]) -> nn.Module:
    """
    Create a 3D Vision Transformer model based on configuration.
    
    Args:
        config: Configuration dictionary containing model parameters
        
    Returns:
        Initialized 3D ViT model
    """
    model = ViT3D(
        num_labels=config['model']['num_labels'],
        freeze_layers=config['model']['freeze_layers'],
        input_size=config['dataset']['input_size'],
        patch_size=config['model']['patch_size'],
        dropout_rate=config['model'].get('dropout_rate', 0.1)
    )
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info("Model moved to GPU")
    
    return model

__all__ = ['create_model', 'ViT3D']
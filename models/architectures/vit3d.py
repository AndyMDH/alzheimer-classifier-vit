"""
Multi-view Vision Transformer for Alzheimer's detection using transfer learning.
"""

import torch
import torch.nn as nn
from transformers import ViTModel
import logging
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class MultiViewViT(nn.Module):
    def __init__(
            self,
            num_labels: int,
            freeze_layers: bool = True,
            input_size: int = 224,
            dropout_rate: float = 0.1
    ):
        super().__init__()

        self.input_size = input_size

        logger.info(f"Initializing MultiViewViT with:")
        logger.info(f"- Input size: {input_size}x{input_size}")
        logger.info(f"- Views: Axial, Sagittal, Coronal")
        logger.info(f"- Number of classes: {num_labels}")

        # Load pre-trained ViT (now using a larger variant)
        self.vit = ViTModel.from_pretrained(
            'google/vit-large-patch16-224-in21k',  # Using larger model for better features
            add_pooling_layer=False
        )
        hidden_size = self.vit.config.hidden_size  # 1024 for large model

        # Channel projection to convert 1-channel MRI to 3-channel input
        self.channel_proj = nn.Sequential(
            nn.Conv2d(1, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.GELU()
        )

        # Adaptive slice selection
        self.slice_attention = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(16, 3, kernel_size=1),  # Output 3 attention maps for 3 views
            nn.Softmax(dim=2)  # Softmax along depth dimension
        )

        # Feature fusion module
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_labels)
        )

        # Initialize weights
        self._init_weights()

        # Freeze pre-trained layers if specified
        if freeze_layers:
            self._freeze_pretrained_layers()

        # Log model statistics
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

    def _init_weights(self):
        """Initialize custom layers."""
        for m in [self.channel_proj, self.fusion, self.classifier]:
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _freeze_pretrained_layers(self):
        """Freeze pretrained ViT layers."""
        for param in self.vit.parameters():
            param.requires_grad = False

    def _get_weighted_slices(self, x):
        """Get attention-weighted slices from each view."""
        B, C, D, H, W = x.shape

        # Generate attention weights
        attention = self.slice_attention(x)  # [B, 3, D, H, W]

        # Extract weighted slices for each view
        axial = (x[:, :, :, :, :] * attention[:, 0:1, :, :, :]).sum(dim=2)
        sagittal = (x[:, :, :, :, :] * attention[:, 1:2, :, :, :]).sum(dim=3)
        coronal = (x[:, :, :, :, :] * attention[:, 2:3, :, :, :]).sum(dim=4)

        return axial, sagittal, coronal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using attention-weighted views."""
        B = x.shape[0]

        # Get weighted slices from each view
        axial, sagittal, coronal = self._get_weighted_slices(x)

        # Process each view
        view_features = []
        for view in [axial, sagittal, coronal]:
            # Project to 3 channels and normalize
            view = self.channel_proj(view)

            # Normalize to ImageNet range
            view = F.interpolate(view, size=(224, 224))
            view = (view - view.mean()) / view.std()

            # Pass through ViT
            outputs = self.vit(pixel_values=view, return_dict=True)
            view_features.append(outputs.last_hidden_state[:, 0])  # Use CLS token

        # Combine view features
        x = torch.cat(view_features, dim=1)
        x = self.fusion(x)
        x = self.classifier(x)

        return x


def create_model(config: dict) -> nn.Module:
    """Create a multi-view ViT model from config."""
    model = MultiViewViT(
        num_labels=config['model']['num_labels'],
        freeze_layers=config['model'].get('freeze_layers', True),
        input_size=config['model'].get('input_size', 224),
        dropout_rate=config['model'].get('dropout_rate', 0.1)
    )
    return model
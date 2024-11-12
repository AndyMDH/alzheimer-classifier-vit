"""
Optimized Multi-view Vision Transformer for faster training and better accuracy.
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
            dropout_rate: float = 0.1,
            use_middle_slices: bool = True  # Added option for faster training
    ):
        super().__init__()

        self.input_size = input_size
        self.use_middle_slices = use_middle_slices

        logger.info(f"Initializing Optimized MultiViewViT with:")
        logger.info(f"- Input size: {input_size}x{input_size}")
        logger.info(f"- Views: Axial, Sagittal, Coronal")
        logger.info(f"- Using middle slices: {use_middle_slices}")
        logger.info(f"- Number of classes: {num_labels}")

        # Load pre-trained ViT (using base model for faster training)
        self.vit = ViTModel.from_pretrained(
            'google/vit-base-patch16-224-in21k',  # Using base model for speed
            add_pooling_layer=False
        )
        hidden_size = self.vit.config.hidden_size  # 768 for base model

        # Efficient channel projection
        self.channel_proj = nn.Sequential(
            nn.Conv2d(1, 3, 1, 1, bias=False),  # Removed bias for speed
            nn.BatchNorm2d(3),
            nn.ReLU()  # Using ReLU instead of GELU for speed
        )

        # Simplified feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Classification head
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Initialize weights
        self._init_weights()

        # Freeze and optimize pre-trained layers
        if freeze_layers:
            self._freeze_pretrained_layers()

        # Log model statistics
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

    def _init_weights(self):
        """Initialize custom layers with simple initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _freeze_pretrained_layers(self):
        """Freeze pretrained ViT layers and optimize memory."""
        self.vit.eval()  # Set to eval mode for inference
        for param in self.vit.parameters():
            param.requires_grad = False

    def _get_middle_slices(self, x):
        """Extract middle slices from each view efficiently."""
        B, C, D, H, W = x.shape

        # Get middle indices
        d_mid = D // 2
        h_mid = H // 2
        w_mid = W // 2

        # Extract slices efficiently
        axial = x[:, :, d_mid, :, :]
        sagittal = x[:, :, :, h_mid, :]
        coronal = x[:, :, :, :, w_mid]

        return axial, sagittal, coronal

    @torch.no_grad()  # Disable gradients for efficiency
    def _normalize_view(self, view):
        """Normalize view efficiently."""
        view = F.interpolate(view, size=(224, 224), mode='bilinear', align_corners=False)
        view = (view - view.mean(dim=[2, 3], keepdim=True)) / (view.std(dim=[2, 3], keepdim=True) + 1e-6)
        return view

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized forward pass."""
        if self.use_middle_slices:
            axial, sagittal, coronal = self._get_middle_slices(x)
        else:
            # Use your original _get_weighted_slices method here
            axial, sagittal, coronal = self._get_weighted_slices(x)

        # Process each view efficiently
        view_features = []
        for view in [axial, sagittal, coronal]:
            # Project and normalize efficiently
            view = self.channel_proj(view)
            view = self._normalize_view(view)

            # Get ViT features
            with torch.set_grad_enabled(not self.vit.training):  # Only compute gradients if not frozen
                outputs = self.vit(pixel_values=view, return_dict=True)
            view_features.append(outputs.last_hidden_state[:, 0])

        # Combine features efficiently
        x = torch.cat(view_features, dim=1)
        x = self.fusion(x)
        x = self.classifier(x)

        return x


def create_model(config: dict) -> nn.Module:
    """Create an optimized multi-view ViT model from config."""
    model = MultiViewViT(
        num_labels=config['model']['num_labels'],
        freeze_layers=config['model'].get('freeze_layers', True),
        input_size=config['model'].get('input_size', 224),
        dropout_rate=config['model'].get('dropout_rate', 0.1),
        use_middle_slices=config['model'].get('use_middle_slices', True)
    )
    return model
"""
2D Vision Transformer model for Alzheimer's detection.
"""

from torch import nn
from transformers import ViTForImageClassification


def create_vit_2d(num_labels: int, freeze_layers: bool = True) -> nn.Module:
    """Create a 2D Vision Transformer model with transfer learning."""
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=num_labels)

    if freeze_layers:
        for param in model.vit.parameters():
            param.requires_grad = False

    # Only train the classification head
    model.classifier = nn.Linear(model.config.hidden_size, num_labels)

    return model
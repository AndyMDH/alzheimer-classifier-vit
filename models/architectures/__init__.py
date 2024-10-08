"""
Initialize the architectures module and provide a unified interface for model creation.
"""

from .vit3d import create_vit_3d
from .vit2d import create_vit_2d
from .cnn3d import create_cnn_3d


def create_model(model_type: str,
                 num_labels: int,
                 freeze_layers: bool = True,
                 input_size: int = 224,
                 patch_size: int = 16):
    """
    Create a model based on the specified type.

    Args:
        model_type (str): Type of the model ('2d_vit', '3d_vit', or '3d_cnn')
        num_labels (int): Number of output labels
        freeze_layers (bool): Whether to freeze pretrained layers
        input_size (int): Size of the input image (assumed to be cubic for 3D)
        patch_size (int): Size of the patches for ViT models (8, 16, or 32)

    Returns:
        nn.Module: The created model
    """
    if model_type == '3d_vit':
        return create_vit_3d(num_labels, freeze_layers, input_size, patch_size)
    elif model_type == '2d_vit':
        return create_vit_2d(num_labels, freeze_layers, input_size, patch_size)
    elif model_type == '3d_cnn':
        return create_cnn_3d(num_labels, freeze_layers)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

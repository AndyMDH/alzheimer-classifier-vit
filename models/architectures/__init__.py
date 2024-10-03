from .vit2d import create_vit_2d
from .vit3d import create_vit_3d
from .cnn3d import create_cnn_3d

def create_model(model_type: str, num_labels: int, freeze_layers: bool) -> nn.Module:
    """Create a model based on the specified type."""
    if model_type == '2d_vit':
        return create_vit_2d(num_labels, freeze_layers)
    elif model_type == '3d_vit':
        return create_vit_3d(num_labels, freeze_layers)
    elif model_type == '3d_cnn':
        return create_cnn_3d(num_labels, freeze_layers)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
"""
Model factory for creating different model architectures.
"""

import logging
from typing import Dict, Any
from .architectures import ViT2D, ViT3D, CNN3D

logger = logging.getLogger(__name__)


def create_model(config: Dict[str, Any], model_type: str = None) -> Any:
    """
    Create a model based on configuration and type.

    Args:
        config: Configuration dictionary
        model_type: Type of model to create ('vit2d', 'vit3d', 'cnn3d')

    Returns:
        Instantiated model
    """
    # Use model type from config if not specified
    model_type = model_type or config['model']['type']

    # Map model types to classes
    model_map = {
        'vit2d': ViT2D,
        'vit3d': ViT3D,
        'cnn3d': CNN3D
    }

    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}. "
                         f"Available types: {list(model_map.keys())}")

    # Get the model class
    model_class = model_map[model_type]

    # Create model instance
    logger.info(f"Creating model of type: {model_type}")
    try:
        model = model_class(
            num_labels=config['model']['num_labels'],
            freeze_layers=config['model'].get('freeze_layers', True),
            input_size=config['model'].get('input_size', 224),
            patch_size=config['model'].get('patch_size', 16),
            dropout_rate=config['model'].get('dropout_rate', 0.1)
        )
        logger.info(f"Successfully created {model_type} model")
        return model

    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        raise
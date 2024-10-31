"""
Initialize the architectures module and provide a unified interface for model creation.
"""

from .vit3d import create_vit_3d
import logging

logger = logging.getLogger(__name__)

def create_model(config: dict):
    """
    Create a model based on configuration.
    
    Args:
        config (dict): Configuration dictionary containing model parameters
        
    Returns:
        nn.Module: The created model
    """
    try:
        model_type = config['model']['type']
        model_config = config['model']
        
        if model_type == 'vit3d':
            model = create_vit_3d(
                num_labels=model_config['num_labels'],
                freeze_layers=model_config['freeze_layers'],
                input_size=model_config['input_size'],
                patch_size=model_config['patch_size'],
                dropout_rate=model_config.get('dropout_rate', 0.1)
            )
            logger.info(f"Created {model_type} model successfully")
            return model
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    except KeyError as e:
        logger.error(f"Missing configuration parameter: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        raise

__all__ = ['create_model']
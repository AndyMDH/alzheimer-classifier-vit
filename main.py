"""
Main script for Alzheimer's detection with model selection.
"""

import sys
import argparse
import yaml
import logging
from pathlib import Path
import torch
import random
import numpy as np
from datetime import datetime
from models import create_model
from data.data_loader import create_data_loaders
from models.train import train_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Alzheimer's Detection Training")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--model', type=str, choices=['vit2d', 'vit3d', 'cnn3d'],
                       default='vit2d', help='Model architecture to use')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                       help='Device to use (overrides config file)')
    parser.add_argument('--patch_size', type=int, default=16,
                       help='Patch size for ViT models')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    return parser.parse_args()

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_experiment(config: dict, model_type: str) -> Path:
    """Setup experiment directories and logging."""
    try:
        if 'paths' not in config:
            config['paths'] = {}

        # Set default paths relative to current directory
        base_dir = Path.cwd()
        default_paths = {
            'output_dir': base_dir / 'output',
            'log_dir': base_dir / 'logs',
            'checkpoint_dir': base_dir / 'checkpoints'
        }

        # Create or verify directories
        for path_key, path in default_paths.items():
            path = Path(path)
            if path.exists() and not path.is_dir():
                path.unlink()
            path.mkdir(parents=True, exist_ok=True)
            config['paths'][path_key] = str(path.absolute())

        # Create experiment directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"{model_type}_{timestamp}"
        exp_dir = Path(config['paths']['output_dir']) / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Create experiment subdirectories
        for subdir in ['checkpoints', 'logs', 'results']:
            (exp_dir / subdir).mkdir(exist_ok=True)

        # Save config to experiment directory
        with open(exp_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info(f"Created experiment directory at: {exp_dir}")
        return exp_dir

    except Exception as e:
        logger.error(f"Error in setup_experiment: {str(e)}")
        raise

def main():
    """Main function to run the training pipeline."""
    args = parse_args()

    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config(args.config)

        # Override config with command line arguments
        if args.device:
            config['training']['device'] = args.device
        config['model']['type'] = args.model
        config['model']['patch_size'] = args.patch_size

        # Device handling
        if not torch.cuda.is_available() and config['training']['device'] == 'cuda':
            logger.warning("CUDA not available, falling back to CPU")
            config['training']['device'] = 'cpu'
        device = torch.device(config['training']['device'])
        logger.info(f"Using device: {device}")

        # Set random seed
        random.seed(config['training']['seed'])
        np.random.seed(config['training']['seed'])
        torch.manual_seed(config['training']['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config['training']['seed'])
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Setup experiment directory
        exp_dir = setup_experiment(config, args.model)

        # Create data loaders
        logger.info("Creating data loaders...")
        mode = '2d' if args.model == 'vit2d' else '3d'
        train_loader, val_loader, test_loader = create_data_loaders(config, mode=mode)
        logger.info("Data loaders created successfully")

        # Create model
        logger.info(f"Creating {args.model} model...")
        model = create_model(config, model_type=args.model)
        model = model.to(device)
        logger.info(f"Model created and moved to {device} successfully")

        # Train model
        logger.info("Starting training...")
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            exp_dir=exp_dir
        )
        logger.info("Training completed successfully")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
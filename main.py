"""
Main script for Alzheimer's detection using Vision Transformers.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
import torch
import logging
import random
import numpy as np
from datetime import datetime
from models.architectures import create_model
from data.data_loader import create_data_loaders, load_config
from models.train import train_model
from models.evaluate import evaluate_model

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

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_experiment(config: dict) -> Path:
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
        exp_name = f"{config['model']['type']}_{timestamp}"
        exp_dir = Path(config['paths']['output_dir']) / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Create experiment subdirectories
        for subdir in ['checkpoints', 'logs', 'results', 'visualizations']:
            (exp_dir / subdir).mkdir(exist_ok=True)

        # Save config to experiment directory
        with open(exp_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info(f"Created experiment directory at: {exp_dir}")
        return exp_dir

    except Exception as e:
        logger.error(f"Error in setup_experiment: {str(e)}")
        raise

def main() -> None:
    """Main function to run the training pipeline."""
    parser = argparse.ArgumentParser(description="Alzheimer's Detection Model Training")
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Path to config file')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                       help='Device to use (overrides config file)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    args = parser.parse_args()

    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config(args.config)
        
        # Device handling
        if args.device:
            config['training']['device'] = args.device
        elif not torch.cuda.is_available() and config['training']['device'] == 'cuda':
            logger.warning("CUDA not available, falling back to CPU")
            config['training']['device'] = 'cpu'
            
        device = torch.device(config['training']['device'])
        logger.info(f"Using device: {device}")
        
        if device.type == 'cuda':
            logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")

        # Set random seed
        set_seed(config['training']['seed'])
        logger.info(f"Set random seed: {config['training']['seed']}")

        # Setup experiment directory
        exp_dir = setup_experiment(config)

        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(config)
        logger.info("Data loaders created successfully")

        # Create model
        logger.info("Creating model...")
        model = create_model(config)
        try:
            model = model.to(device)
            logger.info(f"Model moved to {device} successfully")
        except Exception as e:
            logger.error(f"Error moving model to {device}: {str(e)}")
            if device.type == 'cuda':
                logger.info("Falling back to CPU")
                device = torch.device('cpu')
                config['training']['device'] = 'cpu'
                model = model.to(device)

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

        # Evaluate model
        logger.info("Starting evaluation...")
        results = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device
        )
        
        # Save results
        results_file = exp_dir / 'results' / 'test_results.yaml'
        with open(results_file, 'w') as f:
            yaml.dump(results, f)
        logger.info(f"Results saved to {results_file}")

        return exp_dir

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        if '--debug' in sys.argv:
            import traceback
            traceback.print_exc()
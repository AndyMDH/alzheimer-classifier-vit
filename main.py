# main.py
"""
Main script for Alzheimer's detection using transfer learning on vision transformers and CNNs.
"""

import argparse
import yaml
from pathlib import Path
import torch
from models.architectures import create_model
from data.data_loader import prepare_data
from models.train import train_model
from models.evaluate import evaluate_model
from utils.logger import setup_logger

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Alzheimer's Detection Model Training")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--model_type', type=str, choices=['2d_vit', '3d_vit', '3d_cnn'],
                        help='Type of model to use (overrides config file)')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    logger = setup_logger('alzheimer_detection', Path(config['paths']['log_dir']))

    # Set device
    device = torch.device(config['training']['device'])
    logger.info(f"Using device: {device}")

    # Set model type
    model_type = args.model_type if args.model_type else config['model']['type']
    logger.info(f"Using model type: {model_type}")

    # Prepare data loaders
    try:
        train_loader, val_loader, test_loader = prepare_data(
            data_root=config['dataset']['path'],
            model_type=model_type,
            batch_size=config['dataset']['batch_size'],
            val_ratio=config['dataset']['val_ratio'],
            test_ratio=config['dataset']['test_ratio'],
            spatial_size=(config['dataset']['input_size'],) * 3  # Convert to 3D tuple
        )
        logger.info("Data loaders created successfully")
    except Exception as e:
        logger.error(f"Error creating data loaders: {str(e)}")
        raise

    # Create model
    try:
        model = create_model(
            model_type=model_type,
            num_labels=config['model']['num_labels'],
            freeze_layers=config['model']['freeze_layers'],
            input_size=config['model']['input_size'],
            patch_size=config['model']['patch_size']
        )
        model = model.to(device)
        logger.info(f"Model created successfully and moved to {device}")
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        raise

    # Train model
    try:
        train_model(model, train_loader, val_loader, config['training'])
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

    # Evaluate model
    try:
        results = evaluate_model(model, test_loader, device)
        logger.info(f"Evaluation results: {results}")
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise
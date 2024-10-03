import argparse
import yaml
from models.architectures import create_vit_2d, create_vit_3d, create_cnn_3d
from data.data_loader import prepare_data
from models.train import train_model
from models.evaluate import evaluate_model
from utils.logger import setup_logger


def load_config(config_path):
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

    # Set up logger
    logger = setup_logger('alzheimer_detection', config['paths']['log_dir'])

    # Determine model type (command line argument takes precedence over config file)
    model_type = args.model_type if args.model_type else config['model']['type']
    logger.info(f"Using model type: {model_type}")

    # Prepare data
    train_loader, val_loader, test_loader = prepare_data(
        config['dataset']['name'],
        model_type,
        config['dataset']['batch_size']
    )

    # Create model
    num_labels = config['model']['num_labels']
    if model_type == '2d_vit':
        model = create_vit_2d(num_labels)
    elif model_type == '3d_vit':
        model = create_vit_3d(num_labels)
    elif model_type == '3d_cnn':
        model = create_cnn_3d(num_labels)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train model
    train_model(model, train_loader, val_loader,
                config['training']['max_epochs'],
                config['training']['learning_rate'])

    # Evaluate model
    results = evaluate_model(model, test_loader)
    logger.info(f"Evaluation results: {results}")


if __name__ == "__main__":
    main()
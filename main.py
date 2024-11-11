"""
Main script for Alzheimer's detection using Vision Transformers.
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
from models.architectures import create_model
from data.data_loader import create_data_loaders

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


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


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


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return val_loss / len(val_loader), 100. * correct / total


def train_model(model, train_loader, val_loader, config, device, exp_dir):
    """Training loop."""
    try:
        # Initialize optimizer and criterion
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['optimizer']['weight_decay']
        )
        criterion = torch.nn.CrossEntropyLoss()
        
        # Initialize tracking variables
        best_val_acc = 0.0
        best_val_loss = float('inf')
        val_losses = []
        checkpoint_dir = exp_dir / 'checkpoints'
        
        # Training loop
        for epoch in range(config['training']['epochs']):
            # Training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Get data
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Log progress
                if batch_idx % 5 == 0:
                    logger.info(
                        f'Epoch: {epoch+1}/{config["training"]["epochs"]}, '
                        f'Batch: {batch_idx}/{len(train_loader)}, '
                        f'Loss: {loss.item():.4f}, '
                        f'Acc: {100.*correct/total:.2f}%'
                    )
            
            # Compute epoch statistics
            train_loss = running_loss / len(train_loader)
            train_acc = 100. * correct / total
            
            # Validation phase
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            
            # Log epoch results
            logger.info(
                f'\nEpoch {epoch+1}/{config["training"]["epochs"]}:\n'
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%\n'
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n'
            )
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'config': config
            }
            
            # Save latest checkpoint
            torch.save(
                checkpoint,
                checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
            )
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                torch.save(
                    checkpoint,
                    checkpoint_dir / 'best_model.pt'
                )
                logger.info(f'New best model saved with validation accuracy: {val_acc:.2f}%')
            
            # Early stopping
            if config['training'].get('early_stopping', {}).get('enable', False):
                patience = config['training']['early_stopping']['patience']
                min_delta = config['training']['early_stopping']['min_delta']
                if (epoch > patience and 
                    val_loss > min(val_losses[-patience:]) - min_delta):
                    logger.info(f'Early stopping triggered at epoch {epoch+1}')
                    break
                    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def main():
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
        model = model.to(device)
        logger.info(f"Model moved to {device} successfully")

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
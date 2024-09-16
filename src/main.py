# main.py
import logging
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from data.data_loader import create_dataloaders

from models.architectures.vit.vit3d_m8 import ViT3DModel_M8
from models.architectures.vit.vit3d_b16 import ViT3DModel_B16

from models.architectures.cnn.cnn3d import CNN3DModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Load configuration
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    data_config = config['data']
    train_config = config['train']
    model_config = config['model']

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create data loaders
    dataloaders = create_dataloaders(
        data_dir=data_config['data_dir'],
        json_path=data_config['json_path'],
        train_batch_size=train_config['batch_size'],
        val_batch_size=train_config['val_batch_size'],
        num_workers=train_config['num_workers'],
    )

    # Choose the model to train
    model_type = train_config['model_type']

    if model_type == 'vit_m8':
        model = ViT3DModel_M8(config)
    elif model_type == 'vit_b16':
        model = ViT3DModel_B16(config)
    elif model_type == 'cnn':
        model = CNN3DModel(config)
    else:
        raise ValueError("Invalid model choice")

    # Initialize the TensorBoard logger
    logger_tb = TensorBoardLogger("lightning_logs", name=train_config['experiment_name'])

    # Initialize the Trainer
    trainer = pl.Trainer(
        max_epochs=train_config['num_epochs'],
        logger=logger_tb,
        gpus=1 if torch.cuda.is_available() else 0,
    )

    # Train the model
    trainer.fit(model, dataloaders['train'], dataloaders['val'])

# models.py
import pytorch_lightning as pl
import torch
import torch.nn as nn
from monai.networks.nets import ViT


class ViT3DLarge32(pl.LightningModule):
    def __init__(self, config):
        super(ViT3DLarge32, self).__init__()
        # Extract model configurations from the config dictionary
        model_config = config['model']
        vit_config = model_config['vit_m8']

        # Define the ViT model with specific parameters for M8
        self.model = ViT(
            in_channels=model_config['in_channels'],
            img_size=tuple(model_config['img_size']),  # For example, (128, 128, 128)
            patch_size=tuple(vit_config['patch_size']),  # For example, (16, 16, 16)
            hidden_size=vit_config['hidden_size'],  # For example, 512
            mlp_dim=vit_config['mlp_dim'],  # For example, 2048
            num_layers=vit_config['num_layers'],  # For example, 8
            num_heads=vit_config['num_heads'],  # For example, 8
            classification=True,
            num_classes=model_config['num_classes'],
            dropout_rate=vit_config.get('dropout_rate', 0.1),
        )

        # Load pretrained weights if available
        pretrained_weights = vit_config.get('pretrained_weights')
        if pretrained_weights:
            state_dict = torch.load(pretrained_weights)
            self.model.load_state_dict(state_dict)

        # Define loss function
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = config['train']['learning_rate']

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs = batch['image']
        labels = batch['label'].long().squeeze()
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch['image']
        labels = batch['label'].long().squeeze()
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

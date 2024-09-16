import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from timm.models.vision_transformer import VisionTransformer
# Import both models
from src.models.architectures.vit3d_b16 import ViT3DB16
from src.models.architectures.vit3d_m8 import ViT3DM8
from src.data.preprocess import get_preprocessing_transforms

class ViT3DLightning(pl.LightningModule):
    def __init__(self, model_name='vit_b16', learning_rate=1e-4, num_classes=2, use_pretrained=True):
        super(ViT3DLightning, self).__init__()
        self.save_hyperparameters()

        # Initialize the model based on model_name
        if model_name == 'vit_b16':
            self.model = ViT3DB16(num_classes=num_classes)
        elif model_name == 'vit_m8':
            self.model = ViT3DM8(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        if use_pretrained:
            # Transfer learning from 2D pre-trained ViT
            vit_pretrained = VisionTransformer.from_pretrained('vit_base_patch16_224')
            # Load the pre-trained transformer weights into the model's transformer
            self.model.transformer.load_state_dict(vit_pretrained.state_dict(), strict=False)
            print("Loaded pre-trained weights for the transformer!")

        self.criterion = torch.nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        val_loss = self.criterion(outputs, labels)
        self.log('val_loss', val_loss)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

# DataLoader (replace [...] with your actual dataset)
train_transforms = get_preprocessing_transforms()
train_dataset = [...]  # Your dataset with preprocessing applied
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Validation loader (replace [...] with actual validation dataset)
val_dataset = [...]
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Initialize the Lightning model (choose the model you want to train: 'vit_b16' or 'vit_m8')
model = ViT3DLightning(model_name='vit_b16', learning_rate=1e-4, num_classes=2, use_pretrained=True)

# Add checkpointing callback
checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min')

# Initialize the PyTorch Lightning trainer
trainer = pl.Trainer(max_epochs=10, callbacks=[checkpoint_callback], gpus=1)  # Set gpus=1 if using GPU

# Train the model
trainer.fit(model, train_loader, val_loader)

import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from monai.data import Dataset, CacheDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    ScaleIntensityd,
    Resized,
    RandRotate90d,
    RandFlipd,
    ToTensord,
)
from monai.networks.nets import ViT


class AlzheimerDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Define preprocessing and augmentation transforms
        self.train_transforms = Compose([
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            ScaleIntensityd(keys=["image"]),
            Resized(keys=["image"], spatial_size=(128, 128, 128)),
            RandRotate90d(keys=["image"], prob=0.8, spatial_axes=[0, 2]),
            RandFlipd(keys=["image"], spatial_axis=0, prob=0.5),
            ToTensord(keys=["image", "label"])
        ])

        self.val_transforms = Compose([
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            ScaleIntensityd(keys=["image"]),
            Resized(keys=["image"], spatial_size=(128, 128, 128)),
            ToTensord(keys=["image", "label"])
        ])

        # Create dataset
        train_files = [{"image": os.path.join(self.data_dir, "train", f),
                        "label": label} for f, label in train_data]
        val_files = [{"image": os.path.join(self.data_dir, "val", f),
                      "label": label} for f, label in val_data]

        self.train_ds = CacheDataset(data=train_files, transform=self.train_transforms)
        self.val_ds = CacheDataset(data=val_files, transform=self.val_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          num_workers=self.num_workers)


class ViT3DModule(pl.LightningModule):
    def __init__(self, num_classes=3, lr=1e-4):
        super().__init__()
        self.model = ViT(
            in_channels=1,
            img_size=(128, 128, 128),
            patch_size=(16, 16, 16),
            num_classes=num_classes,
            pos_embed='conv'
        )
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }


def main():
    # Dataset information
    data_dir = "/path/to/your/data"
    num_classes = 3  # Adjust based on your classification task

    # Create data module
    data_module = AlzheimerDataModule(data_dir, batch_size=8)

    # Create model
    model = ViT3DModule(num_classes=num_classes)

    # Create logger
    logger = TensorBoardLogger("lightning_logs", name="alzheimer_vit")

    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='alzheimer_vit-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss'
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=100,
        gpus=1 if torch.cuda.is_available() else 0,
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    # Train model
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
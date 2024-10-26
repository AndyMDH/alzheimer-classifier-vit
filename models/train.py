"""
Training module for Alzheimer's detection models.
"""

import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from monai.engines import SupervisedTrainer
from monai.handlers import StatsHandler, TensorBoardStatsHandler, CheckpointSaver
from monai.utils import set_determinism


def train_model(model, train_loader, val_loader, config):
    """Train the model using MONAI's SupervisedTrainer."""
    device = torch.device(config['device'])
    model = model.to(device)

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, **config['lr_scheduler'])
    early_stopper = EarlyStopper(**config['early_stopping'])

    best_val_loss = float('inf')

    for epoch in range(config['max_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        # Update learning rate
        scheduler.step(val_loss)

        # Early stopping
        if early_stopper.early_stop(val_loss):
            print("Early stopping triggered")
            break

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{config['checkpoint_dir']}/best_model.pt")

        print(f"Epoch {epoch + 1}/{config['max_epochs']}, "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
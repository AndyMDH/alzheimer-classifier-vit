"""
Training module for Alzheimer's detection models.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class AverageMeter:
    """Compute and store the average and current value."""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class EarlyStopping:
    """Early stopping handler."""
    def __init__(self, patience: int = 7, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop

def save_checkpoint(state: Dict[str, Any], is_best: bool, exp_dir: Path) -> None:
    """Save model checkpoint."""
    checkpoint_path = exp_dir / 'checkpoints' / 'last_checkpoint.pt'
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = exp_dir / 'checkpoints' / 'best_model.pt'
        torch.save(state, best_path)
        logger.info(f"Saved best model checkpoint to {best_path}")

def plot_training_progress(train_losses: list, val_losses: list, exp_dir: Path) -> None:
    """Plot and save training progress."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(exp_dir / 'results' / 'training_progress.png')
    plt.close()

def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> float:
    """Train for one epoch."""
    model.train()
    losses = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}', leave=False)
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update statistics
        losses.update(loss.item(), images.size(0))
        pbar.set_postfix({'train_loss': f'{losses.avg:.4f}'})
    
    return losses.avg

@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Validate the model."""
    model.eval()
    losses = AverageMeter()
    correct = 0
    total = 0
    
    for batch in tqdm(val_loader, desc='Validating', leave=False):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Update statistics
        losses.update(loss.item(), images.size(0))
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    logger.info(f'Validation Accuracy: {accuracy:.2f}%')
    return losses.avg

def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    exp_dir: Path
) -> None:
    """Train the model."""
    # Setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['optimizer']['weight_decay']
    )
    
    # Setup scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config['training']['scheduler']['T_0'],
        eta_min=config['training']['scheduler'].get('min_lr', 1e-6)
    )
    
    # Setup early stopping
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping']['patience'],
        min_delta=config['training']['early_stopping']['min_delta']
    )
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(config['training']['epochs']):
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch
        )
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step()
        
        # Log progress
        logger.info(
            f'Epoch {epoch + 1}/{config["training"]["epochs"]} - '
            f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
            f'LR: {optimizer.param_groups[0]["lr"]:.6f}'
        )
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'config': config
        }, is_best, exp_dir)
        
        # Early stopping
        if early_stopping(val_loss):
            logger.info(f'Early stopping triggered after epoch {epoch + 1}')
            break
    
    # Plot training progress
    plot_training_progress(train_losses, val_losses, exp_dir)
    logger.info('Training completed')
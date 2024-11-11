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

def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> float:
    """Train for one epoch."""
    model.train()
    losses = AverageMeter()
    
    pbar = tqdm(loader, desc=f'Epoch {epoch + 1}')
    for batch_idx, batch in enumerate(pbar):
        # Get data
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
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Validate the model."""
    model.eval()
    losses = AverageMeter()
    
    for batch in tqdm(loader, desc='Validating'):
        # Get data
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Update statistics
        losses.update(loss.item(), images.size(0))
    
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
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(config['training']['epochs']):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
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
        
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'config': config
        }
        
        if is_best:
            torch.save(checkpoint, exp_dir / 'checkpoints' / 'best_model.pt')
            logger.info(f'Saved best model with val_loss: {best_val_loss:.4f}')
        
        torch.save(checkpoint, exp_dir / 'checkpoints' / 'last_model.pt')
        
        # Plot progress
        plot_training_progress(
            train_losses, 
            val_losses,
            exp_dir / 'results' / 'training_progress.png'
        )
        
    logger.info('Training completed')

def plot_training_progress(
    train_losses: list,
    val_losses: list,
    save_path: Path
) -> None:
    """Plot and save training progress."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

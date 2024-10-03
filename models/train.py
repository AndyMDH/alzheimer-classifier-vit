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

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def early_stop(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train_model(model, train_loader, val_loader, config):
    """Train the model using MONAI's SupervisedTrainer."""
    set_determinism(seed=config['seed'])

    loss_function = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['lr_scheduler']['factor'],
                                  patience=config['lr_scheduler']['patience'])
    early_stopper = EarlyStopper(patience=config['early_stopping']['patience'],
                                 min_delta=config['early_stopping']['min_delta'])

    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()
        inputs, targets = batch["image"], batch["label"]
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}

    def validate_step(engine, batch):
        model.eval()
        with torch.no_grad():
            inputs, targets = batch["image"], batch["label"]
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
        return {"val_loss": loss.item()}

    trainer = SupervisedTrainer(
        device=torch.device(config['device']),
        max_epochs=config['max_epochs'],
        train_data_loader=train_loader,
        network=model,
        optimizer=optimizer,
        loss_function=loss_function,
        prepare_batch=lambda batch: (batch["image"], batch["label"]),
        train_step=train_step,
        validation_data_loader=val_loader,
        validation_step=validate_step,
        amp=config.get('amp', True)
    )

    @trainer.on(SupervisedTrainer.EPOCH_COMPLETED)
    def update_lr_and_check_early_stop(engine):
        metrics = engine.state.metrics
        val_loss = metrics.get("val_loss", None)
        if val_loss is not None:
            scheduler.step(val_loss)
            if early_stopper.early_stop(val_loss):
                print("Early stopping triggered")
                engine.terminate()

    stats_handler = StatsHandler(output_transform=lambda x: None)
    tb_stats_handler = TensorBoardStatsHandler(log_dir=config['tensorboard_log_dir'])
    checkpoint_handler = CheckpointSaver(save_dir=config['checkpoint_dir'],
                                         save_dict={"model": model},
                                         save_interval=1, n_saved=3)

    trainer.add_event_handler(trainer.EPOCH_COMPLETED, stats_handler)
    trainer.add_event_handler(trainer.EPOCH_COMPLETED, tb_stats_handler)
    trainer.add_event_handler(trainer.EPOCH_COMPLETED, checkpoint_handler)

    trainer.run()
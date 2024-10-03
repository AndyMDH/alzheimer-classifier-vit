"""
Training module for Alzheimer's detection models.
"""

from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from monai.engines import SupervisedTrainer
from monai.handlers import StatsHandler, TensorBoardStatsHandler, CheckpointSaver

def train_model(model, train_loader, val_loader, config):
    """Train the model using MONAI's SupervisedTrainer."""
    loss_function = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])

    trainer = SupervisedTrainer(
        device=config['device'],
        max_epochs=config['max_epochs'],
        train_data_loader=train_loader,
        val_data_loader=val_loader,
        network=model,
        optimizer=optimizer,
        loss_function=loss_function,
        inferer=None,
        amp=True,
    )

    val_interval = 1
    stats_handler = StatsHandler(output_transform=lambda x: None)
    tb_stats_handler = TensorBoardStatsHandler(log_dir=config['tensorboard_log_dir'])
    checkpoint_handler = CheckpointSaver(save_dir=config['checkpoint_dir'],
                                         save_dict={"model": model},
                                         save_interval=1, n_saved=3)

    trainer.add_event_handler(event_name=trainer.EVENT_EPOCH, handler=stats_handler)
    trainer.add_event_handler(event_name=trainer.EVENT_EPOCH, handler=tb_stats_handler)
    trainer.add_event_handler(event_name=trainer.EVENT_EPOCH, handler=checkpoint_handler)

    trainer.run()
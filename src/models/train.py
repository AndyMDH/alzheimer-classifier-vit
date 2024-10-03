from monai.engines import SupervisedTrainer
from monai.handlers import StatsHandler, TensorBoardStatsHandler, CheckpointSaver
from monai.losses import DiceLoss

def train_model(model, train_loader, val_loader, max_epochs, lr, device):
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr)

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=max_epochs,
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
    tb_stats_handler = TensorBoardStatsHandler(log_dir="./runs")
    checkpoint_handler = CheckpointSaver(save_dir="./checkpoints", save_dict={"model": model}, save_interval=1, n_saved=10)

    trainer.add_event_handler(event_name=trainer.EVENT_EPOCH, handler=stats_handler)
    trainer.add_event_handler(event_name=trainer.EVENT_EPOCH, handler=tb_stats_handler)
    trainer.add_event_handler(event_name=trainer.EVENT_EPOCH, handler=checkpoint_handler)

    trainer.run()
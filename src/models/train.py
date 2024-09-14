import pytorch_lightning as pl
from src.models.architectures.vit3d_b16 import ViT3D_B16
from utils.data_module import AlzheimerDataModule
from src.utils.train_utils import get_callbacks, get_logger


def train(config):
    model = ViT3D_B16(num_classes=config.num_classes)
    data_module = AlzheimerDataModule(config.data_dir, config.batch_size)

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        gpus=config.gpus,
        logger=get_logger(config),
        callbacks=get_callbacks(config)
    )

    trainer.fit(model, data_module)
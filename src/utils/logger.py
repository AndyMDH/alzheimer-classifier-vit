# src/utils/logger.py
from torch.utils.tensorboard import SummaryWriter


def get_tensorboard_logger(log_dir="logs"):
    """
    Initializes a TensorBoard SummaryWriter to log metrics and losses.

    Args:
        log_dir (str): Directory where logs will be saved.

    Returns:
        SummaryWriter object
    """
    writer = SummaryWriter(log_dir)
    return writer

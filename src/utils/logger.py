# src/utils/logger.py
from torch.utils.tensorboard import SummaryWriter

def get_logger(log_dir="logs"):
    writer = SummaryWriter(log_dir)
    return writer

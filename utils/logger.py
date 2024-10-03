"""
Logging utility for Alzheimer's detection project.
"""

import logging
from pathlib import Path


def setup_logger(name: str, log_file: Path, level=logging.INFO) -> logging.Logger:
    """Set up logger with file and console handlers."""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
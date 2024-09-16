import logging
from typing import Callable

from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotate90d,
    RandZoomd,
    Rand3DElasticd,
    RandAdjustContrastd,
)

# Configure logging
logger = logging.getLogger(__name__)


def get_augmentation_transforms() -> Callable:
    """
    Returns the augmentation transforms to be applied to the training data.

    Returns:
        Callable: A composed transform function.
    """
    logger.info("Creating augmentation transforms.")
    transforms = Compose([
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=[0, 1, 2]),
        RandRotate90d(keys=['image', 'label'], prob=0.5, max_k=3),
        RandZoomd(keys=['image', 'label'], prob=0.5, min_zoom=0.9, max_zoom=1.1),
        Rand3DElasticd(
            keys=['image', 'label'],
            prob=0.5,
            sigma_range=(5, 8),
            magnitude_range=(100, 200),
            mode=('bilinear', 'nearest'),
            padding_mode='zeros',
        ),
        RandAdjustContrastd(keys=['image'], prob=0.5, gamma=(0.9, 1.1)),
    ])
    return transforms

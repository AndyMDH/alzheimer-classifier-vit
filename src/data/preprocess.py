import logging
from typing import Callable

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityd,
    ResizeWithPadOrCropd,
    ToTensord,
)

# Configure logging
logger = logging.getLogger(__name__)


def get_preprocessing_transforms() -> Callable:
    """
    Returns the preprocessing transforms to be applied to the data.

    Returns:
        Callable: A composed transform function.
    """
    logger.info("Creating preprocessing transforms.")
    transforms = Compose([
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label']),  # Ensures channel-first format
        Orientationd(keys=['image', 'label'], axcodes='RAS'),  # Standardizes orientation
        Spacingd(
            keys=['image', 'label'],
            pixdim=(1.0, 1.0, 1.0),
            mode=('bilinear', 'nearest'),  # 'bilinear' for images, 'nearest' for labels
        ),
        ScaleIntensityd(keys=['image']),  # Normalizes image intensities to [0, 1]
        ResizeWithPadOrCropd(keys=['image', 'label'], spatial_size=(128, 128, 128)),
        ToTensord(keys=['image', 'label']),
    ])
    return transforms

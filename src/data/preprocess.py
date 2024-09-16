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
        EnsureChannelFirstd(keys=['image', 'label']),  # Handles images with or without a channel dimension
        Orientationd(keys=['image', 'label'], axcodes='RAS'),  # Ensure consistent orientation
        Spacingd(
            keys=['image', 'label'],
            pixdim=(1.0, 1.0, 1.0),
            mode=('bilinear', 'nearest'),  # 'nearest' for labels to avoid interpolation artifacts
        ),
        ScaleIntensityd(keys=['image']),  # Normalize intensity to [0, 1]
        ResizeWithPadOrCropd(keys=['image', 'label'], spatial_size=(128, 128, 128)),
        # Uncomment the following line if you want to standardize intensity
        # NormalizeIntensityd(keys=['image']),  # Standardize intensity (zero mean, unit variance)
        ToTensord(keys=['image', 'label']),
    ])
    return transforms

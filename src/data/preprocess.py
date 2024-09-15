import monai.transforms as T
from monai.transforms import Compose, Resize, NormalizeIntensity, RandFlip, RandRotate90, ToTensor


def get_preprocessing_transforms(target_size=(128, 128, 128)):
    """
    Define preprocessing transforms for 3D data.

    Args:
        target_size (tuple): Target size to resize the 3D images.

    Returns:
        A composed list of transforms.
    """
    return Compose([
        Resize(spatial_size=target_size),
        NormalizeIntensity(nonzero=True),  # Normalize intensity for non-zero pixels
        RandFlip(prob=0.5, spatial_axis=(0, 1, 2)),  # Random flip along each axis
        RandRotate90(prob=0.5, spatial_axes=(0, 1)),  # Randomly rotate 90 degrees
        ToTensor()  # Convert to PyTorch tensor
    ])

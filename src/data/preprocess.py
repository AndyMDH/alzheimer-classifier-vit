import monai.transforms as T
from monai.transforms import Compose, RandFlip, RandRotate90, NormalizeIntensity, Resize, ToTensor


def preprocess_3d_image(image, target_size=(128, 128, 128)):
    """
    Preprocess a 3D image, including resizing, normalization, and augmentation.

    Args:
        image (numpy array or tensor): The 3D image to preprocess.
        target_size (tuple): Desired output size for the image.

    Returns:
        Processed tensor.
    """
    # Define a series of transforms
    transform = Compose([
        Resize(spatial_size=target_size),  # Resize to a fixed 3D volume size
        NormalizeIntensity(nonzero=True),  # Normalize intensity to 0 mean, 1 std dev
        RandFlip(spatial_axis=(0, 1, 2)),  # Random flipping along all three axes
        RandRotate90(prob=0.5, spatial_axes=(0, 1)),  # Random 90 degree rotation
        ToTensor()  # Convert to PyTorch tensor
    ])

    return transform(image)

# Example usage with a 3D image (assuming `image` is loaded elsewhere)
# processed_image = preprocess_3d_image(image)

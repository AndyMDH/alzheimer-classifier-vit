"""
Data loading and preprocessing module for Alzheimer's detection project.
"""

from pathlib import Path
from typing import List, Tuple
import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split
from monai.data import Dataset, DataLoader, CacheDataset
from monai.transforms import (
    Compose, AddChannel, ScaleIntensity, Resize, RandRotate90, RandFlip,
    RandGaussianNoise, RandAdjustContrast, NormalizeIntensity, ThresholdIntensity
)


class ADNIDataset(Dataset):
    def __init__(self, file_list: List[Path], transform: Compose = None, model_type: str = '3d'):
        super().__init__()
        self.file_list = file_list
        self.transform = transform
        self.model_type = model_type

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        label = file_path.parent.parent.name  # Assuming directory name is the label

        try:
            adni = nib.load(str(file_path))
            image = adni.get_fdata()

            if self.model_type == '2d':
                middle_slice_idx = image.shape[2] // 2
                image = image[:, :, middle_slice_idx]
                image = np.expand_dims(image, axis=0)

            if self.transform:
                image = self.transform(image)

            return {"image": image, "label": label}
        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            return None


def create_monai_dataset(file_list: List[Path], transforms: Compose, model_type: str,
                         cache_rate: float = 0.1) -> Dataset:
    """Create a MONAI dataset from a list of file paths."""
    return CacheDataset(data=[{"image": f} for f in file_list], transform=transforms, cache_rate=cache_rate)


def create_data_loaders(dataset: Dataset, batch_size: int, shuffle: bool = False,
                        num_workers: int = 4) -> DataLoader:
    """Create DataLoader with given dataset and batch size."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_transforms(model_type: str, spatial_size: List[int] = [224, 224, 224]) -> Compose:
    """Get MONAI transforms based on model type."""
    common_transforms = [
        AddChannel(),
        ScaleIntensity(),
        NormalizeIntensity(nonzero=True),
        ThresholdIntensity(threshold=0, above=True),  # Basic skull stripping
        RandGaussianNoise(prob=0.2, mean=0.0, std=0.1),
        RandAdjustContrast(prob=0.2)
    ]

    if model_type == '2d_vit':
        return Compose(common_transforms + [
            Resize(spatial_size[:2]),
            RandRotate90(prob=0.5, spatial_axes=[0, 1]),
            RandFlip(prob=0.5, spatial_axis=1),
        ])
    else:  # 3D transforms for 3D ViT and 3D CNN
        return Compose(common_transforms + [
            Resize(spatial_size),
            RandRotate90(prob=0.5, spatial_axes=[0, 1]),
            RandFlip(prob=0.5, spatial_axis=0),
        ])


def prepare_data(data_dir: str, model_type: str, batch_size: int,
                 val_ratio: float = 0.15, test_ratio: float = 0.15,
                 input_size: int = 224) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Prepare datasets and dataloaders with automatic splitting."""
    transforms = get_transforms(model_type, spatial_size=[input_size, input_size, input_size])

    # Get all ADNI files
    data_dir = Path(data_dir)
    all_files = list(data_dir.rglob('*.nii'))

    # Split the data
    train_files, test_files = train_test_split(all_files, test_size=test_ratio, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=val_ratio / (1 - test_ratio), random_state=42)

    # Create datasets
    train_ds = create_monai_dataset(train_files, transforms, model_type)
    val_ds = create_monai_dataset(val_files, transforms, model_type)
    test_ds = create_monai_dataset(test_files, transforms, model_type)

    # Create data loaders
    train_loader = create_data_loaders(train_ds, batch_size, shuffle=True)
    val_loader = create_data_loaders(val_ds, batch_size)
    test_loader = create_data_loaders(test_ds, batch_size)

    return train_loader, val_loader, test_loader

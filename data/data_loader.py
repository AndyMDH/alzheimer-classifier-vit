"""
Data loading and preprocessing module for Alzheimer's detection project.
"""

from pathlib import Path
from typing import Tuple
import nibabel as nib
import numpy as np
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose, AddChannel, ScaleIntensity, Resize, RandRotate90, RandFlip
)

class NIfTIDataset(Dataset):
    def __init__(self, data_dir: str, transform: Compose = None, model_type: str = '3d'):
        self.data_dir = Path(data_dir)
        self.file_list = list(self.data_dir.rglob('*.nii.gz'))
        self.transform = transform
        self.model_type = model_type

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        label = file_path.parent.name  # Assuming directory name is the label

        nifti = nib.load(str(file_path))
        image = nifti.get_fdata()

        if self.model_type == '2d':
            middle_slice_idx = image.shape[2] // 2
            image = image[:, :, middle_slice_idx]
            image = np.expand_dims(image, axis=0)

        if self.transform:
            image = self.transform(image)

        return {"image": image, "label": label}

def create_monai_dataset(data_dir: str, transforms: Compose, model_type: str) -> Dataset:
    """Create a MONAI dataset from local NIfTI files."""
    return NIfTIDataset(data_dir, transforms, model_type)

def create_data_loaders(dataset: Dataset, batch_size: int, num_workers: int = 4) -> DataLoader:
    """Create DataLoader with given dataset and batch size."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

def get_transforms(model_type: str, spatial_size: Tuple[int, int, int] = (224, 224, 224)) -> Compose:
    """Get MONAI transforms based on model type."""
    if model_type == '2d_vit':
        return Compose([
            AddChannel(),
            ScaleIntensity(),
            Resize(spatial_size[:2]),
            RandRotate90(prob=0.5, spatial_axes=[0, 1]),
            RandFlip(prob=0.5, spatial_axis=1),
        ])
    else:  # 3D transforms for 3D ViT and 3D CNN
        return Compose([
            AddChannel(),
            ScaleIntensity(),
            Resize(spatial_size),
            RandRotate90(prob=0.5, spatial_axes=[0, 1]),
            RandFlip(prob=0.5, spatial_axis=0),
        ])

def prepare_data(data_dir: str, model_type: str, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Prepare datasets and dataloaders."""
    transforms = get_transforms(model_type)

    train_ds = create_monai_dataset(Path(data_dir) / 'train', transforms, model_type)
    val_ds = create_monai_dataset(Path(data_dir) / 'val', transforms, model_type)
    test_ds = create_monai_dataset(Path(data_dir) / 'test', transforms, model_type)

    train_loader = create_data_loaders(train_ds, batch_size)
    val_loader = create_data_loaders(val_ds, batch_size)
    test_loader = create_data_loaders(test_ds, batch_size)

    return train_loader, val_loader, test_loader
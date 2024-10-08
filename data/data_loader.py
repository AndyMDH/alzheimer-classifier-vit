import os
from pathlib import Path
from typing import Tuple
import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split
from monai.transforms.compose import Compose
from monai.transforms import AddChannel
from monai.transforms.intensity.array import (
    ScaleIntensity, NormalizeIntensity, ThresholdIntensity,
    RandGaussianNoise, RandAdjustContrast
)
from monai.transforms.spatial.array import Resize, RandRotate90, RandFlip
from monai.data import Dataset
from torch.utils.data import DataLoader, Subset


class ADNIDataset(Dataset):
    def __init__(self, data_root: str, transform: Compose = None, model_type: str = '3d'):
        super().__init__()
        self.data_root = Path(data_root)
        self.transform = transform
        self.model_type = model_type
        self.file_list = self._create_file_list()

    def _create_file_list(self):
        file_list = []
        for label in ['AD', 'MCI', 'CN']:
            label_dir = self.data_root / 'raw' / label
            for file_name in os.listdir(label_dir):
                if file_name.endswith('.nii'):
                    file_list.append((label_dir / file_name, label))
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path, label = self.file_list[idx]
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


def get_transforms(model_type: str, spatial_size: Tuple[int, int, int] = (224, 224, 224)) -> Compose:
    common_transforms = [
        AddChannel(),
        ScaleIntensity(),
        NormalizeIntensity(nonzero=True),
        ThresholdIntensity(threshold=0.0, above=True),  # Basic skull stripping
        RandGaussianNoise(prob=0.2, mean=0.0, std=0.1),
        RandAdjustContrast(prob=0.2)
    ]

    if model_type == '2d':
        return Compose(common_transforms + [
            Resize(spatial_size[:2]),
            RandRotate90(prob=0.5, spatial_axes=(0, 1)),
            RandFlip(prob=0.5, spatial_axis=1),
        ])
    else:  # 3D transforms
        return Compose(common_transforms + [
            Resize(spatial_size),
            RandRotate90(prob=0.5, spatial_axes=(0, 1)),
            RandFlip(prob=0.5, spatial_axis=0),
        ])


def create_data_loader(dataset: Dataset, batch_size: int, shuffle: bool = True,
                       num_workers: int = 4) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def prepare_data(data_root: str, model_type: str, batch_size: int,
                 val_ratio: float = 0.15, test_ratio: float = 0.15,
                 spatial_size: Tuple[int, int, int] = (224, 224, 224)) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transforms = get_transforms(model_type, spatial_size=spatial_size)

    full_dataset = ADNIDataset(data_root, transform=transforms, model_type=model_type)

    # Split the data
    train_val_indices, test_indices = train_test_split(
        range(len(full_dataset)), test_size=test_ratio, random_state=42,
        stratify=[item[1] for item in full_dataset.file_list]
    )
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_ratio / (1 - test_ratio), random_state=42,
        stratify=[full_dataset.file_list[i][1] for i in train_val_indices]
    )

    # Create subset datasets from the previously split data
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # Create data loaders
    train_loader = create_data_loader(train_dataset, batch_size, shuffle=True)
    val_loader = create_data_loader(val_dataset, batch_size, shuffle=False)
    test_loader = create_data_loader(test_dataset, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

"""
Data loading and preprocessing module for Alzheimer's detection.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import yaml
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import nibabel as nib
import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    CropForegroundd,
    ScaleIntensityd,
    NormalizeIntensityd,
    ResizeWithPadOrCropd,
    RandRotate90d,
    RandFlipd,
    RandAffined,
    RandGaussianNoised,
    EnsureTyped,
    ToTensord,
    Lambda
)
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ADNIDataset(Dataset):
    """Dataset for loading ADNI data in both 2D and 3D modes."""

    def __init__(
        self,
        config: Dict[str, Any],
        transform: Optional[Compose] = None,
        split: str = 'train',
        mode: str = '3d'
    ):
        self.data_root = Path(config['dataset']['path'])
        self.transform = transform
        self.split = split
        self.mode = mode
        self.file_list = self._create_file_list()
        self.label_to_idx = {'AD': 0, 'CN': 1, 'MCI': 2}

        logger.info(f"Initialized {split} dataset with {len(self.file_list)} samples in {mode} mode")

    def _create_file_list(self) -> List[Tuple[Path, str]]:
        """Create list of file paths and labels."""
        file_list = []
        raw_dir = self.data_root / 'raw'

        for label in ['AD', 'CN', 'MCI']:
            label_dir = raw_dir / label
            if not label_dir.exists():
                logger.warning(f"Directory not found: {label_dir}")
                continue

            for file_path in label_dir.glob('*.nii*'):
                file_list.append((file_path, label))

        if not file_list:
            raise RuntimeError(f"No .nii or .nii.gz files found in {raw_dir}")
        return file_list

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset."""
        file_path, label = self.file_list[idx]

        try:
            data_dict = {
                'image': str(file_path),
                'label': self.label_to_idx[label]
            }

            if self.transform:
                data_dict = self.transform(data_dict)

            return data_dict

        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise

def get_transforms(config: Dict[str, Any], split: str, mode: str = '3d') -> Compose:
    """Get transforms based on configuration and mode."""
    # Set spatial size based on mode
    if mode == '2d':
        spatial_size = (config['dataset']['input_size'],) * 2
    else:
        spatial_size = (config['dataset']['input_size'],) * 3

    common_transforms = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=config['dataset']['preprocessing']['voxel_spacing'],
            mode="bilinear"
        ),
        CropForegroundd(
            keys=["image"],
            source_key="image",
            margin=config['dataset']['preprocessing']['crop_margin']
        )
    ]

    # Add mode-specific resize
    if mode == '2d':
        resize_transform = [
            # Extract center slice for 2D
            Lambda(lambda x: {
                'image': x['image'][:, x['image'].shape[1]//2, :, :]
                if len(x['image'].shape) == 4
                else x['image'],
                'label': x['label']
            }),
            ResizeWithPadOrCropd(
                keys=["image"],
                spatial_size=spatial_size
            )
        ]
    else:
        resize_transform = [
            ResizeWithPadOrCropd(
                keys=["image"],
                spatial_size=spatial_size
            )
        ]

    # Add remaining transforms
    post_transforms = [
        ScaleIntensityd(keys=["image"]),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        EnsureTyped(keys=["image", "label"]),
        ToTensord(keys=["image", "label"])
    ]

    # Combine transforms
    transforms = common_transforms + resize_transform + post_transforms

    # Add augmentation for training
    if split == 'train':
        if mode == '2d':
            augmentation = [
                RandRotate90d(
                    keys=["image"],
                    prob=0.5,
                    spatial_axes=(0, 1)
                ),
                RandFlipd(
                    keys=["image"],
                    prob=0.5,
                    spatial_axis=0
                ),
                RandAffined(
                    keys=["image"],
                    prob=0.5,
                    rotate_range=[np.pi/12] * 2,
                    scale_range=[0.1] * 2,
                    mode="bilinear"
                )
            ]
        else:
            augmentation = [
                RandRotate90d(
                    keys=["image"],
                    prob=0.5,
                    spatial_axes=(0, 1)
                ),
                RandFlipd(
                    keys=["image"],
                    prob=0.5,
                    spatial_axis=0
                ),
                RandAffined(
                    keys=["image"],
                    prob=0.5,
                    rotate_range=[np.pi/12] * 3,
                    scale_range=[0.1] * 3,
                    mode="bilinear"
                )
            ]
        transforms = transforms + augmentation

    return Compose(transforms)

def create_data_loaders(
    config: Dict[str, Any],
    mode: str = '3d'
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders with specified mode."""

    # Create transforms
    train_transforms = get_transforms(config, 'train', mode)
    val_transforms = get_transforms(config, 'val', mode)
    test_transforms = get_transforms(config, 'test', mode)

    # Create datasets
    train_dataset = ADNIDataset(config, transform=train_transforms, split='train', mode=mode)
    val_dataset = ADNIDataset(config, transform=val_transforms, split='val', mode=mode)
    test_dataset = ADNIDataset(config, transform=test_transforms, split='test', mode=mode)

    # Split data
    train_size = 1 - config['dataset']['val_ratio'] - config['dataset']['test_ratio']
    labels = [label for _, label in train_dataset.file_list]

    train_idx, temp_idx = train_test_split(
        range(len(train_dataset)),
        train_size=train_size,
        stratify=labels,
        random_state=config['training']['seed']
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=[labels[i] for i in temp_idx],
        random_state=config['training']['seed']
    )

    def collate_fn(batch):
        batch_data = {}
        for key in batch[0].keys():
            if key == 'image':
                # Stack images
                images = torch.stack([item[key] for item in batch])

                # Handle channel dimension based on mode
                if mode == '2d' and len(images.shape) == 3:  # [B, H, W]
                    images = images.unsqueeze(1)  # Add channel dim [B, C, H, W]
                elif mode == '3d' and len(images.shape) == 4:  # [B, D, H, W]
                    images = images.unsqueeze(1)  # Add channel dim [B, C, D, H, W]

                batch_data[key] = images
            else:
                batch_data[key] = torch.tensor([item[key] for item in batch])
        return batch_data

    # Create loaders
    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=0 if os.name == 'nt' else 4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        Subset(val_dataset, val_idx),
        batch_size=config['dataset']['batch_size'],
        shuffle=False,
        num_workers=0 if os.name == 'nt' else 4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        Subset(test_dataset, test_idx),
        batch_size=config['dataset']['batch_size'],
        shuffle=False,
        num_workers=0 if os.name == 'nt' else 4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Log split sizes
    logger.info(f"Dataset splits - Train: {len(train_idx)}, "
                f"Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Verify shapes
    try:
        train_batch = next(iter(train_loader))
        expected_shape = "B, C, H, W" if mode == '2d' else "B, C, D, H, W"
        logger.info(f"Train batch image shape ({expected_shape}): {train_batch['image'].shape}")
        logger.info(f"Train batch label shape: {train_batch['label'].shape}")
    except Exception as e:
        logger.warning(f"Could not verify train batch shapes: {str(e)}")

    return train_loader, val_loader, test_loader
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
    RandGaussianNoised
)
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class ADNIDataset(Dataset):
    """Custom Dataset for loading ADNI data."""

    def __init__(
        self,
        config: Dict[str, Any],
        transform: Optional[Compose] = None,
        split: str = 'train'
    ):
        """
        Initialize the dataset.
        
        Args:
            config: Configuration dictionary
            transform: MONAI transforms to apply
            split: Dataset split ('train', 'val', or 'test')
        """
        self.data_root = Path(config['dataset']['path'])
        self.transform = transform
        self.split = split
        self.file_list = self._create_file_list()
        self.label_to_idx = {'AD': 0, 'CN': 1, 'MCI': 2}
        
        logger.info(f"Initialized {split} dataset with {len(self.file_list)} samples")
        
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

def get_transforms(config: Dict[str, Any], split: str) -> Compose:
    """
    Get preprocessing transforms based on configuration.
    
    Args:
        config: Configuration dictionary
        split: Dataset split ('train', 'val', or 'test')
    """
    # Get spatial size
    spatial_size = (config['dataset']['input_size'],) * 3
    
    # Common transforms for all splits
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
        ),
        ResizeWithPadOrCropd(
            keys=["image"],
            spatial_size=spatial_size
        ),
        ScaleIntensityd(keys=["image"]),
        NormalizeIntensityd(keys=["image"], nonzero=True)
    ]
    
    # Add augmentation for training
    if split == 'train':
        augmentation_transforms = [
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
                rotate_range=(np.pi/12, np.pi/12, np.pi/12),
                scale_range=(0.1, 0.1, 0.1),
                mode="bilinear"
            ),
            RandGaussianNoised(
                keys=["image"],
                prob=0.2,
                mean=0.0,
                std=0.1
            )
        ]
        transforms = common_transforms + augmentation_transforms
    else:
        transforms = common_transforms
    
    return Compose(transforms)

def create_data_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders."""
    # Create transforms
    train_transforms = get_transforms(config, 'train')
    val_transforms = get_transforms(config, 'val')
    test_transforms = get_transforms(config, 'test')
    
    # Create datasets
    train_dataset = ADNIDataset(config, transform=train_transforms, split='train')
    
    # Split data
    train_size = 1 - config['dataset']['val_ratio'] - config['dataset']['test_ratio']
    
    # Get all labels for stratification
    labels = [label for _, label in train_dataset.file_list]
    
    # Create splits
    train_idx, temp_idx = train_test_split(
        range(len(train_dataset)),
        train_size=train_size,
        stratify=labels,
        random_state=config['training']['seed']
    )
    
    val_size = len(temp_idx) // 2
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=[labels[i] for i in temp_idx],
        random_state=config['training']['seed']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        Subset(ADNIDataset(config, transform=train_transforms), train_idx),
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        Subset(ADNIDataset(config, transform=val_transforms), val_idx),
        batch_size=config['dataset']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        Subset(ADNIDataset(config, transform=test_transforms), test_idx),
        batch_size=config['dataset']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Log split sizes
    logger.info(f"Dataset splits - Train: {len(train_idx)}, "
                f"Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    return train_loader, val_loader, test_loader
"""
Data loading and preprocessing module for Alzheimer's detection project.
"""

import os
import warnings
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import yaml
import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    CropForegroundd,
    ScaleIntensityd,
    NormalizeIntensityd,
    SaveImaged
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Filter warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    project_root = get_project_root()
    config_file = project_root / config_path

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Update paths relative to project root
    config['dataset']['path'] = str(project_root / 'adni')
    config['paths'] = {
        'data': {
            'raw': str(project_root / 'adni/raw'),
            'processed': str(project_root / 'adni/processed'),
            'metadata': str(project_root / 'metadata/adni.csv')
        }
    }

    return config

def validate_data_paths(config: Dict[str, Any]) -> bool:
    """
    Validate that data paths exist and contain required files.

    Args:
        config: Configuration dictionary

    Returns:
        bool: True if validation passes
    """
    raw_dir = Path(config['paths']['data']['raw'])

    # Check if raw directory exists
    if not raw_dir.exists():
        logger.error(f"Raw data directory not found: {raw_dir}")
        return False

    # Check for class directories and files
    total_files = 0
    for label in ['AD', 'CN', 'MCI']:
        label_dir = raw_dir / label
        if not label_dir.exists():
            logger.error(f"Class directory not found: {label_dir}")
            continue

        files = list(label_dir.glob('*.nii*'))
        total_files += len(files)
        logger.info(f"Found {len(files)} files in {label} directory")

    if total_files == 0:
        logger.error("No .nii or .nii.gz files found in any class directory")
        return False

    logger.info(f"Found total of {total_files} files")
    return True

def preprocess_dataset(config: Dict[str, Any]) -> None:
    """
    Preprocess and save the entire dataset once.
    """
    raw_dir = Path(config['paths']['data']['raw'])
    processed_dir = Path(config['paths']['data']['processed'])

    # Create processed directory structure
    processed_dir.mkdir(parents=True, exist_ok=True)
    for label in ['AD', 'CN', 'MCI']:
        (processed_dir / label).mkdir(exist_ok=True)

    preprocess_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(1.5, 1.5, 1.5),
            mode=("bilinear"),
        ),
        CropForegroundd(
            keys=["image"],
            source_key="image",
            margin=10
        ),
        ScaleIntensityd(keys=["image"]),
        NormalizeIntensityd(keys=["image"], nonzero=True),
    ])

    # Process each class directory
    total_processed = 0
    for label in ['AD', 'CN', 'MCI']:
        label_dir = raw_dir / label
        if not label_dir.exists():
            continue

        files = list(label_dir.glob('*.nii*'))
        logger.info(f"Processing {len(files)} files for {label}")

        for file_path in files:
            try:
                # Create output path
                output_path = processed_dir / label / f"{file_path.stem}_processed.nii.gz"

                # Skip if already processed
                if output_path.exists():
                    logger.debug(f"Skipping {file_path.name} - already processed")
                    total_processed += 1
                    continue

                # Process file
                data_dict = {"image": str(file_path)}
                processed = preprocess_transforms(data_dict)

                # Save processed file
                nib.save(
                    nib.Nifti1Image(processed["image"].numpy(), np.eye(4)),
                    str(output_path)
                )
                total_processed += 1
                logger.debug(f"Processed: {file_path.name}")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")

    logger.info(f"Successfully processed {total_processed} files")

class ADNIDataset(Dataset):
    """Custom Dataset for loading preprocessed ADNI data."""

    def __init__(self, config: Dict[str, Any], transform: Optional[Compose] = None):
        """Initialize the dataset."""
        self.data_root = Path(config['paths']['data']['processed'])
        self.transform = transform
        self.file_list = self._create_file_list()
        self.label_to_idx = {'AD': 0, 'CN': 1, 'MCI': 2}

        if len(self.file_list) == 0:
            raise RuntimeError("No processed files found. Run preprocessing first.")

        logger.info(f"Dataset initialized with {len(self.file_list)} samples")

    def _create_file_list(self) -> list:
        """Create list of preprocessed file paths and labels."""
        file_list = []
        for label in ['AD', 'CN', 'MCI']:
            label_dir = self.data_root / label
            if label_dir.exists():
                for file_path in label_dir.glob('*_processed.nii.gz'):
                    file_list.append((file_path, label))
        return file_list

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_path, label = self.file_list[idx]
        data_dict = {
            'image': str(file_path),
            'label': self.label_to_idx[label]
        }
        if self.transform:
            data_dict = self.transform(data_dict)
        return data_dict

def create_data_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders."""
    # Basic transforms for loading preprocessed data
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"])
    ])

    # Create dataset
    dataset = ADNIDataset(config, transform=transforms)

    # Calculate split sizes
    total_size = len(dataset)
    test_size = int(total_size * config['dataset']['test_ratio'])
    val_size = int(total_size * config['dataset']['val_ratio'])
    train_size = total_size - test_size - val_size

    # Split indices
    indices = list(range(total_size))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Create data loaders
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=config['dataset']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=config['dataset']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    logger.info(f"Created data loaders - Train: {len(train_indices)}, "
                f"Val: {len(val_indices)}, Test: {len(test_indices)}")

    return train_loader, val_loader, test_loader

def main():
    """Main function to preprocess data and test the data loading pipeline."""
    config = load_config()

    # Validate data paths
    if not validate_data_paths(config):
        logger.error("Data path validation failed. Please check your data directory structure.")
        return

    # Preprocess the dataset
    logger.info("Starting data preprocessing...")
    preprocess_dataset(config)

    # Create and test data loaders
    logger.info("Creating data loaders...")
    try:
        train_loader, val_loader, test_loader = create_data_loaders(config)

        # Test a single batch
        batch = next(iter(train_loader))
        logger.info(f"Successfully loaded batch with shape: {batch['image'].shape}")

    except Exception as e:
        logger.error(f"Error creating data loaders: {str(e)}")
        raise

if __name__ == "__main__":
    main()
"""
Optimized data loading and preprocessing module for Alzheimer's detection project.
"""

import os
import warnings
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
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
    ResizeWithPadOrCropd,
    RandAffined,
    RandGaussianNoised,
    RandAdjustContrastd,
    EnsureTyped,
    ToTensord
)
from monai.data import ThreadDataLoader
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Filter warnings
warnings.filterwarnings('ignore', message='.*TorchScript.*', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

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

def preprocess_dataset(config: Dict[str, Any]) -> None:
    """
    Preprocess the dataset with consistent sizing and save processed files.
    """
    raw_dir = Path(config['paths']['data']['raw'])
    processed_dir = Path(config['paths']['data']['processed'])
    
    # Create processed directory structure
    processed_dir.mkdir(parents=True, exist_ok=True)
    for label in ['AD', 'CN', 'MCI']:
        (processed_dir / label).mkdir(exist_ok=True)
    
    # Define preprocessing transforms
    spatial_size = (config['dataset']['input_size'],) * 3
    preprocess_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(1.5, 1.5, 1.5),
            mode="bilinear"
        ),
        CropForegroundd(
            keys=["image"],
            source_key="image",
            margin=10
        ),
        ResizeWithPadOrCropd(
            keys=["image"],
            spatial_size=spatial_size,
            mode="constant"
        ),
        ScaleIntensityd(keys=["image"]),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        EnsureTyped(keys=["image"]),
        ToTensord(keys=["image"])
    ])

    # Process each class
    total_processed = 0
    skipped = 0
    
    for label in ['AD', 'CN', 'MCI']:
        label_dir = raw_dir / label
        if not label_dir.exists():
            continue
        
        processed_label_dir = processed_dir / label
        files = list(label_dir.glob('*.nii*'))
        logger.info(f"Processing {len(files)} files for {label}")
        
        for file_path in tqdm(files, desc=f"Processing {label}"):
            try:
                output_path = processed_label_dir / f"{file_path.stem}_processed.nii.gz"
                
                if output_path.exists():
                    skipped += 1
                    continue
                
                # Process file
                data = preprocess_transforms({"image": str(file_path)})
                processed_image = data["image"][0].numpy()  # Remove batch dimension
                
                # Save processed file
                nib.save(
                    nib.Nifti1Image(processed_image, np.eye(4)),
                    str(output_path)
                )
                total_processed += 1
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
    
    logger.info(f"Preprocessing complete:")
    logger.info(f"- Newly processed: {total_processed}")
    logger.info(f"- Skipped existing: {skipped}")

class ADNIDataset(Dataset):
    """Custom Dataset for loading preprocessed ADNI data."""

    def __init__(self, config: Dict[str, Any], transform: Optional[Compose] = None):
        """Initialize the dataset."""
        self.data_root = Path(config['paths']['data']['processed'])
        self.transform = transform
        self.file_list = self._create_file_list()
        self.label_to_idx = {'AD': 0, 'CN': 1, 'MCI': 2}
        
        logger.info(f"Dataset initialized with {len(self.file_list)} samples")

    def _create_file_list(self) -> List[Tuple[Path, str]]:
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

def get_data_transforms(config: Dict[str, Any]) -> Dict[str, Compose]:
    """
    Get data loading transforms for preprocessed data.
    """
    common_transforms = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityd(keys=["image"]),
        EnsureTyped(keys=["image", "label"]),
        ToTensord(keys=["image", "label"])
    ]
    
    train_transforms = common_transforms + [
        RandAffined(
            keys=["image"],
            prob=0.5,
            rotate_range=(np.pi/12, np.pi/12, np.pi/12),
            scale_range=(0.1, 0.1, 0.1),
            mode="bilinear",
            padding_mode="zeros"
        ),
        RandGaussianNoised(keys=["image"], prob=0.2),
        RandAdjustContrastd(keys=["image"], prob=0.2)
    ]
    
    return {
        'train': Compose(train_transforms),
        'val': Compose(common_transforms)
    }

def create_data_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders."""
    transforms = get_data_transforms(config)
    
    # Create dataset
    dataset = ADNIDataset(config, transform=transforms['train'])
    
    # Calculate split sizes
    total_size = len(dataset)
    test_size = int(total_size * config['dataset']['test_ratio'])
    val_size = int(total_size * config['dataset']['val_ratio'])
    train_size = total_size - test_size - val_size
    
    # Get all labels for stratification
    labels = [dataset.label_to_idx[label] for _, label in dataset.file_list]
    
    # Split indices with stratification
    train_idx, temp_idx = train_test_split(
        range(total_size),
        train_size=train_size,
        stratify=labels,
        random_state=config['training']['seed']
    )
    
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_size/(test_size + val_size),
        stratify=[labels[i] for i in temp_idx],
        random_state=config['training']['seed']
    )
    
    # Create data loaders
    train_loader = ThreadDataLoader(
        Subset(ADNIDataset(config, transform=transforms['train']), train_idx),
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = ThreadDataLoader(
        Subset(ADNIDataset(config, transform=transforms['val']), val_idx),
        batch_size=config['dataset']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = ThreadDataLoader(
        Subset(ADNIDataset(config, transform=transforms['val']), test_idx),
        batch_size=config['dataset']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"Created data loaders - Train: {len(train_idx)}, "
                f"Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    return train_loader, val_loader, test_loader

def main():
    """Main function to preprocess data and test the data loading pipeline."""
    try:
        # Load configuration
        config = load_config()
        
        # Check if preprocessing is needed
        raw_dir = Path(config['paths']['data']['raw'])
        processed_dir = Path(config['paths']['data']['processed'])
        
        raw_files = sum(len(list((raw_dir / label).glob('*.nii*')))
                       for label in ['AD', 'CN', 'MCI'])
        processed_files = sum(len(list((processed_dir / label).glob('*_processed.nii.gz')))
                            for label in ['AD', 'CN', 'MCI'])
        
        if processed_files < raw_files:
            logger.info("Starting preprocessing...")
            preprocess_dataset(config)
        
        # Create and test data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(config)
        
        # Test batch loading
        batch = next(iter(train_loader))
        logger.info(f"Successfully loaded batch:")
        logger.info(f"- Image shape: {batch['image'].shape}")
        logger.info(f"- Label shape: {batch['label'].shape}")
        logger.info(f"- Labels in batch: {batch['label'].tolist()}")
        
        # Memory usage
        if torch.cuda.is_available():
            logger.info(f"- GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
import logging
from typing import Dict

from monai.transforms import Compose
from monai.data import DataLoader, CacheDataset, load_decathlon_datalist
import monai

from preprocess import get_preprocessing_transforms
from augmentation import get_augmentation_transforms

# Configure logging
logger = logging.getLogger(__name__)


def get_train_transforms():
    """
    Combines preprocessing and augmentation transforms for training data.

    Returns:
        Callable: A composed transform function.
    """
    logger.info("Creating training transforms.")
    transforms = Compose([
        get_preprocessing_transforms(),
        get_augmentation_transforms(),
    ])
    return transforms


def get_val_transforms():
    """
    Returns preprocessing transforms for validation data.

    Returns:
        Callable: A composed transform function.
    """
    logger.info("Creating validation transforms.")
    transforms = get_preprocessing_transforms()
    return transforms


def create_dataloaders(
        data_dir: str,
        json_path: str,
        train_batch_size: int = 2,
        val_batch_size: int = 1,
        num_workers: int = 4,
) -> Dict[str, DataLoader]:
    """
    Creates data loaders for training and validation datasets.

    Args:
        data_dir (str): The directory where the data is stored.
        json_path (str): Path to the JSON file containing data splits.
        train_batch_size (int): Batch size for the training loader.
        val_batch_size (int): Batch size for the validation loader.
        num_workers (int): Number of worker threads for data loading.

    Returns:
        Dict[str, DataLoader]: A dictionary containing 'train' and 'val' data loaders.
    """
    logger.info("Creating data loaders.")

    # Load data lists from JSON
    datasets = load_decathlon_datalist(
        data_list_file_path=json_path,
        is_training=True,
        data_list_key="training",
        base_dir=data_dir,
    )

    # Split data into training and validation sets
    train_files, val_files = monai.data.utils.partition_dataset(
        data=datasets,
        ratios=[0.8, 0.2],  # 80% training, 20% validation
        shuffle=True,
        seed=42,
    )

    # Create transforms
    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()

    # Create datasets
    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=1.0,
        num_workers=num_workers,
    )
    val_ds = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_rate=1.0,
        num_workers=num_workers,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return {'train': train_loader, 'val': val_loader}

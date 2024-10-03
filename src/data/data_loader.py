from datasets import load_dataset
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Resized,
    RandRotate90d,
    RandFlipd
)
import torch


def load_huggingface_dataset(dataset_name, split='train'):
    """Load dataset from Hugging Face."""
    return load_dataset(dataset_name, split=split)


def create_monai_dataset(hf_dataset, transforms):
    """Create a MONAI dataset from a Hugging Face dataset."""
    data = [{"image": item["image"], "label": item["label"]} for item in hf_dataset]
    return Dataset(data=data, transform=transforms)


def create_data_loaders(dataset, batch_size, num_workers=4):
    """Create DataLoader with given dataset and batch size."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def get_transforms(model_type, spatial_size=(224, 224, 224)):
    """Get MONAI transforms based on model type."""
    if model_type == '2d_vit':
        return Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Resized(keys=["image"], spatial_size=spatial_size[:2]),
            ScaleIntensityd(keys=["image"]),
            RandRotate90d(keys=["image"], prob=0.5, spatial_axes=[0, 1]),
            RandFlipd(keys=["image"], spatial_axis=1),
        ])
    else:  # 3D transforms for 3D ViT and 3D CNN
        return Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Resized(keys=["image"], spatial_size=spatial_size),
            ScaleIntensityd(keys=["image"]),
            RandRotate90d(keys=["image"], prob=0.5, spatial_axes=[0, 1]),
            RandFlipd(keys=["image"], spatial_axis=0),
        ])


def prepare_data(dataset_name, model_type, batch_size):
    """Prepare datasets and dataloaders."""
    # Load dataset from Hugging Face
    hf_dataset = load_huggingface_dataset(dataset_name)

    # Get MONAI transforms
    transforms = get_transforms(model_type)

    # Create MONAI datasets
    train_ds = create_monai_dataset(hf_dataset['train'], transforms)
    val_ds = create_monai_dataset(hf_dataset['validation'], transforms)
    test_ds = create_monai_dataset(hf_dataset['test'], transforms)

    # Create data loaders
    train_loader = create_data_loaders(train_ds, batch_size)
    val_loader = create_data_loaders(val_ds, batch_size)
    test_loader = create_data_loaders(test_ds, batch_size)

    return train_loader, val_loader, test_loader
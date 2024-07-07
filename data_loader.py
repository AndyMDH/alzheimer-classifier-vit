import os
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose, LoadNiftid, AddChanneld, ScaleIntensityd,
    RandRotate90d, RandFlipd, ToTensord
)

def create_data_loader(data_dir, batch_size):
    transforms = Compose([
        LoadNiftid(keys=["image", "label"]),
        AddChanneld(keys=["image"]),
        ScaleIntensityd(keys=["image"]),
        RandRotate90d(keys=["image"], prob=0.8, spatial_axes=[0, 2]),
        RandFlipd(keys=["image"], spatial_axis=0),
        ToTensord(keys=["image", "label"])
    ])

    train_files = [
        {"image": os.path.join(data_dir, f"image{i}.nii.gz"), "label": i % 2}
        for i in range(100)  # Assume 100 images for this example
    ]
    train_ds = Dataset(data=train_files, transform=transforms)
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
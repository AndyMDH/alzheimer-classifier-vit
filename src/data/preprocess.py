from monai.transforms import (
    Compose, LoadImaged, AddChanneld, ScaleIntensityd,
    Resized, RandRotate90d, RandFlipd, ToTensord
)

def get_train_transforms():
    return Compose([
        LoadImaged(keys=["image"]),
        AddChanneld(keys=["image"]),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(128, 128, 128)),
        RandRotate90d(keys=["image"], prob=0.8, spatial_axes=[0, 2]),
        RandFlipd(keys=["image"], spatial_axis=0, prob=0.5),
        ToTensord(keys=["image", "label"])
    ])

def get_val_transforms():
    return Compose([
        LoadImaged(keys=["image"]),
        AddChanneld(keys=["image"]),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(128, 128, 128)),
        ToTensord(keys=["image", "label"])
    ])
from monai.transforms import (
    LoadImaged,
    AddChanneld,
    NormalizeIntensityd,
    SpatialPadd,
    RandRotate90d,
    RandFlipd,
    ToTensord,
)


def get_transforms(image_size):
    return Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            NormalizeIntensityd(keys=["image"]),
            SpatialPadd(keys=["image"], spatial_size=image_size),
            RandRotate90d(keys=["image"], prob=0.5),
            RandFlipd(keys=["image"], prob=0.5),
            ToTensord(keys=["image"]),
        ]
    )

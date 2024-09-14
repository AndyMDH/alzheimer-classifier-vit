# src/models/architectures/vit3d_b16.py
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer

class ViT3D(VisionTransformer):
    def __init__(self, img_size=(128, 128, 128), patch_size=16, num_classes=2):
        super(ViT3D, self).__init__(img_size=img_size, patch_size=patch_size, num_classes=num_classes, in_chans=1)
        # Initialize a 3D version of ViT, overriding specific layers if needed

    def forward(self, x):
        x = super().forward(x)
        return x

def vit3d_b16():
    model = ViT3D(img_size=(128, 128, 128), patch_size=16, num_classes=2)
    return model

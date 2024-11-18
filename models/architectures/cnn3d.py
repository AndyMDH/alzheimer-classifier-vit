"""
models/cnn_3d.py - 3D CNN for Alzheimer's detection with transfer learning.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import logging

logger = logging.getLogger(__name__)

class CNN3D(nn.Module):
    def __init__(
        self,
        num_labels: int,
        freeze_layers: bool = True,
        input_size: int = 224,
        patch_size: int = 16,
        dropout_rate: float = 0.1
    ):
        super().__init__()

        # Load pretrained 2D ResNet
        resnet = models.resnet50(pretrained=True)

        # Convert first conv layer to 3D
        self.conv1 = nn.Conv3d(
            1, 64,
            kernel_size=(7, 7, 7),
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False
        )

        # Initialize from 2D weights
        with torch.no_grad():
            self.conv1.weight.copy_(
                resnet.conv1.weight.unsqueeze(2).repeat(1, 1, 7, 1, 1) / 7
            )

        # Convert other layers to 3D
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Convert ResNet blocks to 3D
        def convert_layer(layer2d):
            blocks = []
            for block in layer2d:
                blocks.append(
                    Block3D(
                        block.conv1.in_channels,
                        block.conv1.out_channels,
                        stride=block.stride[0]
                    )
                )
            return nn.Sequential(*blocks)

        self.layer1 = convert_layer(resnet.layer1)
        self.layer2 = convert_layer(resnet.layer2)
        self.layer3 = convert_layer(resnet.layer3)
        self.layer4 = convert_layer(resnet.layer4)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_labels)
        )

        if freeze_layers:
            self._freeze_layers()

        # Log model statistics
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

    def _freeze_layers(self):
        """Freeze early layers."""
        frozen_layers = [self.conv1, self.bn1, self.layer1, self.layer2]
        for layer in frozen_layers:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class Block3D(nn.Module):
    """3D version of ResNet block."""

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_planes, planes, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(planes)

        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    in_planes, planes,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm3d(planes)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out
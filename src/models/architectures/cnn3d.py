import torch.nn as nn
import torchvision.models as models


class CNN3D(nn.Module):
    def __init__(self, num_labels, pretrained=True):
        super().__init__()
        # Start with a 2D ResNet and modify for 3D
        resnet = models.resnet50(pretrained=pretrained)

        # Modify the first convolutional layer for 3D input
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Use other ResNet layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer_3d(resnet.layer1)
        self.layer2 = self._make_layer_3d(resnet.layer2)
        self.layer3 = self._make_layer_3d(resnet.layer3)
        self.layer4 = self._make_layer_3d(resnet.layer4)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(2048, num_labels)

    def _make_layer_3d(self, layer):
        new_layer = nn.Sequential()
        for i, bottleneck in enumerate(layer):
            new_bottleneck = nn.Sequential()
            for name, module in bottleneck.named_children():
                if isinstance(module, nn.Conv2d):
                    # Replace 2D convolutions with 3D
                    new_conv = nn.Conv3d(
                        in_channels=module.in_channels,
                        out_channels=module.out_channels,
                        kernel_size=module.kernel_size[0],
                        stride=module.stride[0],
                        padding=module.padding[0],
                        bias=module.bias
                    )
                    new_bottleneck.add_module(name, new_conv)
                else:
                    new_bottleneck.add_module(name, module)
            new_layer.add_module(str(i), new_bottleneck)
        return new_layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def create_cnn_3d(num_labels):
    """Create a 3D CNN model."""
    return CNN3D(num_labels)
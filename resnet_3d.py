import torch
from monai.networks.nets import ResNet
from config import config
from data_loader import create_data_loader
from train_utils import train_model, save_results

def create_resnet_model(device):
    return ResNet(
        block="basic",
        layers=[2, 2, 2, 2],
        block_inplanes=[64, 128, 256, 512],
        spatial_dims=3,
        n_input_channels=1,
        num_classes=config['num_classes']
    ).to(device)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = create_data_loader(config['data_dir'], config['batch_size'])
    model = create_resnet_model(device)
    history = train_model(model, train_loader, config, device)
    save_results(model, history, "ResNet3D", config)

if __name__ == "__main__":
    main()
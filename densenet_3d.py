import torch
from monai.networks.nets import DenseNet121
from config import config
from data_loader import create_data_loader
from train_utils import train_model, save_results

def create_densenet_model(device):
    return DenseNet121(
        spatial_dims=3,
        in_channels=1,
        out_channels=config['num_classes']
    ).to(device)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = create_data_loader(config['data_dir'], config['batch_size'])
    model = create_densenet_model(device)
    history = train_model(model, train_loader, config, device)
    save_results(model, history, "DenseNet3D", config)

if __name__ == "__main__":
    main()
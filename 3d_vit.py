import torch
import torch.nn as nn
from monai.networks.nets import ViT
from config import config
from data_loader import create_data_loader
from train_utils import train_model, save_results


def create_vit_model(device):
    model = ViT(
        in_channels=1,
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        hidden_size=config['hidden_size'],
        mlp_dim=config['mlp_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        pos_embed='conv',
        classification=True,
        num_classes=config['num_classes'],
        pretrained=True,
        spatial_dims=3,
    ).to(device)

    for name, param in model.named_parameters():
        if "classification_head" not in name:
            param.requires_grad = False
    model.classification_head = nn.Linear(model.hidden_size, config['num_classes']).to(device)

    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = create_data_loader(config['data_dir'], config['batch_size'])
    model = create_vit_model(device)
    history = train_model(model, train_loader, config, device)
    save_results(model, history, "ViT", config)


if __name__ == "__main__":
    main()
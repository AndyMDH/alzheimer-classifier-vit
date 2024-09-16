# src/models/train.py
import torch
from torch.utils.data import DataLoader
from src.models.architectures.vit3d_b16 import ViT3DB16
from src.data.preprocess import get_preprocessing_transforms
from src.utils.logger import get_tensorboard_logger
from src.utils.train_utils import save_checkpoint


def train_model(model_name='vit_b16', epochs=10, batch_size=4, learning_rate=1e-4):
    # Initialize model
    if model_name == 'vit_b16':
        model = ViT3DB16()
    elif model_name == 'vit_m8':
        from src.models.architectures.vit3d_m8 import ViT3DM8
        model = ViT3DM8()
    elif model_name == 'vit_l32':
        from src.models.architectures.vit3d_m8 import ViT3L32
        model = ViT3DL32()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # DataLoader (replace [...] with your dataset)
    transforms = get_preprocessing_transforms()
    train_dataset = [...]  # Load your dataset here and apply transforms
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # TensorBoard logger
    writer = get_tensorboard_logger()

    best_loss = float('inf')  # Track best loss for checkpointing

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

        # Save checkpoint
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                        is_best=(avg_loss < best_loss))
        best_loss = min(avg_loss, best_loss)

    writer.close()

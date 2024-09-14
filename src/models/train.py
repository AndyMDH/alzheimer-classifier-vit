# src/models/train.py
import torch
from torch.utils.data import DataLoader
from vit3d_b16 import vit3d_b16
from monai.transforms import Compose, LoadImage, RandRotate90, ToTensor

def train_model():
    model = vit3d_b16()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Dummy dataset loader (replace with real MRI dataset)
    train_loader = DataLoader([...], batch_size=4, shuffle=True)

    model.train()
    for epoch in range(10):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/10], Loss: {loss.item()}")

if __name__ == "__main__":
    train_model()

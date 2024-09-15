# src/models/evaluate.py
import torch
from torch.utils.data import DataLoader
from src.models.architectures.vit3d_b16 import ViT3DB16
from src.data.preprocess import get_preprocessing_transforms

def evaluate_model(model_name='vit_b16'):
    if model_name == 'vit_b16':
        model = ViT3DB16()
    elif model_name == 'vit_m8':
        from src.models.architectures.vit3d_m8 import ViT3DM8
        model = ViT3DM8()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Load checkpoint
    checkpoint = torch.load('best_model.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])

    # DataLoader (replace [...] with your dataset)
    transforms = get_preprocessing_transforms()
    test_dataset = [...]  # Load your test dataset here and apply transforms
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")

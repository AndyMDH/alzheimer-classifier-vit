import torch
from monai.utils import set_determinism
from src.data.data_loader import prepare_data
from src.models.architectures.vit2d import create_vit_2d
from src.models.architectures.vit3d import create_vit_3d
from src.models.architectures.cnn3d import create_cnn_3d
from src.models.train import train_model
from src.models.evaluate import evaluate_model
from src.utils.logger import setup_logger

def main():
    # Set deterministic training for reproducibility
    set_determinism(seed=0)

    # Set up logger
    logger = setup_logger('alzheimer_detection', 'logs/alzheimer_detection.log')

    # Set parameters
    dataset_name = 'your_dataset_name'
    model_type = '3d_vit'  # Options: '2d_vit', '3d_vit', '3d_cnn'
    batch_size = 32

    # Prepare data
    train_loader, val_loader, test_loader = prepare_data(dataset_name, model_type, batch_size)

    # Create model
    num_labels = 4  # Assuming 4 classes for Alzheimer's stages
    if model_type == '2d_vit':
        model = create_vit_2d(num_labels)
    elif model_type == '3d_vit':
        model = create_vit_3d(num_labels)
    elif model_type == '3d_cnn':
        model = create_cnn_3d(num_labels)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Train model
    train_model(model, train_loader, val_loader, max_epochs=50, lr=1e-4, device=device)

    # Evaluate model
    results = evaluate_model(model, test_loader, device)
    logger.info(f"Evaluation results: {results}")

if __name__ == "__main__":
    main()
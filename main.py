import logging

import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import create_dataloaders

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    device,
    num_epochs=10,
):
    """
    Trains the model using the provided data loaders, criterion, and optimizer.

    Args:
        model: The neural network model to train.
        dataloaders: A dictionary containing 'train' and 'val' data loaders.
        criterion: The loss function.
        optimizer: The optimizer for updating model weights.
        device: The device to run the training on ('cpu' or 'cuda').
        num_epochs (int): Number of training epochs.

    Returns:
        The trained model.
    """
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data
            for batch in dataloaders[phase]:
                inputs = batch['image'].to(device)
                labels = batch['label'].to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # Adjust outputs and labels as per your task (e.g., apply softmax, reshape, etc.)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimize
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            logger.info(f"{phase.capitalize()} Loss: {epoch_loss:.4f}")

    return model

if __name__ == "__main__":
    # Paths to your data directory and JSON file
    data_dir = '/path/to/data_directory'
    json_path = '/path/to/dataset.json'

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create data loaders
    dataloaders = create_dataloaders(
        data_dir=data_dir,
        json_path=json_path,
        train_batch_size=2,
        val_batch_size=1,
        num_workers=4,
    )

    # Initialize your 3D ViT model
    model = ...  # Replace with your model initialization
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Adjust as per your task
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    trained_model = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=10,
    )

    # Save the trained model
    torch.save(trained_model.state_dict(), 'trained_model.pth')

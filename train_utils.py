import os
import json
import torch
import torch.nn as nn
import torch.optim as optim


def train_model(model, train_loader, config, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    history = []
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0
        for batch_data in train_loader:
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        history.append({"epoch": epoch + 1, "loss": avg_loss})
        print(f'Epoch {epoch + 1}/{config["num_epochs"]}, Loss: {avg_loss:.4f}')

    return history


def save_results(model, history, model_type, config):
    os.makedirs(config['output_dir'], exist_ok=True)

    torch.save(model.state_dict(), os.path.join(config['output_dir'], f'{model_type}_model.pth'))

    with open(os.path.join(config['output_dir'], f'{model_type}_history.json'), 'w') as f:
        json.dump(history, f)

    with open(os.path.join(config['output_dir'], f'{model_type}_config.json'), 'w') as f:
        json.dump(config, f)
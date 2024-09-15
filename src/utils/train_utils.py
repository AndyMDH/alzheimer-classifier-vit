import torch

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves the model state."""
    torch.save(state, filename)
    if is_best:
        torch.save(state, 'best_model.pth.tar')

def load_checkpoint(model, optimizer, filename='checkpoint.pth.tar'):
    """Loads the model and optimizer states."""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer

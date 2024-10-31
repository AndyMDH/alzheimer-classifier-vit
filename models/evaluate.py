"""
Evaluation module for Alzheimer's detection models.
"""

import torch
import torch.nn as nn
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    exp_dir = None
) -> Dict[str, Any]:
    """
    Evaluate the model and compute metrics.
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        exp_dir: Optional experiment directory for saving plots
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    
    # Initialize lists to store predictions and labels
    all_preds = []
    all_labels = []
    all_probs = []
    
    # Evaluate batches
    for batch in tqdm(test_loader, desc='Evaluating'):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        
        # Get predictions
        _, predicted = outputs.max(1)
        
        # Store results
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    conf_matrix = confusion_matrix(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    # Calculate ROC AUC (one-vs-rest for multiclass)
    roc_auc = roc_auc_score(
        np.eye(3)[all_labels],
        all_probs,
        multi_class='ovr'
    )
    
    # Get detailed classification report
    class_report = classification_report(
        all_labels,
        all_preds,
        target_names=['AD', 'CN', 'MCI'],
        output_dict=True
    )
    
    # Create results dictionary
    results = {
        'accuracy': (all_preds == all_labels).mean(),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report
    }
    
    # Plot confusion matrix if exp_dir is provided
    if exp_dir is not None:
        plot_confusion_matrix(
            conf_matrix,
            ['AD', 'CN', 'MCI'],
            exp_dir / 'results' / 'confusion_matrix.png'
        )
        
        # Plot ROC curves
        plot_roc_curves(
            np.eye(3)[all_labels],
            all_probs,
            ['AD', 'CN', 'MCI'],
            exp_dir / 'results' / 'roc_curves.png'
        )
    
    # Log results
    logger.info("Evaluation Results:")
    logger.info(f"Accuracy: {results['accuracy']:.4f}")
    logger.info(f"Precision: {results['precision']:.4f}")
    logger.info(f"Recall: {results['recall']:.4f}")
    logger.info(f"F1 Score: {results['f1']:.4f}")
    logger.info(f"ROC AUC: {results['roc_auc']:.4f}")
    
    return results

def plot_confusion_matrix(
    cm: np.ndarray,
    classes: list,
    save_path: Path
) -> None:
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: list,
    save_path: Path
) -> None:
    """Plot and save ROC curves."""
    from sklearn.metrics import roc_curve, auc
    
    plt.figure(figsize=(10, 8))
    
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr, tpr,
            label=f'{classes[i]} (AUC = {roc_auc:.2f})'
        )
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
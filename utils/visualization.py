```python
"""
Visualization utilities for Alzheimer's detection.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, List
import seaborn as sns
from monai.visualize import blend_images
import logging

logger = logging.getLogger(__name__)


def visualize_batch(
        batch: dict,
        num_samples: int = 4,
        save_path: Optional[Path] = None
) -> None:
    """
    Visualize a batch of 3D images.

    Args:
        batch: Dictionary containing 'image' and 'label'
        num_samples: Number of samples to visualize
        save_path: Optional path to save the visualization
    """
    images = batch['image'].cpu().numpy()
    labels = batch['label'].cpu().numpy()

    num_samples = min(num_samples, len(images))
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

    for i in range(num_samples):
        # Get middle slices
        x_mid = images[i, 0, images.shape[2] // 2, :, :]
        y_mid = images[i, 0, :, images.shape[3] // 2, :]
        z_mid = images[i, 0, :, :, images.shape[4] // 2]

        # Plot slices
        axes[i, 0].imshow(x_mid, cmap='gray')
        axes[i, 0].set_title(f'Sample {i + 1} (Label: {labels[i]})\nSagittal')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(y_mid, cmap='gray')
        axes[i, 1].set_title('Coronal')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(z_mid, cmap='gray')
        axes[i, 2].set_title('Axial')
        axes[i, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved batch visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_attention(
        model: torch.nn.Module,
        image: torch.Tensor,
        save_path: Optional[Path] = None
) -> None:
    """
    Visualize attention maps from the model.

    Args:
        model: The ViT model
        image: Input image tensor
        save_path: Optional path to save the visualization
    """
    model.eval()
    with torch.no_grad():
        # Get attention maps
        outputs = model.vit(image, output_attentions=True)
        attention_maps = outputs.attentions[-1]  # Last layer attention

        # Average attention across heads
        attention = attention_maps.mean(1)

        # Get attention for CLS token
        cls_attention = attention[0, 0, 1:]  # Skip CLS token

        # Reshape attention to match image patches
        patch_size = model.patch_size
        num_patches = int(np.cbrt(len(cls_attention)))
        attention_map = cls_attention.view(num_patches, num_patches, num_patches)

        # Visualize middle slices
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(image[0, 0, image.shape[2] // 2].cpu(), cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Attention map
        im = axes[1].imshow(attention_map[attention_map.shape[0] // 2].cpu(),
                            cmap='hot', interpolation='nearest')
        axes[1].set_title('Attention Map')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])

        # Blend
        blended = blend_images(image[0, 0, image.shape[2] // 2].cpu(),
                               attention_map[attention_map.shape[0] // 2].cpu(),
                               alpha=0.5)
        axes[2].imshow(blended, cmap='gray')
        axes[2].set_title('Blended')
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved attention visualization to {save_path}")
        else:
            plt.show()

        plt.close()


def plot_patches(
        model: torch.nn.Module,
        image: torch.Tensor,
        save_path: Optional[Path] = None
) -> None:
    """
    Visualize how the image is divided into patches.

    Args:
        model: The ViT model
        image: Input image tensor
        save_path: Optional path to save the visualization
    """
    patch_size = model.patch_size

    # Get middle slice
    middle_slice = image[0, 0, image.shape[2] // 2].cpu().numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(middle_slice, cmap='gray')

    # Draw grid
    for i in range(0, middle_slice.shape[0], patch_size):
        plt.axhline(y=i, color='r', linestyle='-', alpha=0.3)
        plt.axvline(x=i, color='r', linestyle='-', alpha=0.3)

    plt.title(f'Patch Grid (Size: {patch_size}x{patch_size})')
    plt.axis('off')

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved patch visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_predictions(
        model: torch.nn.Module,
        batch: dict,
        save_path: Optional[Path] = None
) -> None:
    """
    Visualize model predictions.

    Args:
        model: The model
        batch: Batch of data
        save_path: Optional path to save the visualization
    """
    model.eval()
    with torch.no_grad():
        images = batch['image']
        labels = batch['label']

        # Get predictions
        outputs = model(images)
        _, predicted = outputs.max(1)

        # Get probabilities
        probabilities = torch.softmax(outputs, dim=1)

        # Visualize
        num_samples = min(4, len(images))
        fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))

        class_names = ['AD', 'CN', 'MCI']

        for i in range(num_samples):
            # Image
            axes[i, 0].imshow(images[i, 0, images.shape[2] // 2].cpu(), cmap='gray')
            axes[i, 0].set_title(f'True: {class_names[labels[i]]
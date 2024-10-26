# notebooks/visualize_preprocessing.py

import os
import sys
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from tqdm import tqdm

# Add project root to path to import config
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from data.data_loader import load_config, get_transforms


def plot_brain_slices(img_data, title, slice_nums=None):
    """Plot sagittal, coronal, and axial slices of a brain scan."""
    if slice_nums is None:
        # Get middle slices by default
        slice_nums = [img_data.shape[i] // 2 for i in range(3)]

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3)

    # Plot sagittal slice (x plane)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_data[slice_nums[0], :, :], cmap='gray')
    ax1.set_title(f'Sagittal (x={slice_nums[0]})')
    ax1.axis('off')

    # Plot coronal slice (y plane)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(img_data[:, slice_nums[1], :], cmap='gray')
    ax2.set_title(f'Coronal (y={slice_nums[1]})')
    ax2.axis('off')

    # Plot axial slice (z plane)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(img_data[:, :, slice_nums[2]], cmap='gray')
    ax3.set_title(f'Axial (z={slice_nums[2]})')
    ax3.axis('off')

    plt.suptitle(title)
    return fig


def plot_intensity_histogram(img_data, title):
    """Plot intensity histogram of the image."""
    plt.figure(figsize=(10, 4))
    sns.histplot(img_data.ravel(), bins=100)
    plt.title(f'Intensity Distribution - {title}')
    plt.xlabel('Intensity')
    plt.ylabel('Count')
    return plt.gcf()


def visualize_preprocessed_pair(raw_path, processed_path, output_dir):
    """Compare raw and preprocessed versions of the same image."""
    # Load images
    raw_img = nib.load(raw_path)
    processed_img = nib.load(processed_path)

    raw_data = raw_img.get_fdata()
    processed_data = processed_img.get_fdata()

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot slices
    fig_raw = plot_brain_slices(raw_data, 'Raw Image')
    fig_raw.savefig(output_dir / 'raw_slices.png', bbox_inches='tight', dpi=300)

    fig_processed = plot_brain_slices(processed_data, 'Processed Image')
    fig_processed.savefig(output_dir / 'processed_slices.png', bbox_inches='tight', dpi=300)

    # Plot histograms
    fig_hist_raw = plot_intensity_histogram(raw_data, 'Raw Image')
    fig_hist_raw.savefig(output_dir / 'raw_histogram.png', bbox_inches='tight', dpi=300)

    fig_hist_processed = plot_intensity_histogram(processed_data, 'Processed Image')
    fig_hist_processed.savefig(output_dir / 'processed_histogram.png', bbox_inches='tight', dpi=300)

    # Print image statistics
    stats = {
        'Raw': {
            'shape': raw_data.shape,
            'mean': np.mean(raw_data),
            'std': np.std(raw_data),
            'min': np.min(raw_data),
            'max': np.max(raw_data)
        },
        'Processed': {
            'shape': processed_data.shape,
            'mean': np.mean(processed_data),
            'std': np.std(processed_data),
            'min': np.min(processed_data),
            'max': np.max(processed_data)
        }
    }

    return stats


def main():
    """Main function to visualize preprocessing effects."""
    # Load configuration
    config = load_config()

    # Setup paths
    raw_dir = Path(config['paths']['data']['raw'])
    processed_dir = Path(config['paths']['data']['processed'])
    output_dir = project_root / 'output' / 'preprocessing_visualization'

    # Create visualization directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample one image from each class
    classes = ['AD', 'CN', 'MCI']

    for class_name in classes:
        print(f"\nProcessing {class_name} examples...")

        # Get a sample image
        raw_files = list((raw_dir / class_name).glob('*.nii*'))
        if not raw_files:
            print(f"No files found for class {class_name}")
            continue

        # Process first file found
        raw_path = raw_files[0]
        processed_path = processed_dir / class_name / f"{raw_path.stem}_processed.nii.gz"

        if not processed_path.exists():
            print(f"Processed file not found: {processed_path}")
            continue

        # Create class-specific output directory
        class_output_dir = output_dir / class_name

        # Visualize the pair
        stats = visualize_preprocessed_pair(
            raw_path,
            processed_path,
            class_output_dir
        )

        # Print statistics
        print(f"\n{class_name} Image Statistics:")
        for img_type, img_stats in stats.items():
            print(f"\n{img_type}:")
            for stat_name, stat_value in img_stats.items():
                print(f"{stat_name}: {stat_value}")


if __name__ == "__main__":
    main()
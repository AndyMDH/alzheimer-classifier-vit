"""
Visualization script for comparing raw and preprocessed MRI data.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*TorchScript.*')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib
import yaml
from typing import Dict, Any, Tuple, Optional
import logging
from tqdm import tqdm
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    CropForegroundd,
    ScaleIntensityd,
    NormalizeIntensityd,
    ResizeWithPadOrCropd
)

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        project_root = get_project_root()
        config_file = project_root / config_path
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found at {config_file}")
            
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            
        logger.info(f"Loaded configuration from {config_file}")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

def get_preprocessing_transforms(config: Dict[str, Any]) -> Compose:
    """Get preprocessing transforms pipeline."""
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes=config['dataset']['preprocessing']['orientation']),
        Spacingd(
            keys=["image"],
            pixdim=config['dataset']['preprocessing']['voxel_spacing'],
            mode="bilinear"
        ),
        CropForegroundd(
            keys=["image"],
            source_key="image",
            margin=config['dataset']['preprocessing']['crop_margin'],
            allow_smaller=False
        ),
        ScaleIntensityd(keys=["image"]),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        ResizeWithPadOrCropd(
            keys=["image"],
            spatial_size=[config['dataset']['input_size']] * 3
        )
    ])

def load_and_preprocess(file_path: Path, transforms: Compose) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load and preprocess a single file, returning both raw and processed versions."""
    try:
        # Load raw data
        nifti_img = nib.load(str(file_path))
        raw_data = nifti_img.get_fdata()
        
        # Apply preprocessing
        data_dict = transforms({"image": str(file_path)})
        processed_data = data_dict["image"][0].numpy()
        
        return raw_data, processed_data
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return None, None

def plot_comparison(raw_data: np.ndarray, processed_data: np.ndarray, 
                   output_path: Path, class_name: str) -> None:
    """Create a comparison figure of raw and processed data."""
    # Use non-interactive backend
    plt.switch_backend('agg')
    
    try:
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(f'Raw vs Processed Comparison: {class_name}', fontsize=16)

        # Function to get middle slices
        def get_middle_slices(volume):
            return (
                volume[volume.shape[0]//2, :, :],  # Sagittal
                volume[:, volume.shape[1]//2, :],  # Coronal
                volume[:, :, volume.shape[2]//2]   # Axial
            )

        # Get slices
        raw_slices = get_middle_slices(raw_data)
        processed_slices = get_middle_slices(processed_data)

        # Plot titles
        titles = ['Sagittal', 'Coronal', 'Axial']
        
        # Plot raw data
        for i, (slice_data, title) in enumerate(zip(raw_slices, titles), 1):
            ax = plt.subplot(2, 3, i)
            im = ax.imshow(slice_data, cmap='gray')
            ax.set_title(f'Raw - {title}')
            ax.axis('off')
            plt.colorbar(im, ax=ax)

        # Plot processed data
        for i, (slice_data, title) in enumerate(zip(processed_slices, titles), 4):
            ax = plt.subplot(2, 3, i)
            im = ax.imshow(slice_data, cmap='gray')
            ax.set_title(f'Processed - {title}')
            ax.axis('off')
            plt.colorbar(im, ax=ax)

        # Add statistics
        stats_text = (
            f'Raw Shape: {raw_data.shape}\n'
            f'Processed Shape: {processed_data.shape}\n\n'
            f'Raw Range: [{raw_data.min():.2f}, {raw_data.max():.2f}]\n'
            f'Processed Range: [{processed_data.min():.2f}, {processed_data.max():.2f}]\n\n'
            f'Raw Mean ± Std: {raw_data.mean():.2f} ± {raw_data.std():.2f}\n'
            f'Processed Mean ± Std: {processed_data.mean():.2f} ± {processed_data.std():.2f}'
        )
        plt.figtext(0.02, 0.02, stats_text, fontsize=10, family='monospace')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
    except Exception as e:
        logger.error(f"Error plotting comparison: {str(e)}")
        if 'fig' in locals():
            plt.close(fig)

def main():
    """Main function to visualize preprocessing comparisons."""
    try:
        # Load configuration
        config = load_config()
        
        # Setup paths
        raw_dir = Path(config['paths']['data']['raw'])
        output_dir = Path(config['paths']['output_dir']) / 'preprocessing_comparison'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing files from {raw_dir}")
        logger.info(f"Saving outputs to {output_dir}")
        
        # Get preprocessing transforms
        transforms = get_preprocessing_transforms(config)
        
        # Process each class
        for class_name in ['AD', 'CN', 'MCI']:
            class_dir = raw_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Directory not found: {class_dir}")
                continue
                
            # Get all files
            files = list(class_dir.glob('*.nii*'))
            if not files:
                logger.warning(f"No files found in {class_dir}")
                continue
                
            # Sample files
            sample_files = np.random.choice(files, size=min(5, len(files)), replace=False)
            logger.info(f"Processing {len(sample_files)} samples from {class_name}")
            
            for file_path in tqdm(sample_files, desc=f"Processing {class_name}", 
                                ncols=100, leave=True):
                try:
                    raw_data, processed_data = load_and_preprocess(file_path, transforms)
                    if raw_data is not None and processed_data is not None:
                        output_path = output_dir / f"{class_name}_{file_path.stem}_comparison.png"
                        plot_comparison(raw_data, processed_data, output_path, class_name)
                        logger.debug(f"Saved comparison for {file_path.name}")
                except Exception as e:
                    logger.error(f"Error processing {file_path.name}: {str(e)}")
                    continue
        
        logger.info(f"Completed processing. Comparisons saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()

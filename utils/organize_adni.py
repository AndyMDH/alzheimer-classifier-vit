# utils/organize_adni.py
import pandas as pd
import shutil
from pathlib import Path


def find_nifti_file(base_dir: Path, subject_id: str) -> Path:
    """
    Find a NIfTI file for a given subject by searching through directories.

    Args:
        base_dir: Base directory of ADNI dataset
        subject_id: Subject ID to search for (e.g., '031_S_0830')

    Returns:
        Path to the NIfTI file if found, None otherwise
    """
    # Search recursively for files matching the pattern
    pattern = f"ADNI_{subject_id}_MR_*N3__Scaled*.nii"
    matching_files = list(base_dir.rglob(pattern))

    if matching_files:
        return matching_files[0]
    return None


def organize_adni_data(csv_path: str, source_dir: str, destination_dir: str):
    """
    Organize ADNI data into AD/CN/MCI folders using metadata CSV

    Args:
        csv_path: Path to the metadata CSV file
        source_dir: Root directory containing ADNI dataset
        destination_dir: Where to organize files by diagnosis
    """
    # Read metadata
    print(f"Reading metadata from {csv_path}")
    df = pd.read_csv(csv_path)

    # Create destination directories
    dest_dir = Path(destination_dir)
    for dx in ['AD', 'CN', 'MCI']:
        (dest_dir / 'raw' / dx).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dest_dir}/raw/{dx}")

    # Track progress
    total_files = len(df)
    files_processed = 0
    files_not_found = 0
    files_copied = 0

    # Convert source_dir to Path object
    source_path = Path(source_dir)

    # Group files by diagnosis
    for _, row in df.iterrows():
        files_processed += 1

        # Find the NIfTI file for this subject
        nifti_file = find_nifti_file(source_path, row['Subject'])

        if nifti_file is None:
            print(f"Warning: File not found for subject {row['Subject']}")
            files_not_found += 1
            continue

        # Determine destination directory based on Group
        diagnosis = row['Group'].upper()  # AD, CN, or MCI
        dest_path = dest_dir / 'raw' / diagnosis / nifti_file.name

        # Copy file
        print(f"[{files_processed}/{total_files}] Copying {nifti_file.name} to {diagnosis} folder")
        shutil.copy2(nifti_file, dest_path)
        files_copied += 1

    # Print summary
    print("\nData organization complete!")
    print(f"Total files processed: {files_processed}")
    print(f"Files copied successfully: {files_copied}")
    print(f"Files not found: {files_not_found}")


if __name__ == "__main__":
    organize_adni_data(
        # Path to your CSV file
        csv_path='metadata/adni.csv',

        # Path to root directory of ADNI dataset
        source_dir=r'C:/Users/andyh/Downloads/ADNI1_Complete 1Yr 3T',

        # Path to your project's adni directory
        destination_dir='adni'
    )

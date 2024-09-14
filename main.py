from src.data.download_data import download_mri_data
from src.models.train import train_model

if __name__ == "__main__":
    # Download data
    download_mri_data()

    # Train model
    train_model()

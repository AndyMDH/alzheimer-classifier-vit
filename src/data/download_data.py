# src/data/download_data.py
import os
import requests

def download_mri_data(save_path="data/train"):
    url = "http://example-dataset-link.com"  # Replace with actual dataset URL
    response = requests.get(url)
    with open(os.path.join(save_path, "mri_data.zip"), "wb") as f:
        f.write(response.content)
    # Extract and preprocess further if needed

if __name__ == "__main__":
    download_mri_data()

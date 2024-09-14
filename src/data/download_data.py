import os
import requests

def download_dataset(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)

def main():
    dataset_url = "https://example.com/alzheimer_dataset.zip"
    save_path = os.path.join("data", "alzheimer_dataset.zip")
    download_dataset(dataset_url, save_path)

if __name__ == "__main__":
    main()
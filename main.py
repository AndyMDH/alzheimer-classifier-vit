from src.models.train import train
from src.utils.config import load_config

def main():
    config = load_config()
    train(config)

if __name__ == "__main__":
    main()
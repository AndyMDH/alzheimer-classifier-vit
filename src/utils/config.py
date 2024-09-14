from dataclasses import dataclass

@dataclass
class Config:
    data_dir: str = "./data"
    num_classes: int = 3
    batch_size: int = 32
    max_epochs: int = 100
    gpus: int = 1

def load_config():
    return Config()
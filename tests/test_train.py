# src/tests/test_train.py
import unittest
from src.models.train import train_model

class TestTraining(unittest.TestCase):
    def test_train(self):
        train_model(model_name='vit_b16', epochs=1, batch_size=1)

if __name__ == "__main__":
    unittest.main()

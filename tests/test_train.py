import unittest
import torch
from models.architectures.vit3d import create_vit_3d
from models.train import train_model
from monai.data import DataLoader, Dataset


class TestTrain(unittest.TestCase):
    def setUp(self):
        self.num_labels = 4
        self.batch_size = 2
        self.image_size = (1, 224, 224, 224)
        self.num_samples = 10

    def create_dummy_data(self):
        data = [
            {"image": torch.randn(*self.image_size), "label": torch.randint(0, self.num_labels, (1,))}
            for _ in range(self.num_samples)
        ]
        dataset = Dataset(data)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def test_train_model(self):
        model = create_vit_3d(self.num_labels)
        train_loader = self.create_dummy_data()
        val_loader = self.create_dummy_data()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            train_model(model, train_loader, val_loader, max_epochs=1, lr=1e-4, device=device)
            self.assertTrue(True)  # If we reach here without errors, the test passes
        except Exception as e:
            self.fail(f"Training failed with error: {str(e)}")


if __name__ == '__main__':
    unittest.main()
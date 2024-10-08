import unittest
import torch
from models.architectures import create_model
from models.train import train_model
from torch.utils.data import DataLoader, TensorDataset


class TestTrain(unittest.TestCase):
    def setUp(self):
        self.num_classes = 3  # AD, MCI, CN
        self.batch_size = 2
        self.image_size_2d = (1, 224, 224)  # (C, H, W)
        self.image_size_3d = (1, 224, 224, 224)  # (C, H, W, D)
        self.model_types = ['2d', '3d']
        self.num_epochs = 2

    def create_dummy_data(self, model_type):
        if model_type == '2d':
            x = torch.randn(10, *self.image_size_2d)
        else:  # 3d
            x = torch.randn(10, *self.image_size_3d)
        y = torch.randint(0, self.num_classes, (10,))
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def test_train_model(self):
        for model_type in self.model_types:
            model = create_model(model_type, self.num_classes)
            train_loader = self.create_dummy_data(model_type)
            val_loader = self.create_dummy_data(model_type)

            config = {
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'max_epochs': self.num_epochs,
                'learning_rate': 0.001,
                'early_stopping': {'patience': 5, 'min_delta': 0.01},
                'lr_scheduler': {'factor': 0.1, 'patience': 2},
                'seed': 42,
                'tensorboard_log_dir': 'logs/',
                'checkpoint_dir': 'checkpoints/'
            }

            try:
                train_model(model, train_loader, val_loader, config)
            except Exception as e:
                self.fail(f"Training failed for {model_type} model with error: {str(e)}")


if __name__ == '__main__':
    unittest.main()

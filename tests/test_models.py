import unittest
import torch
from models.architectures import create_model


class TestModels(unittest.TestCase):
    def setUp(self):
        self.num_classes = 3  # AD, MCI, CN
        self.batch_size = 2
        self.image_size_2d = (1, 224, 224)  # (C, H, W)
        self.image_size_3d = (1, 224, 224, 224)  # (C, H, W, D)
        self.model_types = ['2d', '3d']

    def test_model_creation(self):
        for model_type in self.model_types:
            model = create_model(model_type, self.num_classes)
            self.assertIsNotNone(model)

    def test_model_forward_pass(self):
        for model_type in self.model_types:
            model = create_model(model_type, self.num_classes)

            if model_type == '2d':
                x = torch.randn(self.batch_size, *self.image_size_2d)
            else:  # 3d
                x = torch.randn(self.batch_size, *self.image_size_3d)

            output = model(x)
            self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_model_output_range(self):
        for model_type in self.model_types:
            model = create_model(model_type, self.num_classes)

            if model_type == '2d':
                x = torch.randn(self.batch_size, *self.image_size_2d)
            else:  # 3d
                x = torch.randn(self.batch_size, *self.image_size_3d)

            output = model(x)

            # Check if output is a valid probability distribution
            self.assertTrue(torch.all(output >= 0))
            self.assertTrue(torch.all(output <= 1))
            self.assertTrue(torch.allclose(output.sum(dim=1), torch.tensor(1.0)))


if __name__ == '__main__':
    unittest.main()

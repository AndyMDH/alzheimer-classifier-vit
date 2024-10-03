import unittest
import torch
from models.architectures.vit2d import create_vit_2d
from models.architectures.vit3d import create_vit_3d
from models.architectures.cnn3d import create_cnn_3d

class TestModels(unittest.TestCase):
    def setUp(self):
        self.num_labels = 4
        self.batch_size = 2
        self.image_size_2d = (3, 224, 224)
        self.image_size_3d = (1, 224, 224, 224)

    def test_vit_2d(self):
        model = create_vit_2d(self.num_labels)
        x = torch.randn(self.batch_size, *self.image_size_2d)
        output = model(x)
        self.assertEqual(output.logits.shape, (self.batch_size, self.num_labels))

    def test_vit_3d(self):
        model = create_vit_3d(self.num_labels)
        x = torch.randn(self.batch_size, *self.image_size_3d)
        output = model(x)
        self.assertEqual(output.shape, (self.batch_size, self.num_labels))

    def test_cnn_3d(self):
        model = create_cnn_3d(self.num_labels)
        x = torch.randn(self.batch_size, *self.image_size_3d)
        output = model(x)
        self.assertEqual(output.shape, (self.batch_size, self.num_labels))

if __name__ == '__main__':
    unittest.main()
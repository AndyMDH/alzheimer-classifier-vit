import unittest
import os
from pathlib import Path
from data.data_loader import create_monai_dataset, get_transforms, prepare_data

class TestDataset(unittest.TestCase):
    def setUp(self):
        self.data_dir = Path('path/to/test/dataset')
        self.model_type = '3d_vit'

    def test_create_monai_dataset(self):
        transforms = get_transforms(self.model_type)
        dataset = create_monai_dataset(self.data_dir / 'train', transforms, self.model_type)
        self.assertIsNotNone(dataset)
        self.assertTrue(len(dataset) > 0)

    def test_get_transforms(self):
        transforms_2d = get_transforms('2d_vit')
        transforms_3d = get_transforms('3d_vit')
        self.assertIsNotNone(transforms_2d)
        self.assertIsNotNone(transforms_3d)
        self.assertNotEqual(transforms_2d, transforms_3d)

    def test_prepare_data(self):
        train_loader, val_loader, test_loader = prepare_data(str(self.data_dir), self.model_type, batch_size=2)
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)

if __name__ == '__main__':
    unittest.main()
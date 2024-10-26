import unittest
from pathlib import Path
from data.data_loader import ADNIDataset, get_transforms, prepare_data


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.data_dir = Path('path/to/test/dataset')
        self.model_types = ['2d', '3d']

    def test_adni_dataset(self):
        for model_type in self.model_types:
            transforms = get_transforms(model_type)
            dataset = ADNIDataset(str(self.data_dir), transform=transforms, model_type=model_type)
            self.assertIsNotNone(dataset)
            self.assertTrue(len(dataset) > 0)

            # Test __getitem__
            item = dataset[0]
            self.assertIn('image', item)
            self.assertIn('label', item)

            if model_type == '2d':
                self.assertEqual(len(item['image'].shape), 3)  # (C, H, W)
            else:
                self.assertEqual(len(item['image'].shape), 4)  # (C, H, W, D)

    def test_get_transforms(self):
        for model_type in self.model_types:
            transforms = get_transforms(model_type)
            self.assertIsNotNone(transforms)

    def test_prepare_data(self):
        for model_type in self.model_types:
            train_loader, val_loader, test_loader = prepare_data(
                data_root=str(self.data_dir),
                model_type=model_type,
                batch_size=2,
                val_ratio=0.15,
                test_ratio=0.15
            )
            self.assertIsNotNone(train_loader)
            self.assertIsNotNone(val_loader)
            self.assertIsNotNone(test_loader)

            # Test the shape of the batches
            for loader in [train_loader, val_loader, test_loader]:
                batch = next(iter(loader))
                self.assertIn('image', batch)
                self.assertIn('label', batch)
                self.assertEqual(batch['image'].shape[0], 2)  # batch size
                if model_type == '2d':
                    self.assertEqual(len(batch['image'].shape), 4)  # (B, C, H, W)
                else:
                    self.assertEqual(len(batch['image'].shape), 5)  # (B, C, H, W, D)


if __name__ == '__main__':
    unittest.main()

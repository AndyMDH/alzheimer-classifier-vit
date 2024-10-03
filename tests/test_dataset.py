import unittest
from src.data.data_loader import load_huggingface_dataset, create_monai_dataset, get_transforms

class TestDataset(unittest.TestCase):
    def setUp(self):
        self.dataset_name = 'your_dataset_name'
        self.model_type = '3d_vit'

    def test_load_huggingface_dataset(self):
        dataset = load_huggingface_dataset(self.dataset_name)
        self.assertIsNotNone(dataset)
        self.assertTrue('train' in dataset)
        self.assertTrue('validation' in dataset)
        self.assertTrue('test' in dataset)

    def test_create_monai_dataset(self):
        hf_dataset = load_huggingface_dataset(self.dataset_name)
        transforms = get_transforms(self.model_type)
        monai_dataset = create_monai_dataset(hf_dataset['train'], transforms)
        self.assertIsNotNone(monai_dataset)
        self.assertTrue(len(monai_dataset) > 0)

    def test_get_transforms(self):
        transforms_2d = get_transforms('2d_vit')
        transforms_3d = get_transforms('3d_vit')
        self.assertIsNotNone(transforms_2d)
        self.assertIsNotNone(transforms_3d)
        self.assertNotEqual(transforms_2d, transforms_3d)

if __name__ == '__main__':
    unittest.main()
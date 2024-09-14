import unittest
from utils.data_module import AlzheimerDataModule

class TestDataset(unittest.TestCase):
    def setUp(self):
        self.data_module = AlzheimerDataModule("./data", batch_size=32)
        self.data_module.setup()

    def test_dataset_length(self):
        self.assertGreater(len(self.data_module.train_ds), 0)
        self.assertGreater(len(self.data_module.val_ds), 0)

    def test_batch_shape(self):
        batch = next(iter(self.data_module.train_dataloader()))
        self.assertEqual(batch["image"].shape[0], 32)
        self.assertEqual(batch["image"].shape[1:], (1, 128, 128, 128))

if __name__ == '__main__':
    unittest.main()
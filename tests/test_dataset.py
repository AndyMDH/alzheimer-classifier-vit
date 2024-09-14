import unittest
import numpy as np
from src.data.preprocess import preprocess_3d_image


class TestDataset(unittest.TestCase):

    def setUp(self):
        # Create a dummy 3D image (e.g., 64x64x64 with random values)
        self.image = np.random.rand(64, 64, 64)

    def test_preprocessing(self):
        # Test the preprocessing function
        processed_image = preprocess_3d_image(self.image)

        # Assert the processed image has the correct shape
        self.assertEqual(processed_image.shape, (1, 128, 128, 128))

        # Additional checks can include data type and value range checks
        self.assertTrue(isinstance(processed_image, torch.Tensor))
        self.assertTrue(processed_image.max() <= 1.0)
        self.assertTrue(processed_image.min() >= 0.0)


if __name__ == "__main__":
    unittest.main()

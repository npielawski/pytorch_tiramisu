"""Checks for the classes and functions in layers.blurpool"""
import unittest
import torch
from tiramisu.layers.blurpool import BlurPool2d
import torch_unittest


class TestBlurPool(torch_unittest.TestCase):
    """Tests the BlurPool2d main class."""

    def test_dirac(self) -> None:
        """Checks that the result of the BlurPool2d is right on an example."""
        # Preparation
        data = torch.zeros(size=(1, 1, 8, 8), dtype=torch.float32)
        data[:, :, 2:6, 2:6] = 1.0  # White square in the middle
        blurpool_layer = BlurPool2d(1)
        # Calculation
        result = blurpool_layer(data)
        # Check
        expected = torch.tensor(
            [
                [
                    [
                        [0.0625, 0.2500, 0.2500, 0.0625],
                        [0.2500, 1.0000, 1.0000, 0.2500],
                        [0.2500, 1.0000, 1.0000, 0.2500],
                        [0.0625, 0.2500, 0.2500, 0.0625],
                    ]
                ]
            ],
            dtype=torch.float32,
        )
        self.assertTensorAlmostEqual(result, expected)

    def test_dense(self) -> None:
        """Checks that the output of the BlurPool2d has the right size (w/ 4D tensors)."""
        in_channels = 8
        # Preparation
        data = torch.zeros(size=(16, in_channels, 256, 256), dtype=torch.float32)
        blurpool_layer = BlurPool2d(in_channels)
        # Calculation
        result = blurpool_layer(data)
        # Check
        self.assertEqual(result.shape, torch.Size((16, in_channels, 128, 128)))


if __name__ == "__main__":
    unittest.main()

"""Addition of assertions for accounting for PyTorch objects.
"""
import unittest
import torch


class TestCase(unittest.TestCase):
    """Overloading for unittest to account for PyTorch classes."""

    def assertTensorAlmostEqual(  # pylint: disable=invalid-name # fits original unittest case
        self, first: torch.Tensor, second: torch.Tensor, places: int = 7
    ) -> None:
        """Checks that each element of first is approx. equal to each element of second.

        Args:
            first (torch.Tensor): The first Tensor
            second (torch.Tensor): The second Tensor to check
            places (int): Precision to use (how many decimal shall be used?).
                Defaults to 7.
        """
        epsilon = 10 ** (-places)
        is_close = torch.isclose(first, second, atol=epsilon)
        count_matching = torch.sum(is_close).item()
        count_total = is_close.numel()
        self.assertTrue(
            torch.all(is_close),
            f"{first} != {second}, {count_matching}/{count_total} elements matching.",
        )

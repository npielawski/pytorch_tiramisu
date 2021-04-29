"""Blur Pooling layer to replace max pooling.

Blur pooling replaces the max pooling operation because this operation isn't completely
shift-invariant (the output varies in unexpected ways when the input tensor values are
shifted). The new operation consists of a regular max pooling followed by a blur
operation (gaussian kernel).

TODO:
    * Expand to 3-dimensional data.
    * Allow different filter sizes (now constrained to 3x3).
"""
import torch
import torch.nn as nn


class BlurPool2d(nn.Sequential):
    """Blur Pooling Layer (MaxPool2d replacement)

    Adds a blurring operation after the max pooling such that the layer is
    actually shift-invariant (invariance to small translations of the input).

    Official webpage: https://richzhang.github.io/antialiased-cnns/
    Paper: https://arxiv.org/abs/1904.11486

    Note:
        Improves shift-invariance, might remove artifacts (for segmentation) and might
        improve performance. Slower than regular max pooling.

    Arguments:
        in_features (int): The number of channels of the input tensor.
    """

    __constants__ = ["in_features"]
    _blur_kernel = torch.tensor(
        [[1 / 16, 2 / 16, 1 / 16], [2 / 16, 4 / 16, 2 / 16], [1 / 16, 2 / 16, 1 / 16]]
    )

    def __init__(self, in_features: int):
        """
        Args:
            in_features (int): The number of channels in the input
        """
        super().__init__()
        self.in_features = in_features

        self.add_module("maxpool", nn.MaxPool2d(2, stride=1))
        blurpool = nn.Conv2d(
            in_features,
            in_features,
            kernel_size=3,
            padding=1,
            stride=2,
            bias=False,
            groups=in_features,
        )
        blurpool.weight = torch.nn.Parameter(
            self._blur_kernel.repeat(in_features, 1, 1, 1), requires_grad=False
        )
        self.add_module("blurpool", blurpool)

    def extra_repr(self) -> str:
        return "in_features={}".format(self.in_features)

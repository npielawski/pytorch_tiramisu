"""Contains a base model for deep learning models to inherit from and have additional
functionalities, such as summaries (like with Keras) and easier initialization.
"""
from __future__ import annotations

from abc import ABC
from collections.abc import Callable
import os
from typing import BinaryIO, IO, Tuple, Union

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """Creates a base class for Deep Learning models with nice features.

    The class allows any inheriting model to have printed summaries, and easy
    initialization of certain layers.
    """

    def __str__(self) -> str:
        """Returns the name of the current class."""
        return type(self).__qualname__

    def get_param_count(self) -> Tuple[int, int]:
        """Returns the number of parameters of the model.

        Returns (tuple):
            (The number of trainable params, The number of non-trainable params)
        """
        param_count_nogrd = 0
        param_count_grd = 0
        for param in self.parameters():
            if param.requires_grad:
                param_count_grd += param.size().numel()
            else:
                param_count_nogrd += param.size().numel()
        return param_count_grd, param_count_nogrd

    def summary(
        self, half: bool = False, printer: Callable[[str], None] = print
    ) -> None:
        """Logs some information about the neural network.
        Args:
            printer: The printing function to use.
        """
        layers_count = len(list(self.modules()))
        printer(f"Model {self} has {layers_count} layers.")
        param_grd, param_nogrd = self.get_param_count()
        param_total = param_grd + param_nogrd
        printer(f"-> Total number of parameters: {param_total:n}")
        printer(f"-> Trainable parameters:       {param_grd:n}")
        printer(f"-> Non-trainable parameters:   {param_nogrd:n}")
        approx_size = param_total * (2.0 if half else 4.0) * 10e-7
        printer(f"Uncompressed size of the weights: {approx_size:.1f}MB")

    def save(self, filename: Union[str, os.PathLike, BinaryIO, IO[bytes]]) -> None:
        """Saves the model"""
        torch.save(self, filename)

    def half(self) -> BaseModel:
        """Transforms all the weights of the model in half precision.

        Note: this function fixes an issue on BatchNorm being half precision.
            See: https://discuss.pytorch.org/t/training-with-half-precision/11815/2
        """
        super().half()
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()
        return self

    def initialize_kernels(
        self,
        initializer: Callable[[torch.Tensor], torch.Tensor],
        conv: bool = False,
        linear: bool = False,
        batchnorm: bool = False,
    ) -> None:
        """Initializes the chosen set of kernels of the model.

        Args:
            initializer: a function that will apply the weight initialization.
            conv: Will initialize the kernels of the convolutions.
            linear: Will initialize the kernels of the linear layers.
            batchnorm: Will initialize the kernels of the batch norm layers.
        """
        for layer in self.modules():
            if linear and isinstance(layer, nn.Linear):
                initializer(layer.weight)
                continue
            if conv and isinstance(layer, nn.Conv2d):
                initializer(layer.weight)
                continue
            if batchnorm and isinstance(layer, (nn.BatchNorm2d, nn.GroupNorm)):
                initializer(layer.weight)
                continue

    def initialize_biases(
        self,
        initializer: Callable[[torch.Tensor], torch.Tensor],
        conv: bool = False,
        linear: bool = False,
        batchnorm: bool = False,
    ) -> None:
        """Initializes the chosen set of biases of the model.

        Args:
            initializer: A function that will apply the weight initialization.
            conv: Will initialize the biases of the convolutions.
            linear: Will initialize the biases of the linear layers.
            batchnorm: Will initialize the biases of the batch norm layers.
            **kwargs: Extra arguments to pass to the initializer function.
        """
        for layer in self.modules():
            if layer.bias is None:
                continue
            if linear and isinstance(layer, nn.Linear):
                initializer(layer.bias)  # type: ignore
            if conv and isinstance(layer, nn.Conv2d):
                initializer(layer.bias)  # type: ignore
            if batchnorm and isinstance(layer, (nn.BatchNorm2d, nn.GroupNorm)):
                initializer(layer.bias)  # type: ignore

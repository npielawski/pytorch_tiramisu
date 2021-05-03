"""Implementation of the Tiramisu Deep Neural Network.

The file contains the modules and main class to instantiate a Tiramisu model. The
model isn't the same as in the original paper and has various improvements, such as
checkpointing, different types of upsampling, etc.

TODO:
    * Add unit-tests.
    * Make the Neural Network work at any input size (currently only works with
      powers of 2).
    * Add a default bank and make NN work with 3-D data.
"""
from collections.abc import Callable
from enum import Enum, auto
from functools import partial
from typing import Dict, List, Tuple

# Deep Learning imports
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

# Local imports
from .model import BaseModel


class ModuleName(Enum):
    """Module names that define which layers to use in the Tiramisu model."""

    CONV = auto()  # Convolution operations
    CONV_INIT = auto()  # Initial (1st) conv. operation. Kernel size must be provided.
    CONV_FINAL = auto()  # Final convolution. 1x1 kernel and reduce output to C classes.
    BATCHNORM = auto()  # Batch normalization
    POOLING = auto()  # Pooling operation (must reduce input size by a factor of two)
    # Note: if the size is odd, round *up* to the closest integer.
    DROPOUT = auto()  # Dropout
    UPSAMPLE = auto()  # Upsampling operation (must be by a factor of two)
    ACTIVATION = auto()  # Activation function to use everywhere
    ACTIVATION_FINAL = auto()  # Act. function at the last layer (e.g.softmax)


ModuleBankType = Dict[ModuleName, Callable[..., nn.Module]]

UPSAMPLE_NEAREST = lambda in_channels, out_channels: nn.Sequential(
    nn.UpsamplingNearest2d(),
    nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        padding=1,
    ),
    nn.ReLU(inplace=True),
)
# Pixel shuffle see: https://arxiv.org/abs/1609.05158
UPSAMPLE_PIXELSHUFFLE = lambda in_channels, out_channels: nn.Sequential(
    nn.Conv2d(
        in_channels=in_channels,
        out_channels=4 * out_channels,
        kernel_size=3,
        padding=1,
    ),
    nn.ReLU(inplace=True),
    nn.PixelShuffle(2),
)
UPSAMPLE_TRANPOSE = lambda in_channels, out_channels: nn.Sequential(
    nn.ConvTranspose2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=2,
        padding=0,
    ),
    nn.ReLU(inplace=True),
)
DEFAULT_MODULE_BANK: ModuleBankType = {
    ModuleName.CONV: nn.Conv2d,
    ModuleName.CONV_INIT: partial(nn.Conv2d, kernel_size=3, padding=1),
    ModuleName.CONV_FINAL: nn.Conv2d,
    ModuleName.BATCHNORM: nn.BatchNorm2d,
    ModuleName.POOLING: nn.MaxPool2d,
    ModuleName.DROPOUT: partial(nn.Dropout2d, p=0.2, inplace=True),
    ModuleName.UPSAMPLE: UPSAMPLE_NEAREST,
    ModuleName.ACTIVATION: partial(nn.ReLU, inplace=True),
    ModuleName.ACTIVATION_FINAL: nn.Identity,
}


def _denselayer_factory(
    norm: nn.Module,
    activation: nn.Module,
    conv: nn.Module,
) -> Callable[..., torch.Tensor]:
    """This returns a callback to implement checkpointing."""

    def bn_function(x: torch.Tensor) -> torch.Tensor:
        x = norm(x)
        x = activation(x)
        x = conv(x)
        return x

    return bn_function


class DenseLayer(nn.Module):
    """A complete BN-ReLU-Conv-DropOut layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        module_bank: ModuleBankType,
        checkpoint: bool = False,
    ):
        """A superlayer containing all the basic blocs together.

        Performs a BatchNorm-ReLU-Conv-DropOut set of operation. Each individual part
        can be replaced by modifying the bank entries. More details in DenseUNet class.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            module_bank (ModuleBankType): The layers to use for common operations.
            checkpoint (bool, optional): Memory checkpointing. Defaults to False.
        """
        super().__init__()
        self.checkpoint = checkpoint
        # Standard Tiramisu Layer (BN-ReLU-Conv-DropOut)
        self.add_module(
            "batchnorm",
            module_bank[ModuleName.BATCHNORM](num_features=in_channels),
        )
        self.add_module("relu", module_bank[ModuleName.ACTIVATION]())
        self.add_module(
            "conv",
            module_bank[ModuleName.CONV](
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.add_module("dropout", module_bank[ModuleName.DROPOUT]())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the foward pass of the layer."""
        assert isinstance(self.batchnorm, nn.Module)
        assert isinstance(self.relu, nn.Module)
        assert isinstance(self.conv, nn.Module)
        assert isinstance(self.dropout, nn.Module)
        bn_function = _denselayer_factory(self.batchnorm, self.relu, self.conv)
        if self.checkpoint and x.requires_grad:
            x = cp.checkpoint(bn_function, x)
        else:
            x = bn_function(x)
        x = self.dropout(x)
        return x


class DenseBlock(nn.ModuleDict):
    """Implementation of the denseblock of the Tiramisu, with all the skip connections."""

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        nb_layers: int,
        module_bank: ModuleBankType,
        skip_block: bool = False,
        checkpoint: bool = False,
        prefix: str = "",
    ):  # pylint: disable=too-many-arguments
        """DenseBlock

        Args:
            in_channels (int): Number of channels in the input tensor.
            growth_rate (int): Number of channels in each DenseLayer.
            nb_layers (int): Number of DenseLayers to use.
            module_bank (ModuleBankType): The layers to use for common operations.
            skip_block (bool, optional): Adds a skip-connection over the full
                DenseBlock. Defaults to False.
            checkpoint (bool, optional): Memory checkpointing. Defaults to False.
            prefix (str, optional): Name prefix (pref. in snake case). Defaults to "".
        """
        super().__init__()
        for i in range(nb_layers):
            self.add_module(
                f"{prefix}dense_layer_{i+1}",
                DenseLayer(
                    in_channels + i * growth_rate,
                    growth_rate,
                    module_bank,
                    checkpoint,
                ),
            )

        self.skip_block = skip_block

    def forward(self, inp: torch.Tensor) -> torch.Tensor:  # type: ignore # pylint: disable=arguments-differ
        """Computes the forward pass of the denseblock."""
        skip_connections = [inp]

        for _, layer in self.items():
            out = layer(skip_connections)
            skip_connections.append(out)

        if self.skip_block:
            # Returns all of the x's, except for the first x (inp), concatenated
            # As we are not supposed to have skip connections over the full dense block.
            # See original tiramisu paper for more details
            return torch.cat(skip_connections[1:], 1)

        # Returns all of the x's concatenated
        return torch.cat(skip_connections, 1)


class TransitionDown(nn.Sequential):
    """A layer with a convolution and some operation that divides the resolution by 2."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        module_bank: ModuleBankType,
    ):
        """Layer dedicated to reducing the output width and height by a factor of two.

        Note: The POOLING module must downsize the input by two-folds.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            module_bank (ModuleBankType): The layers to use for common operations.
        """
        super().__init__()
        self.add_module("batchnorm", module_bank[ModuleName.BATCHNORM](in_channels))
        self.add_module("relu", module_bank[ModuleName.ACTIVATION]())
        self.add_module(
            "conv",
            module_bank[ModuleName.CONV](
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )
        self.add_module("dropout", module_bank[ModuleName.DROPOUT]())
        self.add_module("pool", module_bank[ModuleName.POOLING]())


def center_crop(layer: torch.Tensor, max_height: int, max_width: int) -> torch.Tensor:
    """ Crops a given layer to a certain size by removing equal margins all around."""
    _, _, height, width = layer.size()
    xy1 = (width - max_width) // 2
    xy2 = (height - max_height) // 2
    return layer[:, :, xy2 : (xy2 + max_height), xy1 : (xy1 + max_width)]


class TransitionUp(nn.Module):
    """Layer that increases the spatial resolution by a factor of 2."""

    def __init__(
        self, in_channels: int, out_channels: int, module_bank: ModuleBankType
    ):
        """Layer dedicated to increasing the output width and height by a factor of two.

        Note: The UPSAMPLE module must increase the scale of the input by two-folds.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            module_bank (ModuleBankType): The layers to use for common operations.
        """
        super().__init__()
        self.upsampling_layer = module_bank[ModuleName.UPSAMPLE](
            in_channels, out_channels
        )

    def forward(self, inp: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Computes the forward pass of the layer."""
        out = self.upsampling_layer(inp)
        out = center_crop(out, skip.size(2), skip.size(3))
        out = torch.cat([skip, out], 1)
        return out


class DenseUNet(BaseModel):
    """DensUNet
    Paper: The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for
    Semantic Segmentation
    URL: https://arxiv.org/pdf/1611.09326.pdf
    Notes:
        Coded with the help of https://github.com/bfortuner/pytorch_tiramisu
        MIT License - Copyright (c) 2018 Brendan Fortuner
        and the help of https://github.com/keras-team/keras-contrib
        MIT License - Copyright (c) 2017 Fariz Rahman
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        init_conv_filters: int = 48,
        structure: Tuple[List[int], int, List[int]] = (
            [4, 4, 4, 4, 4],  # Down blocks
            4,  # bottleneck layers
            [4, 4, 4, 4, 4],  # Up blocks
        ),
        growth_rate: int = 12,
        compression: float = 1.0,
        early_transition: bool = False,
        include_top: bool = True,
        checkpoint: bool = False,
        module_bank: ModuleBankType = None,
    ):  # pylint: disable=too-many-arguments,too-many-locals
        """Creates a Tiramisu/Fully Convolutional DenseNet Neural Network for
        dense image prediction (e.g. segmentatation or pixel regression).

        Args:
            in_channels (int): The number of channels of the input images
            out_channels (int, optional): The number of predicted channels.
                Defaults to 1.
            init_conv_filters (int, optional): The number of filters of the
                very first layer. Defaults to 48.
            structure (Tuple[List[int], int, List[int]], optional): The number
                of DenseBlocks that compose the U-Net-shaped model. The tuple
                corresponds to the number and size of each dense block in,
                respectively, the encoding part, the bottleneck, and the
                decoding part. Defaults to (
                    [4, 4, 4, 4, 4],  # Down blocks
                    4,  # bottleneck layers
                    [4, 4, 4, 4, 4],  # Up blocks
                ).
            growth_rate (int, optional): The rate at which the DenseBlocks
                layers grow. Defaults to 12.
            compression (float, optional): Optimization where each of the
                DenseBlocks channels are reduced by a factor between 0 and 100%.
                Defaults to 1.0 (no compression).
            early_transition (bool, optional): Optimization where the input is
                downscaled by a factor of two after the first layer without
                skip-connection. Reduces memory usage. Defaults to False.
            include_top (bool, optional): Includes the final convolution and
                activation function. Defaults to True.
            checkpoint (bool, optional): Activates memory checkpointing, which
                reduces memory usage but increases latency. Defaults to False.
                See: https://arxiv.org/pdf/1707.06990.pdf
            module_bank (ModuleBankType, optional): The layers to use for
                common operations. Defaults to None (=DEFAULT_MODULE_BANK).
        """

        super().__init__()
        self.out_channels = out_channels
        self.init_conv_filters = init_conv_filters
        self.down_blocks = structure[0]
        self.bottleneck_layers = structure[1]
        self.up_blocks = structure[2]
        self.growth_rate = growth_rate
        self.compression = compression
        self.early_transition = early_transition
        self.include_top = include_top
        self.module_bank = module_bank or DEFAULT_MODULE_BANK

        channels_count = init_conv_filters
        # Keeps track of the number of channels for each skip-connection.
        skip_connections: List[int] = []

        # First layer
        self.conv_init = self.module_bank[ModuleName.CONV_INIT](
            in_channels=in_channels, out_channels=init_conv_filters
        )

        if early_transition:
            self.early_transition_down = TransitionDown(
                in_channels=channels_count,
                out_channels=int(channels_count * compression),
                module_bank=self.module_bank,
            )
            channels_count = int(channels_count * compression)

        # Downsampling part
        self.layers_down = nn.ModuleDict()
        self.transitions_down = nn.ModuleDict()
        for i, block_size in enumerate(self.down_blocks):
            self.layers_down.add_module(
                f"layers_down_{i}",
                DenseBlock(
                    in_channels=channels_count,
                    growth_rate=growth_rate,
                    nb_layers=block_size,
                    module_bank=self.module_bank,
                    skip_block=False,
                    checkpoint=checkpoint,
                    prefix="layers_down_{i}_",
                ),
            )
            channels_count += growth_rate * block_size
            skip_connections.insert(0, channels_count)
            self.transitions_down.add_module(
                f"transition_down_{i}",
                TransitionDown(
                    in_channels=channels_count,
                    out_channels=int(channels_count * compression),
                    module_bank=self.module_bank,
                ),
            )
            channels_count = int(channels_count * compression)

        # Bottleneck
        self.bottleneck = DenseBlock(
            in_channels=channels_count,
            growth_rate=growth_rate,
            nb_layers=self.bottleneck_layers,
            module_bank=self.module_bank,
            skip_block=True,
            checkpoint=checkpoint,
            prefix="bottleneck_",
        )
        prev_block_channels = growth_rate * self.bottleneck_layers
        channels_count += prev_block_channels

        # Upsampling part
        self.transitions_up = nn.ModuleDict()
        self.layers_up = nn.ModuleDict()
        for i, block_size in enumerate(self.up_blocks[:-1]):
            self.transitions_up.add_module(
                f"transition_up_{i}",
                TransitionUp(
                    in_channels=prev_block_channels,
                    out_channels=prev_block_channels,
                    module_bank=self.module_bank,
                ),
            )
            channels_count = prev_block_channels + skip_connections[i]
            self.layers_up.add_module(
                f"layers_up_{i}",
                DenseBlock(
                    in_channels=channels_count,
                    growth_rate=growth_rate,
                    nb_layers=block_size,
                    module_bank=self.module_bank,
                    skip_block=True,
                    checkpoint=checkpoint,
                    prefix="layers_up_{i}_",
                ),
            )
            prev_block_channels = growth_rate * block_size
            channels_count += prev_block_channels

        self.transitions_up.add_module(
            "transition_up_last",
            TransitionUp(
                in_channels=prev_block_channels,
                out_channels=prev_block_channels,
                module_bank=self.module_bank,
            ),
        )
        channels_count = prev_block_channels + skip_connections[-1]
        self.layers_up.add_module(
            "layers_up_last",
            DenseBlock(
                in_channels=channels_count,
                growth_rate=growth_rate,
                nb_layers=self.up_blocks[-1],
                module_bank=self.module_bank,
                skip_block=False,
                checkpoint=checkpoint,
                prefix="layers_up_last_",
            ),
        )
        channels_count += growth_rate * self.up_blocks[-1]

        if early_transition:
            self.early_transition_up = TransitionUp(
                in_channels=channels_count,
                out_channels=channels_count,
                module_bank=self.module_bank,
            )
            channels_count += init_conv_filters

        # Last layer
        if include_top:
            self.conv_final = self.module_bank[ModuleName.CONV_FINAL](
                in_channels=channels_count,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            self.activation_final = self.module_bank[ModuleName.ACTIVATION_FINAL]()

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Computes the forward pass of the Tiramisu network."""
        x = self.conv_init(inp)

        transition_skip = None
        if self.early_transition:
            transition_skip = x
            x = self.early_transition_down(x)

        skip_connections = []
        for layer_down, transition_down in zip(
            self.layers_down.values(), self.transitions_down.values()
        ):
            x = layer_down(x)
            skip_connections.append(x)
            x = transition_down(x)

        x = self.bottleneck(x)

        for transition_up, layer_up in zip(
            self.transitions_up.values(), self.layers_up.values()
        ):
            skip = skip_connections.pop()
            x = transition_up(x, skip)
            x = layer_up(x)

        if self.early_transition:
            x = self.early_transition_up(x, skip=transition_skip)

        if self.include_top:
            # Computation of the final 1x1 convolution
            y_pred = self.conv_final(x)
            return self.activation_final(y_pred)

        return x

    def get_channels_count(self) -> List[int]:
        """Counts the number of out channels for each DenseBlocks and transitions.

        Returns: The list containing for each layers the number of (input) channels.
        """
        channels_count = [self.init_conv_filters]
        skip_connections: List[int] = []

        if self.early_transition:
            channels_count.append(int(channels_count[-1] * self.compression))

        # Downsampling part
        for block_size in self.down_blocks:
            channels_count.append(channels_count[-1] + self.growth_rate * block_size)
            skip_connections.insert(0, channels_count[-1])
            channels_count.append(int(channels_count[-1] * self.compression))

        # Bottleneck
        prev_block_channels = self.growth_rate * self.bottleneck_layers
        channels_count.append(channels_count[-1] + prev_block_channels)

        # Upsampling part
        for i, block_size in enumerate(self.up_blocks[:-1]):
            channels_count.append(prev_block_channels + skip_connections[i])
            prev_block_channels = self.growth_rate * block_size
            channels_count.append(channels_count[-1] + prev_block_channels)

        channels_count.append(prev_block_channels + skip_connections[-1])
        channels_count.append(
            channels_count[-1] + self.growth_rate * self.up_blocks[-1]
        )

        if self.early_transition:
            channels_count.append(channels_count[-1] + self.init_conv_filters)

        if self.include_top:
            channels_count.append(self.out_channels)

        return channels_count

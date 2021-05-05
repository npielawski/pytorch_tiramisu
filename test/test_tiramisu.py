"""Checks for the classes and functions in tiramisu"""
from functools import partial
import os
import unittest
import warnings
import numpy as np
import skimage
import skimage.io as skio
import skimage.transform as sktr
import torch
from torch import nn
from torch import optim
import torch_unittest
from tiramisu import ModuleName, DEFAULT_MODULE_BANK
from tiramisu import DenseLayer, DenseBlock, DenseUNet

OUTPUT_FOLDER = "test/output"


class TestDenseLayer(torch_unittest.TestCase):
    """Checks for the DenseLayer class."""

    def test_structure(self) -> None:
        """Checks that the layers of DenseLayer are correct and in good order."""
        dense_layer = DenseLayer(
            in_channels=8,
            out_channels=16,
            module_bank=DEFAULT_MODULE_BANK,
            checkpoint=False,
        )
        modules = [type(module).__name__ for module in dense_layer.children()]
        self.assertListEqual(modules, ["BatchNorm2d", "ReLU", "Conv2d", "Dropout2d"])

    def test_run_evalmode(self) -> None:
        """Checks the forward pass of the DenseLayer in eval mode."""
        torch.random.manual_seed(0)
        in_channels = 8
        # Preparation
        data = torch.ones(size=(1, in_channels, 1, 1), dtype=torch.float32)
        dense_layer = DenseLayer(
            in_channels=in_channels,
            out_channels=16,
            module_bank=DEFAULT_MODULE_BANK,
            checkpoint=False,
        ).eval()  # Disables batchnorm and dropout
        # Calculation
        result = dense_layer(data)
        # Check
        conv2d = dense_layer.conv
        assert isinstance(conv2d, nn.Conv2d)
        # The result can be computed in closed form
        weights, biases = conv2d.weight[:, :, 1, 1].sum(1), conv2d.bias
        self.assertTensorAlmostEqual(result.ravel(), weights + biases, places=4)

    def test_run_trainmode(self, checkpoint: bool = False) -> None:
        """Checks the forward pass of the DenseLayer in train mode."""
        torch.random.manual_seed(0)
        in_channels, out_channels = 8, 16
        # Preparation
        data = torch.ones(size=(2, in_channels, 1, 1), dtype=torch.float32)
        dense_layer = DenseLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            module_bank=DEFAULT_MODULE_BANK,
            checkpoint=checkpoint,
        )
        dropped = torch.zeros(out_channels, dtype=torch.bool)
        dropped[[3, 9, 13]] = True
        # Calculation
        result = dense_layer(data)
        # Check
        # Note that the batchnormalization makes data = 0.0
        conv2d, dropout = dense_layer.conv, dense_layer.dropout
        assert isinstance(conv2d, nn.Conv2d)
        assert isinstance(dropout, nn.Dropout2d)
        # The result can be computed in closed form
        biases = conv2d.bias
        assert isinstance(biases, torch.Tensor)
        biases_scaled = biases / (1.0 - dropout.p)  # Dropout rescales the tensor
        expected = torch.where(dropped, torch.zeros_like(biases_scaled), biases_scaled)
        self.assertTensorAlmostEqual(result[0].ravel(), expected, places=4)

    def test_run_trainmode_checkpoint(self) -> None:
        """Checks the forward pass of the DenseLayer in train mode with checkpointing."""
        self.test_run_trainmode(checkpoint=True)


class TestDenseBlock(torch_unittest.TestCase):
    """Checks for the DenseBlock class."""

    def test_structure(self) -> None:
        """Checks that the DenseLayers of the DenseBlock are consistent."""
        nb_layers = 4
        dense_layer = DenseBlock(
            in_channels=8,
            growth_rate=16,
            nb_layers=nb_layers,
            module_bank=DEFAULT_MODULE_BANK,
            skip_block=False,
            checkpoint=False,
            prefix="",
        )
        modules = [name for name, _ in dense_layer.named_children()]
        expected = [f"dense_layer_{i}" for i in range(1, 5)]
        self.assertListEqual(modules, expected)


def nchw2hwc(image: torch.Tensor) -> torch.Tensor:
    """Transforms an image from torch to regular format (NCHW to HWC)."""
    return np.transpose(image[0], (1, 2, 0))


def hwc2nchw(image: np.ndarray) -> np.ndarray:
    """Transforms an image from regular format to torch format (HWC to NCHW)."""
    return np.transpose(image, (2, 0, 1))[np.newaxis]


class TestDenseUNet(torch_unittest.TestCase):
    """Checks for the DenseUNet class."""

    def setUp(self) -> None:
        """Sets up the DenseUNet model with the settings from the example in readme."""
        torch.random.manual_seed(0)

        module_bank = DEFAULT_MODULE_BANK.copy()
        # Dropout
        module_bank[ModuleName.DROPOUT] = partial(nn.Dropout2d, p=0.2, inplace=True)
        # Every activation in the model is going to be a GELU (Gaussian Error Linear
        # Units function). GELU(x) = x * Φ(x)
        # See: https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
        module_bank[ModuleName.ACTIVATION] = nn.GELU
        # Example for image reconstruction:
        module_bank[ModuleName.ACTIVATION_FINAL] = nn.Sigmoid
        # Example for segmentation:
        # module_bank[ModuleName.ACTIVATION_FINAL] = partial(nn.LogSoftmax, dim=1)
        # Example for regression (default):
        # module_bank[ModuleName.ACTIVATION_FINAL] = nn.Identity

        self.model = DenseUNet(
            in_channels=1,  # Grayscale
            out_channels=3,  # 3-channel output (RGB)
            init_conv_filters=8,  # Number of channels outputted by the 1st convolution
            structure=(
                [2, 2],  # Down blocks
                2,  # bottleneck layers
                [2, 2],  # Up blocks
            ),
            growth_rate=8,  # Growth rate of the DenseLayers
            compression=1.0,  # No compression
            early_transition=False,  # No early transition
            include_top=True,  # Includes last layer and activation
            checkpoint=False,  # No memory checkpointing
            module_bank=module_bank,  # Modules to use
        )
        # Creating an optimizer
        # assert isinstance(self.model.parameters(), torch.Tensor)
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3)

        # Loading the test image
        try:
            image = skimage.img_as_float32(skio.imread("test/data/woman_in_pink.jpg"))
        except FileNotFoundError as e:
            raise Exception(
                "Please run ./download_mockdata.sh before running this test."
            ) from e
        image = sktr.rescale(
            image[230 : 230 + 512, 60 : 60 + 512],  # Taking an even sized patch.
            scale=0.25,  # Makes the image of size 128x128 for faster computations.
            multichannel=2,  # Specifies the image is RGB
        )
        image_grayscale = image.mean(axis=-1, keepdims=True)

        self.X = torch.tensor(hwc2nchw(image_grayscale), dtype=torch.float32)
        self.y_true = torch.tensor(hwc2nchw(image), dtype=torch.float32)

        self.loss = nn.BCELoss()

        # Running the model (single training step)
        self.optimizer.zero_grad()
        self.y_pred_before_step = self.model(self.X)
        # Loss function
        self.loss_before_step = self.loss(
            self.y_pred_before_step.flatten(start_dim=1),
            self.y_true.flatten(start_dim=1),
        )
        self.loss_before_step.backward()
        self.optimizer.step()

        self.y_pred_after_step = self.model(self.X)
        self.loss_after_step = self.loss(
            self.y_pred_after_step.flatten(start_dim=1),
            self.y_true.flatten(start_dim=1),
        )

        # The prediction of the neural network will be outputed here
        if not os.path.exists(OUTPUT_FOLDER):
            os.mkdir(OUTPUT_FOLDER)

        # Saving the images
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skio.imsave(
                os.path.join(OUTPUT_FOLDER, "groundtruth.png"),
                skimage.img_as_ubyte(image),
            )
            skio.imsave(
                os.path.join(OUTPUT_FOLDER, "input.png"),
                skimage.img_as_ubyte(image_grayscale),
            )
            skio.imsave(
                os.path.join(OUTPUT_FOLDER, "prediction_before_step.png"),
                skimage.img_as_ubyte(nchw2hwc(self.y_pred_before_step.detach())),
            )
            skio.imsave(
                os.path.join(OUTPUT_FOLDER, "prediction_after_step.png"),
                skimage.img_as_ubyte(nchw2hwc(self.y_pred_after_step.detach())),
            )

    def test_structure_main(self) -> None:
        """Checks that the main structure of the model is consistent."""
        main_modules = [name for name, _ in self.model.named_children()]
        expected = [
            "conv_init",
            "layers_down",
            "transitions_down",
            "bottleneck",
            "transitions_up",
            "layers_up",
            "conv_final",
            "activation_final",
        ]
        self.assertListEqual(main_modules, expected)

    def test_loss_improvement(self) -> None:
        """Checks that the loss improved after training one step."""
        self.assertGreater(self.loss_before_step, self.loss_after_step)

    def test_gradients(self) -> None:
        """Checks that the gradients exist everywhere in the model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                with self.subTest(name=name):
                    self.assertIsNotNone(param)
                    self.assertNotEqual(torch.sum(param.grad ** 2).item(), 0.0)

    def test_image101(self) -> None:
        """Checks for inputing images that do not have a power of 2 for size."""
        X = self.X[:, :, :101, :101]
        y_pred = self.model(X)
        self.assertEqual(X.size()[2:], y_pred.size()[2:])


if __name__ == "__main__":
    unittest.main()

[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/) [![DOI](https://zenodo.org/badge/242668685.svg)](https://zenodo.org/badge/latestdoi/242668685)

# 🔥 Better Tiramisu for PyTorch 🔥

Implementation of the Tiramisu Neural network for PyTorch with new features such
as:
* Memory-efficient version (trade-off between memory and speed).
* Different types of upsampling (deconvolution, upsampling and pixel shuffle).
* Different types of pooling (max-pooling, avg-pooling, blur-pooling).
* The depth and width of the Tiramisu is fully configurable.
* Early-transition can be enabled when the input images are big.
* The activation function of the last layer can be disabled or modified.

## Getting Started

The package can be installed from the repository with:
```console
$ pip3 install git+https://github.com/npielawski/pytorch_tiramisu
```

You can try the model in Python with:
```py
from tiramisu import DenseUNet

model = DenseUNet(
    in_channels=3,
    nb_classes=1,
    init_conv_size=3,
    init_conv_filters=48,
    init_conv_stride=1,
    down_blocks=(4, 4, 4, 4, 4),
    bottleneck_layers=4,
    up_blocks=(4, 4, 4, 4, 4),
    growth_rate=12,
    compression=1.0,
    dropout_rate=0.2,
    upsampling_type="upsample",
    early_transition=False,
    transition_pooling="max",
    batch_norm="batchnorm",
    include_top=True,
    activation_func=None,
    efficient=False,
)

# Initializes all the convolutional kernel weights.
model.initialize_kernels(nn.init.kaiming_uniform_, conv=True)
# Shows some information about the model.
model.summary()
```

This example tiramisu network has a depth of len(down_blocks) = 5.


## Documentation

The parameters of the constructor are explained as following:
* nb_classes: The number of classes to predict.
* in_channels: The number of channels of the input images.
* init_conv_size: The size of the very first first layer.
* init_conv_filters: The number of filters of the very first layer.
* init_conv_stride: The stride of the very first layer.
* down_blocks: The number of DenseBlocks and their size in the
    compressive part.
* bottleneck_layers: The number of DenseBlocks and their size in the
    bottleneck part.
* up_blocks: The number of DenseBlocks and their size in the
    reconstructive part.
* growth_rate: The rate at which the DenseBlocks layers grow.
* compression: Optimization where each of the DenseBlocks layers are reduced
    by a factor between 0 and 1. (1.0 does not change the original arch.)
* dropout_rate: The dropout rate to use.
* upsampling_type: The type of upsampling to use in the TransitionUp layers.
    available options: ["upsample" (default), "deconv", "pixelshuffle"]
    For Pixel shuffle see: https://arxiv.org/abs/1609.05158
* early_transition: Optimization where the input is downscaled by a factor
    of two after the first layer. You can thus reduce the numbers of down
    and up blocks by 1.
* transition_pooling: The type of pooling to use during the transitions.
    available options: ["max" (default), "avg", "blurpool"]
* batch_norm: Type of batch normalization to use.
    available options: ["batchnorm" (default), None]
    For FRN see: https://arxiv.org/pdf/1911.09737v1.pdf
* include_top (bool): Including the top layer, with the last convolution
    and softmax/sigmoid (True) or returns the embeddings for each pixel
    of the input image (False).
* activation_func (func): Activation function to use at the end of the model.
* efficient (bool): Memory efficient version of the Tiramisu.
    See: https://arxiv.org/pdf/1707.06990.pdf


## Tips and tricks

* Make sure the features you are interested in fit approximately the perceptive field.
For instance, if you have an object that measures 50 pixels, you need at approx. 6
levels of resolution in down/up blocks. Or use early transition, which down samples
the input by two.
* If you need to reduce the memory footprint, trying out the efficient version,
enabling the early transition is a great way to start. Then, using compression,
reducing the growth rate and finally the number of dense blocks in the down/up blocks.
* Use upsampling instead of deconvolution, seriously. Deconvolutions are hard to
manage and create a lot of gridding artefacts.
* Use blurpooling if you want the neural network to be shift-invariant (good accuracy
even when shifting the input).

## Built With

* [Pytorch](https://pytorch.org/) - Version 1.4.0 (for memory efficient version)


## Authors

* **Nicolas Pielawski** - *Initial work*

See also the list of [contributors](https://github.com/npielawski/torch_tiramisu/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

Acknowledging and citing is appreciated and encouraged.

Zenodo record: https://zenodo.org/record/3685491

Cite as:
```
Nicolas Pielawski. (2020, February 24). npielawski/pytorch_tiramisu: Better Tiramisu 1.0 (Version 1.0). Zenodo. http://doi.org/10.5281/zenodo.3685491
```
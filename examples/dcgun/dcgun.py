import torch
import torch.nn as nn

import src.famlinn


class Generator(nn.Module):
    def __init__(self, latent_dim=100, batchnorm=True):
        """A generator for mapping a latent space to a sample space.
        The sample space for this generator is single-channel, 28x28 images
        with pixel intensity ranging from -1 to +1.
        Args:
            latent_dim (int): latent dimension ("noise vector")
            batchnorm (bool): Whether or not to use batch normalization
        """
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.batchnorm = batchnorm
        self._init_modules()

    def _init_modules(self):
        """Initialize the modules."""
        # Project the input
        # self.linear1 = nn.Linear(self.latent_dim, 256*7*7, bias=False)
        self.linear1 = src.famlinn.TorchTensorSkipModule(nn.Linear(self.latent_dim, 256 * 7 * 7, bias=False), 1)
        self.bn1d1 = nn.BatchNorm1d(256 * 7 * 7) if self.batchnorm else None
        self.leaky_relu = nn.LeakyReLU()

        # Convolutions
        self.conv1 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=False)
        self.bn2d1 = nn.BatchNorm2d(128) if self.batchnorm else None

        self.conv2 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False)
        self.bn2d2 = nn.BatchNorm2d(64) if self.batchnorm else None

        self.conv3 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=1,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False)
        self.tanh = nn.Tanh()

        self.sview = src.famlinn.TorchTensorSmartView((-1, 256, 7, 7))

    def forward(self, x):
        """Forward pass; map latent vectors to samples."""
        intermediate = self.linear1(x)
        intermediate = self.bn1d1(intermediate)
        intermediate = self.leaky_relu(intermediate)

        # intermediate = intermediate.view((-1, 256, 7, 7))
        intermediate = self.sview(intermediate)

        intermediate = self.conv1(intermediate)
        if self.batchnorm:
            intermediate = self.bn2d1(intermediate)
        intermediate = self.leaky_relu(intermediate)

        intermediate = self.conv2(intermediate)
        if self.batchnorm:
            intermediate = self.bn2d2(intermediate)
        intermediate = self.leaky_relu(intermediate)

        intermediate = self.conv3(intermediate)
        output_tensor = self.tanh(intermediate)
        return output_tensor


class Discriminator(nn.Module):
    def __init__(self):
        """A discriminator for discerning real from generated images.
        Images must be single-channel and 28x28 pixels.
        Output activation is Sigmoid.
        """
        super(Discriminator, self).__init__()
        self._init_modules()

    def _init_modules(self):
        """Initialize the modules."""
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=5,
            stride=2,
            padding=2,
            bias=True)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout_2d = nn.Dropout2d(0.3)

        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=5,
            stride=2,
            padding=2,
            bias=True)

        self.linear1 = nn.Linear(128 * 7 * 7, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

        self.sview = src.famlinn.TorchTensorSmartView((-1, 128 * 7 * 7))

    def forward(self, x):
        """Forward pass; map samples to confidence they are real [0, 1]."""
        intermediate = self.conv1(x)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.dropout_2d(intermediate)

        intermediate = self.conv2(intermediate)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.dropout_2d(intermediate)

        # intermediate = intermediate.view((-1, 128*7*7))
        intermediate = self.sview(intermediate)
        intermediate = self.linear1(intermediate)
        output_tensor = self.sigmoid(intermediate)

        return output_tensor

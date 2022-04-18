import src.famlinn

import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.onnx


class LeNetOriginal(nn.Module):

    def __init__(self):
        super(LeNetOriginal, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = nnF.max_pool2d(nnF.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = nnF.max_pool2d(nnF.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = nnF.relu(self.fc1(x))
        x = nnF.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.mp1 = nn.MaxPool2d((2, 2))
        self.mp2 = nn.MaxPool2d(2)
        self.rl1 = nn.ReLU()
        self.rl2 = nn.ReLU()
        self.rl3 = nn.ReLU()
        self.rl4 = nn.ReLU()
        self.flat = src.famlinn.TorchTensorFlatten(1)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.mp1(self.rl1(self.conv1(x)))
        # If the size is a square, you can specify with a single number
        x = self.mp2(self.rl2(self.conv2(x)))
        x = self.flat(x)  # flatten all dimensions except the batch dimension
        x = self.fc1(x)
        x = self.rl3(x)
        x = self.fc2(x)
        x = self.rl4(x)
        x = self.fc3(x)
        return x

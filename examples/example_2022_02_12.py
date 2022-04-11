from src.famlinn import *

import torch
import torch.nn as nn
import torch.nn.functional as nnF


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.flatten = lambda x: torch.flatten(x, 1)

    def forward(self, x):
        x = nnF.max_pool2d(nnF.relu(self.conv1(x)), (2, 2))
        x = nnF.max_pool2d(nnF.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = nnF.relu(self.fc1(x))
        x = nnF.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NetF(nn.Module):

    def __init__(self):
        super(NetF, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.mp1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.mp2 = nn.MaxPool2d(2)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.mp2(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = x.resharp()
        x = self.relu2(x)
        x = self.fc3(x)
        return x


def forward_hook_example():
    print("fromNet example: ")
    n = NetF()
    arg = torch.rand((1, 1, 32, 32))

    def fd_hook(module):
        def hook(module, input, output):
            print(module)

        return hook

    n.apply(lambda module: module.register_forward_hook(fd_hook(module)))
    resOrig = n(arg)
    print("Original: ", resOrig)
    famlinn = FAMLINN()
    famlinn.hook_net(n, arg)
    famlinn.pprint()
    resFamlinn = famlinn.eval(arg)
    print("Famlinn: ", resFamlinn)


def example():
    print("fromNet example: ")
    n = Net()
    arg = torch.rand((1, 1, 32, 32))
    resOrig = n(arg)
    print("Original: ", resOrig)
    famlinn = FAMLINN()
    famlinn.from_net(n, arg)
    famlinn.pprint()
    resFamlinn = famlinn.eval(arg)
    print("Famlinn: ", resFamlinn)


# RESNET-50
# U-NET
# concat, add, reshape
def hook_net_example():
    print("hookNet example: ")
    n = NetF()
    arg = torch.rand((1, 1, 32, 32))
    resOrig = n(arg)
    famlinn = FAMLINN()
    famlinn.hook_linear_net(n, arg)
    resFamlinn = famlinn.eval(arg)
    famlinn.pprint()
    print("Original: ", resOrig)
    print("Famlinn: ", resFamlinn)

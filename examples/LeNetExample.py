from src.famlinn import *

import torch
import torch.nn as nn
import torch.nn.functional as nnF
import numpy as np
import torch.onnx
import onnx
import onnxruntime


def getLayers(model):
    """
    get each layer's name and its module
    :param model:
    :return: each layer's name and its module
    """
    layers = []

    def unfoldLayer(model):
        """
        unfold each layer
        :param model: the given model or a single layer
        :param root: root name
        :return:
        """

        # get all layers of the model
        layer_list = list(model.named_children())
        for item in layer_list:
            module = item[1]
            sublayer = list(module.named_children())
            sublayer_num = len(sublayer)

            # if current layer contains sublayers, add current layer name on its sublayers
            if sublayer_num == 0:
                layers.append(module)
            # if current layer contains sublayers, unfold them
            elif isinstance(module, torch.nn.Module):
                unfoldLayer(module)

    unfoldLayer(model)
    return layers


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            # nn.Dropout(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(50, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 320)
        x = self.fc_layers(x)
        return x


class LeNet2(nn.Module):

    def __init__(self):
        super(LeNet2, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.after1 = nn.Sequential(nn.ReLU(), nn.MaxPool2d((2, 2)))
        self.after2 = nn.Sequential(nn.ReLU(), nn.MaxPool2d(2))
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.after1(self.conv1(x))
        # If the size is a square, you can specify with a single number
        x = self.after2(self.conv2(x))
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


def sampleData():
    return torch.rand((1, 1, 28, 28))


def cloneExample():
    print("Clone example: ")
    n = LeNet2()
    arg = torch.rand((1, 1, 32, 32))
    resOrig = n(arg)
    print("Original: ", resOrig)
    t = arg
    l = getLayers(n)
    print("Layers", l)
    # for i in l:
    #    print("Go througt", i)
    #    t = i(t)
    # print("Layer on", t)


"""
* Создание FAMLINN по pyTorch nn.Model
   . Получение архитектуры с учетом операции над тензорами и выходом слоев
   . Установление порядка слоев
   . Получение весов (на самом деле важные веса есть)
* Создание FAMLINN по hiddenLayers
   . Реализация "расклейки" и "антирепликации" слоев
   . Нахождение стока и истока сети
   . Получение весов и их назначение
   . Получение функций по названию
Устройство pyTorch:
   . _modules содержит все веса??
"""


def hlExample():
    import hiddenlayer as hl
    print("HL example: ")
    n = Net()
    arg = torch.rand((1, 1, 28, 28))
    params = list(n.parameters())
    resOrig = n(arg)
    print("Original: ", resOrig)
    h = hl.build_graph(n, arg)
    # h.theme = hl.graph.THEMES["blue"].copy()
    # h.save('LetNet', format='png')
    print("Layer on", h)
    for i in h.edges: print("  edge", i)
    for i in h.nodes: print("  node", i)


def convertExample():
    print("Convert example: ")
    n = Net()
    # https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
    torch.onnx.export(n,  # model being run
                      sampleData(),  # model input (or a tuple for multiple inputs)
                      "super_resolution.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})

    onnx_model = onnx.load("super_resolution.onnx")
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession("super_resolution.onnx")

    arg = sampleData()

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(arg)}
    ort_outs = ort_session.run(None, ort_inputs)
    print("Original: ", n(arg))
    print("ONNX_converted: ", ort_outs)

    for node in onnx_model.graph.node:
        print(node)
        print("--------------------------------")

    # https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
    # https://pythonrepo.com/repo/fumihwh-onnx-pytorch
    # https://github.com/onnx/onnx/blob/main/docs/IR.md#graphs
    # https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html


def fromNetExample(useLeNet=True):
    print("fromNet example: ")
    n = Net()
    arg = torch.rand((1, 1, 28, 28))
    if useLeNet:
        n = LeNet()
        arg = torch.rand((1, 1, 32, 32))
    resOrig = n(arg)
    print("Original: ", resOrig)
    famlinn = FAMLINN()
    famlinn.from_net(n, arg)
    famlinn.pprint()
    resFamlinn = famlinn.eval(arg)
    print("Famlinn: ", resFamlinn)

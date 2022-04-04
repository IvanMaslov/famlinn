import pprint

import torch
import examples.unet.unet


def dump_pytorch_graph(graph):
    """List all the nodes in a PyTorch graph."""
    f = "{:25} {:40}   {} -> {}"
    print(f.format("kind", "scopeName", "inputs", "outputs"))
    for node in graph.nodes():
        print(f.format(node.kind(), node.scopeName(),
                       [i.unique() for i in node.inputs()],
                       [i.unique() for i in node.outputs()]
                       ))


# https://github.com/mateuszbuda/brain-segmentation-pytorch
def do_research():
    n = examples.unet.unet.UNet()
    arg = torch.rand((1, 3, 256, 256))
    trace, out = torch.jit._get_trace_graph(n, arg)
    torch_graph = torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
    dump_pytorch_graph(torch_graph)
    pprint.pprint(torch_graph)
    resOrig = n(arg)
    print("Original: ", resOrig)


"""
 * https://discuss.pytorch.org/t/set-model-weights-to-preset-tensor-with-torch/35051/2
 * https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/dcgan_faces_tutorial.ipynb
 * https://towardsdatascience.com/how-to-build-a-dcgan-with-pytorch-31bfbf2ad96a
 * https://github.com/mateuszbuda/brain-segmentation-pytorch
 * https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/
 * https://github.com/milesial/Pytorch-UNet
 * https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
 * https://pytorch.org/hub/pytorch_vision_resnet/
 * https://github.com/Lornatang/ResNet-PyTorch/blob/master/resnet_pytorch/utils.py
 * https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
 * https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
 * https://stackoverflow.com/questions/54846905/pytorch-get-all-layers-of-model
 * https://programmersought.com/article/42964056129/
 * https://shap.readthedocs.io/en/latest/example_notebooks/image_examples/image_classification/PyTorch%20Deep%20Explainer%20MNIST%20example.html
 * https://colab.research.google.com/github/d2l-ai/d2l-en-colab/blob/master/chapter_convolutional-neural-networks/lenet.ipynb#scrollTo=y5dg3qjD41In
 * https://github.com/marload/LeNet-keras/blob/master/lenet.py
 * https://stackabuse.com/tensorflow-neural-network-tutorial/
 * https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html
"""
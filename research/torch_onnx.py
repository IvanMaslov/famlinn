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

import torch

import examples.unet.unet
import src.famlinn


# https://github.com/mateuszbuda/brain-segmentation-pytorch
def example():
    n = examples.unet.unet.UNet()

    arg = torch.rand((1, 3, 256, 256))
    trace, out = torch.jit._get_trace_graph(n, arg)
    torch_graph = torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
    resOrig = n(arg)
    print("Original: ", resOrig)

    famlinn = src.famlinn.FAMLINN()
    famlinn.hook_net(n, arg)
    famlinn.pprint()
    resFamlinn = famlinn.eval(arg, verbose=True)
    print("Famlinn: ", resFamlinn)

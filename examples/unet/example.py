import torch

import examples.unet.unet
import src.famlinn


# https://github.com/mateuszbuda/brain-segmentation-pytorch
def example():
    n = examples.unet.unet.UNet()
    arg = torch.rand((1, 3, 256, 256))
    resOrig = n(arg)
    print("Original: ", resOrig)

    famlinn = src.famlinn.FAMLINN()
    famlinn.hook_linear_net(n, arg)
    famlinn.pprint()
    resFamlinn = famlinn.eval(arg)
    print("Famlinn: ", resFamlinn)

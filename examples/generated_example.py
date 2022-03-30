import torch

import examples.resources.tmp
import src.famlinn


# https://github.com/mateuszbuda/brain-segmentation-pytorch
def example():
    n = examples.resources.tmp.Net()

    arg = torch.randn(1, 3, 256, 256)
    #arg = torch.randn(1, 3, 224, 224)
    #arg = torch.randn(10, 100)
    resOrig = n(arg)
    print("Original: ", resOrig)

    famlinn = src.famlinn.FAMLINN()
    famlinn.hook_net(n, arg)
    famlinn.pprint()
    resFamlinn = famlinn.eval(arg, verbose=False)
    print("Famlinn: ", resFamlinn)

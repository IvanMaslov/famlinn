import torch

import examples.resnet.resnet
import src.famlinn


# https://github.com/mateuszbuda/brain-segmentation-pytorch
def example():
    n = examples.resnet.resnet.ResNet101()

    arg = torch.randn(1, 3, 224, 224)
    resOrig = n(arg)
    print("Original: ", resOrig)

    famlinn = src.famlinn.FAMLINN()
    famlinn.hook_net(n, arg)
    famlinn.pprint()
    resFamlinn = famlinn.eval(arg, verbose=False)
    print("Famlinn: ", resFamlinn)

    famlinn.export('D:\\ITMO\\FAMLINN\\examples\\resources\\tmp.py')

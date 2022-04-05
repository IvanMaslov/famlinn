import torch

import examples.vgg19.vgg19
import src.famlinn


# https://github.com/mateuszbuda/brain-segmentation-pytorch
def example():
    VGG_types = {
        "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        "VGG13": [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            "M",
            512,
            512,
            "M",
            512,
            512,
            "M",
        ],
        "VGG16": [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
            "M",
        ],
        "VGG19": [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
            512,
            "M",
        ],
    }

    n = examples.vgg19.vgg19.VGG(VGG_types['VGG19'])

    arg = torch.randn(1, 3, 224, 224)
    resOrig = n(arg)
    print("Original: ", resOrig)

    famlinn = src.famlinn.FAMLINN()
    famlinn.hook_net(n, arg)
    famlinn.pprint()
    resFamlinn = famlinn.eval(arg, verbose=False)
    print("Famlinn: ", resFamlinn)

    famlinn.export('D:\\ITMO\\FAMLINN\\examples\\resources\\tmp.py',
                   'D:\\ITMO\\FAMLINN\\examples\\resources\\weights')

import torch

import examples.dcgun.dcgun
import src.famlinn


def example_discriminator():
    n = examples.dcgun.dcgun.Discriminator()

    arg = torch.randn(1, 1, 28, 28)
    resOrig = n(arg)
    print("Original: ", resOrig)

    famlinn = src.famlinn.FAMLINN()
    famlinn.hook_net(n, arg)
    famlinn.pprint()
    resFamlinn = famlinn.eval(arg, verbose=False)
    print("Famlinn: ", resFamlinn)


def example_generator():
    n = examples.dcgun.dcgun.Generator()

    arg = torch.randn(10, 100)
    resOrig = n(arg)
    print("Original: ", resOrig)

    famlinn = src.famlinn.FAMLINN()
    famlinn.hook_net(n, arg)
    famlinn.pprint()
    resFamlinn = famlinn.eval(arg, verbose=False)
    print("Famlinn: ", resFamlinn)

    famlinn.export('D:\\ITMO\\FAMLINN\\examples\\resources\\tmp.py',
                   'D:\\ITMO\\FAMLINN\\examples\\resources\\weights')

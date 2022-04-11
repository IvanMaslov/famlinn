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
    famlinn.hook_net(n, arg)
    famlinn.pprint()
    resFamlinn = famlinn.eval(arg)
    print("Famlinn: ", resFamlinn)

    famlinn.export('D:\\ITMO\\FAMLINN\\examples\\resources\\tmp.py',
                   'D:\\ITMO\\FAMLINN\\examples\\resources\\weights')


def test(seed, arg=torch.randn(1, 3, 256, 256)):
    n = examples.unet.unet.UNet()

    seed()
    resOrig = n(arg)

    famlinn = src.famlinn.FAMLINN()
    famlinn.hook_net(n, arg)
    famlinn.export('D:\\ITMO\\FAMLINN\\examples\\resources\\unet\\src.py',
                   'D:\\ITMO\\FAMLINN\\examples\\resources\\unet\\weights')

    seed()
    resFamlinn = famlinn.eval(arg)
    assert torch.equal(resOrig, resFamlinn)
    print("TEST_CONVERT_UNET: OK(", resOrig.view(-1)[0], resFamlinn.view(-1)[0], ')')


def test_gen(seed, arg=torch.randn(1, 3, 256, 256)):
    import examples.resources.unet.src
    n = examples.unet.unet.UNet()
    seed()
    resOrig = n(arg)
    # print("Original: ", resOrig)

    nRead = examples.resources.unet.src.Net()
    nRead.read('D:\\ITMO\\FAMLINN\\examples\\resources\\unet\\weights')
    seed()
    resR = nRead(arg)
    # print("Read: ", resR)

    famlinn = src.famlinn.FAMLINN()
    famlinn.hook_net(nRead, arg)
    # famlinn.pprint()
    seed()
    resFamlinn = famlinn.eval(arg)
    # print("FamlinnRead: ", resFamlinn)
    # assert torch.equal(resOrig, resR), str(resOrig) + str(resR)
    assert torch.equal(resR, resFamlinn), str(resR) + str(resFamlinn)
    print("TEST_WEIGHTS_UNET: OK(", resOrig.view(-1)[0], resR.view(-1)[0], resFamlinn.view(-1)[0], ')')

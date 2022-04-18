import torch

import examples.unet.unet
import src.famlinn
from src.utils import Perf


def test(seed, arg=torch.randn(1, 3, 256, 256)):
    n = examples.unet.unet.UNet()

    seed()
    with Perf("EVAL_ORIGINAL(unet)"):
        resOrig = n(arg)

    famlinn = src.famlinn.FAMLINN()
    with Perf("MAKE_FAMLINN(unet)"):
        famlinn.hook_net(n, arg)
    with Perf("SAVE_FAMLINN(unet)"):
        famlinn.export('D:\\ITMO\\FAMLINN\\examples\\resources\\unet\\src.py',
                       'D:\\ITMO\\FAMLINN\\examples\\resources\\unet\\weights')

    seed()
    with Perf("CALC_FAMLINN(unet)"):
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

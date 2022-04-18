import torch

import examples.resnet.resnet
import src.famlinn
from src.utils import Perf


def test(seed, arg=torch.randn(1, 3, 224, 224)):
    n = examples.resnet.resnet.ResNet101()

    seed()
    with Perf("EVAL_ORIGINAL(resnet)"):
        resOrig = n(arg)
    famlinn = src.famlinn.FAMLINN()
    with Perf("MAKE_FAMLINN(resnet)"):
        famlinn.hook_net(n, arg)
    with Perf("SAVE_FAMLINN(resnet)"):
        famlinn.export('D:\\ITMO\\FAMLINN\\examples\\resources\\resnet\\src.py',
                       'D:\\ITMO\\FAMLINN\\examples\\resources\\resnet\\weights')

    seed()
    with Perf("CALC_FAMLINN(resnet)"):
        resFamlinn = famlinn.eval(arg)
    assert torch.equal(resOrig, resFamlinn)
    print("TEST_CONVERT_RESNET: OK(", resOrig.view(-1)[0], resFamlinn.view(-1)[0], ')')


def test_gen(seed, arg=torch.randn(1, 3, 224, 224)):
    import examples.resources.resnet.src
    n = examples.resnet.resnet.ResNet101()
    seed()
    resOrig = n(arg)
    # print("Original: ", resOrig)

    nRead = examples.resources.resnet.src.Net()
    nRead.read('D:\\ITMO\\FAMLINN\\examples\\resources\\resnet\\weights')
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
    print("TEST_WEIGHTS_RESNET: OK(", resOrig.view(-1)[0], resR.view(-1)[0], resFamlinn.view(-1)[0], ')')

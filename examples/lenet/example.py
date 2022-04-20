import torch

import examples.lenet.lenet
import src.famlinn
from src.utils import Perf


def test(seed, arg=torch.randn(1, 1, 32, 32)):
    n = examples.lenet.lenet.LeNet()

    seed()
    with Perf("EVAL_ORIGINAL(lenet)"):
        resOrig = n(arg)
    famlinn = src.famlinn.FAMLINN()
    with Perf("MAKE_FAMLINN(lenet)"):
        famlinn.hook_net(n, arg)
    with Perf("SAVE_FAMLINN(lenet)"):
        famlinn.export('D:\\ITMO\\FAMLINN\\examples\\resources\\lenet\\src.py',
                       'D:\\ITMO\\FAMLINN\\examples\\resources\\lenet\\weights')

    torch.save(n, 'D:\\ITMO\\FAMLINN\\examples\\resources\\lenet\\original')
    seed()
    with Perf("CALC_FAMLINN(lenet)"):
        resFamlinn = famlinn.eval(arg)
    assert torch.equal(resOrig, resFamlinn)
    print("TEST_CONVERT_LENET: OK(", resOrig.view(-1)[0], resFamlinn.view(-1)[0], ')')


def test_gen(seed, arg=torch.randn(1, 1, 32, 32)):
    import examples.resources.lenet.src
    n = examples.lenet.lenet.LeNet()
    seed()
    resOrig = n(arg)
    # print("Original: ", resOrig)

    nRead = examples.resources.lenet.src.Net()
    nRead.read('D:\\ITMO\\FAMLINN\\examples\\resources\\lenet\\weights')
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
    print("TEST_WEIGHTS_LENET: OK(", resOrig.view(-1)[0], resR.view(-1)[0], resFamlinn.view(-1)[0], ')')

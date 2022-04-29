import torch

import examples.vgg19.vgg19
import src.famlinn
from src.utils import Perf

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


def test(seed, arg=torch.randn(1, 3, 224, 224)):
    n = examples.vgg19.vgg19.VGG(VGG_types['VGG19'])

    seed()
    with Perf("EVAL_ORIGINAL(vgg19)"):
        resOrig = n(arg)

    famlinn = src.famlinn.FAMLINN()
    with Perf("MAKE_FAMLINN(vgg19)"):
        famlinn.hook_net(n, arg)
    with Perf("SAVE_FAMLINN(vgg19)"):
        famlinn.export('D:\\ITMO\\FAMLINN\\examples\\resources\\vgg19\\src.py',
                       'D:\\ITMO\\FAMLINN\\examples\\resources\\vgg19\\weights')

    torch.save(n, 'D:\\ITMO\\FAMLINN\\examples\\resources\\vgg19\\original')
    seed()
    with Perf("CALC_FAMLINN(vgg19)"):
        resFamlinn = famlinn.eval(arg)
    assert torch.equal(resOrig, resFamlinn), str(resOrig) + str(resFamlinn)
    print("TEST_CONVERT_VGG19: OK(", resOrig.view(-1)[0], resFamlinn.view(-1)[0], ')')


def test_gen(seed, arg=torch.randn(1, 3, 224, 224)):
    import examples.resources.vgg19.src
    nRead = examples.resources.vgg19.src.Net()
    nRead.read('D:\\ITMO\\FAMLINN\\examples\\resources\\vgg19\\weights')
    seed()
    resR = nRead(arg)
    # print("Read: ", resR)

    famlinn = src.famlinn.FAMLINN()
    famlinn.hook_net(nRead, arg)
    # famlinn.pprint()
    seed()
    resFamlinn = famlinn.eval(arg)
    # print("FamlinnRead: ", resFamlinn)
    assert torch.equal(resR, resFamlinn), str(resR) + str(resFamlinn)
    print("TEST_WEIGHTS_VGG19: OK(", resR.view(-1)[0], resFamlinn.view(-1)[0], ')')

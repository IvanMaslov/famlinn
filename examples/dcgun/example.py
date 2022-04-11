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
    resFamlinn = famlinn.eval(arg)
    print("Famlinn: ", resFamlinn)


def example_generator():
    n = examples.dcgun.dcgun.Generator()

    arg = torch.randn(1, 1, 28, 28)
    resOrig = n(arg)
    print("Original: ", resOrig)

    famlinn = src.famlinn.FAMLINN()
    famlinn.hook_net(n, arg)
    famlinn.pprint()
    resFamlinn = famlinn.eval(arg)
    print("Famlinn: ", resFamlinn)

    famlinn.export('D:\\ITMO\\FAMLINN\\examples\\resources\\tmp.py',
                   'D:\\ITMO\\FAMLINN\\examples\\resources\\weights')


def test_generator(seed, arg=torch.randn(10, 100)):
    n = examples.dcgun.dcgun.Generator()

    seed()
    resOrig = n(arg)

    famlinn = src.famlinn.FAMLINN()
    famlinn.hook_net(n, arg)
    famlinn.export('D:\\ITMO\\FAMLINN\\examples\\resources\\gGun\\src.py',
                   'D:\\ITMO\\FAMLINN\\examples\\resources\\gGun\\weights')

    seed()
    resFamlinn = famlinn.eval(arg)
    assert torch.equal(resOrig, resFamlinn), str(resOrig) + str(resFamlinn)
    print("TEST_CONVERT_G_GUN: OK(", resOrig.view(-1)[0], resFamlinn.view(-1)[0], ')')


def test_gen_generator(seed, arg=torch.randn(10, 100)):
    import examples.resources.gGun.src
    n = examples.dcgun.dcgun.Generator()
    seed()
    resOrig = n(arg)
    # print("Original: ", resOrig)

    nRead = examples.resources.gGun.src.Net()
    nRead.read('D:\\ITMO\\FAMLINN\\examples\\resources\\gGun\\weights')
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
    print("TEST_WEIGHTS_G_GUN: OK(", resOrig.view(-1)[0], resR.view(-1)[0], resFamlinn.view(-1)[0], ')')


def test_discriminator(seed, arg=torch.randn(1, 1, 28, 28)):
    n = examples.dcgun.dcgun.Discriminator()

    seed()
    resOrig = n(arg)

    famlinn = src.famlinn.FAMLINN()
    famlinn.hook_net(n, arg)
    famlinn.export('D:\\ITMO\\FAMLINN\\examples\\resources\\dGun\\src.py',
                   'D:\\ITMO\\FAMLINN\\examples\\resources\\dGun\\weights')

    seed()
    resFamlinn = famlinn.eval(arg)
    assert torch.equal(resOrig, resFamlinn), str(resOrig) + str(resFamlinn)
    print("TEST_CONVERT_D_GUN: OK(", resOrig.view(-1)[0], resFamlinn.view(-1)[0], ')')


def test_gen_discriminator(seed, arg=torch.rand(1, 1, 28, 28)):
    import examples.resources.dGun.src
    n = examples.dcgun.dcgun.Discriminator()
    seed()
    resOrig = n(arg)
    # print("Original: ", resOrig)

    nRead = examples.resources.dGun.src.Net()
    nRead.read('D:\\ITMO\\FAMLINN\\examples\\resources\\dGun\\weights')
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
    print("TEST_WEIGHTS_D_GUN: OK(", resOrig.view(-1)[0], resR.view(-1)[0], resFamlinn.view(-1)[0], ')')

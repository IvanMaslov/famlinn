import pathlib

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
        famlinn.export(pathlib.Path('D:\\ITMO\\FAMLINN\\examples\\resources\\resnet\\src.py'),
                       pathlib.Path('D:\\ITMO\\FAMLINN\\examples\\resources\\resnet\\weights'))

    torch.save(n, pathlib.Path('D:\\ITMO\\FAMLINN\\examples\\resources\\resnet\\original'))
    seed()
    with Perf("CALC_FAMLINN(resnet)"):
        resFamlinn = famlinn.eval(arg)
    assert torch.equal(resOrig, resFamlinn)
    print("TEST_CONVERT_RESNET: OK(", resOrig.view(-1)[0], resFamlinn.view(-1)[0], ')')


def test_gen(seed, arg=torch.randn(1, 3, 224, 224)):
    import examples.resources.resnet.src
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
    assert torch.equal(resR, resFamlinn), str(resR) + str(resFamlinn)
    print("TEST_WEIGHTS_RESNET: OK(", resR.view(-1)[0], resFamlinn.view(-1)[0], ')')


def bench_onnx(arg):
    n = examples.resnet.resnet.ResNet101()
    with Perf("ONNX(resnet)"):
        import onnx
        import onnxruntime
        pth = pathlib.Path('D:\\ITMO\\FAMLINN\\examples\\resources\\resnet\\onnx')
        with Perf("ONNX_SAVE(resnet)"):
            nOnnx = torch.onnx.export(n, arg, pth)
        with Perf("ONNX_LOAD_CHECK(resnet)"):
            nOnnx = onnx.load(str(pth))
            onnx.checker.check_model(nOnnx)
        with Perf("ONNX_LOAD_RUN(resnet)"):
            ort_session = onnxruntime.InferenceSession(str(pth))

            def to_numpy(tensor):
                return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(arg)}
            with Perf("ONNX_LOAD_RUN_ONLY(resnet)"):
                ort_outs = ort_session.run(None, ort_inputs)

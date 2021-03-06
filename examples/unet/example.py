import pathlib

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
        famlinn.export(pathlib.Path('D:\\ITMO\\FAMLINN\\examples\\resources\\unet\\src.py'),
                       pathlib.Path('D:\\ITMO\\FAMLINN\\examples\\resources\\unet\\weights'))

    torch.save(n, pathlib.Path('D:\\ITMO\\FAMLINN\\examples\\resources\\unet\\original'))
    seed()
    with Perf("CALC_FAMLINN(unet)"):
        resFamlinn = famlinn.eval(arg)
    assert torch.equal(resOrig, resFamlinn)
    print("TEST_CONVERT_UNET: OK(", resOrig.view(-1)[0], resFamlinn.view(-1)[0], ')')


def test_gen(seed, arg=torch.randn(1, 3, 256, 256)):
    import examples.resources.unet.src
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
    assert torch.equal(resR, resFamlinn), str(resR) + str(resFamlinn)
    print("TEST_WEIGHTS_UNET: OK(", resR.view(-1)[0], resFamlinn.view(-1)[0], ')')


def bench_onnx(arg):
    n = examples.unet.unet.UNet()
    with Perf("ONNX(unet)"):
        import onnx
        import onnxruntime
        pth = pathlib.Path('D:\\ITMO\\FAMLINN\\examples\\resources\\unet\\onnx')
        with Perf("ONNX_SAVE(unet)"):
            nOnnx = torch.onnx.export(n, arg, pth)
        with Perf("ONNX_LOAD_CHECK(unet)"):
            nOnnx = onnx.load(str(pth))
            onnx.checker.check_model(nOnnx)
        with Perf("ONNX_LOAD_RUN(unet)"):
            ort_session = onnxruntime.InferenceSession(str(pth))
            def to_numpy(tensor):
                return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(arg)}
            with Perf("ONNX_LOAD_RUN_ONLY(unet)"):
                ort_outs = ort_session.run(None, ort_inputs)

import torch

import examples.unet.example
import examples.lenet.example
import examples.resnet.example
import examples.vgg19.example
import examples.dcgun.example
import research.torch_onnx


def seed():
    import numpy as np
    np.random.seed(200)
    import random as rnd
    rnd.seed(200)
    torch.manual_seed(200)
    torch.cuda.manual_seed(200)


def run_research():
    research.torch_onnx.do_research()


def run_unet():
    examples.unet.example.example()


def run_resnet():
    examples.resnet.example.example()


def run_vgg19():
    examples.vgg19.example.example()


def run_dcgun():
    examples.dcgun.example.example_discriminator()
    examples.dcgun.example.example_generator()


def test():
    arg = torch.randn(1, 1, 32, 32)
    examples.lenet.example.test(seed, arg)
    examples.lenet.example.test_gen(seed, arg)
    arg = torch.randn(1, 3, 256, 256)
    examples.unet.example.test(seed, arg)
    examples.unet.example.test_gen(seed, arg)
    examples.unet.example.bench_onnx(arg)
    arg = torch.randn(1, 3, 224, 224)
    examples.resnet.example.test(seed, arg)
    examples.resnet.example.test_gen(seed, arg)
    examples.resnet.example.bench_onnx(arg)
    examples.vgg19.example.test(seed, arg)
    examples.vgg19.example.test_gen(seed, arg)
    arg = torch.randn(1, 1, 28, 28)
    examples.dcgun.example.test_discriminator(seed, arg)
    examples.dcgun.example.test_gen_discriminator(seed, arg)
    arg = torch.randn(10, 100)
    examples.dcgun.example.test_generator(seed, arg)
    examples.dcgun.example.test_gen_generator(seed, arg)


if __name__ == '__main__':
    test()


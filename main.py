import torch

import examples.unet.example
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
    examples.unet.example.test(seed)
    examples.unet.example.test_gen(seed)
    examples.resnet.example.test(seed)
    examples.resnet.example.test_gen(seed)
    examples.vgg19.example.test(seed)
    examples.vgg19.example.test_gen(seed)
    examples.dcgun.example.test_discriminator(seed)
    examples.dcgun.example.test_gen_discriminator(seed)
    examples.dcgun.example.test_generator(seed)
    examples.dcgun.example.test_gen_generator(seed)


if __name__ == '__main__':
    test()


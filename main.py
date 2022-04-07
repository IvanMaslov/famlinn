import torch

import examples.unet.example
import examples.resnet.example
import examples.vgg19.example
import examples.dcgun.example
import examples.generated_example
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


def run_gen():
    examples.generated_example.example()


if __name__ == '__main__':
    seed()
    # run_unet()
    # run_resnet()
    run_vgg19()
    # run_dcgun()
    # run_gen()


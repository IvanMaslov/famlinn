import examples.unet.example
import examples.resnet.example
import examples.vgg19.example
import research.torch_onnx


def run_research():
    research.torch_onnx.do_research()


def run_unet():
    examples.unet.example.example()


def run_resnet():
    examples.resnet.example.example()


def run_vgg19():
    examples.vgg19.example.example()


if __name__ == '__main__':
    # run_unet()
    run_resnet()
    # run_vgg19()

import examples.unet.example
import examples.resnet.example
import research.torch_onnx


def run_research():
    research.torch_onnx.do_research()


def run_unet():
    examples.unet.example.example()


def run_resnet():
    examples.resnet.example.example()


if __name__ == '__main__':
    # run_unet()
    run_resnet()
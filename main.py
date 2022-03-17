import examples.unet.example
import research.torch_onnx


def r():
    research.torch_onnx.do_research()
    exit(0)


def t():
    examples.unet.example.example()
    exit(0)


if __name__ == '__main__':
    # r()
    t()

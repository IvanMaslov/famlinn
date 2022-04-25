# FAMLINN

import torch
import torch.nn as nn
from typing import List

import shutil
import os
import sys


class Container:
    __NEXT_ID = 0

    def __init__(self):
        Container.__NEXT_ID += 1
        self.id = Container.__NEXT_ID
        self.data = None

    def write(self, data): self.data = data

    def read(self): return self.data

    def get_id(self): return self.id

    @staticmethod
    def null_id():
        Container.__NEXT_ID = 0


class Storage:

    def __init__(self):
        self.inputContainer = Container()
        self.outputContainer = self.inputContainer
        self.data = {self.inputContainer.get_id(): self.inputContainer}

    def get_container(self, container_id: int, is_input: bool = False, is_output: bool = False):
        if is_input:
            return self.inputContainer
        if is_output:
            return self.outputContainer
        if container_id not in self.data:
            return None
        return self.data[container_id]

    def add_container(self, set_output: bool = True) -> Container:
        cont = Container()
        self.data[cont.get_id()] = cont
        if set_output:
            self.outputContainer = cont
        return cont

    def input_id(self):
        return self.inputContainer.get_id()

    def output_id(self):
        return self.outputContainer.get_id()

    def pprint(self):
        print("Input container ", self.inputContainer.get_id())
        print("Output container", self.outputContainer.get_id())


class Node:

    def __init__(self, funct, res, args=None, storage: Storage = None, label=""):
        assert funct is not None, "Can not make Node with no function"
        assert storage is not None, "Can not make Node with no storage"
        if args is None:
            args = []

        self.funct = funct
        self.args = args
        self.res = res
        self.storage = storage
        self.label = label

    def save_params(self, path: str):
        torch.save(self.funct.module, path)

    def eval(self):
        arguments = [self.storage.get_container(i).read() for i in self.args]
        self.res.write(self.funct(arguments))

    def pprint(self):
        print("Node from {} to {} ({})".format(self.args, self.res.get_id(), self.label))


# Formatted AutoML Iterable Neural Network
class NeuralNetwork:

    def __init__(self):
        Container.null_id()
        self.storage = Storage()
        self.nodes = []

    def add_layer(self, function, input_layers=None, label="") -> int:
        if input_layers is None:
            input_layers = []
        res = self.storage.add_container()
        if len(input_layers) == 0:
            input_layers = [self.storage.input_id()]
        node = Node(function, res, input_layers, self.storage, label=label)
        self.nodes.append(node)
        return res.get_id()

    def add_var(self, data) -> int:
        res = self.storage.add_container(False)
        res.write(data)
        return res.get_id()

    def eval(self, data):
        self.storage.get_container(self.storage.input_id()).write(data)
        for i in self.nodes:
            i.eval()
        return self.storage.get_container(self.storage.output_id()).read()

    def pprint(self):
        print("Print Neural Network:")
        self.storage.pprint()
        for i in self.nodes:
            i.pprint()


class TorchTensorCat(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, *tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat(tensor, dim=self.dim)

    def __str__(self):
        return "TorchTensorCat(" + str(self.dim) + ")"


class TorchTensorSmartReshape(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, tensor1: torch.Tensor) -> torch.Tensor:
        return tensor1.reshape(tensor1.size(0), -1)


class TorchTensorFlatten(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.flatten(tensor, self.dim)

    def __str__(self):
        return "TorchTensorFlatten(" + str(self.dim) + ")"


class TorchTensorAdd(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
        return tensor1 + tensor2


class TorchTensorTo1D(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(tensor.size(0), -1)


class TorchTensorSmartView(nn.Module):

    def __init__(self, arg):
        super().__init__()
        self.arg = arg

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(self.arg)

    def __str__(self):
        return "TorchTensorSmartView(" + str(self.arg) + ")"


class TorchTensorSkipModule(nn.Module):

    def __init__(self, module: nn.Module, skip: int):
        super().__init__()
        self.skip = skip
        self.module = module

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.module.forward(tensor)

    def __str__(self):
        return "TorchTensorSkipModule(" + str(self.module) + ", " + str(self.skip) + ")"


class Evaluator:
    def __init__(self, module: nn.Module):
        self.module = module

    def __call__(self, *args, **kwargs):
        return self.module(*args[0])

    def __str__(self):
        return str(self.module)


class CodeGen:

    def __init__(self, output=None):
        self.output = output
        self.result = []

    def add_line(self, line: str):
        self.result.append(line)

    def add_lines(self, lines: List[str]):
        self.result.extend(lines)

    def tabulate_add(self, codegen):
        self.result.extend(list(map(lambda x: "    " + x, codegen.result)))

    def write(self):
        with open(self.output, 'w') as file:
            file.write("\n".join(self.result))


global hook_list


class FAMLINN(NeuralNetwork):

    def __init__(self):
        self._graph_hook_map = {}
        super().__init__()

    def hook_net(self, net: nn.Module, arg):
        return self.hook_net_hidden_layer(net, arg)

    def hook_net_hidden_layer(self, net: nn.Module, arg):
        global hook_list
        hook_list = []

        def add_hook(module):
            global hook_list
            if isinstance(module, torch.Tensor):
                return
            if isinstance(module, nn.Sequential):
                return
            hook_list.append(module.register_forward_hook(self._generate_hook_list(module, net)))

        net.apply(add_hook)
        self.active_hook = True
        net(arg)
        self.active_hook = False
        for hook in hook_list:
            hook.remove()

        #trace, out = torch.jit._get_trace_graph(net, arg)
        #torch_graph = torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
        import onnx
        nOnnx = torch.onnx.export(net, arg, 'tmp.onnx')
        model = onnx.load_model('tmp.onnx')
        nodes = [(node.input, node.output, node.name) for node in model.graph.node]
        import pprint
        pprint.pprint(nodes)
        list_id = 0
        torch_id_to_container = {}
        for node in nodes:
            inp = node[0]
            outp = node[1]
            input_containers = []
            output_container = outp[0]

            module = self._graph_hook_map[list_id]

            for i in inp:
                if i in torch_id_to_container:
                    input_containers.append(torch_id_to_container[i])
            if len(input_containers) == 0:
                input_containers = [self.storage.output_id()]
            added_container = self.add_layer(Evaluator(module), input_containers, str(module))
            list_id += 1
            torch_id_to_container[output_container] = added_container
        self.pprint()

    def _generate_hook_list(self, module, net):
        global cnt
        cnt = 0

        def __hook(_module, _input, _output):
            global cnt
            if self.active_hook and not isinstance(_module, net.__class__) and not hasattr(_module, 'famlinn_ignore'):
                self._graph_hook_map[cnt] = _module
                cnt += 1

        return __hook

    def export(self, output_src: str, output_weights: str):
        codegen = CodeGen(output_src)
        self.generate(codegen)
        codegen.write()
        self.write_weights(output_weights)

    def generate(self, codegen: CodeGen) -> None:
        codegen.add_lines([
            "from torch.nn import *",
            "from src.famlinn import *",
            "import base64",
            "",
            "",
            "class Net(nn.Module):",
        ])
        codegen.tabulate_add(self.generate_constructor())
        codegen.add_line("")
        codegen.tabulate_add(self.generate_forward())
        codegen.add_line("")
        codegen.tabulate_add(self.generate_read())
        codegen.add_line("")

    def generate_constructor(self) -> CodeGen:
        res_codegen = CodeGen()
        res_codegen.add_lines([
            "def __init__(self):",
            "    super().__init__()",
        ])
        codegen = CodeGen()
        for i, node in enumerate(self.nodes):
            line = "self.{} = {}".format('node_' + str(i), node.label)
            codegen.add_line(line)
        res_codegen.tabulate_add(codegen)
        return res_codegen

    def generate_forward(self) -> CodeGen:
        res_codegen = CodeGen()
        res_codegen.add_lines([
            "def forward(self, arg):",
            "    res_0 = arg",
        ])
        codegen = CodeGen()
        for i, node in enumerate(self.nodes):
            args = '[' + ",".join(map(lambda x: 'res_{}'.format(x - 1), node.args)) + ']'
            line = "res_{} = self.node_{}(*{})  # {}".format(i + 1, i, args, node.label)
            codegen.add_line(line)
        codegen.add_line("return res_{}".format(len(self.nodes)))
        res_codegen.tabulate_add(codegen)
        return res_codegen

    def generate_read(self) -> CodeGen:
        res_codegen = CodeGen()
        res_codegen.add_lines([
            "def read(self, weights_path):",
        ])
        for i, node in enumerate(self.nodes):
            res_codegen.add_lines([
                '        self.node_{} = torch.load(weights_path + \'Detailed\\\\node{}\')'.format(i, i)
            ])
        return res_codegen

    def write_weights(self, output_weights: str):
        shutil.rmtree(output_weights + 'Detailed', ignore_errors=True)
        os.mkdir(output_weights + 'Detailed')
        for i, node in enumerate(self.nodes):
            node.save_params(output_weights + 'Detailed\\node' + str(i))

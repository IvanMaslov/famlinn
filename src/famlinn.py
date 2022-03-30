import abc
import sys

import torch
import torch.nn as nn
import hiddenlayer as hl
from typing import Any, Tuple, List


class Container:

    __NEXT_ID = 0

    def __init__(self):
        print("FuncIn: Init Container", file=sys.stderr)
        Container.__NEXT_ID += 1
        self.id = Container.__NEXT_ID
        self.data = None

    def write(self, data): self.data = data

    def read(self): return self.data

    def getId(self): return self.id

    @staticmethod
    def nullId():
        Container.__NEXT_ID = 0


class Storage:

    def __init__(self):
        print("FuncIn: Init Storage", file=sys.stderr)

        self.inputContainer = Container()
        self.outputContainer = self.inputContainer
        self.data = {self.inputContainer.getId(): self.inputContainer}

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
        self.data[cont.getId()] = cont
        if set_output:
            self.outputContainer = cont
        return cont

    def inputId(self): return self.inputContainer.getId()

    def outputId(self): return self.outputContainer.getId()

    def pprint(self):
        print("Input container ", self.inputContainer.getId())
        print("Output container", self.outputContainer.getId())


class Node:

    def __init__(self, funct, res, args = [], storage: Storage = None, label=""):
        print("FuncIn: Init Node", file=sys.stderr)
        assert funct is not None, "Can not make Node with no function"
        assert storage is not None, "Can not make Node with no storage"

        self.funct = funct
        self.args = args
        self.res = res
        self.storage = storage
        self.label = label

    def eval(self, verbose=False):
        arguments = [self.storage.get_container(i).read() for i in self.args]
        if verbose:
            print("Evaluating funct={} from args={}".format(self.funct, arguments))
        self.res.write(self.funct(arguments))

    def pprint(self):
        print("Node from {} to {} ({})".format(self.args, self.res.getId(), self.label))


# Format AutoML Iteratable Neural Network
class NeuralNetwork:

    def __init__(self):
        print("FuncIn: Init NeuralNetwork", file=sys.stderr)

        Container.nullId()
        self.storage = Storage()
        self.nodes = []

    def add_layer(self, function, input_layers=[], label="") -> int:
        res = self.storage.add_container()
        if len(input_layers) == 0:
            input_layers = [self.storage.inputId()]
        node = Node(function, res, input_layers, self.storage, label=label)
        self.nodes.append(node)
        return res.getId()

    def add_var(self, data) -> int:
        res = self.storage.add_container(False)
        res.write(data)
        return res.getId()

    def eval(self, data, verbose=False):
        self.storage.get_container(self.storage.inputId()).write(data)
        for i in self.nodes:
            i.eval(verbose)
        return self.storage.get_container(self.storage.outputId()).read()

    def pprint(self):
        print("Print Neural Network:")
        self.storage.pprint()
        for i in self.nodes: i.pprint()


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
        #if isinstance(self.module, TorchTensorCat):
            #return self.module((args[0][0], args[0][1]))
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


class FAMLINN(NeuralNetwork):

    def __init__(self):
        self._graph_hook_map = {}
        super().__init__()

    def from_net(self, net: nn.Module, arg: Any):
        h = hl.build_graph(net, arg)
        graph = []
        outputs = []
        for node in h.nodes:
            graph.append(h.nodes[node])
            outputs.append(node)
        n = len(outputs)
        edges = [[] for _ in outputs]
        edgesR = [[] for _ in outputs]
        for edge in h.edges:
            id1 = 0
            id2 = 0
            for i in range(len(outputs)):
                o = outputs[i]
                if o == edge[0]:
                    id1 = i
                if o == edge[1]:
                    id2 = i
            edges[id1].append(id2)
            edgesR[id2].append(id1)
        from_id = 0
        for i in range(n):
            if len(edgesR[i]) == 0:
                from_id = i
        bfs = [(0, from_id)]
        cnt = 0
        mapping = {}

        def generate_f(gid): return lambda x: str(graph[gid]) + '\n' + str(x[0])

        while len(bfs) > 0:
            t = bfs[0]
            dist = t[0]
            i = t[1]
            cnt += 1
            print(mapping, edgesR[i])
            mapping[i] = self.add_layer(generate_f(i), list(map(lambda x: mapping[x], edgesR[i])))
            bfs = bfs[1:]
            for j in edges[i]:
                bfs.append((dist + 1, j))
            bfs.sort()
        return

    def hook_net(self, net: nn.Module, arg):
        return self.hook_net_hidden_layer(net, arg)
        # return self.hook_linear_net(net, arg)

    def hook_linear_net(self, net: nn.Module, arg):
        def add_hook(module):
            print(module, file=sys.stderr)
            if isinstance(module, torch.Tensor):
                return
            if isinstance(module, nn.Sequential):
                return
            module.register_forward_hook(self._generate_hook_graph(module, net))

        net.apply(add_hook)
        self.active_hook = True
        net(arg)
        self.active_hook = False


    def hook_net_hidden_layer(self, net: nn.Module, arg):
        def add_hook(module):
            print(module, file=sys.stderr)
            if isinstance(module, torch.Tensor):
                return
            if isinstance(module, nn.Sequential):
                return
            module.register_forward_hook(self._generate_hook_list(module, net))

        net.apply(add_hook)
        self.active_hook = True
        net(arg)
        self.active_hook = False

        trace, out = torch.jit._get_trace_graph(net, arg)
        torch_graph = torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
        list_id = 0
        torch_id_to_container = {}
        skip = 0
        last_skip_input = 0
        last_skip_output = 0
        for node in torch_graph.nodes():
            inp = [i.unique() for i in node.inputs()]
            outp = [i.unique() for i in node.outputs()]
            input_containers = []
            output_container = outp[0]

            module = self._graph_hook_map[list_id]
            print(node, module, list_id)
            no_skip = True
            if skip > 0:
                skip -= 1
                if skip == 0:
                    inp = [last_skip_input]
                    no_skip = False
                else:
                    continue
            if no_skip and isinstance(module, nn.AvgPool2d):
                last_skip_input = last_skip_output
                skip = 8
                continue
            if no_skip and isinstance(module, nn.Dropout2d):
                last_skip_input = last_skip_output
                skip = 2
                continue
            if no_skip and isinstance(module, TorchTensorSmartView):
                last_skip_input = last_skip_output
                skip = 1
                continue
            if no_skip and isinstance(module, TorchTensorSmartReshape):
                last_skip_input = last_skip_output
                skip = 7
                continue
            if no_skip and isinstance(module, TorchTensorSkipModule):
                last_skip_input = last_skip_output
                skip = module.skip
                continue
            last_skip_output = output_container

            for i in inp:
                if i in torch_id_to_container:
                    input_containers.append(torch_id_to_container[i])
            if len(input_containers) == 0:
                input_containers = [self.storage.outputId()]
            added_container = self.add_layer(Evaluator(module), input_containers, str(module))
            list_id += 1
            torch_id_to_container[output_container] = added_container

    def _generate_hook_last(self, module, net):
        def __hook(_module, _input, _output):
            if self.active_hook and not isinstance(_module, net.__class__):
                self.add_layer(lambda x: module(*x), [self.storage.outputId()])

        return __hook

    def _generate_hook_list(self, module, net):
        global cnt
        cnt = 0

        def __hook(_module, _input, _output):
            global cnt
            if self.active_hook and not isinstance(_module, net.__class__) and not hasattr(_module, 'famlinn_ignore'):
                self._graph_hook_map[cnt] = _module
                cnt += 1

        return __hook

    def _generate_hook_graph(self, _module, net):
        def __hook(module, input, output):
            if not self.active_hook or isinstance(module, net.__class__):
                return

            #print("Hooked ", module)
            #print("In", input)
            #print("Out", output)

            res = []

            if isinstance(module, TorchTensorCat):
                for i in input[0]:
                    for container_id in self._graph_hook_map:
                        tnsr = self._graph_hook_map[container_id]
                        try:
                            if torch.all(tnsr.eq(i)):
                                res.append(container_id)
                        except (RuntimeError, TypeError):
                            pass

            if len(res) == 0:
                res = [self.storage.outputId()]
            next_container_id = self.add_layer(lambda x: module(*x), res, _module)
            self._graph_hook_map[next_container_id] = output

        return __hook

    def export(self, output: str):
        codegen = CodeGen(output)
        self.generate(codegen)
        codegen.write()

    def generate(self, codegen: CodeGen) -> None:
        codegen.add_lines([
            "from torch.nn import *",
            "from src.famlinn import *",
            "",
            "",
            "class Net(nn.Module):",
        ])
        codegen.tabulate_add(self.generate_constructor())
        codegen.add_line("")
        codegen.tabulate_add(self.generate_forward())

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
            args = '[' + ",".join(map(lambda x: 'res_{}'.format(x-1), node.args)) + ']'
            res = node.res.getId()
            line = "res_{} = self.node_{}(*{})  # {}".format(i+1, i, args, node.label)
            codegen.add_line(line)
        codegen.add_line("return res_{}".format(len(self.nodes)))
        res_codegen.tabulate_add(codegen)
        return res_codegen

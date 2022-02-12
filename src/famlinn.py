import abc
import sys

import torch.nn as nn
import hiddenlayer as hl
from typing import Any


class Container():

    __NEXT_ID = 0

    def __init__(self):
        print("FuncIn: Init Container", file=sys.stderr)
        Container.__NEXT_ID += 1
        self.id = Container.__NEXT_ID
        self.data = None

    def write(self, data): self.data = data

    def read(self): return self.data

    def getId(self): return self.id


class Storage:

    def __init__(self):
        print("FuncIn: Init Storage", file=sys.stderr)

        self.inputContainer = Container()
        self.outputContainer = self.inputContainer
        self.data = {}
        self.data[self.inputContainer.getId()] = self.inputContainer

    def get_container(self, id:int, isInput:bool=False, isOutput:bool=False):
        if isInput: return self.inputContainer
        if isOutput: return self.outputContainer
        if id not in self.data: return None
        return self.data[id]

    def add_container(self, setOutput=True) -> Container:
        cont = Container()
        self.data[cont.getId()] = cont
        if setOutput: self.outputContainer = cont
        return cont

    def inputId(self): return self.inputContainer.getId()

    def outputId(self): return self.outputContainer.getId()

    def pprint(self):
        print("Input container ", self.inputContainer.getId())
        print("Output container", self.outputContainer.getId())

class Node:

    def __init__(self, funct, res:int, args=[], storage:Storage=None):
        print("FuncIn: Init Node", file=sys.stderr)
        assert funct is not None, "Can not make Node with no function"
        assert storage is not None, "Can not make Node with no storage"

        self.funct = funct
        self.args = args
        self.res = res
        self.storage = storage

    def eval(self): self.res.write(self.funct([self.storage.get_container(i).read() for i in self.args]))

    def pprint(self):
        print("Node from {} to {}".format(self.args, self.res.getId()))


# Format AutoML Iteratable Neural Network
class NeuralNetwork():

    def __init__(self):
        print("FuncIn: Init NeuralNetwork", file=sys.stderr)

        self.storage = Storage()
        self.nodes = []

    def add_layer(self, function, input_layers=[]) -> int:
        res = self.storage.add_container()
        if len(input_layers) == 0:
            input_layers = [self.storage.inputId()]
        node = Node(function, res, input_layers, self.storage)
        self.nodes.append(node)
        return res.getId()

    def add_var(self, data) -> int:
        res = self.storage.add_container(False)
        res.write(data)
        return res.getId()

    def from_net(self, net: nn.Module, arg: Any):
        h = hl.build_graph(net, arg)
        graph = []
        outputs = []
        for node in h.nodes:
            graph.append(h.nodes[node])
            outputs.append(node)
        n = len(outputs)
        edges = [[] for i in outputs]
        edgesR = [[] for i in outputs]
        for edge in h.edges:
            id1 = 0
            id2 = 0
            for i in range(len(outputs)):
                o = outputs[i]
                if o == edge[0]: id1 = i
                if o == edge[1]: id2 = i
            edges[id1].append(id2)
            edgesR[id2].append(id1)
        fromId = 0
        for i in range(n):
            if len(edgesR[i]) == 0:
                fromId = i
        bfs = [(0, fromId)]
        cnt = 0
        mapping ={}
        def generateF(i):
            return lambda x: str(graph[i]) + '\n' + str(x[0])
        while len(bfs) > 0:
            t = bfs[0]
            dist = t[0]
            i = t[1]
            cnt += 1
            mapping[i] = cnt
            bfs = bfs[1:]
            self.add_layer(generateF(i), list(map(lambda x: mapping[x], edgesR[i])))
            for j in edges[i]:
                bfs.append((dist + 1, j))
            bfs.sort()
        return

    def eval(self, data):
        self.storage.get_container(self.storage.inputId()).write(data)
        for i in self.nodes: i.eval()
        return self.storage.get_container(self.storage.outputId()).read()

    def pprint(self):
        print("Print Neural Network:")
        self.storage.pprint()
        for i in self.nodes: i.pprint()


class FAMLINN(NeuralNetwork):
    pass
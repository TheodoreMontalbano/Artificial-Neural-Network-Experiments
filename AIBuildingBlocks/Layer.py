import numpy as np
from AIBuildingBlocks import Neuron


class Layer:
    # List of nodes on this layer
    nodes = None
    # How many nodes this layer has
    size = None

    # activationFunction: The activation function to use
    # nodes: List of nodes for this layer
    # nodeNumber: number of nodes for this layer randomly generated requires edgeNumber
    # edgeNumber: number of edges of the previous layer
    def __init__(self, activationFunction=None, nodes=[], nodeNumber=0, edgeNumber=0):
        if nodes:
            self.nodes = nodes
            self.size = len(nodes)
            return self
        if nodeNumber and edgeNumber:
            self.nodes = []
            self.size = edgeNumber
            for i in range(nodeNumber):
                self.nodes.append(Neuron.Neuron(activationFunction, edgeNumber))
            return self
        return -1

    # dot products the incoming vector against each of our node's edge weights
    # returns an np array of the products
    def process(self, incomingVector):
        outputVector = []
        for i in range(len(self.nodes)):
            outputVector.append(self.nodes[i].dotProd(incomingVector))
        return np.array(outputVector)

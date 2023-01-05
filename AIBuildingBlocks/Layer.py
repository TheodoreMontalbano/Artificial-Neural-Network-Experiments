import numpy as np
from AIBuildingBlocks import Neuron


class Layer:
    # List of nodes on this layer
    _nodes = None
    # The activation function this layer uses
    _activationFunction = None

    # activationFunction: The activation function to use
    # nodes: List of nodes for this layer
    # nodeNumber: number of nodes for this layer randomly generated requires edgeNumber
    # edgeNumber: number of edges of the previous layer
    def __init__(self, nodeNumber, edgeNumber, activationFunction=None):
        self._nodes = []
        for i in range(nodeNumber):
            self._nodes.append(Neuron.Neuron(activationFunction, edgeNumber))
        self._activationFunction = self._nodes[0].getActivationFunction()
        return

    # Gets the activation function this layer uses
    def getActivationFunction(self):
        return self._activationFunction

    # Sets the activation function this layer uses
    def setActivationFunction(self, func):
        self._activationFunction = func
        for i in self._nodes:
            i.setActivationFunction(func)

    # dot products the incoming vector against each of our node's edge weights
    # returns an np array of the products
    def process(self, incomingVector):
        outputVector = []
        for i in range(len(self._nodes)):
            outputVector.append(self._nodes[i].dotProd(incomingVector))
        return np.array(outputVector)

    # How many nodes this layer has
    def getSize(self):
        return len(self._nodes)

    # Gets the number of edgeweights this Layers has at each node
    def getEdgeWeightNumber(self):
        return self._nodes[0].getSize()

    # deletes an edgeweight from each of this layer's nodes at index
    def deleteEdgeWeight(self, index):
        for i in self._nodes:
            i.deleteEdgeWeight(index)

    # appends an edgeweight to each of this Layer's nodes at index
    def addEdgeWeight(self, newVal):
        for i in self._nodes:
            i.appendEdgeWeight(newVal)

    # Changes the value of edgeweight j of node i
    def setEdgeWeight(self, i, j, newVal):
        self._nodes[i].editEdgeWeight(j, newVal)

    # Gets edgeweight j at node i
    def getEdgeWeight(self, i, j):
        return self._nodes[i].getEdgeweight(j)

    # deletes Neuron i
    def deleteNeuron(self, i):
        self._nodes.pop(i)

    # Appends a neuron to this layer
    def appendNeuron(self):
        self._nodes.insert(self.getSize() - 1, Neuron.Neuron(self._activationFunction, self._nodes[0].getSize()))

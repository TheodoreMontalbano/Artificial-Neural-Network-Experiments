from AIBuildingBlocks import Layer
import random
from copy import deepcopy


class NeuralNetwork:
    # Layers of neural network (Should not include input layer)
    _layers = None
    # Shape of neural network (Sizes of all layers)
    _shape = None

    # shape: the shape of the NN
    # activationFunctions: The activation functions to use for each Layer
    # bound: The bound to abide by when creating a randomized NN
    def __init__(self, shape=[], activationFunctions=None, bound=0):
        if shape:
            self._shape = shape
            self._layers = NeuralNetwork._createLayersFromShape(shape, activationFunctions)
        elif bound:
            self._shape = []
            for i in range(2, random.randint(2, bound - 1) + 1):
                self._shape.append(random.randint(1, bound - 1) + 1)
            self._layers = NeuralNetwork._createLayersFromShape(self._shape)
        return

    # processes a vector of data and creates an output
    def processVector(self, vector):
        for i in self._layers:
            vector = i.process(vector)
        return vector

    # appends a layer to the NN
    def addLayer(self, activationFunction=None, nodeNumber=1):
        return self.addLayerAtIndex(len(self._shape) - 1, activationFunction, nodeNumber)

    # Adds the layer at the specified index
    def addLayerAtIndex(self, index, activationFunction=None, nodeNumber=1):
        if index == len(self._shape) - 1:
            self._layers.append(Layer.Layer(nodeNumber, self._shape[len(self._shape) - 1], activationFunction))
            self._shape.append(nodeNumber)
        elif index == 0:
            # Inserting the old input layer as a layer
            # It should have nodeNumber edges as that is the new input layer and have
            # The current input layer nodes
            self._layers.insert(index, Layer.Layer(self._shape[index], nodeNumber, activationFunction))
            self._shape.insert(index, nodeNumber)
        else:
            self._layers.insert(index - 1, Layer.Layer(nodeNumber, self._shape[index - 1], activationFunction))
            size = self._layers[index].getSize()
            if nodeNumber > size:
                for i in range(nodeNumber - size):
                    self._layers[index].addEdgeWeight(random.random())
            elif nodeNumber < size:
                for i in range(size - nodeNumber):
                    self._layers[index].deleteEdgeWeight(self._layers[index].getEdgeWeightNumber() - 1)
            self._shape.insert(index, nodeNumber)

    # Deletes the layer at index
    def deleteLayerAtIndex(self, index):
        # Index should never be equal to the output or input layers never 0 or len(self._shape) - 1
        if self._shape[index + 1] > self._shape[index - 1]:
            for i in range(self._shape[index + 1] - self._shape[index - 1]):
                self._layers[index].deleteEdgeWeight(self._layers[index + 1].getEdgeWeightNumber() - 1)
        elif self._shape[index + 1] < self._shape[index - 1]:
            for i in range(self._shape[index - 1] - self._shape[index + 1]):
                self._layers[index].addEdgeWeight(random.random())
        self._layers.pop(index - 1)
        self._shape.pop(index)

    # Returns the value of shape at the given index
    def getShapeAtIndex(self, index):
        return self._shape[index]

    # The size of this neural network
    def getSize(self):
        return len(self._shape)

    # Returns the activation function for the layer at the specified index
    def getActivationFunction(self, i):
        # Need to subtract 1 as there should always be 1 less layer than shape size
        return self._layers[i - 1].getActivationFunction()

    # Creates Layers for a neural network based on a specified shape
    @staticmethod
    def _createLayersFromShape(shape, activationFunctions=None):
        layers = []
        if activationFunctions:
            for i in range(1, len(shape)):
                layers.append(Layer.Layer(shape[i], shape[i - 1], activationFunctions[i - 1]))
        else:
            for i in range(1, len(shape)):
                layers.append(Layer.Layer(shape[i], shape[i - 1], None))
        return layers

    # sets edgeweight k at node j at layer i to newVal
    # i should never be 0 because input layer does not have edgeweights
    def editEdgeWeight(self, i, j, k, newVal):
        self._layers[i - 1].setEdgeWeight(j, k, newVal)

    # Returns edgeweight k at node j at layer i
    # i should never be 0 because input layer does not have edgeweights
    def getEdgeWeight(self, i, j, k):
        return self._layers[i - 1].getEdgeWeight(j, k)

    # deletes neuron j at layer i
    def deleteNeuron(self, i, j):
        # Adjusting shape
        self._shape[i] = self._shape[i] - 1
        # Delete the neuron from the current layer
        self._layers[i - 1].deleteNeuron(j)
        # Adjust the layer that comes after this one if it exists
        if i < self.getSize():
            self._layers[i].deleteEdgeWeight(random.randint(0, self._shape[i + 1] - 1))

    # adds a neuron to layer i
    def addNeuron(self, i):
        # Adjusting shape
        self._shape[i] = self._shape[i] + 1
        # Add the neuron to the current layer
        self._layers[i - 1].appendNeuron()
        # Adjust the layer that comes after this one if it exists
        if i < self.getSize():
            self._layers[i].addEdgeWeight(random.random())

    # returns this neural network's shape
    def getShape(self):
        return deepcopy(self._shape)

    # Sets the activationFunction at layer i to func
    def setActivationFunction(self, i, func):
        self._layers[i - 1].setActivationFunction(func)

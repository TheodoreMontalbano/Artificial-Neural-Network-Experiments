from AIBuildingBlocks import Layer
import random


# Creates Layers for a neural network based on a specified shape
def createLayersFromShape(shape, activationFunctions=None):
    if activationFunctions:
        layers = []
        for i in range(1, len(shape)):
            layers.append(Layer.Layer(activationFunctions[i - 1], [], shape[i], shape[i - 1]))
    else:
        layers = []
        for i in range(1, len(shape)):
            layers.append(Layer.Layer(None, [], shape[i], shape[i - 1]))
    return layers


class NeuralNetwork:
    # Layers of neural network
    layers = None
    # Shape of neural network (Sizes of all layers)
    shape = None

    # activationFunctions: The activation functions to use for each Layer
    # layers: the layers of the NN requires inputLayer
    # inputLayer: How the NN will recieve input
    # shape: the shape of the NN
    # bound: The bound to abide by when creating a randomized NN
    def __init__(self, activationFunctions=[], layers=[], inputLayer=0, shape=[], bound=0):
        if layers and inputLayer:
            self.layers = layers
            self.shape = []
            self.shape.append(inputLayer)
            for i in layers:
                self.shape.append(i.size)
            return self
        if shape and activationFunctions:
            self.shape = shape
            self.layers = createLayersFromShape(shape, activationFunctions)
            return self
        elif shape:
            self.shape = shape
            self.layers = createLayersFromShape(shape)
            return self
        if bound:
            self.shape = []
            for i in range(random.randint(bound - 1) + 1):
                self.shape.append(random.randint(bound - 1) + 1)
            self.layers = createLayersFromShape(self.shape)
            return self
        if inputLayer:
            self.shape.append(inputLayer)
            return self

    def processVector(self, vector):
        for i in self.layers:
            vector = i.process(vector)
        return vector

    def addLayer(self, activationFunction, nodeNumber):
        self.layers.append(Layer.Layer(activationFunction), [], nodeNumber, self.shape[len(self.shape) - 1])
        self.shape.append(nodeNumber)

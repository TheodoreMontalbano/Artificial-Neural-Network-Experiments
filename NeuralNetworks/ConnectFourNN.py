from Math import MathFunctions
import numpy as np
from GameInterfaces import INNPlayer
from AIBuildingBlocks import NeuralNetwork
from Games import ConnectFour


class ConnectFourNN(INNPlayer.INNPlayer):
    _brain = None
    _inputLayer = 42
    _name = None
    _outputFunction = None
    _outputLayer = 1

    def __init__(self, name, shape):
        self._outputFunction = ConnectFourNN.ConnectFourOutputLayer
        length = len(shape)
        if shape[length - 1] == 1:
            activationFunctions = [None for i in range(length)]
            activationFunctions[length - 1] = self._outputFunction
            self._brain = NeuralNetwork.NeuralNetwork(shape, activationFunctions)
        else:
            self._brain = NeuralNetwork.NeuralNetwork(shape)
        self._name = name
        if not self._brain.getShapeAtIndex(0) == self._inputLayer:
            self._brain.addLayerAtIndex(0, None, self._inputLayer)
        if self._brain.getShapeAtIndex(self._brain.getSize() - 1) == self._outputLayer:
            self._brain.setActivationFunction(self._brain.getSize() - 1, self._outputFunction)
        else:
            self._brain.addLayer(self._outputFunction, self._outputLayer)

    # sets edgeweight k at node j at layer i to newVal
    # i should never be 0 because input layer does not have edgeweights
    def setEdgeWeight(self, i, j, k, newVal):
        self._brain.editEdgeWeight(i, j, k, newVal)

    # Returns edgeweight k at node j at layer i
    # i should never be 0 because input layer does not have edgeweights
    def getEdgeWeight(self, i, j, k):
        return self._brain.getEdgeWeight(i, j, k)

    # Returns the name of this AI
    def getName(self):
        return self._name

    # deletes neuron j at layer i - 1
    def deleteNeuron(self, i, j):
        self._brain.deleteNeuron(i, j)

    # adds a neuron at layer i
    def addNeuron(self, i):
        self._brain.addNeuron(i)

    # returns the value of this Ai's shape at the index
    def getShapeAtIndex(self, index):
        return self._brain.getShapeAtIndex(index)

    # returns the size of this neural network. -> How many layers
    def getSize(self):
        return self._brain.getSize()

    # Returns the game that this NN is meant to be trained on
    @staticmethod
    def getGame():
        return ConnectFour.ConnectFour

    # Function used as the activation function for the output Layer of the connect Four game
    @staticmethod
    def ConnectFourOutputLayer(x):
        return np.floor(MathFunctions.aSigmoid(x, 7))

    # add a layer at the given layer
    def addLayerAtIndex(self, index, activationFunction=None, nodeNumber=1):
        self._brain.addLayerAtIndex(index, activationFunction, nodeNumber)

    # deletes the layer at the given index
    def deleteLayerAtIndex(self, index):
        self._brain.deleteLayerAtIndex(index)

    # This player makes a move
    def makeMove(self, state):
        return self._brain.processVector(state)

    # This is a robot so...
    def isRobot(self):
        return True

    # Returns the shape of NN
    def getShape(self):
        return self._brain.getShape()

    # Returns the name of the game
    @staticmethod
    def getGameName():
        return "ConnectFour"

    # The activation function at layer i
    def getActivationFunction(self, i):
        return self._brain.getActivationFunction(i)

    # sets the activation function at layer i
    def setActivationFunction(self, i, func):
        self._brain.setActivationFunction(i, func)

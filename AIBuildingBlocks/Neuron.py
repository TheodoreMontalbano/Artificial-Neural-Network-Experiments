import numpy as np
import random
from Math import MathFunctions


class Neuron:
    # Lists of weights for incoming edges - np array
    _edgeWeights = None
    # The activation function of the Neuron
    _activationFunction = None

    # activationFunction: The activation function to use
    # edgeNumber: number of edges you want
    # edgeWeights: weights of all edges input as np array
    # bound: bound of how many possible edges randomly generated you want
    def __init__(self, activationFunction, edgeNumber, edgeWeights=[], bound=5):
        if activationFunction:
            self._activationFunction = activationFunction
        else:
            self._activationFunction = MathFunctions.sigmoid
        if edgeWeights:
            self._edgeWeights = edgeWeights
            return
        if edgeNumber:
            self._edgeWeights = np.random.rand(edgeNumber)
            for i in range(self._edgeWeights.size):
                self._edgeWeights[i] = self._edgeWeights[i] * 2 - 1
            return
        self._edgeWeights = np.random.rand(random.randint(bound - 1) + 1)
        for i in range(self._edgeWeights.size):
            self._edgeWeights[i] = self._edgeWeights[i] * 2 - 1
        return

    # Takes in an np array and multiplies by all edge weights using dot product
    def dotProd(self, incomingVector):
        return self._activationFunction(np.dot(incomingVector, self._edgeWeights))

    # Changes the value of a particular edgeweight
    def editEdgeWeight(self, index, newVal):
        self._edgeWeights[index] = newVal

    # adds an edgeWeight to the neuron with value newVal
    def appendEdgeWeight(self, newVal):
        self._edgeWeights = np.append(self._edgeWeights, newVal)

    # deletes an edgeweight at index from the neuron
    def deleteEdgeWeight(self, index):
        self._edgeWeights = np.delete(self._edgeWeights, index)

    # returns the activation function used for these edgeweights
    def getActivationFunction(self):
        return self._activationFunction

    # sets the activation function used for this node
    def setActivationFunction(self, func):
        self._activationFunction = func

    # Returns the size of these edgeweights
    def getSize(self):
        return self._edgeWeights.size

    # Returns the edgeweight at index i
    def getEdgeweight(self, i):
        return self._edgeWeights[i]
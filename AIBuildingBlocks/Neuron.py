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
    # edgeWeights: weights of all edges input at np array
    # bound: bound of how many possible edges randomly generated you want
    def __init__(self, activationFunction, edgeNumber, edgeWeights, bound):
        if activationFunction:
            self.activationFunction = activationFunction
        else:
            self.activationFunction = MathFunctions.sigmoid
        if edgeWeights:
            self.edgeWeights = edgeWeights
            return self
        if edgeNumber:
            self.edgeWeights = np.ndarray(shape=edgeNumber, dtype=float)
            return self
        self.edgeWeights = np.ndarray(shape=(random.randint(bound - 1) + 1), dtype=float)
        return self

    # Takes in an np array and multiplies by all edge weights using dot product
    def dotProd(self, incomingVector):
        return self.activationFunction(np.dot(incomingVector, self.edgeWeights))

import numpy as np


# Returns 1 / (1 + e^(-x))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Returns a * sigmoid(x)
def aSigmoid(x, a):
    return a * sigmoid(x)
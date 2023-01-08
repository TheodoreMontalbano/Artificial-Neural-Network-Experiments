import numpy as np
import random


# Returns 1 / (1 + e^(-x))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Returns a * sigmoid(x)
def aSigmoid(x, a):
    return a * sigmoid(x)


# selects num to select randomly with a left skew from popSize
def selectLeftSkewRandomly(numToSelect, popSize):
    selected = []
    unselected = []
    for i in range(popSize):
        if popSize - i == numToSelect:
            selected.append(i)
            numToSelect = numToSelect - 1
        elif numToSelect > 0:
            save = random.randint(0, 10)
            if save:
                selected.append(i)
                numToSelect = numToSelect - 1
            else:
                unselected.append(i)
        else:
            unselected.append(i)
    return [selected, unselected]


# selects num randomly with right skew from pop
def selectRightSkewRandomly(numToSelect, popSize):
    selected = []
    unselected = []
    for i in range(popSize):
        if popSize - i == numToSelect:
            selected.append(popSize - i - 1)
            numToSelect = numToSelect - 1
        elif numToSelect > 0:
            save = random.randint(0, 10)
            if save:
                selected.append(popSize - i - 1)
                numToSelect = numToSelect - 1
            else:
                unselected.append(popSize - i - 1)
        else:
            unselected.append(popSize - i - 1)
    return [selected, unselected]


# VOne of ConnectFourOutputLayer
def ConnectFourOutputLayerVOne(x):
    return np.floor(aSigmoid(x, 7))


# VTwo ConnectFour OutputLayerVTwo
def ConnectFourOutputLayerVTwo(x):
    return max(min(x, 6), 0)

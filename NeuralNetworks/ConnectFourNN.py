from Math import MathFunctions
import numpy as np
from GameInterfaces import IPlayer


class ConnectFourNN(IPlayer.IPlayer):
    Name = None
    outputLayer = 1
    brain = None

    def __init__(self, Name, brain, filePath=""):
        self.Name = Name
        self.brain = brain
        self.brain.addLayer(ConnectFourOutputLayer, self.outputLayer)

    # If this is an AI return true
    # if this is a player don't
    def isRobot(self):
        return True

    # queries player for a move based on state
    def makeMove(self, state):
        return self.brain.process(state)

    @staticmethod
    def getInputLayer():
        return 42


# Function used as the activation function for the output Layer of the connect Four game
def ConnectFourOutputLayer(x):
    return np.floor(MathFunctions.aSigmoid(x, 8))

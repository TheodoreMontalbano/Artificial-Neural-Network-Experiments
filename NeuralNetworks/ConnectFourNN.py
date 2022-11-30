from Math import MathFunctions
import numpy as np
from GameInterfaces import IPlayer


class ConnectFourNN(IPlayer.IPlayer):
    # If this is an AI return true
    # if this is a player don't
    def isRobot(self):
        return True

    # queries player for a move based on state
    def makeMove(self, state):
        pass


# Function used as the activation function for the output Layer of the connect Four game
def ConnectFourOutputLayer(x):
    return np.ceil(MathFunctions.aSigmoid(x, 8))

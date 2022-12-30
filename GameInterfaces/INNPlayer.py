from GameInterfaces import IPlayer


class INNPlayer(IPlayer.IPlayer):
    # The Name of this NN
    _name = None
    # The input layer this NN requires to play this game
    _inputLayer = None
    # The output layer this NN requires to play this game
    _outputLayer = None
    # The brain the NN requires to play this game
    _brain = None
    # The activation function the output layer requires
    _outputFunction = None

    # If this is an AI return true
    # if this is a player don't
    def isRobot(self):
        pass

    # queries player for a move based on state
    def makeMove(self, state):
        pass

    # returns the shape of this NN's brain
    def getShape(self):
        pass

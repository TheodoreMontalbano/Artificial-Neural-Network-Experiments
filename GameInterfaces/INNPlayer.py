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

    # region Inherited methods

    # The activation function at layer i
    def getActivationFunction(self, i):
        return self._brain.getActivationFunction(i)

    # sets the activation function at layer i
    def setActivationFunction(self, i, func):
        self._brain.setActivationFunction(i, func)

    # Returns the shape of NN
    def getShape(self):
        return self._brain.getShape()

    # add a layer at the given layer
    def addLayerAtIndex(self, index, activationFunction=None, nodeNumber=1):
        self._brain.addLayerAtIndex(index, activationFunction, nodeNumber)

    # deletes the layer at the given index
    def deleteLayerAtIndex(self, index):
        self._brain.deleteLayerAtIndex(index)

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

    # If this is an AI return true
    # if this is a player don't
    def isRobot(self):
        return True

    # sets edgeweight k at node j at layer i to newVal
    # i should never be 0 because input layer does not have edgeweights
    def setEdgeWeight(self, i, j, k, newVal):
        self._brain.editEdgeWeight(i, j, k, newVal)

    # Returns edgeweight k at node j at layer i
    # i should never be 0 because input layer does not have edgeweights
    def getEdgeWeight(self, i, j, k):
        return self._brain.getEdgeWeight(i, j, k)

    # endregion

    # region Methods needing implementation

    # Returns the name of the game
    @staticmethod
    def getGameName():
        pass

    # queries player for a move based on state
    def makeMove(self, state):
        pass

    # Returns the game that this NN is meant to be trained on
    @staticmethod
    def getGame():
        pass

    # Returns list of basic players for guidance for AI
    # Only necessary if you want to use launchpad of genetic alg
    @staticmethod
    def getLaunchPadPlayers():
        pass
    # endregion

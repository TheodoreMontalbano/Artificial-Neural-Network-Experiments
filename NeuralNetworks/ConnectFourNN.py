from Math import MathFunctions
from GameInterfaces import INNPlayer
from AIBuildingBlocks import NeuralNetwork
from Games import ConnectFour
from Player import MockPlayer
from random import randint


class ConnectFourNN(INNPlayer.INNPlayer):
    _brain = None
    _inputLayer = 84
    _name = None
    _outputFunction = None
    _outputLayer = 1

    def __init__(self, name, shape):
        self._outputFunction = MathFunctions.ConnectFourOutputLayerVOne
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

    # region Implemented methods

    # Returns the game that this NN is meant to be trained on
    @staticmethod
    def getGame():
        return ConnectFour.ConnectFour

    # This player makes a move
    def makeMove(self, state):
        # This logic makes it so red is always "good", black always "bad"
        temp = []
        if not ConnectFourNN._isPlayerOne(state):
            temp = ConnectFourNN._translateState(state, True)
        else:
            temp = ConnectFourNN._translateState(state, False)
        return self._brain.processVector(temp)

    # Returns the name of the game
    @staticmethod
    def getGameName():
        return "ConnectFour"

    # Returns players to Launch
    @staticmethod
    def getLaunchPadPlayers():
        players = []
        for i in range(7):
            players.append(MockPlayer.MockPlayer(i, lambda state: i))
            players.append(MockPlayer.MockPlayer(i, lambda state: (i + int(ConnectFourNN._sumMoves(state) / 2)) % 7))
            players.append(MockPlayer.MockPlayer(i, lambda state: (i - int(ConnectFourNN._sumMoves(state) / 2)) % 7))
        players.append(MockPlayer.MockPlayer(0, lambda state: randint(0, 6)))
        return players

    # endregion

    # region Extra special handling methods

    # returns whether we are playerOne or not
    @staticmethod
    def _isPlayerOne(state):
        boardSum = ConnectFourNN._sumMoves(state)
        if boardSum % 2 == 0:
            return True
        else:
            return False

    # Sums the total number of moves made
    @staticmethod
    def _sumMoves(state):
        boardSum = 0
        for i in state:
            if not i == 0:
                boardSum = boardSum + 1
        return boardSum

    # translates the state, so that 0 -> 41 holds the positions of our pieces
    # and 42 -> 83 hold the position of our opponents pieces
    @staticmethod
    def _translateState(state, switchColors):
        temp = [0 for i in range(84)]
        # First Change state into a proper format for temp
        for i in range(len(state)):
            if state[i] == 1:
                temp[i] = 1
            elif state[i] == 2:
                temp[i + 42] = 1
        if switchColors:
            for i in range(len(state)):
                currVal = temp[i]
                temp[i] = temp[i + 42]
                temp[i + 42] = currVal
        return temp

    # endregion

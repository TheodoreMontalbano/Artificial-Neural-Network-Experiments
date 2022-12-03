from GameInterfaces import IArrayGame
from copy import deepcopy
from Enums import GameState, InvalidMoveCases
from NeuralNetworks import ConnectFourNN


class ConnectFour(IArrayGame.IArrayGame):
    # Current state of the game
    state = None
    stateVector = None
    players = None
    currPlayer = None
    currEmpty = None
    showGame = None

    def __init__(self, playerOne, playerTwo, showGame=False):
        # 6 row, 7 col
        temp = [0 for i in range(6)]
        self.state = [deepcopy(temp) for i in range(7)]
        self.stateVector = [0 for i in range(42)]
        self.players = []
        self.players.append(playerOne)
        self.players.append(playerTwo)
        self.currPlayer = 0
        self.currEmpty = [0 for i in range(7)]
        self.showGame = showGame

    # The current player makes a move
    def makeMove(self, move):
        if 0 > move or move > 6 or self.currEmpty[move] == 6:
            if self.players[self.currPlayer].isRobot():
                return InvalidMoveCases.InvalidMoveCases.AIInvalid
            else:
                return InvalidMoveCases.InvalidMoveCases.PlayerInvalid
        self.state[move][self.currEmpty[move]] = self.currPlayer + 1
        self.stateVector[move + self.currEmpty[move] * 7]
        self.currEmpty[move] = self.currEmpty[move] + 1
        return InvalidMoveCases.InvalidMoveCases.ValidMove

    # Has the players play a simulation of the game
    def playGame(self):
        move = -1
        isOver = GameState.GameState.NotOver
        moveValidity = InvalidMoveCases.InvalidMoveCases.ValidMove
        while isOver == GameState.GameState.NotOver:
            # Make a move
            move = self.players[self.currPlayer].makeMove(self.stateVector)
            moveValidity = self.makeMove(move)
            while moveValidity == InvalidMoveCases.InvalidMoveCases.PlayerInvalid:
                if 0 > move or move > 6:
                    print("Invalid move: please choose a number between 1 and 7")
                else:
                    print("Invalid move: Please choose a column that is not full")
                moveValidity = self.makeMove(self.players[self.currPlayer].makeMove(self.state))
            # If the AI makes an invalid move it counts as a loss
            if moveValidity == InvalidMoveCases.InvalidMoveCases.AIInvalid:
                return (self.currPlayer + 1) % 2
            if self.showGame:
                self.show()
            isOver = self.gameState()
            if isOver == GameState.GameState.Win:
                return self.currPlayer
            elif isOver == GameState.GameState.Tie:
                return -1
            self.currPlayer = (self.currPlayer + 1) % 2

    # Checks if the game is over
    def gameState(self):
        if self._checkHorizontal() or self._checkVertical() or self._checkDiagonal():
            return GameState.GameState.Win

        i = 0
        isTie = True
        while i < 7 and isTie:
            if self.currEmpty[i] != 6:
                isTie = False
            i = i + 1
        if isTie:
            return GameState.GameState.Tie
        else:
            return GameState.GameState.NotOver

    # Checks the horizontals of the board for a win
    def _checkHorizontal(self):
        count = 0
        isWin = False
        for i in range(6):
            count = 0
            for j in range(6):
                if self.state[j][i] == self.state[j + 1][i] and self.state[j][i] != 0:
                    count = count + 1
                    if count == 3:
                        isWin = True
                else:
                    count = 0
        return isWin

    # Checks the verticals of the board for a win
    def _checkVertical(self):
        isWin = False
        count = 0
        for i in range(7):
            count = 0
            for j in range(5):
                if self.state[i][j] == self.state[i][j + 1] and self.state[i][j] != 0:
                    count = count + 1
                    if count == 3:
                        isWin = True
                else:
                    count = 0
        return isWin

    # Checks the diagonals of the board for a win
    def _checkDiagonal(self):
        isWin = False
        count = 0
        # Upper 1/1 diagonal
        for i in range(4):
            count = 0
            for j in range(i, 5):
                if self.state[j - i][j] == self.state[j + 1 - i][j + 1] and self.state[j - i][j] != 0:
                    count = count + 1
                    if count == 3:
                        isWin = True
                else:
                    count = 0
        # Lower 1/1 diagonal
        for i in range(1, 4):
            count = 0
            for j in range(i, 6):
                if self.state[j][j - i] == self.state[j + 1][j - i + 1] and self.state[j][j - i] != 0:
                    count = count + 1
                    if count == 3:
                        isWin = True
                else:
                    count = 0
        # Upper -1/1 diagonal
        for i in range(4):
            count = 0
            for j in range(i, 5):
                if self.state[6 - (j - i)][j] == self.state[6 - (j + 1 - i)][(j + 1)] \
                        and self.state[6 - (j - i)][j] != 0:
                    count = count + 1
                    if count == 3:
                        isWin = True
                else:
                    count = 0
        # Lower -1/1 diagonal
        for i in range(1, 4):
            count = 0
            for j in range(i, 6):
                if self.state[6 - j][(j - i)] == self.state[6 - (j + 1)][j - i + 1] and self.state[6 - j][j - i] != 0:
                    count = count + 1
                    if count == 3:
                        isWin = True
                else:
                    count = 0
        return isWin

    # Outputs the game's state to the viewer
    def show(self):
        for i in range(1, 7):
            for j in range(7):
                print('{:3}'.format(translate(self.state[j][6 - i])), end='')
            print()

    # Creates a new ConnectFourNN from a NN with the correct input layer
    @staticmethod
    def createNewAI(Name, brain):
        return ConnectFourNN.ConnectFourNN(Name, brain)

    # Gets the input layer for a ConnectFour NN
    @staticmethod
    def getInputLayer():
        return ConnectFourNN.ConnectFourNN.getInputLayer()


def translate(x):
    if x == 1:
        return 'R'
    if x == 2:
        return 'B'
    return '_'

from GameInterfaces import IArrayGame
from copy import deepcopy
from Enums import GameState, InvalidMoveCases


class ConnectFour(IArrayGame.IArrayGame):
    # Current state of the game
    state = None
    players = None
    currPlayer = None
    currEmpty = None
    showGame = None
    status = None

    def __init__(self, playerOne, playerTwo, showGame):
        # 6 row, 7 col
        temp = [0 for i in range(6)]
        self.state = [deepcopy(temp) for i in range(7)]
        self.players = []
        self.players.append(playerOne)
        self.players.append(playerTwo)
        self.currPlayer = 0
        self.currEmpty = [0 for i in range(7)]
        self.showGame = showGame
        self.status = GameState.NotOver

    # The current player makes a move
    def makeMove(self, move):
        # TODO What if player makes an illegal move
        if (0 > move > 6) or self.currEmpty[move] == 6:
            if self.players[self.currPlayer].isRobot():
                return InvalidMoveCases.InvalidMoveCases.AIInvalid
            else:
                return InvalidMoveCases.InvalidMoveCases.PlayerInvalid
        self.state[self.currEmpty[move]][move] = self.currPlayer + 1
        return InvalidMoveCases.InvalidMoveCases.ValidMove

    # Has the players play a simulation of the game
    def playGame(self):
        isOver = GameState.GameState.NotOver
        moveValidity = InvalidMoveCases.InvalidMoveCases.ValidMove
        while isOver == GameState.GameState.NotOver:
            # Make a move
            moveValidity = self.makeMove(self.players[self.currPlayer].makeMove(self.state))
            while moveValidity == InvalidMoveCases.InvalidMoveCases.PlayerInvalid:
                print("Invalid move: please choose a number between 1 and 7")
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
            if self.showGame:
                self.show()
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
                if self.state[i][j] == self.state[i][j + 1] and self.state[i][j] != 0:
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
        for j in range(7):
            count = 0
            for i in range(5):
                if self.state[i][j] == self.state[i + 1][j] and self.state[i][j] != 0:
                    count = count + 1
                    if count == 3:
                        isWin = True
                else:
                    count = 0
        return isWin

    # Checks the diagonals of the board for a win
    # TODO Check this works
    def _checkDiagonal(self):
        isWin = False
        count = 0
        for i in range(1, 13):
            count = 0
            for j in range(min(13 - i, i)):
                if self.state[j + i][j - i] == self.state[j + i + 1][j - i + 1] and self.state[j + i][j - i] != 0:
                    count = count + 1
                    if count == 3:
                        isWin = True
                else:
                    count = 0
        count = 0
        for i in range(1, 13):
            count = 0
            for j in range(min(13 - i, i)):
                if self.state[i + j][i - j] == self.state[j + i + 1][j - i - 1] and self.state[j + i][j - i] != 0:
                    count = count + 1
                    if count == 3:
                        isWin = True
                else:
                    count = 0
        return isWin

    # Outputs the gamestate to the viewer
    def show(self):
        pass

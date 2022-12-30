from GameInterfaces import IPlayer


class Player(IPlayer.IPlayer):
    _name = None

    def __init__(self, name):
        self._name = name

    # If this is an AI return true
    # if this is a player don't
    def isRobot(self):
        return False

    # queries player for a move based on state
    def makeMove(self, state):
        return int(input("Please input the move you would like to make")) - 1

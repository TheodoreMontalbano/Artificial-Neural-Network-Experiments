from GameInterfaces import IPlayer


class MockPlayer(IPlayer.IPlayer):
    _name = None
    _moveFunction = None

    def __init__(self, name, moveFunction):
        self._name = name
        self._moveFunction = moveFunction

    # If this is an AI return true
    # if this is a player don't
    def isRobot(self):
        return True

    # queries player for a move based on state
    def makeMove(self, state):
        return self._moveFunction(state)

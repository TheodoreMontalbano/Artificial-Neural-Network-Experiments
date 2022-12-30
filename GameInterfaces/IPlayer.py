class IPlayer:
    _name = None

    # If this is an AI return true
    # if this is a player don't
    def isRobot(self):
        pass

    # queries player for a move based on state
    def makeMove(self, state):
        pass

    def getName(self):
        return self._name

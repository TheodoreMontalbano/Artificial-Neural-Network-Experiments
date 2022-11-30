from GameInterfaces import IPlayer


class Player(IPlayer.IPlayer):
    # If this is an AI return true
    # if this is a player don't
    def isRobot(self):
        return False

    # queries player for a move based on state
    def makeMove(self, state):
        print("Please input the move you would like to make")
        return int(input("Please input the move you would like to make"))

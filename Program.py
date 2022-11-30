from Games import ConnectFour
from Player import Player


def main():
    playerOne = Player.Player()
    playerTwo = Player.Player()
    game = ConnectFour.ConnectFour(playerOne, playerTwo, True)
    result = game.playGame() + 1
    print("Player " + str(result) + " Wins!")


# Run the program
if __name__ == "__main__":
    main()

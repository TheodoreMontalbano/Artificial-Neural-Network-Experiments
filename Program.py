from Games import ConnectFour
from Player import Player


def main():
    playerOne = Player.Player(input("What is Player One's name?"))
    playerTwo = Player.Player(input("What is Player Two's name?"))
    game = ConnectFour.ConnectFour(playerOne, playerTwo, True)
    result = game.playGame() + 1
    print("Player " + str(result) + " Wins!")


# Run the program
if __name__ == "__main__":
    main()

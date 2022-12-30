from Games import ConnectFour
from Player import Player
from LearningAlgorithms import ArrayGameGeneticAlgorithm
from AIBuildingBlocks import NeuralNetwork
from NeuralNetworks import ConnectFourNN


def main():
    choice = ""
    while True:
        choice = input("Choose an option \n 1. play a game \n 2. train an AI \n")
        if int(choice) == 1:
            choice = input("What game would you like to play \n 1. ConnectFour \n")
            if int(choice) == 1:
                playGame(ConnectFour.ConnectFour)
            else:
                print("Invalid selection")
        elif int(choice) == 2:
            choice = input("What game would you like to train an AI for \n 1. ConnectFour \n")
            if int(choice) == 1:
                trainAI(ConnectFourNN.ConnectFourNN, ArrayGameGeneticAlgorithm.GeneticAlgorithm)
        else:
            print("Invalid Selection")
            test()
            break


def trainAI(nNPlayer, learningAlgorithm):
    learnState = learningAlgorithm(nNPlayer)
    while True:
        choice = input("Choose an option \n 1. Go to next gen \n 2. Go forward n generations \n")
        if int(choice) == 1:
            learnState.nextGen()
        elif int(choice) == 2:
            learnState.nextNGens(int(input("How many generation should be done")))
        else:
            print("Invalid Choice")
            break


def playGame(game):
    playerOne = Player.Player(input("What is Player One's name?"))
    playerTwo = Player.Player(input("What is Player Two's name?"))
    toPlay = game(playerOne, playerTwo, True)
    result = toPlay.playGame() + 1
    print("Player " + str(result) + " Wins!")


def test():
    parentOne = NeuralNetwork.NeuralNetwork([2, 3, 4, 5])
    ConnectFourNN.ConnectFourNN("test", parentOne)
    print("hi")


# Run the program
if __name__ == "__main__":
    main()

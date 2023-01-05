from Games import ConnectFour
from Player import Player
from LearningAlgorithms import ArrayGameGeneticAlgorithm
from AIBuildingBlocks import NeuralNetwork
from NeuralNetworks import ConnectFourNN
from numpy import float64
from Math import MathFunctions


def main():
    choice = ""
    while True:
        choice = input("Choose an option \n 1. play a game \n 2. train an AI \n")
        if int(choice) == 1:
            choice = input("What game would you like to play \n 1. ConnectFour \n")
            if int(choice) == 1:
                playGame(ConnectFourNN.ConnectFourNN)
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


# returns the AI saved in the given file
def loadAI(filePath, nNPlayer):
    file = open("SavedNeuralNetworks/" + filePath + ".txt", "r")
    fileContents = file.readlines()
    shape = fileContents[0][8:-2].split(", ")
    for i in range(len(shape)):
        shape[i] = int(shape[i])
    toReturn = nNPlayer(filePath, shape)
    layerIndex = -1
    nodeIndex = -1
    edgeIndex = -1
    for i in range(1, len(fileContents)):
        if fileContents[i][0:6] == "Layer":
            layerIndex = int(fileContents[i][8:-2])
        elif fileContents[i][4:8] == "Node":
            nodeIndex = int(fileContents[i][9:-2])
        elif fileContents[i][8:19] == "EdgeWeight":
            edgeIndex = int(fileContents[i][21:fileContents[i].index(":")])
            nNPlayer.setEdgeWeight(layerIndex, nodeIndex, edgeIndex
                                   , float64(fileContents[i][fileContents[i].index(": "):-1]))
        elif fileContents[i][4:22] == "ActivationFunction":
            nNPlayer.setActivationFunction(layerIndex, eval(fileContents[i][fileContents[i].index(": ") + 1:-1]))
    return toReturn


def trainAI(nNPlayer, learningAlgorithm):
    learnState = learningAlgorithm(nNPlayer)
    while True:
        choice = input("Choose an option \n "
                       "1. Go to next gen \n "
                       "2. Go forward n generations \n "
                       "3. Save an AI from the current generation \n"
                       "4. Have a show match between two AI in the current generation \n")
        if int(choice) == 1:
            learnState.nextGen()
        elif int(choice) == 2:
            learnState.nextNGens(int(input("How many generation should be done ")))
        elif int(choice) == 3:
            learnState.saveAI(input("What is the name of the AI to save "))
            print("Saved!")
        elif int(choice) == 4:
            learnState.showMatch(input("What is the name of the AI for playerOne ")
                                 , input("What is the name of the AI for playerTwo "))
        else:
            print("Invalid Choice")
            break


def playGame(nNPlayer):
    if input("Would you like to have an AI play for Player One") == "yes":
        filePath = input("What is the file name for this AI: ")
        playerOne = loadAI(filePath, nNPlayer)
    else:
        playerOne = Player.Player(input("What is Player One's name?"))
    if input("Would you like to have an AI play for Player Two") == "yes":
        filePath = input("What is the file name for this AI: ")
        playerTwo = loadAI(filePath, nNPlayer)
    else:
        playerTwo = Player.Player(input("What is Player Two's name?"))
    toPlay = nNPlayer.getGame()(playerOne, playerTwo, True)
    result = toPlay.playGame() + 1
    if result == 1:
        print(str(playerOne.getName()) + " Wins!")
    elif result == 2:
        print(str(playerTwo.getName()) + " Wins!")
    else:
        print("Tie game no contest")


def test():
    parentOne = NeuralNetwork.NeuralNetwork([2, 3, 4, 5])
    ConnectFourNN.ConnectFourNN("test", parentOne)
    print("hi")


# Run the program
if __name__ == "__main__":
    main()

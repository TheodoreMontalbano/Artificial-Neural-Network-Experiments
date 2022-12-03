import random
from AIBuildingBlocks import NeuralNetwork
from numpy import floor
from copy import deepcopy

class GeneticAlgorithm:
    Id = 0
    currAI = None
    AiDict = None
    genSize = None
    game = None
    fitnessTracker = None
    killPerGen = None

    def __init__(self, game, genSize=100, bound=5, killPerGen=33):
        self.bound = bound
        self.game = game
        self.genSize = genSize
        self.currAI = []
        self.currAI = [game.createNewAI(self.genID(), self.genBrain()) for i in range(genSize)]
        self.fitnessTracker = [0 for i in range(genSize)]
        self.AiDict = {}
        for i in range(genSize):
            self.AiDict[i + 1] = i
        if killPerGen > genSize:
            killPerGen = floor(genSize * .33)

    def genBrain(self):
        layerNum = random.randint(self.bound) + 1
        shape = []
        for i in range(layerNum):
            shape.append(random.randint(self.bound) + 1)
        return NeuralNetwork.NeuralNetwork([], [], self.game.getInputLayer(), shape)

    def genID(self):
        self.Id = self.Id + 1
        return self.Id

    # TODO FINISH implement
    def nextGen(self, show=True):
        self.assessFitness()
        # TODO Kill Bottom X%
        # TODO Reproduce
        # TODO output stats
        # TODO reset for next Gen

    # Assesses the fitness of all AI
    def assessFitness(self):
        currGame = None
        removedList = []
        coinFlip = 0
        result = -1
        playerOne = None
        playerTwo = None
        for i in range(self.genSize):
            for j in range(i + 1, self.genSize):
                coinFlip = random.randint(0, 1)
                if coinFlip:
                    playerOne = self.currAI[i]
                    playerTwo = self.currAI[j]
                else:
                    playerOne = self.currAI[j]
                    playerTwo = self.currAI[i]
                currGame = self.game.ConnectFour(playerOne, playerTwo)
                result = currGame.playGame()
                if result == -1:
                    self.fitnessTracker[self.AiDict[playerOne.Name]] = \
                        self.fitnessTracker[self.AiDict[playerOne.Name]] + 1 / 2
                    self.fitnessTracker[self.AiDict[playerTwo.Name]] = \
                        self.fitnessTracker[self.AiDict[playerTwo.Name]] + 1 / 2
                elif result == 0:
                    self.fitnessTracker[self.AiDict[playerOne.Name]] = \
                        self.fitnessTracker[self.AiDict[playerOne.Name]] + 1
                    self.fitnessTracker[self.AiDict[playerTwo.Name]] = \
                        self.fitnessTracker[self.AiDict[playerTwo.Name]] - 1
                else:
                    self.fitnessTracker[self.AiDict[playerOne.Name]] = \
                        self.fitnessTracker[self.AiDict[playerOne.Name]] - 1
                    self.fitnessTracker[self.AiDict[playerTwo.Name]] = \
                        self.fitnessTracker[self.AiDict[playerTwo.Name]] + 1

    def killStragglers(self):
        temp = [0, 0]
        # ID, fitness
        fitnessRatings = [deepcopy(temp) for i in range(self.genSize)]
        for i in range(self.genSize):
            fitnessRatings[i][0] = self.currAI[i].Name
            fitnessRatings[i][1] = self.fitnessTracker[i]
        fitnessRatings.sort(key=lambda x: x[1])
        #TODO Kill rate 1/2


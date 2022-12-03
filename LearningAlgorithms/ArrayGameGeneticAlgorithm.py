import random
from AIBuildingBlocks import NeuralNetwork


class GeneticAlgorithm:
    Id = 0
    currAI = None
    AiDict = None
    genSize = None
    game = None

    def __init__(self, game, genSize=100, bound=5):
        self.bound = bound
        self.game = game
        self.genSize = genSize
        self.currAI = []
        self.currAI = [game.createNewAI(self.genID(), self.genBrain()) for i in range(genSize)]
        self.AiDict = {}
        for i in range(genSize):
            self.AiDict[i] = i

    def genBrain(self):
        layerNum = random.randint(self.bound) + 1
        shape = []
        for i in range(layerNum):
            shape.append(random.randint(self.bound) + 1)
        return NeuralNetwork.NeuralNetwork([], [], self.game.getInputLayer(), shape)

    def genID(self):
        self.Id = self.Id + 1
        return self.Id

    #TODO FINISH implement
    def nextGen(self, gens=1):
        currGame = None
        removedList = []
        coinFlip = 0
        result = -1
        for i in range(self.genSize):
            for j in range(i + 1, self.genSize):
                coinFlip = random.randint(0, 1)
                if coinFlip:
                    currGame = self.game.ConnectFour(self.AiDict[i], self.AiDict[j])
                    result = currGame.playGame()
                else:
                    currGame = self.game.ConnectFour(self.AiDict[j], self.AiDict[i])
                    result = currGame.playGame()



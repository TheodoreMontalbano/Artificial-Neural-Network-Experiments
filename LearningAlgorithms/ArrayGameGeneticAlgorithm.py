import random
from AIBuildingBlocks import NeuralNetwork, Neuron
from numpy import floor
from copy import deepcopy
from Math import MathFunctions


class GeneticAlgorithm:
    # curr Id of last AI
    Id = 0
    # List of currAI
    currAI = None
    # Tracker of the positions of All AI with certain ID
    AiDict = None
    # Size of population
    genSize = None
    # game the AIs play
    game = None
    # List of fitness of all AI
    fitnessTracker = None
    # TODO Make this kill percent??
    killPerGen = None
    # 2 value array symbolizing wanted chances of mutation arr[0]/arr[1]
    mutationChance = None
    # We show the topShowPerGenAI every Gen
    showPerGen = None
    # The current Generation
    genCount = 0

    # game: The game to train the AI on. Should be an IArrayGame. Should be the game class.
    # genSize: How many AI to have in each generation
    # bound: max amount of layers to randomly initiate initially
    # killPerGen: How many AI to kill each generation
    # mutationChance: Fraction of how likely a mutation is to occur
    def __init__(self, game, genSize=100, bound=5, killPerGen=33, mutationChance=[1, 10], showPerGen=10):
        self.mutationChance = mutationChance
        self.bound = bound
        self.game = game
        self.genSize = genSize
        self.currAI = [game.createNewAI(self.genID(), self.genBrain()) for i in range(genSize)]
        self.fitnessTracker = [0 for i in range(genSize)]
        self.AiDict = {}
        for i in range(genSize):
            self.AiDict[i + 1] = i
        if killPerGen > genSize:
            killPerGen = floor(genSize * .33)
        self.showPerGen = showPerGen

    # Generates a new Neural network game based on the AI input
    def genBrain(self):
        layerNum = random.randint(1, self.bound) + 1
        shape = []
        for i in range(layerNum):
            shape.append(random.randint(1, self.bound) + 1)
        return NeuralNetwork.NeuralNetwork([], [], self.game.getInputLayer(), shape)

    # Generates an ID for a new AI
    def genID(self):
        self.Id = self.Id + 1
        return self.Id

    # Plays out the next generation
    def nextGen(self, show=True):
        self.assessFitness()
        fitnessChart = self.killStragglersAndReproduce()
        self.genCount = self.genCount + 1
        self.outputTopPerformers(fitnessChart)
        self.resetRound()

    # Outputs the top performers of this generation based on the fitness chart
    def outputTopPerformers(self, fitnessChart):
        print("Gen: " + str(self.genCount))
        for i in range(self.showPerGen):
            print("AI: " + fitnessChart[i][0] + "Fitness Score: " + str(fitnessChart[i][1]))

    # Resets relevant variables to get program ready for next generation
    def resetRound(self):
        self.fitnessTracker = [0 for i in range(self.genSize)]

    # plays out the next n gens
    def nextNGens(self, n, show=False):
        for i in range(n - 1):
            self.nextGen(show)
        self.nextGen()

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

    # Kills off those with low fitness and has the remainder reproduce to bring pop to genSize
    def killStragglersAndReproduce(self):
        temp = [0, 0]
        # ID, fitness
        fitnessRatings = [deepcopy(temp) for i in range(self.genSize)]
        for i in range(self.genSize):
            fitnessRatings[i][0] = self.currAI[i].Name
            fitnessRatings[i][1] = self.fitnessTracker[i]
        fitnessRatings.sort(key=lambda x: x[1])
        save = -1
        remainingToKill = self.killPerGen
        # 0 index is ones who were selected, 1 index is ones who were not
        killSelection = MathFunctions.selectLeftSkewRandomly(self.killPerGen, self.genSize)
        reproduceSelection = [[]]
        for i in range(len(killSelection[0])):
            reproduceSelection = MathFunctions.selectRightSkewRandomly(2, self.genSize - self.killPerGen)
            self.replaceAI(killSelection[0][i], self.reproduce(reproduceSelection[0][0], reproduceSelection[0][1]))
        return fitnessRatings

    # takes in two indexes and returns a new NN with values based on the two NN corresponding to those
    # index's reproduction
    def reproduce(self, indexOne, indexTwo):
        nNOne = None
        nNTwo = None
        coinFlip = -1
        # assume nNTwo has more layers
        if len(self.currAI[indexOne].brain.layers) > len(self.currAI[indexTwo].brain.layers):
            nNTwo = self.currAI[indexOne].brain
            nNOne = self.currAI[indexTwo].brain
        else:
            nNOne = self.currAI[indexOne].brain
            nNTwo = self.currAI[indexTwo].brain
        shapeOne = nNOne.shape
        shapeTwo = nNTwo.shape
        newShape = GeneticAlgorithm.createNewShape(shapeOne, shapeTwo)
        childBrain = GeneticAlgorithm.distributeGenes(nNOne, nNTwo, newShape)
        self.possibleAddMutation(childBrain)
        return self.game.createNewAI(self.genID(), childBrain)

    # Possibly add a random mutation to the child
    def possibleAddMutation(self, childBrain):
        mutationType = -1
        if not random.randint(1, self.mutationChance[1]) > self.mutationChance[0]:
            mutationType = random.randint(1, 100)
            if mutationType <= 50:
                randnum = 0
                sum = 0
                index = 0
                for i in childBrain.shape:
                    sum = sum + i
                randnum = random.randint(1, sum)
                sum = 0
                for i in range(len(childBrain.shape)):
                    sum = sum + childBrain.shape[i]
                    if sum > randnum:
                        sum = sum - childBrain.shape[i]
                        index = i
                        break
                neuron = childBrain.layers[index].nodes[randnum - sum]
                neuron.edgeWeights[random.randint(1, childBrain.shape[index - 1])] = random.random()
            elif mutationType <= 90:
                index = random.randint(2, len(childBrain.shape) - 1)
                randNum = random.randint(0, childBrain.shape[index] - 1)
                if mutationType <= 70 and self.shape[index] > 1:
                    # Delete a neuron
                    # Setting up Layer that comes after this one to have the correct edgeweights
                    childBrain.addLayerAtIndex(index + 1, None, self.shape[index] - 1)
                    childBrain.layers.pop(index + 1)
                    # maintain shape
                    childBrain.shape[index] = childBrain.shape[index] - 1
                    # Remove node
                    childBrain.layers[index].nodes.pop(randNum)
                else:
                    # Add a neuron
                    # Setting up Layer that comes after this one to have the correct edgeweights
                    childBrain.addLayerAtIndex(index + 1, None, self.shape[index] + 1)
                    childBrain.layers.pop(index + 1)
                    # maintain shape
                    childBrain.shape[index] = childBrain.shape[index] - 1
                    # Add node
                    childBrain.layers[index].nodes.insert(randNum, Neuron.Neuron(None, childBrain.shape[index - 1]))
            else:
                randNum = random.randint(2, len(childBrain.shape) - 1)
                if mutationType <= 95 or len(childBrain.shape) < 3:
                    # Add a layer randomly
                    childBrain.addLayerAtIndex(randNum, None, random.randint(1, self.bound))
                else:
                    # Remove a layer randomly
                    childBrain.layers.pop(randNum)
                    childBrain.addLayerAtIndex(randNum, None, childBrain.shape[randNum - 1])
                    childBrain.layers.pop(randNum)

    # Distributes genes from parents to child
    # Assumes parentTwo has more Layers than parentOne
    @staticmethod
    def distributeGenes(parentOne, parentTwo, childShape):
        childNN = NeuralNetwork.NeuralNetwork(childShape)
        secParHelpIndex = 0
        for i in range(1, len(childShape)):
            childNN.addLayer(None, childShape[i])
            if len(parentOne.shape) <= i:
                while not parentTwo.shape[secParHelpIndex] == childShape[i]:
                    secParHelpIndex = secParHelpIndex + 1
                for j in range(childShape[i]):
                    for k in range(min(parentTwo.shape[secParHelpIndex - 1], childShape[i - 1])):
                        childNN.layers[i][j].edgeWeights[k] = parentTwo.childNN.layers[i][j].edgeWeights[k]
            else:
                for j in range(childShape[i]):
                    for k in range(childShape[i - 1]):
                        # Cases where only one has a possible gene
                        if k > parentOne[i - 1]:
                            childNN.layers[i][j].edgeWeights[k] = parentTwo.childNN.layers[i][j].edgeWeights[k]
                        elif k > parentTwo[i - 1]:
                            childNN.layers[i][j].edgeWeights[k] = parentOne.childNN.layers[i][j].edgeWeights[k]
                        # Cases where both have possible genes
                        elif random.randint(0, 1):
                            childNN.layers[i][j].edgeWeights[k] = parentOne.childNN.layers[i][j].edgeWeights[k]
                        else:
                            childNN.layers[i][j].edgeWeights[k] = parentTwo.childNN.layers[i][j].edgeWeights[k]
        return childNN

    # Creates a new shape from the two input shapes the new shape will not have an output layer
    # Assumes shapeOne is shorter than shapeTwo
    @staticmethod
    def createNewShape(shapeOne, shapeTwo):
        newShape = []
        for i in range(len(shapeOne) - 1):
            coinFlip = random.randint(0, 1)
            if coinFlip:
                newShape.append(shapeOne[i])
            else:
                newShape.append(shapeTwo[i])
        for i in range(len(shapeTwo) - len(shapeOne)):
            coinFlip = random.randint(0, 1)
            if coinFlip:
                newShape.append(shapeTwo[i + len(shapeOne) - 1])
        return newShape

    # replaces an AI in the population (kills it) with a new AI (births it)
    def replaceAI(self, index, replaceWith):
        self.currAI[index] = replaceWith
        self.AiDict[replaceWith.Name] = index

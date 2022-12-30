import random
from AIBuildingBlocks import NeuralNetwork, Neuron
from numpy import floor
from copy import deepcopy
from Math import MathFunctions
from GameInterfaces import IArrayGame, INNPlayer


class GeneticAlgorithm:
    # curr Id of last AI
    _id = 0
    # List of currAI
    _currAI = None
    # Tracker of the positions of All AI with certain ID
    _aiDict = None
    # Size of population
    _genSize = None
    # game the AIs play
    _game = None
    # List of fitness of all AI
    _fitnessTracker = None
    # TODO Make this kill percent??
    _killPerGen = None
    # 2 value array symbolizing wanted chances of mutation arr[0]/arr[1]
    _mutationChance = None
    # We show the topShowPerGenAI every Gen
    _showPerGen = None
    # The current Generation
    _genCount = 0
    # The neural network game player class
    _nNPlayer = None

    # nNPlayer: The type of NN player to train should be an INNPlayer
    # genSize: How many AI to have in each generation
    # bound: max amount of layers to randomly initiate initially
    # killPerGen: How many AI to kill each generation
    # mutationChance: Fraction of how likely a mutation is to occur
    def __init__(self, nNPlayer, genSize=100, bound=5, killPerGen=33, mutationChance=[1, 10], showPerGen=10):
        self._mutationChance = mutationChance
        self._bound = int(bound)
        self._nNPlayer = INNPlayer.INNPlayer(nNPlayer)
        self._game = IArrayGame.IArrayGame(nNPlayer)
        self._genSize = int(genSize)
        self._currAI = [self._nNPlayer(self.genID(), self.genBrain()) for i in range(genSize)]
        self._fitnessTracker = [0 for i in range(genSize)]
        self._aiDict = {}
        for i in range(genSize):
            self._aiDict[i + 1] = i
        if killPerGen > genSize:
            killPerGen = floor(genSize * .33)
        self._killPerGen = killPerGen
        self._showPerGen = showPerGen

    # Generates a new Neural network game based on the AI input
    def genBrain(self):
        layerNum = random.randint(1, self._bound) + 1
        shape = []
        for i in range(layerNum):
            shape.append(random.randint(1, self._bound) + 1)
        return NeuralNetwork.NeuralNetwork(shape)

    # Generates an ID for a new AI
    def genID(self):
        self._id = self._id + 1
        return self._id

    # Plays out the next generation
    def nextGen(self, show=True):
        self.assessFitness()
        fitnessChart = self.killStragglersAndReproduce()
        self._genCount = self._genCount + 1
        if show:
            self.outputTopPerformers(fitnessChart)
        self.resetRound()

    # Outputs the top performers of this generation based on the fitness chart
    def outputTopPerformers(self, fitnessChart):
        print("Gen: " + str(self._genCount))
        for i in range(self._showPerGen):
            print("AI: " + fitnessChart[i][0] + "Fitness Score: " + str(fitnessChart[i][1]))

    # Resets relevant variables to get program ready for next generation
    def resetRound(self):
        self._fitnessTracker = [0 for i in range(self._genSize)]

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
        for i in range(self._genSize):
            for j in range(i + 1, self._genSize):
                coinFlip = random.randint(0, 1)
                if coinFlip:
                    playerOne = self._currAI[i]
                    playerTwo = self._currAI[j]
                else:
                    playerOne = self._currAI[j]
                    playerTwo = self._currAI[i]
                currGame = self._game(playerOne, playerTwo)
                result = currGame.playGame()
                if result == -1:
                    # Tie game
                    self._fitnessTracker[self._aiDict[playerOne.Name]] = \
                        self._fitnessTracker[self._aiDict[playerOne.Name]] + 1 / 2
                    self._fitnessTracker[self._aiDict[playerTwo.Name]] = \
                        self._fitnessTracker[self._aiDict[playerTwo.Name]] + 1 / 2
                elif result == 0:
                    # Player one wins
                    self._fitnessTracker[self._aiDict[playerOne.Name]] = \
                        self._fitnessTracker[self._aiDict[playerOne.Name]] + 1
                    self._fitnessTracker[self._aiDict[playerTwo.Name]] = \
                        self._fitnessTracker[self._aiDict[playerTwo.Name]] - 1
                else:
                    # Player two wins
                    self._fitnessTracker[self._aiDict[playerOne.Name]] = \
                        self._fitnessTracker[self._aiDict[playerOne.Name]] - 1
                    self._fitnessTracker[self._aiDict[playerTwo.Name]] = \
                        self._fitnessTracker[self._aiDict[playerTwo.Name]] + 1

    # Kills off those with low fitness and has the remainder reproduce to bring pop to genSize
    def killStragglersAndReproduce(self):
        temp = [0, 0]
        # ID, fitness
        fitnessRatings = [deepcopy(temp) for i in range(self._genSize)]
        for i in range(self._genSize):
            fitnessRatings[i][0] = self._currAI[i].getName()
            fitnessRatings[i][1] = self._fitnessTracker[i]
        fitnessRatings.sort(key=lambda x: x[1])
        # 0 index is ones who were selected, 1 index is ones who were not

        #TODO don't reproduce with AI that were killed
        killSelection = MathFunctions.selectLeftSkewRandomly(self._killPerGen, self._genSize)
        reproduceSelection = [[]]
        for i in range(len(killSelection[0])):
            reproduceSelection = MathFunctions.selectRightSkewRandomly(2, self._genSize - self._killPerGen)
            self.replaceAI(killSelection[0][i], self.reproduce(reproduceSelection[0][0], reproduceSelection[0][1]))
        return fitnessRatings

    # takes in two indexes and returns a new NN with values based on the two NN corresponding to those
    # index's reproduction
    def reproduce(self, indexOne, indexTwo):
        nNOne = None
        nNTwo = None
        coinFlip = -1
        # assume nNTwo has more layers
        if self._currAI[indexOne].getSize() > self._currAI[indexTwo].getSize():
            nNTwo = self._currAI[indexOne]
            nNOne = self._currAI[indexTwo]
        else:
            nNOne = self._currAI[indexOne]
            nNTwo = self._currAI[indexTwo]
        newShape = GeneticAlgorithm.createNewShape(nNOne, nNTwo)
        childNN = self.distributeGenes(nNOne, nNTwo, newShape)
        self.possibleAddMutation(childNN)
        return childNN

    # Possibly add a random mutation to the child
    def possibleAddMutation(self, childNN):
        mutationType = -1
        if not random.randint(1, self._mutationChance[1]) > self._mutationChance[0]:
            mutationType = random.randint(1, 100)
            # Change a random edgeweight
            if mutationType <= 50:
                sums = 0
                index = 0
                for i in range(childNN.getSize()):
                    sums = sums + childNN.getShapeAtIndex(i)
                randNum = random.randint(1, sums)
                sums = 0
                for i in range(childNN.getSize()):
                    sums = sums + childNN.getShapeAtIndex(i)
                    if sums > randNum:
                        sums = sums - childNN.getShapeAtIndex(i)
                        index = i
                        break
                childNN.setEdgeWeight(index, randNum - sums
                                      , random.randint(0, childNN.getShapeAtIndex(index - 1)), random.random())
            # Add or delete a random neuron (not input or output layer)
            elif mutationType <= 90:
                index = random.randint(1, len(childNN.getSize()) - 2)
                if mutationType <= 70 and childNN.getShapeAtIndex(index) > 1:
                    randNum = random.randint(0, childNN.getShapeAtIndex() - 1)
                    # Delete a neuron
                    childNN.deleteNeuron(index, randNum)
                else:
                    # Add a neuron
                    childNN.addNeuron(index)
            # Add or delete a random layer (not input or output)
            else:
                randNum = random.randint(1, len(childNN.shape) - 2)
                if mutationType <= 95 or len(childNN.shape) < 3:
                    # Add a layer randomly
                    childNN.addLayerAtIndex(randNum, None, random.randint(1, self._bound))
                else:
                    # Remove a layer randomly
                    childNN.deleteLayerAtIndex(randNum)

    # Distributes genes from parents to child
    # Assumes parentTwo has more Layers than parentOne
    def distributeGenes(self, parentOne, parentTwo, childShape):
        childNN = self._nNPlayer(self.genID(), childShape)
        secParHelpIndex = 0
        # For each layer of childShape
        for i in range(1, len(childShape)):
            # If Parent one is smaller than the current spot of child we are iterating over
            if parentOne.getSize <= i:
                while not parentTwo.getShapeAtIndex(secParHelpIndex) == childShape[i]:
                    secParHelpIndex = secParHelpIndex + 1
                # For each node in that layer
                for j in range(childShape[i]):
                    # For each edgeweight in that node
                    for k in range(childShape[i - 1]):
                        childNN.setEdgeWeight(i, j, k, parentTwo.getEdgeWeight(i, j, k))
            else:
                # For each node in that layer
                for j in range(childShape[i]):
                    # For each edgeweight in that node
                    for k in range(childShape[i - 1]):
                        # Cases where only one has a possible gene
                        if k >= parentOne.getShapeAtIndex(i - 1):
                            childNN.setEdgeWeight(i, j, k, parentTwo.getEdgeWeight(i, j, k))
                        elif k >= parentTwo.getShapeAtIndex(i - 1):
                            childNN.setEdgeWeight(i, j, k, parentOne.getEdgeWeight(i, j, k))
                        # Cases where both have possible genes
                        elif random.randint(0, 1):
                            childNN.setEdgeWeight(i, j, k, parentOne.getEdgeWeight(i, j, k))
                        else:
                            childNN.setEdgeWeight(i, j, k, parentTwo.getEdgeWeight(i, j, k))
        return childNN

    # Creates a new shape from the two input shapes the new shape will not have an output layer
    # Assumes shapeOne is shorter than shapeTwo
    @staticmethod
    def createNewShape(nNOne, nNTwo):
        newShape = []
        for i in range(nNOne.getSize()):
            coinFlip = random.randint(0, 1)
            if coinFlip:
                newShape.append(nNOne.getShapeAtIndex(i))
            else:
                newShape.append(nNTwo.getShapeAtIndex(i))
        for i in range(nNTwo.getSize() - nNOne.getSize()):
            coinFlip = random.randint(0, 1)
            if coinFlip:
                newShape.append(nNTwo.getShapeAtIndex(i + len(nNOne) - 1))
        return newShape

    # replaces an AI in the population (kills it) with a new AI (births it)
    def replaceAI(self, index, replaceWith):
        self._currAI[index] = replaceWith
        self._aiDict[replaceWith.getName()] = index

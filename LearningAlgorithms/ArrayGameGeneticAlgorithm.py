import random
from AIBuildingBlocks import NeuralNetwork, Neuron
from GameInterfaces import IArrayGame, INNPlayer
from numpy import floor
from copy import deepcopy
from Math import MathFunctions
from math import sqrt


class GeneticAlgorithm:
    # curr Id of last AI
    _id: int = 0
    # List of currAI
    _currAI = None
    # Tracker of the positions of All AI with certain ID
    _aiDict = None
    # Size of population
    _genSize: int = None
    # game the AIs play
    _game: IArrayGame = None
    # List of fitness of all AI
    _fitnessTracker = None
    # TODO Make this kill percent??
    _killPerGen: int = None
    # 2 value array symbolizing wanted chances of mutation arr[0]/arr[1]
    _mutationChance = None
    # We show the topShowPerGenAI every Gen
    _showPerGen = None
    # The current Generation
    _genCount: int = 0
    # The neural network game player class
    _nNPlayer: INNPlayer = None
    # The max amount of layers to generate
    _layerBound: int = None
    # The algorithm to perform crossover with
    _crossoverAlgorithm: str = None
    # Whether to use the Launchpad players or not
    _useLaunchPad: bool = False
    # End evolution
    _endEvolution: bool = False
    # Stagnated evolution Count
    _stagnateEvolCount: int = 0

    # nNPlayer: The type of NN player to train should be an INNPlayer
    # genSize: How many AI to have in each generation
    # layerBound: max amount of layers to randomly initiate initially
    # nodeBound: max amount of nodes to randomly generate
    # killPerGen: How many AI to kill each generation
    # mutationChance: Fraction of how likely a mutation is to occur
    # showPerGen: amount of top AI to show per gen
    # crossoverAlgorithm: the algorithm used to apply crossover
    def __init__(self, nNPlayer: INNPlayer, genSize: int = 100, layerBound=3, nodeBound=50, killPerGen=33,
                 mutationChance=[1, 10]
                 , showPerGen=10, crossoverAlgorithm="Uniform", useLaunchPad=False):
        self._useLaunchPad = useLaunchPad
        self._mutationChance = mutationChance
        self._crossoverAlgorithm = crossoverAlgorithm
        self._nodeBound = nodeBound
        self._layerBound = layerBound
        self._nNPlayer = nNPlayer
        self._game = self._nNPlayer.getGame()
        self._genSize = int(genSize)
        self._currAI = [self._nNPlayer(self.genID(), self.genBrain()) for i in range(genSize)]
        self._fitnessTracker = [0 for i in range(genSize)]
        self._aiDict = {}
        for i in range(genSize):
            self._aiDict[i + 1] = i
        if killPerGen > genSize:
            killPerGen = int(floor(genSize * .33))
            print("ERROR: Kill per gen is greater than Gen size. Making Kill per gen 1/3 Gen Size")
        self._killPerGen = killPerGen
        self._showPerGen = showPerGen

    # Generates a new Neural network game based on the AI input
    def genBrain(self):
        layerNum = random.randint(2, self._layerBound)
        shape = []
        for i in range(layerNum):
            shape.append(random.randint(1, self._nodeBound) + 1)
        return shape

    # Generates an ID for a new AI
    def genID(self):
        self._id = self._id + 1
        return self._id

    # Plays out the next generation
    def nextGen(self, show=True):
        if self._useLaunchPad:
            self.assessFitnessWithLaunchPad()
        else:
            self.assessFitness()
        fitnessChart = self.killStragglersAndReproduce()
        self._genCount = self._genCount + 1
        if show:
            self.outputTopPerformers(fitnessChart)
        self.resetRound()

    # Outputs the top performers of this generation based on the fitness chart
    def outputTopPerformers(self, fitnessChart):
        stdDev = 0
        average = 0
        fitnessChart.sort(key=lambda x: x[1], reverse=True)
        for i in range(len(fitnessChart) - self._killPerGen):
            average = average + fitnessChart[i][1]
        average = average / (self._genSize - self._killPerGen)
        for i in range(len(fitnessChart) - self._killPerGen):
            stdDev = stdDev + abs(fitnessChart[i][1] - average)
        stdDev = stdDev / (self._genSize - self._killPerGen)
        print("Gen: " + str(self._genCount))
        print("Mean Score of top players: " + str(average))
        print("Average deviation of top players : " + str(stdDev))
        for i in range(self._showPerGen):
            print("AI: " + str(fitnessChart[i][0]) + " Fitness Score: " + str(fitnessChart[i][1])
                  + " Shape: " + str(fitnessChart[i][2]))
        if not self._useLaunchPad and int(stdDev) == 0:
            self._stagnateEvolCount = self._stagnateEvolCount + 1
            if self._stagnateEvolCount >= max(floor(self._genCount * .1), 3):
                self._endEvolution = True
        elif not self._useLaunchPad:
            self._stagnateEvolCount = 0
        elif int(average) + 1 == 2 * len(self._nNPlayer.getLaunchPadPlayers()) and int(stdDev) == 0:
            print()
            print("LaunchPad disabled")
            self._useLaunchPad = False
        print()

    # Resets relevant variables to get program ready for next generation
    def resetRound(self):
        self._fitnessTracker = [0 for i in range(self._genSize)]

    # plays out the next n gens
    def nextNGens(self, n, show=False):
        for i in range(n - 1):
            self.nextGen()
        self.nextGen()

    def repeatUntilEndCondition(self):
        self._endEvolution = False
        while not self._endEvolution:
            self.nextGen()

    # Assesses the fitness of all AI
    def assessFitness(self):
        gameOne = None
        removedList = []
        coinFlip = 0
        result = -1
        playerOne = None
        playerTwo = None
        for i in range(self._genSize):
            # TODO if this works calculate sample size based on pop
            sample = random.sample([k for k in range(self._genSize)], 100)
            # TODO test for j in range(i + 1, self._genSize):
            for j in sample:
                playerOne = self._currAI[i]
                playerTwo = self._currAI[j]
                gameOne = self._game(playerOne, playerTwo)
                gameTwo = self._game(playerTwo, playerOne)
                # TODO testing random selection of AI
                self.scoreAI(None, playerOne, gameTwo.playGame())
                self.scoreAI(playerOne, None, gameOne.playGame())
                # self.scoreAI(playerTwo, playerOne, gameTwo.playGame())
                # self.scoreAI(playerOne, playerTwo, gameOne.playGame())

    # Assesses Fitness Of AI using the launch pad players
    def assessFitnessWithLaunchPad(self):
        players = self._nNPlayer.getLaunchPadPlayers()
        for i in range(self._genSize):
            for j in players:
                gameOne = self._game(self._currAI[i], j)
                gameTwo = self._game(j, self._currAI[i])
                self.scoreAI(None, self._currAI[i], gameTwo.playGame())
                self.scoreAI(self._currAI[i], None, gameOne.playGame())

    # We assume p1 goes first
    def scoreAI(self, playerOne, playerTwo, result):
        if result == 0:
            # Player one wins
            if playerOne:
                self._fitnessTracker[self._aiDict[playerOne.getName()]] = \
                    self._fitnessTracker[self._aiDict[playerOne.getName()]] + 1
            if playerTwo:
                self._fitnessTracker[self._aiDict[playerTwo.getName()]] = \
                    self._fitnessTracker[self._aiDict[playerTwo.getName()]] - 1
        else:
            # Player two wins
            if playerTwo:
                self._fitnessTracker[self._aiDict[playerTwo.getName()]] = \
                    self._fitnessTracker[self._aiDict[playerTwo.getName()]] + 1
            if playerOne:
                self._fitnessTracker[self._aiDict[playerOne.getName()]] = \
                    self._fitnessTracker[self._aiDict[playerOne.getName()]] - 1

    # Kills off those with low fitness and has the remainder reproduce to bring pop to genSize
    def killStragglersAndReproduce(self):
        temp = [0, 0, 0]
        # ID, fitness
        fitnessRatings = [deepcopy(temp) for i in range(self._genSize)]
        for i in range(self._genSize):
            fitnessRatings[i][0] = self._currAI[i].getName()
            fitnessRatings[i][1] = self._fitnessTracker[i]
            fitnessRatings[i][2] = self._currAI[i].getShape()
        fitnessRatings.sort(key=lambda x: x[1])
        # 0 index is ones who were selected, 1 index is ones who were not
        killSelection = MathFunctions.selectLeftSkewRandomly(self._killPerGen, self._genSize)
        reproduceSelection = [[]]
        toReplace = -1
        parentOne = -1
        parentTwo = -1
        for i in range(len(killSelection[0])):
            reproduceSelection = MathFunctions.selectRightSkewRandomly(2, self._genSize - self._killPerGen)
            toReplace = self._aiDict[fitnessRatings[killSelection[0][i]][0]]
            parentOne = self._aiDict[fitnessRatings[killSelection[1][reproduceSelection[0][0]]][0]]
            parentTwo = self._aiDict[fitnessRatings[killSelection[1][reproduceSelection[0][1]]][0]]
            self.replaceAI(toReplace, self.reproduce(parentOne, parentTwo))
        return fitnessRatings

    # takes in two indexes and returns a new NN with values based on the two NN corresponding to those
    # index's reproduction
    def reproduce(self, indexOne, indexTwo):
        if self._crossoverAlgorithm == "Uniform":
            # assume nNTwo has more layers
            if self._currAI[indexOne].getSize() > self._currAI[indexTwo].getSize():
                nNTwo = self._currAI[indexOne]
                nNOne = self._currAI[indexTwo]
            else:
                nNOne = self._currAI[indexOne]
                nNTwo = self._currAI[indexTwo]
            newShape = GeneticAlgorithm.createNewShape(nNOne, nNTwo)
            childNN = self.uniformCrossover(nNOne, nNTwo, newShape)
            self.possibleAddMutation(childNN)
            return childNN
        elif self._crossoverAlgorithm == "OnePointCrossover":
            nNOne = None
            nNTwo = None
            coinFlip = random.randint(0, 1)
            if coinFlip:
                nNTwo = self._currAI[indexOne]
                nNOne = self._currAI[indexTwo]
            else:
                nNOne = self._currAI[indexOne]
                nNTwo = self._currAI[indexTwo]
            childNN = self.onePointCrossover(nNOne, nNTwo)
            self.possibleAddMutation(childNN)
            return childNN
        else:
            return -1

    # Does one point crossover to create a childNN from the two input neural networks
    def onePointCrossover(self, nNOne, nNTwo):
        cutOne = random.randint(1, nNOne.getSize() - 1)
        cutTwo = random.randint(1, nNTwo.getSize() - 2)
        newShape = nNOne.getShape()[:cutOne] + nNTwo.getShape()[cutTwo:]
        childNN = self._nNPlayer(self.genID(), newShape)
        for i in range(1, cutOne):
            for j in range(newShape[i]):
                for k in range(newShape[i - 1]):
                    childNN.setEdgeWeight(i, j, k, nNOne.getEdgeWeight(i, j, k))
        for i in range(cutOne, len(newShape)):
            for j in range(newShape[i]):
                for k in range(min(newShape[i - 1], nNTwo.getShapeAtIndex(i - cutOne + cutTwo - 1))):
                    childNN.setEdgeWeight(i, j, k, nNTwo.getEdgeWeight(i - cutOne + cutTwo, j, k))
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
                for i in range(1, childNN.getSize()):
                    sums = sums + childNN.getShapeAtIndex(i)
                randNum = random.randint(1, sums)
                secSum = 0
                for i in range(1, childNN.getSize()):
                    secSum = secSum + childNN.getShapeAtIndex(i)
                    if secSum > randNum:
                        secSum = secSum - childNN.getShapeAtIndex(i)
                        index = i
                        break
                childNN.setEdgeWeight(index, randNum - secSum
                                      , random.randint(0, childNN.getShapeAtIndex(index - 1) - 1)
                                      , random.random() * 2 - 1)
            # Add or delete a random neuron (not input or output layer)
            elif mutationType <= 90:
                index = random.randint(1, childNN.getSize() - 2)
                if mutationType <= 70 and childNN.getShapeAtIndex(index) > 1:
                    randNum = random.randint(0, childNN.getShapeAtIndex(index) - 1)
                    # Delete a neuron
                    # TODO this is bugged
                    # childNN.deleteNeuron(index, randNum)
                else:
                    # Add a neuron
                    childNN.addNeuron(index)
            # Add or delete a random layer (not input or output)
            else:
                randNum = random.randint(1, childNN.getSize() - 2)
                if mutationType <= 95 or childNN.getSize() < 3:
                    # Add a layer randomly
                    childNN.addLayerAtIndex(randNum, None, random.randint(1, self._nodeBound))
                # TODO this is bugged
            #    else:
            #        # Remove a layer randomly
            #        childNN.deleteLayerAtIndex(randNum)

    # Distributes genes from parents to child
    # Assumes parentTwo has more Layers than parentOne
    def uniformCrossover(self, parentOne, parentTwo, childShape):
        childNN = self._nNPlayer(self.genID(), childShape)
        secParHelpIndex = 0
        # For each layer of childShape
        for i in range(1, len(childShape)):
            # If Parent one is smaller than the current spot of child we are iterating over
            if parentOne.getSize() <= i:
                while not parentTwo.getShapeAtIndex(secParHelpIndex) == childShape[i]:
                    secParHelpIndex = secParHelpIndex + 1
                # For each node in that layer
                for j in range(childShape[i]):
                    # For each edgeweight in that node
                    # TODO check this logic do we need the first if?
                    if i < parentTwo.getSize() and j < parentTwo.getShapeAtIndex(i):
                        for k in range(childShape[i - 1]):
                            # TODO Check if we need the if else
                            if k < parentTwo.getShapeAtIndex(i - 1):
                                childNN.setEdgeWeight(i, j, k, parentTwo.getEdgeWeight(i, j, k))
                            else:
                                childNN.setEdgeWeight(i, j, k, random.random())
                    else:
                        for k in range(childShape[i - 1]):
                            childNN.setEdgeWeight(i, j, k, random.random())
            else:
                # For each node in that layer
                # TODO add handling for different node counts
                for j in range(childShape[i]):
                    # If only parent one has enough nodes to account
                    if j >= parentOne.getShapeAtIndex(i):
                        for k in range(childShape[i - 1]):
                            if k < parentTwo.getShapeAtIndex(i - 1):
                                childNN.setEdgeWeight(i, j, k, parentTwo.getEdgeWeight(i, j, k))
                            # Child has too many nodes for parent one
                            else:
                                childNN.setEdgeWeight(i, j, k, random.random())
                    # If only parent two has enough nodes to contribute genes
                    elif j >= parentTwo.getShapeAtIndex(i):
                        for k in range(childShape[i - 1]):
                            if k < parentOne.getShapeAtIndex(i - 1):
                                childNN.setEdgeWeight(i, j, k, parentOne.getEdgeWeight(i, j, k))
                            # Child has too many nodes for parent one
                            else:
                                childNN.setEdgeWeight(i, j, k, random.random())
                    else:
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
                newShape.append(nNTwo.getShapeAtIndex(i + nNOne.getSize() - 1))
        return newShape

    # replaces an AI in the population (kills it) with a new AI (births it)
    def replaceAI(self, index, replaceWith):
        self._currAI[index] = replaceWith
        self._aiDict[replaceWith.getName()] = index

    # Opens a file to save the AI in
    def saveAI(self, name):
        indent = "    "
        toSave = self._currAI[self._aiDict[int(name)]]
        fileName = "Gen-" + self._nNPlayer.getGameName() + "-" + str(toSave.getName()) + ".txt"
        file = open("SavedNeuralNetworks/" + fileName, "w")
        file.write("Shape: " + str(toSave.getShape()) + '\n')
        for i in range(1, toSave.getSize()):
            file.write("Layer " + str(i) + ":\n")
            file.write(indent + "ActivationFunction: "
                       + "MathFunctions." + toSave.getActivationFunction(i).__name__ + "\n")
            for k in range(toSave.getShapeAtIndex(i)):
                file.write(indent + "Node " + str(k) + ":\n")
                for j in range(toSave.getShapeAtIndex(i - 1)):
                    file.write(indent + indent + "EdgeWeight " + str(j) + ": " + str(toSave.getEdgeWeight(i, k, j))
                               + "\n")
        file.close()

    # Has two AI play a match that can be viewed
    def showMatch(self, playerOneId, playerTwoId):
        nNOne = self._currAI[self._aiDict[int(playerOneId)]]
        nNTwo = self._currAI[self._aiDict[int(playerTwoId)]]
        toShow = self._game(nNOne, nNTwo, True)
        result = toShow.playGame()
        if result == 0:
            print("Player One Wins")
        elif result == 1:
            print("Player Two Wins")
        else:
            print("No Contest")

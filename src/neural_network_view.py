"""
View used to render our neural network
"""

from itertools import izip
from numpy.random import randint
from basicNN import NeuralNetwork
import sys
from graph_tool.all import *

# create our Neural Network Model
dataFile = sys.argv[1]
neural_net = NeuralNetwork(dataFile) 
neural_net.initialMatrixWeights()
neural_net.trainAndTestNetwork()
nbrOfInputs = len(neural_net.targetValues[0])
nbrOfOutputs = len(neural_net.inputValues[0])
nbrOfHiddenUnits = neural_net.numHiddenUnits
inputsToHiddenWeights = n.getInputHiddenWeightsAsList()
hiddenToOutputsWeights = n.getHiddenTargetWeightsAsList()

# create our graph
g = Graph()

# create the input nodes of our graph
inputs = []
for i in range(nbrOfInputs):
    inputs.append(g.add_vertex())

# create the hidden nodes of our graph
hidden_units = []
for i in range(neural_net.numHiddenUnits):
    hidden_units.append(g.add_vertex())

# create the output nodes of our graph
outputs = []
for i in range(nbrOfOutputs):
    outputs.append(g.add_vertex())

# properly assign the edge weights
# first assign edges from the inputs to hidden layer
for row in range(nbrOfInputs):
	for col in range(nbrOfHiddenUnits):
		e = g.add_edge(inputs[row], hidden_units[col])
		e.color = [inputsToHiddenWeights[row][col], .203, .210, .8]

# second assign edges from the hidden layer to the outputs


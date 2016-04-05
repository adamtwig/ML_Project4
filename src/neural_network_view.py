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
nbrOfInputs = int(len(neural_net.targetValues[0]))
nbrOfOutputs = int(len(neural_net.inputValues[0]))
nbrOfHiddenUnits = int(neural_net.numHiddenUnits)
inputsToHiddenWeights = neural_net.getInputHiddenWeightsAsList()
hiddenToOutputsWeights = neural_net.getHiddenTargetWeightsAsList()

# create our graph
g = Graph()

# create the input nodes of our graph
inputs = []
for i in range(nbrOfInputs):
    inputs.append(g.add_vertex())

# create the hidden nodes of our graph
hidden_units = []
for i in range(nbrOfHiddenUnits):
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

graph_tool.draw(g, vertex_text = g.vertex_index, vertex_font_size=18, output_size=(200, 200), output="test.png")

# second assign edges from the hidden layer to the outputs


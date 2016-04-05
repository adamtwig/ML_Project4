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

# create our graph
g = Graph()

# create the input nodes of our graph
inputs = []
nbrOfInputs = len(neural_net.targetValues[0])
for i in range(nbrOfInputs):
    inputs.append(g.add_vertex())

# create the hidden nodes of our graph
hidden_units = []
for i in range(neural_net.numHiddenUnits):
    hidden_units.append(g.add_vertex())

# create the output nodes of our graph
outputs = []
nbrOfOutputs = len(neural_net.inputValues[0])
for i in range(nbrOfOutputs):
    outputs.append(g.add_vertex())

# properly assign the edge weights


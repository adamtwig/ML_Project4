"""
View used to render our neural network
"""

from itertools import izip
from numpy.random import randint
from basicNN import NeuralNetwork
import sys

# create our Neural Network Model
dataFile = sys.argv[1]
neural_net = NeuralNetwork(dataFile)
neural_net.initialMatrixWeights()
neural_net.train()

# create our graph
g = Graph()
g.add_vertex(100)
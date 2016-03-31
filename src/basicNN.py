"""
@authors: Michael Baldwin, Josh Engelsma, Adam Terwilliger
@date: March 30, 2016
@version: 1.0
This program builds a neural network.
"""

import sys
import math
import numpy as np

class NeuralNetwork(object):
    def __init__(self, dataFile):
        self.dataFile = dataFile
        self.targetValues = []
        self.inputValues = []
        self.readFile()
        self.numHiddenUnits = self.calcNumHiddenUnits()
        self.inputHiddenWeights = np.zeros([])
        self.hiddenTargetWeights = np.zeros([])

    def readFile(self):
        with open(self.dataFile) as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                lineParts = line.split(',')
                target = [lineParts[-1]]
                if i == 0:
                    self.inputNames = lineParts[:-1]
                else:
                    self.targetValues.append(target)
                    self.inputValues.append(lineParts[:-1])

    def calcNumHiddenUnits(self):
        return math.ceil((2.0/3.0)*(len(self.targetValues[0])+len(self.inputValues[0])))


    def initialMatrixWeights(self):
        lowInitial = -1.0 * 10.0**-4
        highInitial = 1.0 * 10.0**-4
        np.random.seed(42)
        self.inputHiddenWeights = np.random.uniform(low=lowInitial, high=highInitial, 
                                    size=(len(self.inputValues[0]),self.numHiddenUnits))
        self.hiddenTargetWeights = np.random.uniform(low=lowInitial, high=highInitial, 
                                    size=(len(self.targetValues[0]),self.numHiddenUnits))

    def feedForward(self):

        currentInput = np.array(self.inputValues[0]).astype(np.float)

        sumOfProductsIH = np.dot(self.inputHiddenWeights,currentInput)

        activationIH = self.sigmoid(sumOfProductsIH)

        print self.hiddenTargetWeights

        sumOfProductsHT = np.dot(self.hiddenTargetWeights,activationIH)

        activationHT =  self.sigmoid(sumOfProductsHT)

        print activationHT

        # one value! :)
        # add biases


    def sigmoid(self, sumOfProducts):
        #return np.exp(sumOfProducts)
        return 1.0 / (1.0 + np.exp(-1.0 * sumOfProducts))


def main(argv):
    if len(argv) < 2:
        print(usage())
    else:
        dataFile = argv[1]
    n = NeuralNetwork(dataFile) 
    n.initialMatrixWeights()
    n.feedForward()

def usage():
    return """
            python neural_network.py [dataFile]
                [dataFile] - the path to the file that will be used to build the net
            """

if __name__ == "__main__":
    main(sys.argv)
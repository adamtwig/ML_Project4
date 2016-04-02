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
        self.learningRate = 0.5

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
                                    size=(self.numHiddenUnits,len(self.inputValues[0]) + 1))
        self.hiddenTargetWeights = np.random.uniform(low=lowInitial, high=highInitial, 
                                    size=(len(self.targetValues[0]), self.numHiddenUnits + 1))

        # set the bias weights and vals to 1
        self.inputHiddenWeights[:, 0] = 1
        self.hiddenTargetWeights[:, 0] = 1

    def addBias(self, currentInput):

        # add the bias unit to our input
        inputWithBias = currentInput
        inputWithBias.insert(0, '1')

        currentInput = np.array(inputWithBias).astype(np.float)

        return currentInput

    def feedForward(self, currentInput):

        sumOfProductsIH = np.dot(self.inputHiddenWeights,currentInput)

        hiddenUnitsValues = np.insert(self.sigmoid(sumOfProductsIH), 0, 1)

        sumOfProductsHT = np.dot(self.hiddenTargetWeights,hiddenUnitsValues)

        sigma =  self.sigmoid(sumOfProductsHT)

        #print sigma

        return hiddenUnitsValues, sigma 

    def sigmoid(self, sumOfProducts):
        return 1.0 / (1.0 + np.exp(-1.0 * sumOfProducts))

    def total_error(self, y, t):
        return 0.5*(t-y)**2

    def sigmoid_error(self, y, t):
        return y*(1.0-y)*(t-y)

    def hidden_error(self, h, w, E):
        return h*(1-h)*(w*E)

    def updateWeights(self, w, eta, E, z):
        return w + eta*E*z

    def backpropagation(self, index, currentInput, hiddenUnitsValues, sigma):

        currentTarget = float(self.targetValues[index][0])
        #print "currentTarget:", currentTarget
        #print "sigma:", sigma[0]
        
        total_error = self.total_error(sigma, currentTarget)

        print total_error

        sigma_error = self.sigmoid_error(total_error[0], currentTarget)

        #print sigma_error

        hiddenUnits_error = self.hidden_error(hiddenUnitsValues,  
                                                self.hiddenTargetWeights, 
                                                    sigma_error)        

        self.hiddenTargetWeights =  self.updateWeights(self.hiddenTargetWeights,
                                        self.learningRate, sigma_error, 
                                        hiddenUnitsValues)

        #print self.hiddenTargetWeights

        self.inputHiddenWeights  =  self.updateWeights(self.inputHiddenWeights,
                                        self.learningRate, hiddenUnits_error, 
                                        currentInput)

        #print self.inputHiddenWeights

def main(argv):
    if len(argv) < 2:
        print(usage())
        sys.exit(-1)
    else:
        dataFile = argv[1]
    n = NeuralNetwork(dataFile) 
    n.initialMatrixWeights()

    for i in range(0,150):

        currentInput = n.addBias(n.inputValues[i])
        hiddenUnitsValues, sigma = n.feedForward(currentInput)
        n.backpropagation(i, currentInput, hiddenUnitsValues, sigma)



def usage():
    return """
            python neural_network.py [dataFile]
                [dataFile] - include data to build the net
            """

if __name__ == "__main__":
    main(sys.argv)
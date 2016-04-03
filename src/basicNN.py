"""
@authors: Michael Baldwin, Josh Engelsma, Adam Terwilliger
@date: March 30, 2016
@version: 1.0
This program builds a neural network.
"""

import sys
import math
import numpy as np
import random

class NeuralNetwork(object):
    def __init__(self, dataFile):
        self.dataFile = dataFile
        self.targetValues = []
        self.inputValues = []
        self.numTargetValues = 1
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
                target = lineParts[-self.numTargetValues:]
                if i == 0:
                    self.inputNames = lineParts[:-self.numTargetValues]
                else:
                    self.targetValues.append(target)
                    self.inputValues.append(lineParts[:-self.numTargetValues])


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

        return hiddenUnitsValues, sigma 

    def sigmoid(self, sumOfProducts):
        return 1.0 / (1.0 + np.exp(-1.0 * sumOfProducts))

    def total_error(self, y, t):
        return 0.5*(t-y)**2

    def sigmoid_error(self, y, t):
        return y*(1.0-y)*(t-y)

    def hidden_error(self, h, w, E):
        return np.dot(E,w)*h*(1-h)

    # http://stackoverflow.com/questions/22949966/dot-product-of-two-1d-vectors-in-numpy
    def updateHTweights(self, w, eta, E, z):
        return w + eta*np.outer(E,z)

    def updateIHweights(self, w, eta, E, z):
        return w + eta*np.outer(E[1:],z)


    def backpropagation(self, index, currentInput, hiddenUnitsValues, sigma):

        currentTarget = self.targetValues[index]
        currentTarget = [float(x) for x in currentTarget]
        #print "currentTarget:", currentTarget
        #print "sigma:", sigma
        
        total_error = self.total_error(sigma, currentTarget)

        print total_error

        sigma_error = self.sigmoid_error(total_error, currentTarget)

        #print sigma_error

        hiddenUnits_error = self.hidden_error(hiddenUnitsValues,  
                                                self.hiddenTargetWeights, 
                                                    sigma_error)        
        #print hiddenUnits_error

        self.hiddenTargetWeights =  self.updateHTweights(self.hiddenTargetWeights,
                                        self.learningRate, sigma_error, 
                                        hiddenUnitsValues)

        #print self.hiddenTargetWeights

        self.inputHiddenWeights  =  self.updateIHweights(self.inputHiddenWeights,
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

    #trainingExampleIndices = random.sample(range(0, 150), 150)

    for i in range(0,3):
    #for i in trainingExampleIndices:

        currentInput = n.addBias(n.inputValues[i])
        hiddenUnitsValues, sigma = n.feedForward(currentInput)
        n.backpropagation(i, currentInput, hiddenUnitsValues, sigma)

    numCorrect = 0

    #for i in range(0,150):
    for i in range(0,3):
        currentInput = np.array(n.inputValues[i]).astype(np.float)        
        hiddenUnitsValues, sigma = n.feedForward(currentInput)
        
        #print sigma
        if n.numTargetValues == 1:
            predicted = sigma
            target = n.targetValues[i]

        else:
            predicted = np.argmax(sigma)        
            target = np.argmax(n.targetValues[i])

        print "predicted:",predicted
        print "target:",target

        if predicted == target:
            numCorrect +=1

    print numCorrect / 4

def usage():
    return """
            python neural_network.py [dataFile]
                [dataFile] - include data to build the net
            """

if __name__ == "__main__":
    main(sys.argv)
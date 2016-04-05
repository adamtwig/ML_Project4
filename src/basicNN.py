"""
@authors: Josh Engelsma, Adam Terwilliger
@date: April 5, 2016
@version: 1.0
This program builds a neural network, tests / trains the network, and outputs the 
weight matrices from every epoch which are used for training viz purposes.
"""

import sys
import math
import numpy as np
import random
import os

class NeuralNetwork(object):
    def __init__(self, dataFile, numTargetValues=3):
        self.dataFile = dataFile
        self.outputFileName = "../output/weights_outputs"
        self.flattenedWeights =  ""
        self.dataFileSize = 0
        self.targetValues = []
        self.inputValues = []
        self.numTargetValues = numTargetValues
        self.numEpochs = 100
        self.readFile()
        self.numHiddenUnits = self.calcNumHiddenUnits()
        self.inputHiddenWeights = np.zeros([])
        self.hiddenTargetWeights = np.zeros([])
        self.learningRate = 0.5
        self.trainTestSplit = 0.75

    def readFile(self):
        """
        @summary: method reads in the training data inputs to be used for training our neural network
        """
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
                self.dataFileSize += 1

    def calcNumHiddenUnits(self):
        """
        @summary: method returns 2/3 the number of inputs plus outputs - the number of hidden values 
                  in out network
        """
        return math.ceil((2.0/3.0)*(len(self.targetValues[0])+len(self.inputValues[0])))

    def initialMatrixWeights(self):
        """
        @summary: method initializes our weights from the input to hidden layer and the 
                  hidden to output layer with random values.
        """
        lowInitial = -0.1
        highInitial = 0.1
        np.random.seed(42)

        self.inputHiddenWeights = np.random.uniform(low=lowInitial, high=highInitial, 
                                    size=(self.numHiddenUnits,len(self.inputValues[0]) + 1))
        self.hiddenTargetWeights = np.random.uniform(low=lowInitial, high=highInitial, 
                                    size=(len(self.targetValues[0]), self.numHiddenUnits + 1))

        # set the bias weights and vals to 1
        self.inputHiddenWeights[:, 0] = 1
        self.hiddenTargetWeights[:, 0] = 1 

    def addBias(self, currentInput):
        """
        @param currentInput: The input vector for this training example
        @summary: method adds bias value of 1 to our input vector
        """
        # add the bias unit to our input
        inputWithBias = currentInput
        inputWithBias.insert(0, '1')

        currentInput = np.array(inputWithBias).astype(np.float)

        return currentInput

    def feedForward(self, currentInput):
        """
        @param currentInput: The input vector
        @summary: Methods feeds our input through the currently weighted neural network
        """
        sumOfProductsIH = np.dot(self.inputHiddenWeights,currentInput)

        hiddenUnitsValues = np.insert(self.sigmoid(sumOfProductsIH), 0, 1)

        sumOfProductsHT = np.dot(self.hiddenTargetWeights,hiddenUnitsValues)

        sigma =  self.sigmoid(sumOfProductsHT)

        return hiddenUnitsValues, sigma 

    def sigmoid(self, sumOfProducts):
        """
        @param sumOfProducts: the results of moving through one layer in the network 
        @summary: method runs the sigmoid function on sigma
        """
        return 1.0 / (1.0 + np.exp(-1.0 * sumOfProducts))

    def total_error(self, y, t):
        """
        @param y: sigma
        @param t: the current target value 
        @summary: method returns the total error for the given sigma given the current target value
        """
        return 0.5*(t-y)**2

    def sigmoid_error(self, y, t):
        """
        @param y: sigma
        @param t: the current target value
        @summary: method returns the target error by using the derivative
        """
        return y*(1.0-y)*(t-y)

    def hidden_error(self, h, w, E):
        """
        @param h: The vector of current hidden Units values
        @param w: the hidden to target weights matrix
        @param E: The target Error
        @summary: method returns the error for respective hidden nodes
        """
        return np.dot(E,w)*h*(1-h)

    def updateHTweights(self, w, eta, E, z):
        """
        @param w: the hidden to target weights matrix
        @param eta: our learning rate.
        @param E: The target units error
        @param z: The hidden units values
        @ref: # http://stackoverflow.com/questions/22949966/dot-product-of-two-1d-vectors-in-numpy
        @summary: method updates the hidden to target weights
        """
        return w + eta*np.outer(E,z)

    def updateIHweights(self, w, eta, E, z):
        """
        @param w: the input to hidden layer weights
        @param eta: the learning rate 
        @param E: the hidden units error
        @param z: The current input vector values
        @summary: method updates the weights of the input to hidden layer nodes
        """
        return w + eta*np.outer(E[1:],z)


    def backpropagation(self, index, currentInput, hiddenUnitsValues, sigma):
        """
        @param index: the index of the training example we are currently using.
        @param currentInput: the current input vector values
        @param hiddenUnitsValues: the values that we summed up by forward prop at hidden layer.
        @param sigma: value of sigma that was calculated during forward prop.
        @summary: method backpropogates our errors and updates our weights accordingly.
        """

        currentTarget = self.targetValues[index]
        currentTarget = [float(x) for x in currentTarget]
        
        total_error = self.total_error(sigma, currentTarget)

        target_error = self.sigmoid_error(sigma, currentTarget)

        hiddenUnits_error = self.hidden_error(hiddenUnitsValues,  
                                                self.hiddenTargetWeights, 
                                                    target_error)        

        self.hiddenTargetWeights =  self.updateHTweights(self.hiddenTargetWeights,
                                        self.learningRate, target_error, 
                                        hiddenUnitsValues)


        self.inputHiddenWeights  =  self.updateIHweights(self.inputHiddenWeights,
                                    self.learningRate, hiddenUnits_error, 
                                        currentInput)

    def trainAndTestNetwork(self):
        """
        @summary: method runs through a specified percentage of our training data, and
                  a specified number of epochs to build the neural network. Then the remaining 
                  data is used for testing purposes.
        """
        # separate our testing and training data...
        trainingExampleIndices = random.sample(range(0, self.dataFileSize-1),
                                                int((self.trainTestSplit)*(self.dataFileSize-1)))
        testExampleIndices = []
        for i in range(self.dataFileSize-1):
            if i not in trainingExampleIndices:
                testExampleIndices.append(i)

        # add our bias to all our examples
        for i, e in enumerate(self.inputValues):
            self.inputValues[i] = self.addBias(e)

        # train our network with training examples over multiple epochs
        for j in range(self.numEpochs):
            self.updateFlattenedWeights()
            for i in trainingExampleIndices:
                currentInput = self.inputValues[i]
                hiddenUnitsValues, sigma = self.feedForward(currentInput)
                self.backpropagation(i, currentInput, hiddenUnitsValues, sigma)

        # test our data
        numCorrect = 0
        for i in testExampleIndices:
            currentInput = np.array(self.inputValues[i]).astype(np.float)        
            hiddenUnitsValues, sigma = self.feedForward(currentInput)

            if self.numTargetValues == 1:
                predicted = int(round(sigma))
                target = int(self.targetValues[i][0])

            else:
                predicted = self.sigmoid(sigma)
                predicted = np.argmax(predicted)        
                target = np.argmax(self.targetValues[i])

            if predicted == target:
                numCorrect +=1

        print ("Dataset:", self.dataFile, "Num Correct:", numCorrect,
                "Num Test:", len(testExampleIndices), "Num Epochs:", self.numEpochs,
                 "Learning Rate:", self.learningRate,
                 "Train/Test Split:", self.trainTestSplit)

    def getInputHiddenWeightsAsList(self):
        """
        @summary: method returns the input to hidden layer weights as a python list.
        """
        return self.inputHiddenWeights.tolist()

    def getHiddenTargetWeightsAsList(self):
        """
        @summary: method returns the hidden to target layer weights as python list.
        """
        return self.hiddenTargetWeights.tolist()

    def updateFlattenedWeights(self):
        """
        @summary: method flattens out weight matrices into a string to be used by our viz.
        """
        lFlattenedWeights = ""
        for x in np.nditer(self.inputHiddenWeights):
            lFlattenedWeights += "{},".format(str(x))
        for x in np.nditer(self.hiddenTargetWeights):
            lFlattenedWeights += "{},".format(str(x))
        self.flattenedWeights += lFlattenedWeights + "\n"

    def writeWeightsToFile(self, fileName):
        """
        @param fileName: name of file you wish to output flattened weights to.
        @summary: method writes out our flattened weights from every epoch to a file.
        """
        with open(fileName, "w") as fh:
            fh.write(self.flattenedWeights)

def main(argv):
    if len(argv) < 2:
        print(usage())
        sys.exit(-1)
    else:
        dataFile = argv[1]
        if len(argv) == 2:
            n = NeuralNetwork(dataFile) 
        if len(argv) > 2:
            numTargetValues = int(argv[2])
            n = NeuralNetwork(dataFile,numTargetValues) 
            n.numEpochs = int(argv[3])
            n.learningRate = float(argv[4])
            n.trainTestSplit = float(argv[5])

    n.initialMatrixWeights()
    n.trainAndTestNetwork()
    
def usage():
    return """
            python neural_network.py [dataFile]
                [dataFile] - include data to build the net
            """

if __name__ == "__main__":
    main(sys.argv)

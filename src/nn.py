import numpy as np

def sigmoid(layer) :
    tmp = 1 / (1 + np.exp(layer))

    return tmp

def sigmoid_derivative(layer) :
    tmp = layer * (1 - layer)

    return tmp

class NeuralNetwork:

    def __init__(self, inputs, results) :
        np.random.seed()

        self.inputs  = inputs
        self.weights = 2 * np.random.rand(self.inputs.shape[1], 1) - 1
        self.results = results
        self.output  = np.zeros(self.results.shape)

    def feedfoward(self) :
        self.output = sigmoid(np.dot(self.inputs, self.weights))

    # Gradient descent - converge towards a local minimum of the loss function
    def backdrop(self) :
        error    = self.results - self.output
        gradient = np.dot(self.inputs.T, (2 * error) * sigmoid_derivative(self.output))

        self.weights += gradient

    def train(self, itter) :

        for i in range(itter) :
            self.feedfoward()
            self.backdrop()

    def print_weights(self) :
        print("weights:\n{}".format(self.weights))

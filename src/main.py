from nn import *
 
if __name__ == "__main__" : 

    training_inputs  = np.array([[0,1,0,0],
                                 [1,0,1,0],
                                 [0,1,0,0],
                                 [0,0,1,0],
                                 [0,0,1,1]])

    training_outputs = np.array([[0,1,0,0,0]]).T

    print("Training Set:\n{0}\nExpected Result:\n{1}".
            format(training_inputs, training_outputs))
    
    neuralNet = NeuralNetwork(training_inputs, training_outputs)

    neuralNet.train(1000)

    print("Cost:\n", neuralNet.output)

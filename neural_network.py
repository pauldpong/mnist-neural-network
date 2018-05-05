import random
import dataloader
import numpy as np


class Network(object):
    def __init__(self, layers):
        self.num_layers = len(layers)
        self.sizes = layers
        self.biases = [np.random.randn(neurons, 1) for neurons in layers[1:3]]
        self.weights = [np.random.rand(layer1_2, layer2_3) for layer1_2, layer2_3 in zip(layers[0:2], layers[1:3])]

    def output(self, prev_out):
        for b, w in zip(self.biases, self.weights):
            new_out = sigmoid(np.dot(prev_out, w) + b)

        return new_out

    def start_network(self, training_data, testing_data, batch_size, generations, learning_rate):

        training = list(training_data)
        testing = list(testing_data)

        num_train = len(training)
        num_test = len(testing)

        for gen in range(generations):
            random.shuffle(training)

            for i in range(0, int(num_train/batch_size), batch_size):
                self.update_weights_biases(training[i:i + batch_size], learning_rate)

            print("Generation {0}: {1} / {2}".format(gen, self.evaluate(testing), num_test))

    # TODO - update the weights and biases for each mini-batch using back prop gradient descent 
    def update_weights_biases(self, batch_data, rate):
        print("hi")

    # TODO - take the updated weights and biases and run it with test-data set and compare with test-labels
    def evaluate(self, testing_data):
        print(3)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    return np.exp(x)/np.power(np.exp(x)+1, 2)


if __name__ == "__main__":
    tr_data, te_data = dataloader.load_data("./MNIST/trainimages.gz", "./MNIST/trainlabels.gz", "./MNIST/10ktest.gz", "./MNIST/10klabels.gz")
    net = Network([784, 30, 10])
    net.start_network(tr_data, te_data, 10, 30, 3.0)






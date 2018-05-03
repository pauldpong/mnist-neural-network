import numpy as np


class Network(object):
    def __init__(self, layers):
        self.num_layers = len(layers)
        self.sizes = layers
        self.biases = [np.random.randn(neurons, 1) for neurons in layers[1:3]]
        self.weights = [np.random.rand(layer1_2, layer2_3) for layer1_2, layer2_3 in zip(layers[0:2], layers[1:3])]

    


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    return np.exp(x)/np.power(np.exp(x)+1, 2)


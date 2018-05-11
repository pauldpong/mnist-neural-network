import random
import dataloader
import numpy as np


class Network(object):
    def __init__(self, layers):
        self.num_layers = len(layers)
        self.sizes = layers
        self.biases = [np.random.randn(neurons, 1) for neurons in layers[1:3]]
        self.weights = [np.random.randn(layer1_2, layer2_3) for layer1_2, layer2_3 in zip(layers[0:2], layers[1:3])]

    def output(self, prev_out):
        for b, w in zip(self.biases, self.weights):
            new_out = sigmoid(np.dot(prev_out, w) + b)

        return new_out

    def start_network(self, training_data, testing_data, batch_size, generations, learning_rate):

        training_set = list(training_data)
        testing = list(testing_data)

        num_train = len(training_set)
        num_test = len(testing)

        for gen in range(generations):
            random.shuffle(training_set)

            for i in range(0, int(num_train/batch_size), batch_size):
                self.update_weights_biases(training_set[i:i + batch_size - 1], learning_rate)

            print("Generation {0}: {1} / {2}".format(gen, self.test(testing), num_test))

    def update_weights_biases(self, batch_data, rate):
        delta_weights = [np.zeros(w.shape) for w in self.weights]
        delta_biases = [np.zeros(b.shape) for b in self.biases]

        for image, label in batch_data:
            p_delta_weights, p_delta_biases = self.backprop(image, label)

            # Sum of the weights and biases gradients
            delta_weights = [dw + pdw for dw, pdw in zip(delta_weights, p_delta_weights)]
            delta_biases = [db + pdb for db, pdb in zip(delta_biases, p_delta_biases)]

        # Update
        self.weights = [old_weights - ((rate/len(batch_data)) * delta_w)
                        for old_weights, delta_w in zip(self.weights, delta_weights)]
        self.biases = [old_biases - ((rate / len(batch_data)) * delta_b)
                       for old_biases, delta_b in zip(self.biases, delta_biases)]

    def backprop(self, image, label):
        image_delta_weights = [np.zeros(w.shape) for w in self.weights]
        image_delta_biases = [np.zeros(b.shape) for b in self.biases]

        activation = image
        activations = [image]

        weighted_z = []

        for w, b in zip(self.weights, self.biases):
            output = np.dot(w, activation) + b
            weighted_z.append(output)

            activation = sigmoid(output)
            activations.append(activation)


    def test(self, testing_data):
        total_correct = 0

        for test_image, label in testing_data:
            if self.evaluate(test_image) == label:
                total_correct += 1

        return total_correct

    def evaluate(self, image):
        # feedfoward


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


if __name__ == "__main__":
    tr_data, te_data = dataloader.load_data("./MNIST/trainimages.gz", "./MNIST/trainlabels.gz", "./MNIST/10ktest.gz", "./MNIST/10klabels.gz")
    net = Network([784, 30, 10])
    net.start_network(tr_data, te_data, 10, 30, 3.0)






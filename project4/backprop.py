# CS 425. Project 3.
# Written by: Ksenia Burova
#
# Back Propagation Algorithm

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math

class Data:
    def __init__(self, filename):
        data = np.genfromtxt(filename, delimiter=',')
        self.features = data[:, :-1]
        self.labels = data[:, [-1]]

        # normalize data
        self.normalize()

        # split data into training, validation and testing data
        # make sure that tum type arrays are 1-dimensional
        self.trainFeatures, self.testFeatures, self.trainLabels, self.testLabels = train_test_split(
            self.features, self.labels, test_size=0.4, random_state=26)
        self.trainLabels = self.trainLabels.ravel()
        self.testLabels = self.testLabels.ravel()

        self.validFeatures, self.testFeatures, self.validLabels, self.testLabels = train_test_split(
            self.testFeatures, self.testLabels, test_size=0.5, random_state=26)
        self.testLabels = self.testLabels.ravel()
        self.validLabels = self.validLabels.ravel()

    def normalize(self):
        M = np.mean(self.features, axis=0)
        D = np.sqrt(np.mean((self.features - M)**2, axis=0))

        self.features = (self.features - M) / D

class BackProp:
    def __init__(self, data, lr, ne, ln, nn):
        self.data = data
        self.learning_rate = lr
        self.number_epochs = ne
        self.num_layers = ln
        self.num_neurons = nn
        self.num_inputs = len(d.features[0])

        self.output_delta = 0
        self.output_h = 0
        self.output_sigma = 0
        self.output_bias_w = np.random.uniform(-0.01, 0.01, 1)
        self.expected_output = 0

        self.input = np.array([])
        self.RMSE = np.array([])

        # 3 dimensional array num_layers x num_neurons x num_of_prev_layer_nodes
        self.hidden_weights = []

        # 1 dim array, num of elem = number of neurons in last layer
        self.output_weights = np.random.uniform(-0.01, 0.01, self.num_neurons[-1])

        # 2 dimensional arrays num_of_layers x num_of_neurons
        self.bias_weights = []
        self.hidden_sigma = []
        self.hidden_h = []
        self.hidden_delta = []

        for i, n in enumerate(self.num_neurons):
            # set bias weights to random values
            self.bias_weights.append(np.random.uniform(-0.01, 0.01, n))

            # fill array with empty dimensions
            self.hidden_sigma.append(np.zeros(n))
            self.hidden_h.append(np.zeros(n))
            self.hidden_delta.append(np.zeros(n))

            hid_w = []
            for j in range(n):
                if i == 0:
                    num_prev = self.num_inputs
                else:
                    num_prev = self.num_neurons[i-1]
                hid_w.append(list(np.random.uniform(-0.01, 0.01, num_prev)))
            self.hidden_weights.append(np.array(hid_w, dtype=object))

        self.bias_weights = np.array(self.bias_weights, dtype=object)
        self.hidden_sigma = np.array(self.hidden_sigma, dtype=object)
        self.hidden_h = np.array(self.hidden_h, dtype=object)
        self.hidden_delta = np.array(self.hidden_delta, dtype=object)
        self.hidden_weights = np.array(self.hidden_weights, dtype=object)

    def compute_outputs(self):
        self.output_delta = 0
        self.output_h = 0
        self.output_sigma = 0

        # reset vals
        for i in range(self.num_layers):
            for j in range(self.num_neurons[i]):
                self.hidden_sigma[i][j] = 0
                self.hidden_delta[i][j] = 0
                self.hidden_h[i][j] = 0

        for i, n in enumerate(self.num_neurons):
            for j in range(n):
                if i == 0:
                    for c in range(self.num_inputs):
                        self.hidden_h[i][j] += self.hidden_weights[i][j][c] * self.input[c]
                else:
                    for c in range(self.num_neurons[i-1]):
                        self.hidden_h[i][j] += self.hidden_weights[i][j][c] * self.hidden_sigma[i-1][c]
                self.hidden_h[i][j] += self.bias_weights[i][j]
                self.hidden_sigma[i][j] = 1/(1 + math.exp(-self.hidden_h[i][j]))

        for i in range(self.num_neurons[-1]):
            self.output_h += self.hidden_sigma[-1][i] * self.output_weights[i]
        self.output_h += self.output_bias_w
        self.output_sigma = 1/(1 + math.exp(-self.output_h))

    def compute_delta(self):
        self.output_delta = self.output_sigma * (1 - self.output_sigma) * (self.expected_output - self.output_sigma)
        for i, n in reversed(list(enumerate(self.num_neurons))):
            for j in range(n):
                if i == self.num_layers-1:
                    self.hidden_delta[i][j] = self.hidden_sigma[i][j] * (1 - self.hidden_sigma[i][j]) * self.output_delta * self.output_weights[j]
                else:
                    for c in range(self.num_neurons[i+1]):
                        self.hidden_delta[i][j] = self.hidden_sigma[i][j] * (1 - self.hidden_sigma[i][j]) * self.hidden_delta[i+1][c] * self.hidden_weights[i+1][c][j]

    def update_weights(self):
        for i, n in enumerate(self.num_neurons):
            for j in range(n):
                if i == 0:
                    for c in range(self.num_inputs):
                        self.hidden_weights[i][j][c] += (self.learning_rate * self.hidden_delta[i][j]) * self.input[c]

                else:
                    for c in range(self.num_neurons[i-1]):
                        self.hidden_weights[i][j][c] += (self.learning_rate * self.hidden_delta[i][j]) * self.hidden_sigma[i - 1][c]
                self.bias_weights[i][j] += self.learning_rate * self.hidden_delta[i][j]

        for j in range(self.num_neurons[-1]):
            self.output_weights[j] += self.learning_rate * self.output_delta * self.hidden_sigma[-1][j]
        self.output_bias_w += self.learning_rate * self.output_delta

    def train_network(self):
        for i, self.input in enumerate(self.data.trainFeatures):
            self.expected_output = self.data.trainLabels[i]

            self.compute_outputs()
            self.compute_delta()
            self.update_weights()

    def validate_network(self):
        sum = 0.0
        num_of_patterns = 0

        for i, self.input in enumerate(self.data.validFeatures):
            self.expected_output = self.data.validLabels[i]
            self.compute_outputs()
            sum += (self.expected_output - self.output_sigma)**2
            num_of_patterns += 1
        rmse = (sum / (2.0 * num_of_patterns))**0.5
        np.append(self.RMSE, rmse)
        print(rmse)

    def test_network(self):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        print('Testing Network')
        for i, self.input in enumerate(self.data.testFeatures):
            self.expected_output = self.data.testLabels[i]
            self.compute_outputs()

            if math.fabs(self.expected_output - self.output_sigma) > (self.RMSE[-1] + 0.01):
                if self.expected_output == 1:
                    fn += 1
                else:
                    fp += 1
            else:
                if self.expected_output == 1:
                    tp += 1
                else:
                    tn += 1
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        print("Accuracy: ", accuracy)

    def print_RMSE(self):
        print('Epoch #, RMSE:')
        for i, rmse in enumerate(self.RMSE):
            if (i % 100) == 0:
                print(i, ', ', rmse)

    def run(self):
        for i in range(self.number_epochs):
            print('Epoch: ', i)
            self.train_network()
            self.validate_network()
        # self.print_RMSE()
        self.test_network()


d = Data('spambase.data')
b = BackProp(d, 0.3, 200, 1, [10])
b.run()
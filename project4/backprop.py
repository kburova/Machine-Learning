# CS 425. Project 3.
# Written by: Ksenia Burova
#
# Back Propagation Algorithm

import numpy as np
from sklearn.model_selection import train_test_split
from scipy.special import expit
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
            self.features, self.labels, test_size=0.5, random_state=26)
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
        self.num_inputs = len(data.features[0])
        self.accuracy = 0.0

        self.output_delta = 0
        self.output_h = 0
        self.output_sigma = 0
        self.output_bias_w = np.random.uniform(-0.1, 0.1, 1)
        self.expected_output = 0

        self.input = np.array([])
        self.RMSE = []

        # 3 dimensional array num_layers x num_neurons x num_of_prev_layer_nodes
        self.hidden_weights = []

        # 1 dim array, num of elem = number of neurons in last layer
        self.output_weights = np.random.uniform(-0.1, 0.1, self.num_neurons[-1])

        # 2 dimensional arrays num_of_layers x num_of_neurons
        self.bias_weights = []
        self.hidden_sigma = []
        self.hidden_h = []
        self.hidden_delta = []

        for i, n in enumerate(self.num_neurons):
            self.bias_weights.append(np.random.uniform(-0.1, 0.1, n))
            self.hidden_sigma.append(np.empty(n, dtype=float))
            self.hidden_h.append(np.empty(n, dtype=float))
            self.hidden_delta.append(np.empty(n, dtype=float))

            if i == 0:
                num_prev = self.num_inputs
            else:
                num_prev = self.num_neurons[i-1]
            self.hidden_weights.append(np.random.uniform(-0.1, 0.1, (n, num_prev)))

    def compute_outputs(self):
        for i, n in enumerate(self.num_neurons):
            if i == 0:
                self.hidden_h[i] = np.sum(self.hidden_weights[i] * self.input, axis=1)

            else:
                self.hidden_h[i] = np.sum(self.hidden_weights[i] * self.hidden_sigma[i-1], axis=1)
            self.hidden_h[i] += self.bias_weights[i]
            self.hidden_sigma[i] = expit(self.hidden_h[i])
        self.output_h = np.sum(self.hidden_sigma[-1] * self.output_weights) + self.output_bias_w
        self.output_sigma = expit(self.output_h)[0]

    def compute_delta(self):
        self.output_delta = self.output_sigma * (1 - self.output_sigma) * (self.expected_output - self.output_sigma)
        for i, n in reversed(list(enumerate(self.num_neurons))):
            if i == self.num_layers-1:
                self.hidden_delta[i] = self.hidden_sigma[i] * (1 - self.hidden_sigma[i]) * self.output_delta * self.output_weights
            else:
                self.hidden_delta[i] = self.hidden_sigma[i] * (1 - self.hidden_sigma[i]) * np.sum(np.transpose(self.hidden_delta[i+1] * np.transpose(self.hidden_weights[i+1][:])), axis=0)

    def update_weights(self):
        for i, n in enumerate(self.num_neurons):
            if i == 0:
                self.hidden_weights[i] = self.hidden_weights[i] + (self.learning_rate * self.hidden_delta[i]).reshape((len(self.hidden_delta[i]), 1)) * self.input
            else:
                self.hidden_weights[i] = self.hidden_weights[i] + (self.learning_rate * self.hidden_delta[i]).reshape((len(self.hidden_delta[i]), 1)) * self.hidden_sigma[i - 1]
            self.bias_weights[i] += self.learning_rate * self.hidden_delta[i]

        self.output_weights += self.learning_rate * self.output_delta * self.hidden_sigma[-1]
        self.output_bias_w += self.learning_rate * self.output_delta

    def train_network(self):
        for i, self.input in enumerate(self.data.trainFeatures):
            self.expected_output = self.data.trainLabels[i]
            self.compute_outputs()
            self.compute_delta()
            self.update_weights()

    def validate_network(self):
        sum = 0.0
        for i, self.input in enumerate(self.data.validFeatures):
            self.expected_output = self.data.validLabels[i]
            self.compute_outputs()
            sum += (self.expected_output - self.output_sigma)**2
        self.RMSE.append((sum / (2.0 * len(self.data.validFeatures)))**0.5)

    def test_network(self, ofile):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
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
        self.accuracy = (tp + tn) / (tp + tn + fp + fn)
        try:
            tpr = tp/(tp + fn)
        except ZeroDivisionError:
            tpr = 0
        try:
            ppv = tp/(tp + fp)
        except ZeroDivisionError:
            ppv = 0
        try:
            tnr = tn/(tn + fp)
        except ZeroDivisionError:
            tnr = 0
        # fscore = ppv*tpr/(ppv+tpr)
        print('Matrix: \n %d %d\n %d %d\n' % (tn, fp, fn, tp), file=ofile)
        print('Accuracy: %0.3f' % self.accuracy, file=ofile)
        print('TPR: %0.3f' % tpr, file=ofile)
        print('PPV: %0.3f' % ppv, file=ofile)
        print('TNR: %0.3f\n' % tnr, file=ofile)
        # print('FScore: %0.3f\n' % fscore, file=ofile)

    def plot_RMSE(self, n, lr_num):
        c = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

        plt.figure(0, figsize=(9, 6))
        label = "LR: %.2f, Accuracy: %.2f" % (self.learning_rate, self.accuracy)
        plt.plot(range(self.number_epochs), self.RMSE, linestyle='-', color=c[(n % lr_num)%10], linewidth=1.2, label=label)

        if (n % lr_num) == (lr_num-1):
            print('Saving picture')
            title = "Layers: %s" % str(self.num_neurons)
            plt.title(title)
            plt.xlabel('Epoch #')
            plt.ylabel('RMSE')
            plt.legend()
            plt.savefig('images/rmse_'+str(int(n/lr_num))+'.png')
            plt.close()

    def run(self, n, lr_num, ofile):
        for i in np.arange(self.number_epochs):
            self.train_network()
            self.validate_network()
        self.test_network(ofile)
        self.plot_RMSE(n, lr_num)

def main():
    outfile = open('results.txt', 'w')
    d = Data('spambase.data')

    layers = [[15],
              [65],
              [100],
              [70, 70],
              [30, 30],
              [60, 70, 80],
              [30, 40, 30],
              [15, 15, 15],
              [10, 10, 10, 10],
              [60, 70, 70, 60],
              [60, 40, 30, 50],
              [15, 15, 15, 15, 15],
              [60, 70, 100, 70, 60]]
    epochs = 500
    lrate = [0.01, 0.1, 0.25, 0.4, 0.65, 0.85, 1.0]

    for i, l in enumerate(layers):
        print('Testing Network %d' % i)
        for k, lr in enumerate(lrate):
            n = i * len(lrate) + k
            print('Problem %d\n' % n, file=outfile)
            try:
                BackProp(d, lr, epochs, len(l), l).run(n, len(lrate), outfile)
            except OverflowError:
                print('Overflow in layers combination %d, epochs = %d, and l. rate = %.2f' % (i, epochs, k), file=outfile)

    outfile.close()


main()

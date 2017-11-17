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
        self.epochs = np.array([6, 10, 30, 60, 200, 650, 1200, 2000, 5000])
        self.performance = np.empty(shape=(len(self.epochs), 5), dtype=float)

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
            # set bias weights to random values
            self.bias_weights.append(np.random.uniform(-0.1, 0.1, n))

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
                hid_w.append(np.random.uniform(-0.1, 0.1, num_prev))
            self.hidden_weights.append(np.array(hid_w))

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
        for s, d, h in zip(self.hidden_sigma, self.hidden_delta, self.hidden_h):
            s.fill(0)
            d.fill(0)
            h.fill(0)

        for i, n in enumerate(self.num_neurons):
            if i == 0:
                self.hidden_h[i] = np.sum(self.hidden_weights[i] * self.input, axis=1)
            else:
                self.hidden_h[i] = np.sum(self.hidden_weights[i] * self.hidden_sigma[i-1], axis=1)

            self.hidden_h[i] = self.hidden_h[i] + self.bias_weights[i]
            # for j in range(len(self.hidden_sigma[i])):
            self.hidden_sigma[i] = self.hidden_sigma[i] + 1/(1 + np.exp(-1 * self.hidden_h[i].astype(float)))

        self.output_h = np.sum(self.hidden_sigma[-1] * self.output_weights) + self.output_bias_w
        self.output_sigma = 1/(1 + math.exp(-self.output_h))

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

        self.bias_weights += self.learning_rate * self.hidden_delta
        self.output_weights = self.output_weights + self.learning_rate * self.output_delta * self.hidden_sigma[-1]
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
        self.RMSE.append((sum / (2.0 * num_of_patterns))**0.5)

    def test_network(self, ofile, ep_num):
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
        tpr = tp/(tp + fn)
        ppv = tp/(tp + fp)
        tnr = tn/(tn + fp)
        fscore = ppv*tpr/(ppv+tpr)

        self.performance[ep_num] = [self.accuracy, tpr, ppv, tnr, fscore]

        print('Accuracy: ', self.accuracy, file=ofile)
        print('TPR: ', tpr, file=ofile)
        print('PPV: ', ppv, file=ofile)
        print('TNR: ', tnr, file=ofile)
        print('FScore: ', fscore, file=ofile)

    def plot_rmse_performance(self, n, lr_num):
        c = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        label = "LR: %.2f" % self.learning_rate
        print('Plot lr # %d' % int(n % lr_num))

        plt.figure(0)
        plt.plot(range(self.number_epochs), self.RMSE, linestyle='-', color=c[(n % lr_num)%10], linewidth=1.0,
                 label=label)
        if (n % lr_num) == (lr_num-1):
            print('Saving picture 0')

            title = "RMSE: Layers: %s" % str(self.num_neurons)
            plt.title(title)
            plt.xlabel('Epoch #')
            plt.ylabel('RMSE')
            plt.legend()
            plt.savefig('images/rmse_'+str(int(n/lr_num))+'.png')
            plt.close()

        plt.figure(1)
        plt.plot(self.epochs, self.performance[:, 0], linestyle='-', color=c[(n % lr_num) % 10], linewidth=1.0,
                 label=label)
        if (n % lr_num) == (lr_num - 1):
            print('Saving picture 1')

            title = "Accuracy: Layers: %s" % str(self.num_neurons)
            plt.title(title)
            plt.xticks(self.epochs)
            plt.xlabel('Epoch #')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig('images/accuracy_' + str(int(n / lr_num)) + '.png')
            plt.close()

    def run(self, n, lr_num, ofile):
        e_num = 0
        for i in range(self.number_epochs):
            self.train_network()
            self.validate_network()
            if self.epochs[e_num] == i:
                self.test_network(ofile, e_num)
        self.plot_rmse_performance(n, lr_num)

def main():
    outfile = open('results.txt', 'w')
    d = Data('spambase.data')

    layers = [ [10],
               [30],
               [65],
               [58],
               [100],
               [10, 10],
               [60, 40],
               [30, 30],
               [58, 58],
               [58, 58, 58],
               [60, 70, 80],
               [30, 40, 30],
               [60, 100, 60],
               [15, 15, 15],
               [10, 10, 10, 10],
               [60, 70, 70, 60],
               [60, 100, 40, 30],
               [60, 40, 30, 50],
               [10, 10, 10, 10, 10],
               [60, 60, 60, 60, 60],
               [60, 70, 100, 70, 60]]

    # epochs = [6, 10, 30, 60, 200, 650, 1200, 2000, 5000, 10000]
    e = 5000
    lrate = [0.01, 0.1, 0.25, 0.4, 0.65, 0.8, 0.9, 1.0]

    for i, l in enumerate(layers):
        # for j, e in enumerate(epochs):
        for k, lr in enumerate(lrate):
            n = i * len(lrate) + k
            print('Problem %d' % n, file=outfile)
            print('Testing Layer %d and lrate %.2f' % (n, lr))
            try:
                BackProp(d, lr, e, len(l), l).run(n, len(lrate), outfile)
            except OverflowError:
                print('Overflow in layers combination %d, epochs = %d, and l. rate = %.2f' % (i, e, k), file=outfile)
            except:
                print('Some other error')
                raise
    outfile.close()


main()

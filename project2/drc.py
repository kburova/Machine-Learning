# CS 425. Project 2. Dimensionality Reduction and Clustering
# Written by: Ksenia Burova
#

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class dr:
    data = []  # each row has 2016 death rates
    countries = []  # all countries that have no missing data
    years = []
    f = 216   # num of features (dimensions)
    s = np.array(())
    V = np.array(())

    # reduced data
    Z = np.array(())

    def __init__(self, filename):
        for i, line in enumerate(open(filename, 'r').readlines()):
            values = line.strip().split(',')
            clean_list = list(filter(bool, values))
            if i == 0:
                self.years = list(map(int, clean_list[1:]))
            elif len(clean_list) != self.f+1:
                continue
            else:
                self.data.append(list(map(float, clean_list[1:])))
                self.countries.append(values[0])

    # factoring data and extracting singular values
    # s has singular values

    def pca(self, k):
        U, self.s, self.V = np.linalg.svd(np.array(self.data))

        # Calculate proportion of varience
        p = []
        sum_sv = sum(self.s**2)
        for i, x in enumerate(self.s):
            p.append( sum( self.s[:i+1]**2 ) / sum_sv)

        plt.figure(1)
        plt.subplot(211)
        plt.plot(self.s, marker='x', linestyle='-', color='r', markersize=4)
        plt.ylabel('Singular Values')
        plt.subplot(212)
        plt.plot(p, marker='x', linestyle='-', color='b', markersize=4)
        plt.ylabel('Proportion Of Variance')
        plt.xlabel('Singular Values #')
        plt.savefig('scree.png')

        self.reduce(k)

    # reduce pca
    # mxk matrix = data * V = mxn * nxk (k columns of V)

    def reduce(self, k):
        self.V = self.V[:, :k]
        self.Z = np.matmul(self.data, self.V)

        plt.figure(2)
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.scatter(self.Z[:, 0], self.Z[:, 1], color='purple', marker='*')
        plt.savefig('scatterPCs.png')

# K-means is fast O(#iter * #clusters * #instences * #dimentions)
def kmean(k, data):
    m = []
    num_of_iter = 0
    MAX = 10000
    randi = np.random.randint(0, len(data), size=k)

    # choose random data points to be k means
    for i in randi:
        m.append(data[i])

    # iterate until converged or too many iterations
    while num_of_iter != MAX:
        clusters = [[] for i in range(k)]
        b = list(range(len(data)))
        m_new = []

        # find nearest m
        for i, x in enumerate(data):
            dist = []
            for j in m:
                dist.append(np.linalg.norm(np.array(x) - np.array(j)))
            label = dist.index(min(dist))
            clusters[label].append(x)
            b[i] = label

        # compute new m
        for c in clusters:
            if len(c) == 0:
                m_new.append(data[np.random.randint(0, len(data), size=1)[0]])
            else:
                m_new.append(list(np.mean(c, axis=0)))

        num_of_iter += 1

        if np.array_equal(m, np.array(m_new)):
            break

        m = m_new

    # plot cluster
    colors = ['pink', 'purple', 'orange', 'yellow', 'green', 'blue', 'red']

    plt.figure(3)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')

    for i, d in enumerate(data):
        plt.scatter(d[0], d[1], color=colors[b[i]], marker='*')

    plt.savefig('Clusters.png')


    if num_of_iter == MAX:
        print("Too many iterations", num_of_iter)
    else:
        print("Converged", num_of_iter)

# Creating instance of the class and calling functions...
# --------------------------------------------------------

d = dr('under5mortalityper1000.csv')
d.pca(2)

kmean(6, d.Z)

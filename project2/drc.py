# CS 425. Project 2. Dimensionality Reduction and Clustering
# Written by: Ksenia Burova
#

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st
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
        sum_sv = sum(self.s)
        for i, x in enumerate(self.s):
            p.append(sum( self.s[:i+1] ) / sum_sv)

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
        self.V.resize(len(self.V), k)
        self.Z = np.matmul(self.data, self.V)

        plt.figure(2)
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.scatter(self.Z[:, 0], self.Z[:, 1], color='purple', marker='*')
        plt.savefig('scatterPCs.png')
        plt.show()

#k-means clustering
# 1.place centroids c1...ck at random locations
# 2. Repeat until convergence:
#   a. for each point xi
#       * find nearest centroid cj (compute distance for every centroid cj, arg minj D(xi,cj)  )
#       * assign point to cluster
#   b. for each cluster j = 1...K (each centroid)  cj(a) = 1/n sum'xi->cj' ( xi(a) ), for a=1..d
#       * recomputes its position: new cj = mean of all points xi assigned to cluster j in previous step
# 3. Stop when none of the cluster assignments change
# K-means is fast O(#iter * #clusters * #instences * #dimentions)
def kmean(k, data):
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink']
    m = [] #centroids
    m_gmeans = []
    num_of_iter = 0
    MAX = 10000
    randi = np.random.randint(0, len(data), size=k)

    # choose random data points to be k centroids
    for i in randi:
        m.append(data[i])
        m_gmeans.append(st.gmean(data[i]))

    plt.figure(3)
    mt = np.transpose(np.array(m))
    plt.scatter(mt[: ,0], mt[:, 1], color='pink', marker='o')
    plt.savefig('kmeans.png')

    # iterate until converged ot too many iterations
    while num_of_iter != MAX:
        clusters = [[] for i in range(k)]
        m_new = []
        # find nearest centroid for each datapoint
        for x in data:
            dist = []
            for j in m:
                dist.append(np.linalg.norm( np.array(x) - np.array(j)))
            label = dist.index(min(dist))
            clusters[label].append(x)

        # compute new m
        for c in clusters:
            if len(c) == 0:
                m_new.append(data[np.random.randint(0, len(data), size=1)[0]])
            m_new.append(list(np.mean(c, axis=0)))

        num_of_iter += 1

        print(m)
        print(m_new)
        if np.array_equal(m, np.array(m_new)):
            break

        m = m_new

        plt.figure(3+1+num_of_iter)
        mt = np.transpose(np.array(m))
        plt.scatter(mt[:, 0], mt[:, 1], color='pink', marker='o')
        name = 'km' + str(num_of_iter)
        plt.savefig(name)

    if num_of_iter == MAX:
        print("Too many iterations", num_of_iter)

    else:
        print("Converged", num_of_iter)

# Creating instance of the class and calling functions...
# --------------------------------------------------------

d = dr('under5mortalityper1000.csv')
d.pca(2)

print(len(d.Z))
print(len(d.Z[0]))
kmean(6, d.Z)

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

    # 2pcs-reduced data
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

        plt.clf()
        plt.xlabel("Years")
        plt.ylabel("Death # per 1000")
        cmap = plt.cm.get_cmap('hsv', 180)
        for i in range(0, len(self.countries)):
            plt.plot(self.years, self.data[len(self.data)-i-1], color=cmap(i), linewidth=0.5 )
        plt.savefig('data.png')
        self.reduce(k)

    # reduce pca
    # mxk matrix = data * V = mxn * nxk (k columns of V)

    def reduce(self, k):
        self.V = self.V[:, :k]
        self.Z = np.matmul(self.data, self.V)

        plt.figure(2)
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.scatter(self.Z[:, 0], self.Z[:, 1], color='red', marker='*')
        plt.savefig('scatter_pcs1.png')
        for i, c in enumerate(self.countries):
            plt.annotate(c, (self.Z[i, 0], self.Z[i, 1]), fontsize=7)
        plt.savefig('scatter_pcs.png')

        # draw PCs vs. year
        plt.clf()
        plt.xlabel('Year')
        plt.ylabel('Eigen vectors')
        plt.scatter(self.years, self.V[:, 0], color='teal', marker="s")
        plt.scatter(self.years, self.V[:, 1], color='plum', marker="s")
        plt.savefig('pcs_year.png')

# K-means is fast O(#iter * #clusters * #instences * #dimentions)

def kmean(k, data, key, z):
    # plot cluster
    colors = ['pink', 'purple', 'orange', 'gold', 'green', 'blue', 'red']
    clusters = []
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

    # plt.figure(3)
    # plt.clf()
    # plt.xlabel('PC 1')
    # plt.ylabel('PC 2')

    # color the scatter plot by clusters
    for i in range(len(z)):
        plt.scatter(z[i, 0], z[i, 1], color=colors[b[i]], marker='*')

    # filename = 'clusters_'+key+str(k)+'.png'
    # plt.savefig(filename)

    if num_of_iter == MAX:
        print("Too many iterations", num_of_iter)
    else:
        print("Converged", num_of_iter)

    # get dunn index
    return [dunn_index(clusters), num_of_iter]


def dunn_index(clusters):
    intra = []
    inter = []

    for i, c in enumerate(clusters):
        # calculate intra distance within each cluster
        if len(c) == 1:
            intra.append(0)
            continue
        dist = []
        x1 = 0
        while x1 < (len(c)-1):
            x2 = x1+1
            while x2 < len(c):
                dist.append(np.linalg.norm(np.array(c[x1]) - np.array(c[x2])))
                x2 += 1
            x1 += 1
        intra.append(max(dist))

        # calculate inter distance within all clusters
        for j in range(i+1, len(clusters)):
            mean1 = list(np.mean(clusters[i], axis=0))
            mean2 = list(np.mean(clusters[j], axis=0))
            inter.append(np.linalg.norm(np.array(mean1) - np.array(mean2)))

    return min(inter)/max(intra)


def run_experiment():

    # initialize data
    d = dr('under5mortalityper1000.csv')
    d.pca(2)
    r = np.matmul(d.data, d.V)
    raw = []
    pc2 = []
    pcs = []
    ind = range(2,8)

    plt.figure(3, figsize=(10, 15))
    plt.clf()
    plt.title("Raw data")
    for k in range(2,8):
        plt.subplot(320+k-1)
        raw.append(kmean(k, d.data, 'raw', d.Z))
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.title(str(k) + 'clusters')
    filename = 'clusters_raw.png'
    plt.savefig(filename)

    plt.clf()
    plt.title("2 PCs data")
    for k in range(2, 8):
        plt.subplot(320 + k - 1)
        pc2.append(kmean(k, d.Z, '2pcs', d.Z))
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.title(str(k) + 'clusters')
    filename = 'clusters_2pcs.png'
    plt.savefig(filename)

    plt.clf()
    plt.title("k PCs data")
    for k in range(2, 8):
        plt.subplot(320 + k - 1)
        pcs.append(kmean(k, r, 'pcs', d.Z))
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.title(str(k) + 'clusters')
    filename = 'clusters_kpcs.png'
    plt.savefig(filename)

    plt.clf()
    plt.subplot(211)
    plt.plot(ind, np.array(raw)[:, 0], marker='+', linestyle='-', color='blue', markersize=4)
    plt.ylabel('Dunn Index')

    plt.subplot(212)
    plt.plot(ind, np.array(raw)[:, 1], marker='+', linestyle='-', color='green', markersize=4)
    plt.ylabel('Number of Iterations')
    plt.xlabel('Number of Clusters')

    plt.savefig('dun_raw.png')

    plt.clf()
    plt.subplot(211)
    plt.plot(ind, np.array(pc2)[:, 0], marker='+', linestyle='-', color='blue', markersize=4)
    plt.ylabel('Dunn Index')

    plt.subplot(212)
    plt.plot(ind, np.array(pc2)[:, 1], marker='+', linestyle='-', color='green', markersize=4)
    plt.ylabel('Number of Iterations')
    plt.xlabel('Number of Clusters')

    plt.savefig('dun_pc2.png')

    plt.clf()
    plt.subplot(211)
    plt.plot(ind, np.array(pcs)[:, 0], marker='+', linestyle='-', color='blue', markersize=4)
    plt.ylabel('Dunn Index')

    plt.subplot(212)
    plt.plot(ind, np.array(pcs)[:, 1], marker='+', linestyle='-', color='green', markersize=4)
    plt.ylabel('Number of Iterations')
    plt.xlabel('Number of Clusters')

    plt.savefig('dun_kpcs.png')
# Creating instance of the class and calling functions...
# --------------------------------------------------------
run_experiment()
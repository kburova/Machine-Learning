# CS 425. Project 3.
# Written by: Ksenia Burova
#
# K-nearest neighbors and decision trees

import math
import numpy as np
from operator import itemgetter

import sys
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# handle data
class Data:
    cancerData = []
    features = []
    labels = []
    pc = 0
    comp = '_0'
    argv = ''
    t = ''
    trainFeatures = []
    testFeatures = []
    validFeatures = []
    trainLabels = []
    testLabels = []
    validLabels = []
    Z = []
    def __init__(self, filename):
        for line in open(filename, 'r').readlines():
            values = line.strip().split(',')
            if '?' not in values:
                self.cancerData.append(list(map(int, values[1:])))

        self.features = np.array(self.cancerData)[:, :9]
        self.labels = np.array(self.cancerData)[:, 9]

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

        print(len(self.testLabels), len(self.validLabels), len(self.trainLabels))

# similarity
def GetDistance(set1, set2):
    s1 = np.array(set1)
    s2 = np.array(set2)
    return np.sqrt(((s1-s2)**2).sum())


# neighbors
def GetNeighbors(k, trainF, test):
    distance = []
    for i in range(len(trainF)):
        dist = GetDistance(test, trainF[i])
        distance.append([dist, int(i)])
    distance.sort(key=itemgetter(0))
    # return indexes of nearest neighbors
    n = list(map(int, np.array(distance)[:k, 1]))
    return n


# response
def Classify(trainL, neighbors):
    benign = 0
    malignant = 0

    for i in neighbors:
        if int(trainL[i]) is 2:
            benign += 1
        else:
            malignant += 1

    if benign > malignant:
        return 2
    else:
        return 4


# check accuracy
def getPerformance(testL, prediction):
    confMatrix = [[0, 0], [0, 0]]
    for i, x in enumerate(testL):
        if int(x) is int(prediction[i]):
            if int(x) == 2:
                confMatrix[0][0] += 1
            else:
                confMatrix[1][1] += 1
        else:
            if int(x) == 2:
                confMatrix[0][1] += 1
            else:
                confMatrix[1][0] += 1
    accuracy = (confMatrix[0][0] + confMatrix[1][1])/(sum(map(sum, confMatrix)))
    if (confMatrix[1][1] + confMatrix[1][0]) == 0:
        TPR = 0
    else:
        TPR = confMatrix[1][1]/(confMatrix[1][1] + confMatrix[1][0])
    if (confMatrix[1][1] + confMatrix[0][1]) == 0:
        PPV = 0
    else:
        PPV = confMatrix[1][1]/(confMatrix[1][1] + confMatrix[0][1])
    if (confMatrix[0][0] + confMatrix[0][1]) == 0:
        TNR = 0
    else:
        TNR = confMatrix[0][0]/(confMatrix[0][0] + confMatrix[0][1])
    if (PPV + TPR) == 0:
        FScore = 0
    else:
        FScore = PPV * TPR / (PPV + TPR)

    return confMatrix, accuracy, TPR, PPV, TNR, FScore


def kNN(k, trFeatures, trLabels, tstFeatures, tstLabels):
    prediction = []

    for t in tstFeatures:
        neighbors = GetNeighbors(k, trFeatures, t)
        prediction.append(Classify(trLabels, neighbors))
    return getPerformance(tstLabels, prediction)


# ..............Decision Trees...............

class Node:
    def __init__(self, attribute, value, left, right, depth):
        self.attribute = attribute
        self.depth = depth
        self.value = value
        self.left = left
        self.right = right
        self.isLeaf = False


class Leaf:
    def __init__(self, labels, depth):
        self.label = self.getClass(labels)
        self.depth = depth
        self.isLeaf = True

    def getClass(self, labels):
        l1 = (np.array(labels) == 2).sum()
        l2 = len(labels) - l1
        if l1 > l2:
            return 2
        else:
            return 4


# Impurity method 1
def entropy(p):
    if p is 0 or p is 1:
        return 0
    elif p is 0.5:
        return 1
    else:
        return -1 * p * np.log2(p) - (1 - p) * np.log2((1 - p))


# Impurity method 2
def gini(p):
    return 2*p*(1-p)


# Impurity method 3
def error(p):
    return 1 - np.max([p, 1 - p])


def NodeImpurity(impurity_f, trainLabels):
    p = float((np.array(trainLabels) == 2).sum()) / len(trainLabels)
    return impurity_f(p)


def split(dataFeatures, dataLabels, index, value):
    lF = []
    rF = []
    lL = []
    rL = []
    for i, d in enumerate(dataFeatures):
        if int(d[index]) < value:
            lF.append(d)
            lL.append(dataLabels[i])
        else:
            rF.append(d)
            rL.append(dataLabels[i])
    return lF, rF, lL, rL


def GenerateTreeThreshold(trainFeatures, trainLabels, threshold, depth, impurity_f):

    gain, attribute = SplitAttribute(trainFeatures, trainLabels, impurity_f)
    if gain == 0 or NodeImpurity(impurity_f, trainLabels) < threshold:
        return Leaf(trainLabels, depth)

    leftF, rightF, leftL, rightL = split(trainFeatures, trainLabels, attribute[0], attribute[1])

    leftNode = GenerateTreeThreshold(leftF, leftL, threshold, depth+1, impurity_f)
    rightNode = GenerateTreeThreshold(rightF, rightL, threshold, depth+1, impurity_f)

    return Node(attribute[0], attribute[1], leftNode, rightNode, max(leftNode.depth, rightNode.depth))


def GenerateTreeDepth(trainFeatures, trainLabels, k, depth, impurity_f):

    gain, attribute = SplitAttribute(trainFeatures, trainLabels, impurity_f)

    if gain == 0 or depth == k:
        return Leaf(trainLabels, depth)

    leftF, rightF, leftL, rightL = split(trainFeatures, trainLabels, attribute[0], attribute[1])
    leftNode = GenerateTreeDepth(leftF, leftL, k, depth+1, impurity_f)
    rightNode = GenerateTreeDepth(rightF, rightL, k, depth+1, impurity_f)

    return Node(attribute[0], attribute[1], leftNode, rightNode, max(leftNode.depth, rightNode.depth))


def SplitAttribute(trainFeatures, trainLabels, impurity_f):
    cur_ent = NodeImpurity(impurity_f, trainLabels)
    gain = 0
    bestf = []

    for fIndex in range(len(trainFeatures[0])):
        attrVals = np.unique([dataPoint[fIndex] for dataPoint in trainFeatures])
        for val in attrVals:
            leftF, rightF, leftL, rightL = split(trainFeatures, trainLabels, fIndex, val)
            if len(leftL) == 0 or len(rightL) == 0:
                continue
            imp = SplitImpurity(leftL, rightL, cur_ent,impurity_f)
            if imp >= gain:
                gain = imp
                bestf = [fIndex, val]

    return gain, bestf


def SplitImpurity(leftL, rightL, cur_ent, impurity_f):
    p1 = float(len(leftL)) / float(len(leftL) + len(rightL))
    p2 = 1 - p1
    return cur_ent - (p1 * NodeImpurity(impurity_f, leftL) + p2 * NodeImpurity(impurity_f, rightL))


def Predict(tree, testFeatures, testLabels):
    print("Tree Depth: ", tree.depth)
    colors = ['orange', 'blue', 'green']
    predictions = []
    for f in testFeatures:
        predictions.append(ClassifyWithTree(f, tree))

    if d.pc == 2 and (testLabels == d.testLabels).all:
        plt.figure(4)
        plt.clf()
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')

        # color the scatter plot by clusters

        for i in range(len(testFeatures)):
            plt.scatter(np.array(testFeatures)[i, 0], np.array(testFeatures)[i, 1], color=colors[predictions[i]-2], marker='*')
        filename = 'images/clusters_'+d.argv+'_'+ d.t +'_.png'
        plt.savefig(filename)

    return getPerformance(testLabels, predictions)

def ClassifyWithTree(sample, node):
    if node.isLeaf is True:
        return node.label

    if sample[node.attribute] < node.value:
        return ClassifyWithTree(sample, node.left)
    else:
        return ClassifyWithTree(sample, node.right)


# ......................Runs...............................

def runKNN():
    performance = []
    tpr = []
    ppv = []
    tnr = []
    fscore = []

    K = [2, 3, 4, 5, 6, 7, 8, 16, 32]

    for k in K:
        cMatrix, accuracy, TPR, PPV, TNR, FScore = kNN(k, d.trainFeatures, d.trainLabels, d.validFeatures, d.validLabels)
        print(cMatrix, accuracy, TPR, PPV, TNR, FScore)
        performance.append(accuracy)
        tpr.append(TPR)
        ppv.append(PPV)
        tnr.append(TNR)
        fscore.append(FScore)

    plotKNN(K, tpr, ppv, tnr, fscore, performance, 'validation_KNN')

    bestK = []
    for i, p in enumerate(performance):
        if p == max(performance):
            bestK.append(K[i])

    print("Best K: ", bestK)

    # run kNN on best chosen K
    for k in bestK:
        cMatrix, accuracy, TPR, PPV, TNR, FScore = kNN(k, d.trainFeatures, d.trainLabels, d.testFeatures, d.testLabels)
        print(cMatrix, accuracy, TPR, PPV, TNR, FScore)


def runDT(type, function):
    performance = []
    tpr = []
    ppv = []
    tnr = []
    fscore = []

    K = [2, 3, 4, 5, 6, 7, 8, 16, 32]
    T = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    if str(type) == 'depth' or str(type) == 'both':
        print('Depth DT')
        d.t = 'Depth'
        for k in K:
            cMatrix, accuracy, TPR, PPV, TNR, FScore = Predict(GenerateTreeDepth(d.trainFeatures, d.trainLabels, k, 0, function), d.validFeatures, d.validLabels)
            print(cMatrix, accuracy, TPR, PPV, TNR, FScore)
            performance.append(accuracy)
            tpr.append(TPR)
            ppv.append(PPV)
            tnr.append(TNR)
            fscore.append(FScore)

        plotKNN(K, tpr, ppv, tnr, fscore, performance, 'validation_DT_Depth_'+d.argv + d.comp)
        bestK = []

        for i, p in enumerate(performance):
            if p == max(performance):
                bestK.append(K[i])

        print("Best K: ", bestK)

        for k in bestK:
            cMatrix, accuracy, TPR, PPV, TNR, FScore = Predict(
                GenerateTreeDepth(d.trainFeatures, d.trainLabels, k, 0, function), d.testFeatures, d.testLabels)
            print(cMatrix, accuracy, TPR, PPV, TNR, FScore)

    performance = []
    tpr = []
    ppv = []
    tnr = []
    fscore = []

    if str(type) == 'threshold' or str(type) == 'both':
        print('TreshDT')
        d.t = 'Thr'
        for t in T:
            cMatrix, accuracy, TPR, PPV, TNR, FScore = Predict(GenerateTreeThreshold(d.trainFeatures, d.trainLabels, t, 0, function), d.validFeatures, d.validLabels)
            print(cMatrix, accuracy, TPR, PPV, TNR, FScore)
            performance.append(accuracy)
            tpr.append(TPR)
            ppv.append(PPV)
            tnr.append(TNR)
            fscore.append(FScore)

        plotKNN(T, tpr, ppv, tnr, fscore, performance, 'validation_DT_Threshold_'+d.argv + d.comp)

        bestT = []
        for i, p in enumerate(performance):
            if p == max(performance):
                bestT.append(T[i])

        print("Best Theshold: ", bestT)

        for t in bestT:
            cMatrix, accuracy, TPR, PPV, TNR, FScore = Predict(GenerateTreeThreshold(d.trainFeatures, d.trainLabels, t, 0, function), d.validFeatures, d.validLabels)
            print(cMatrix, accuracy, TPR, PPV, TNR, FScore)


def plotKNN(K, tpr, ppv, tnr, fscore, performance, type):
    plt.figure(1)
    plt.title('Accuracy vs. k')
    plt.xlabel('k value')
    plt.ylabel('Accuracy')
    plt.xticks(K)
    plt.plot(K, performance, color='purple', marker='*')
    plt.savefig('images/accuracy_'+type+'.png')
    plt.clf()

    plt.figure(2, figsize=(11.5, 8))

    plt.subplot(221)
    plt.xticks(K)
    plt.plot(K, tpr, color='green', marker='*')
    plt.ylabel('Sensitivity (TPR)')
    plt.xlabel('k value')

    plt.subplot(222)
    plt.xticks(K)
    plt.plot(K, ppv, color='blue', marker='*')
    plt.ylabel('Precision (PPV) ')
    plt.xlabel('k value')

    plt.subplot(223)
    plt.xticks(K)
    plt.plot(K, tnr, color='magenta', marker='*')
    plt.ylabel('Specificity (TNR)')
    plt.xlabel('k value')

    plt.subplot(224)
    plt.xticks(K)
    plt.plot(K, fscore, color='orange', marker='*')
    plt.ylabel('F Score')
    plt.xlabel('k value')
    plt.savefig('images/metrics_'+type+'.png')
    plt.clf()


def pca():
    U, s, V = np.linalg.svd(np.array(d.features))

    for k in [2, 4, 6]:
        d.pc = k
        d.comp = '_' + str(k)
        reduce(k, V)

        d.trainFeatures, d.testFeatures, d.trainLabels, d.testLabels = train_test_split(
            d.Z, d.labels, test_size=0.5, random_state=26)
        d.trainLabels = d.trainLabels.ravel()
        d.testLabels = d.testLabels.ravel()

        d.validFeatures, d.testFeatures, d.validLabels, d.testLabels = train_test_split(
            d.testFeatures, d.testLabels, test_size=0.5, random_state=26)
        d.testLabels = d.testLabels.ravel()
        d.validLabels = d.validLabels.ravel()

        d.argv = 'entropy'
        runDT('both', entropy)
        d.argv = 'gini'
        runDT('both', gini)
        d.argv = 'error'
        runDT('both', error)


def reduce(k, V):
    v = V[:, :k]
    d.Z = np.matmul(d.features, v)


# .................RUN ..................
d = Data('breast-cancer-wisconsin.data')

runKNN()
t = sys.argv[1]
if sys.argv[2] is 'entropy':
    func = entropy
    d.argv = 'entropy'
elif sys.argv[2] is 'gini':
    func = gini
    d.argv = 'gini'
else:
    func = error
    d.argv = 'error'

runDT(t, func)
pca()
# CS 425. Project 3.
# Written by: Ksenia Burova
#
# K-nearest neighbors and decision trees

import math
import numpy as np
from operator import itemgetter
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# handle data
class Data:
    cancerData = []
    features = []
    labels = []

    trainFeatures = []
    testFeatures = []
    validFeatures = []
    trainLabels = []
    testLabels = []
    validLabels = []

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
            if int(x) is 2:
                confMatrix[0][0] += 1
            else:
                confMatrix[1][1] += 1
        else:
            if int(x) is 2:
                confMatrix[0][1] += 1
            else:
                confMatrix[1][0] += 1

    accuracy = (confMatrix[0][0] + confMatrix[1][1])/(sum(map(sum, confMatrix)))
    TPR = confMatrix[1][1]/(confMatrix[1][1] + confMatrix[1][0])
    PPV = confMatrix[1][1]/(confMatrix[1][1] + confMatrix[0][1])
    TNR = confMatrix[0][0]/(confMatrix[0][0] + confMatrix[0][1])
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
    def __init__(self, attribute, value, left, right):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.isLeaf = False


class Leaf:
    def __init__(self, labels):
        self.label = self.getClass(labels)
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


def NodeEntropy(type, trainLabels):
    p = float( (np.array(trainLabels) == 2).sum()) / len(trainLabels)
    if int(type) is 1:
        return entropy(p)
    elif int(type) is 2:
        return gini(p)
    else:
        return error(p)


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


def GenerateTree(trainFeatures, trainLabels, k, depth):

    gain, attribute = SplitAttribute(trainFeatures, trainLabels)
    if gain ==0 or NodeEntropy(impurityType, trainLabels) < k:
        return Leaf(trainLabels)

    leftF, rightF, leftL, rightL = split(trainFeatures, trainLabels, attribute[0], attribute[1])

    leftNode = GenerateTree(leftF, leftL, k, depth+1)
    rightNode = GenerateTree(rightF, rightL, k, depth+1)

    return Node(attribute[0], attribute[1], leftNode, rightNode)


def GenerateTreeDepth(trainFeatures, trainLabels, k, depth):

    gain, attribute = SplitAttribute(trainFeatures, trainLabels)
    if gain==0 or depth == k:
        return Leaf(trainLabels)

    leftF, rightF, leftL, rightL = split(trainFeatures, trainLabels, attribute[0], attribute[1])
    leftNode = GenerateTreeDepth(leftF, leftL, k, depth+1)
    rightNode = GenerateTreeDepth(rightF, rightL, k, depth+1)

    return Node(attribute[0], attribute[1], leftNode, rightNode)


def SplitAttribute(trainFeatures, trainLabels):
    cur_ent = NodeEntropy(impurityType, trainLabels)
    gain = 0
    bestf = [0, 0]

    for fIndex in range(len(trainFeatures[0])):
        attrVals = np.unique([dataPoint[fIndex] for dataPoint in trainFeatures])
        for val in attrVals:
            leftF, rightF, leftL, rightL = split(trainFeatures, trainLabels, fIndex, val)

            if len(leftL) == 0 or len(rightL) == 0:
                continue

            e = SplitEntropy(leftL, rightL, cur_ent)
            if e >= gain:
                gain = e
                bestf = [fIndex, val]


    return gain, bestf


def SplitEntropy(leftL, rightL, cur_ent):

    p1 = float(len(leftL)) / float(len(leftL) + len(rightL))
    p2 = 1 - p1

    return cur_ent - (p1*NodeEntropy(impurityType, leftL) + p2*NodeEntropy(impurityType, rightL))


def Predict(tree, testFeatures, testLabels):
    predictions = []
    for f in testFeatures:
        predictions.append(ClassifyWithTree(f, tree))

    confMatrix = [[0, 0], [0, 0]]
    for i, x in enumerate(testLabels):
        if int(x) is int(predictions[i]):
            if int(x) is 2:
                confMatrix[0][0] += 1
            else:
                confMatrix[1][1] += 1
        else:
            if int(x) is 2:
                confMatrix[0][1] += 1
            else:
                confMatrix[1][0] += 1

    accuracy = (confMatrix[0][0] + confMatrix[1][1]) / (sum(map(sum, confMatrix)))
    print(accuracy)


def ClassifyWithTree(sample, node):
    if node.isLeaf is True:
        return node.label

    if sample[node.attribute] < node.value:
        return ClassifyWithTree(sample, node.left)
    else:
        return ClassifyWithTree(sample, node.right)



# .............. Main Call ..................

d = Data('breast-cancer-wisconsin.data')
performance = []
tpr = []
ppv = []
tnr = []
fscore = []

K = [2, 3, 4, 5, 6, 7, 8, 16, 32]

# split training set into into training and validation

for k in K:
    cMatrix, accuracy, TPR, PPV, TNR, FScore = kNN(k, d.trainFeatures, d.trainLabels, d.validFeatures, d.validLabels)
    print(cMatrix, accuracy, TPR, PPV, TNR, FScore)
    performance.append(accuracy)
    tpr.append(TPR)
    ppv.append(PPV)
    tnr.append(TNR)
    fscore.append(FScore)

plt.figure(1)
plt.title('Accuracy vs. k')
plt.xlabel('k value')
plt.ylabel('Accuracy')
plt.xticks(K)
plt.plot(K, performance, color='purple', marker='*')
plt.savefig('images/accuracy.png')

plt.figure(2,figsize=(11.5, 8))

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
plt.savefig('images/metrics.png')

bestK = []
for i, p in enumerate(performance):
    if p == max(performance):
        bestK.append(K[i])

print("Best K: ", bestK)

# run kNN on best chosen K
for k in bestK:
    cMatrix, accuracy, TPR, PPV, TNR, FScore = kNN(k, d.trainFeatures, d.trainLabels, d.testFeatures, d.testLabels)
    print(cMatrix, accuracy, TPR, PPV, TNR, FScore)

impurityType = 0

for k in K:
    Predict(GenerateTreeDepth(d.trainFeatures, d.trainLabels, k, 0), d.validFeatures, d.validLabels)

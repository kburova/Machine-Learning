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
    trainLabels = []
    testLabels = []

    def __init__(self, filename):
        for line in open(filename, 'r').readlines():
            values = line.strip().split(',')
            if '?' not in values:
                self.cancerData.append(list(map(int, values[1:])))

        self.features = np.array(self.cancerData)[:, :9]
        self.labels = np.array(self.cancerData)[:, 9]

        # split data into training/validation and testing data for now using 75/25 rule
        # make sure that tum type arrays are 1-dimensional
        self.trainFeatures, self.testFeatures, self.trainLabels, self.testLabels = train_test_split(
            self.features, self.labels, test_size=0.3, random_state = 100)
        self.trainLabels = self.trainLabels.ravel()
        self.testLabels = self.testLabels.ravel()


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


# .............. Main Call ..................

d = Data('breast-cancer-wisconsin.data')
performance = []
tpr = []
ppv = []
tnr = []
fscore = []

K = [2, 3, 4, 5, 6, 7, 8, 16, 32]

# split training set into into training and validation
trainFeatures, validFeatures, trainLabels, validLabels = train_test_split(
    d.trainFeatures, d.trainLabels, test_size=0.3, random_state=100)
trainLabels = trainLabels.ravel()
validLabels = validLabels.ravel()

for k in K:
    cMatrix, accuracy, TPR, PPV, TNR, FScore = kNN(k, trainFeatures, trainLabels, validFeatures, validLabels)
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

plt.figure(2,figsize=(11, 8))

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


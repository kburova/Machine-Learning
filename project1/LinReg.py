# CS 425. Project 1. Multiple Regression Algorithm
# Written by: Ksenia Burova
#
# The Goal is to calculate 'w' vector
import math
import numpy as np


class MultRegression:
    X = []     # Data, where each line has all params for one r_i (x1^i, x2^i....xn^i)
    XT = []    # Transpose Data params, where each line has one parameter type values for all r's (x1^1, x1^2, x1^3...x1^i)

    stdD = []  # Standard deviations
    M = []     # Means
    w = []     # weights
    r = []     # function output (mpg)
    names = []
    missingLines = []
    # Read all the data from a file, split all into r and  x1, x2, x3 ...xn
    #   r   1. mpg:           continuous
    #   x1  2. cylinders:     multi-valued discrete
    #   x2  3. displacement:  continuous
    #   x3  4. horsepower:    continuous
    #   x4  5. weight:        continuous
    #   x5  6. acceleration:  continuous
    #   x6  7. model year:    multi-valued discrete
    #   x7  8. origin:        multi-valued discrete
    #       9. car name:      string (unique for each instance)
    def __init__(self, filename, toStandardize, toIgnore):
        for line in open(filename, 'r').readlines():
            values = line.split()
            if values[3] == '?':
                self.missingLines.append(line)
            else:
                self.X.append(list(map(float, values[1:8])))
                self.names.append(' '.join(values[8:]).replace('"', ''))
                self.r.append(float(values[0]))
            # self.X.append(list(map(float, values[1:])))
            # self.r.append(float(values[0]))

        self.XT = np.array(self.X).transpose()

        if toIgnore == 0:
            self.mean()
            newVal = self.M[2]   #horsepower mean
            for line in self.missingLines:
                values = line.split()
                values[3] = newVal
                self.X.append(list(map(float, values[1:8])))
                self.names.append(' '.join(values[8:]).replace('"', ''))
                self.r.append(float(values[0]))
            self.XT = np.array(self.X).transpose()

        # Do data standardization if requested
        if toStandardize == 1:
            self.standardize()

    def stdDev(self):
        for i, x in enumerate(self.XT):
            x1 = np.array(x) - self.M[i]   # x_diff = (x - m)
            x1 = [j * j for j in x1]    # x_diff^2
            d = math.sqrt( math.fsum(x1) / len(x1))  # ( sum(x_diff^2) / n)^(0.5)
            self.stdD.append(d)

    def mean(self):
        for x in self.XT:
            m = math.fsum(x)/len(x)
            self.M.append(m)

    def standardize(self):
        stdX = []
        if len(self.M) == 0:
            self.mean()
        self.stdDev()

        for i, x in enumerate(self.XT):
            x1 = (np.array(x) - self.M[i]) / self.stdD[i]
            stdX.append(x1)

        self.XT = np.array(stdX)
        self.X = self.XT.transpose()

    def calcW(self):

        # insert ones as first column in matrix and get equivalent transpose matrix
        self.X = np.insert(self.X, 0, 1, axis=1)
        self.XT = self.X.transpose()

        # follow formula computations here (XT * X)^-1 * XT * r
        XTX = np.matmul(self.XT, self.X)
        Xinv = np.linalg.inv(XTX)
        Xprod = np.matmul(Xinv, self.XT)
        self.w = np.matmul(Xprod, self.r)

        print(self.w)
        print()

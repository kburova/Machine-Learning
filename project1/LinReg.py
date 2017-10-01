# CS 425. Project 1. Multiple Regression Algorithm
# Written by: Ksenia Burova
#
# The Goal is to calculate 'w' vector

from random import uniform
import math
import copy
import numpy as np
import os
import matplotlib.pyplot as plt


class MultRegression:
    X = []     # Data params, where each line has one parameter type values for all r's (x1^1, x1^2, x1^3...x1^i)
    XT = []    # Transpose Data, where each line has all params for one r_i (x1^i, x2^i....xn^i)

    stdX = []  # Standardized data

    stdD = []  # Standard deviations
    M = []     # Means
    w = []     # weights
    r = []     # function output (mpg)
    names = []

    def __init__(self, filename):
        for line in open(filename, 'r').readlines():
            values = line.split()
            if values[3] == '?':
                continue
            else:
                self.XT.append(values[1:8])
                print(values[1:8])
        self.X = np.array(self.XT).transpose()

    # def __init__(self):
    #     print()

    def printMatrix(self,x):
        for i in x:
            print(i)

    def printArray(self,x):
        print(x)

    def stdDev(self):
        for i, x in enumerate(self.X):
            x1 = np.array(x) - self.M[i]   # x_diff = (x - m)
            x1 = [j * j for j in x1]    # x_diff^2
            d = math.sqrt( math.fsum(x1) / len(x1))  # ( sum(x_diff^2) / n)^(0.5)
            self.stdD.append(d)

    def mean(self):
        for x in self.X:
            m = math.fsum(x)/len(x)
            self.M.append(m)

    def standardize(self):
        for i, x in enumerate(self.X):
            x1 = (np.fabs(np.array(x) - self.M[i])) / self.stdD[i]
            self.stdX.append(x1)

# Step I.  Read all the data from a file.
#   r   1. mpg:           continuous
#   x1  2. cylinders:     multi-valued discrete
#   x2  3. displacement:  continuous
#   x3  4. horsepower:    continuous
#   x4  5. weight:        continuous
#   x5  6. acceleration:  continuous
#   x6  7. model year:    multi-valued discrete
#   x7  8. origin:        multi-valued discrete
#       9. car name:      string (unique for each instance)

# Step II. Split all into x1, x2, x3 ...xn

# Step III. Write functions to call matrix manipulations

# Step IV. Data standardization
# Identify the observation (X), the mean (μ) and the standard deviation (σ) in
# the question. Step 2: Plug the values from Step 1 into the formula:
# Standardized value = (X – μ) / σ
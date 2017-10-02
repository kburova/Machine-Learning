# CS 425. Project 1. Multiple Regression Algorithm
# Written by: Ksenia Burova

# Main file:
# Pass filename with data
# Do all the operations

from LinReg import *

mr = MultRegression('auto-mpg.data', 1, 1)

mr.calcW()
import numpy as np

# Functions, which remove special characters from arrays

def removeInf(inputMatrix):

    outputMatrix = inputMatrix

    outputMatrix = np.where(outputMatrix == np.inf, 0, outputMatrix)

    return outputMatrix


def removeNaN(inputMatrix):

    outputMatrix = inputMatrix

    outputMatrix = np.nan_to_num(outputMatrix, 0.0)

    return outputMatrix

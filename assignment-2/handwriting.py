#!/usr/bin/env python2
################################################
# CS434 Machine Learning and Data Mining       #
# Assignment 2                                 #
# Nathan Brahmstadt and Jordan Crane           #
################################################
import math
import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from collections import namedtuple

########## Data Structures ##########

# To check if output is four or nine use labels.four or labels.nine
Labels = namedtuple('Labels', 'four nine')
labels = Labels(0, 1)

########## Main ##########

def main():
    (training_features, training_outputs) = get_data_arrays(
            "data/usps-4-9-train.csv")
    (testing_features, testing_outputs) = get_data_arrays(
            "data/usps-4-9-train.csv")
    batch_gradient_descent(training_features, training_outputs)

########## Data Import ##########

def get_data_arrays(filename):
    file = open(filename)
    (features, outputs) = build_data_arrays(file)
    file.close()
    return (features, outputs)

def build_data_arrays(file):
    (features, outputs) = build_data_lists(file)
    return (np.array(features, dtype=int), np.array(outputs, dtype=int))

def build_data_lists(file):
    (features, outputs) = ([], [])
    for line in file:
        (line_features, line_output) = extract_features_and_output(line)
        features.append(line_features)
        outputs.append(line_output)
    return (features, outputs)

def extract_features_and_output(line):
    features_and_output = map(int, line.split(','))
    return (features_and_output[0:-1], features_and_output[-1])

########## Gradient Descent ##########

def batch_gradient_descent(features, outputs):
    weight = np.zeros_like(features[0])
    learning_rate = 0.01
    epsilon = 0.1
    iterations = 0
    while True:
        weight += learning_rate * update(features, outputs, weight)
        print norm(np.gradient(weight))
        if norm(np.gradient(weight)) < epsilon:
            break

def update(features, outputs, weight):
    return (outputs - sigmoid(weight, features)).dot(features)

def sigmoid(weight, features):
    return 1 / (1 + np.exp(-weight.dot(features.T)))

main()

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
            "data/usps-4-9-test.csv")
    weight = batch_gradient_descent(training_features, training_outputs)
    
    test_weight(training_features, training_outputs, weight)

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
    weight = np.zeros_like(features[0], dtype=float)
    learning_rate = 0.01
    epsilon = 0.1
    iterations = 0
    while True:
        old_norm = norm_of_gradient(weight)
        weight += learning_rate * update(features, outputs, weight)
        new_norm = norm_of_gradient(weight)
        iterations += 1
        if abs(old_norm - new_norm) < epsilon:
            print "Convered at Iteration: " + str(iterations)
            return weight

def norm_of_gradient(weight):
    return norm(np.gradient(weight))            
            
def update(features, outputs, weight):
    return (outputs - sigmoid(weight, features)).dot(features)

def sigmoid(weight, features):
    return 1 / (1 + np.exp(-weight.dot(features.T)))

def test_weight(features, outputs, weight):
        guesses = sigmoid(weight, features)
        right_tally = 0
        wrong_tally = 0
        for i, guess in enumerate(guesses):
            if abs(guess-outputs[i]) <= 0.5:
                right_tally += 1
            else:
                wrong_tally += 1
        acc = float(right_tally) / (right_tally + wrong_tally)
        print "Accuracy: " + str(acc*100) + "%"
        
main()

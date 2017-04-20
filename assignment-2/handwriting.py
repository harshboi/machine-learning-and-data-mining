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
    for i in range(0, 5):

        learning_rate = float(10)/(10**i)
        print "Learning Rate of: " + str(learning_rate)
        weight = batch_gradient_descent(
                training_features,
                training_outputs,
                learning_rate
                )

        train_acc = test_weight(training_features, training_outputs, weight)

        print "Training Data Accuracy: " + str(train_acc*100) + "%"

        test_acc = test_weight(testing_features, testing_outputs, weight)

        print "Testing Data Accuracy: " + str(test_acc*100) + "%"


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

def batch_gradient_descent(features, outputs, learning_rate):
    weight = np.random.rand(features.shape[1]) * 100
    print weight
    #learning_rate = 0.01
    epsilon = 0.0000001
    iterations = 0
    while True:
        d_old = weight
        weight += learning_rate * update(features, outputs, weight)
        d_new = weight
        print d_old - d_new
        iterations += 1
        if norm(d_old - d_new, ord=2) < epsilon:
            print "Convered at Iteration: " + str(iterations)
            return weight

def norm_of_gradient(weight):
    return norm(np.gradient(weight))

def update(features, outputs, weight):
    return (outputs - sigmoid(weight, features)).dot(features)

def sigmoid(weight, features):
    exponents = weight.dot(features.T)
    results = []
    #prevent overflow
    for exponent in exponents:
        if exponent > 0:
            results.append(1 / (1 + math.exp(-exponent)))
        else:
            results.append(1 - 1 / (1 + math.exp(exponent)))
    return np.array(results)

def test_weight(features, outputs, weight):
    guesses = sigmoid(weight, features)
    right_tally = 0
    wrong_tally = 0
    for i, guess in enumerate(guesses):
        if abs(guess-outputs[i]) <= 0.5:
            right_tally += 1
        else:
            wrong_tally += 1
    accuracy = float(right_tally) / (right_tally + wrong_tally)
    return accuracy

main()

#!/usr/bin/env python2
################################################
# CS434 Machine Learning and Data Mining       #
# Assignment 2                                 #
# Nathan Brahmstadt and Jordan Crane           #
################################################
import sys
import math
import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

########## Data Structures ##########

# To check if output is four or nine use labels.four or labels.nine
Data = namedtuple('Data', 'features outputs')
Labels = namedtuple('Labels', 'four nine')
labels = Labels(0, 1)

########## Main ##########

def main():
    training_data = get_data_arrays(
            "data/usps-4-9-train.csv")
    testing_data = get_data_arrays(
            "data/usps-4-9-test.csv")

    #test_fn(training_data, testing_data)

    prob1(training_data, testing_data)

   # prob2(training_data)

   # prob3(training_data, testing_data)

########## Problems ##########

def test_fn(training_data, testing_data):
    learning_rate = 0.01
    weight = batch_gradient_descent(training_data, learning_rate)

def prob1(training_data, testing_data):
    print "***** Problem 1 *****"
    for i in range(5):
        learning_rate = 0.01/(10**i)
        print "\nLearning Rate of: " + str(learning_rate)
        weight = batch_gradient_descent(training_data, learning_rate)

        train_acc = test_weight(training_data, weight)

        print "Training Data Accuracy: " + str(train_acc*100) + "%"

        test_acc = test_weight(testing_data, weight)

        print "Testing Data Accuracy: " + str(test_acc*100) + "%"

def prob2(training_data):
    print "\n***** Problem 2 *****"
    print "....Generating Accuracy Data..."
    final_weight = batch_gradient_descent(training_data, 0.01, True)
    print "Done"

def prob3(training_data, testing_data):
    print "\n***** Problem 3 *****"
    for i in range(7):
        scalar = float(.001)*(10**i)
        learning_rate = 0.1
        print "\nLearning Rate of: " + str(learning_rate)
        print "Using a scaler of : " + str(scalar)
        regularized_weight = batch_gradient_descent(training_data,
                learning_rate, regularization=scalar)
        train_acc = test_weight(training_data, regularized_weight)
        test_acc = test_weight(testing_data, regularized_weight)
        print "Training Data Accuracy with Regularization: " + str(train_acc*100) + "%"
        print "Testing Data Accuracy with Regularization: " + str(test_acc*100) + "%"

########## Data Import ##########

def get_data_arrays(filename):
    file = open(filename)
    (features, outputs) = build_data_arrays(file)
    file.close()
    return Data(features, outputs)

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

def batch_gradient_descent(data, learning_rate, regularization=0):
    weight = np.zeros_like(data.features[0], dtype=float)
    epsilon = 0.1

    iterations = 0

    while True:

        gradient = update(data, weight) + (regularization*weight)

        weight += learning_rate * gradient

        iterations += 1

       # print norm(gradient)

        if norm(gradient) < epsilon or iterations >= 10000:
            print "Convered at Iteration: " + str(iterations)
            return weight

def norm_of_gradient(weight):
    return norm(np.gradient(weight))

def update(data, weight):
    return (data.outputs - sigmoid(weight, data.features)).dot(data.features)

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

def test_weight(data, weight):
    probabilities = sigmoid(weight, data.features)
    guesses = np.round(probabilities)
    m = float(len(data.outputs))
    accuracy = (m - sum(abs(guesses - data.outputs))) / m
    return accuracy

#Really cool function to show the weights
def plot_weight(weight):
    image = plt.imshow(np.reshape(weight, (16, 16)).T)
    plt.show(image)
    while 1:
        pass

main()

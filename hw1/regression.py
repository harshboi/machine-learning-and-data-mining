################################################
# CS434 Machine Learning and Data Mining
# Assignment 1
# Nathan Brahmstadt and Jordan Crane
################################################
import numpy as np
from numpy.linalg import inv
import numpy.random
import random

####### Main ########
def main():
    # Don't print in scientific notation :)
    np.set_printoptions(suppress=True)
    #Setup random number generator
    random.seed(123)

    ### Without Dummy Variables ###
    print_()
    weight = train_without_dummy_variable()
    test_without_dummy_variable(weight)

    ### With Dummy Variables ###
    print_(dummy_variable=True)
    weight = train_with_dummy_variable()
    test_with_dummy_variable(weight)

    ### With Random Features ###
    #For problem 5, add 2, 4, 6, 8, and 10 features with random values
    for number_of_features in range(2, 12, 2):
        print_(random_features=number_of_features)
        weight = train_with_random_features(number_of_features)
        test_with_random_features(weight, number_of_features)

    ### With Scalar Multiplier ###
    #For problem 6, vary the scalar
    scalars_to_test = [.01, .05, .1, .5, 1, 5, 10, 15]
    for scalar in scalars_to_test:
        print_(scalar=scalar)
        weight = train_with_scalar(scalar)
        test_with_scalar(weight)

####### Functions #######
def train_with_dummy_variable(features=None, outputs=None, scalar=0):
    if features is None:
        (features, outputs) = get_data_arrays("data/housing_train.txt")
    features = insert_dummy_variable(features)
    return train(features, outputs, scalar)

def test_with_dummy_variable(weight, features=None, outputs=None, scalar=0):
    if features is None:
        (features, outputs) = get_data_arrays("data/housing_test.txt")
    features = insert_dummy_variable(features)
    test(features, outputs, weight)

def train_without_dummy_variable():
    features, outputs = get_data_arrays("data/housing_train.txt")
    return train(features, outputs)

def test_without_dummy_variable(weight):
    (features, outputs) = get_data_arrays("data/housing_test.txt")
    test(features, outputs, weight)

def train_with_random_features(number_of_features):
    (features, outputs) = get_data_arrays("data/housing_train.txt")
    features = insert_random_features(features, number_of_features)
    return train_with_dummy_variable(features=features, outputs=outputs)

def test_with_random_features(weight, number_of_features):
    (features, outputs) = get_data_arrays("data/housing_test.txt")
    features = insert_random_features(features, number_of_features)
    return test_with_dummy_variable(weight, features=features, outputs=outputs)

def train_with_scalar(scalar):
    return train_with_dummy_variable(scalar=scalar)

def test_with_scalar(weight):
    test_with_dummy_variable(weight)

def train(features, outputs, scalar=0):
    weight = calculate_weight_vector(features, outputs, scalar)
    print("Weight Vector:\n", weight)
    print("Training SSE: ", calculate_sse(features, outputs, weight))
    return weight

def get_data_arrays(filename):
    file = open(filename, 'r')
    (features, outputs) = build_data_arrays(file)
    file.close()
    return features, outputs

def build_data_arrays(file):
    (features, outputs) = build_data_lists(file)
    # Arrays are equivalent to matrices and vectors in Python
    return np.array(features, dtype=float), np.array(outputs, dtype=float)

def build_data_lists(file):
    features = []
    outputs = []
    for line in file:
        (line_features, line_output) = extract_features_and_output(line)
        features.append(line_features)
        outputs.append(line_output)
    return features, outputs

def extract_features_and_output(line):
    features_and_output = line.split()
    return features_and_output[0:-1], features_and_output[-1]

def calculate_weight_vector(X, y, scalar):
    # Formula for weight vector from slides
    return inv(X.T.dot(X) + scalar*np.identity(X.shape[1])).dot(X.T).dot(y)

def calculate_sse(X, y, w):
    # Formula for grad(E(w))) (i.e. SSE) from slides
    return (y-X.dot(w)).T.dot(y-X.dot(w))

def test(features, outputs, weight):
    print("Testing SSE: ", calculate_sse(features, outputs, weight))

def insert_dummy_variable(features):
    return np.hstack((np.ones((1, len(features)), dtype=float).T, features))

def insert_random_features(features, number_of_features):
    for i in range(number_of_features):
        features = insert_random_feature(features, random.uniform(0, 100))
    return features

def insert_random_feature(features, max_value):
    random_array = np.random.rand(1, int(len(features))).dot(max_value)
    return np.hstack((random_array.T, features))

def print_(dummy_variable=False, random_features=0, scalar=0):
    if scalar is not 0:
        print("\n------------------------\
               \nUsing ", scalar, " as a scalar\
               \n------------------------")
    elif dummy_variable is True:
        print("\n-------------------\
               \nWith Dummy Variable\
               \n-------------------")
    elif random_features is not 0:
        print("\n---------------------------\
               \nWith ", random_features, " random features\
               \n---------------------------")
    else:
        print("\n------------------\
               \nWithout Dummy Variable\
               \n------------------")

main()

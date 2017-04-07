################################################
# CS434 Machine Learning and Data Mining
# Assignment 1
# Nathan Brahmstadt and Jordan Crane
################################################
import numpy as np
from numpy.linalg import inv

####### Main ########
def main():
    # Don't print in scientific notation :)
    np.set_printoptions(suppress=True)

    ### Parts 1-3 ###
    print "\n########## WITH DUMMY VARIABLES ##########"
    weight = train_with_dummy_variables()
    print "Weight Vector:"
    print weight
    test_with_dummy_variables(weight)

    ### Part 4 ###
    print "\n########## WITHOUT DUMMY VARIABLES ##########"
    weight_no_dummy_variables = \
        train_without_dummy_variables()
    print "Weight Vector:"
    print weight_no_dummy_variables
    test_without_dummy_variables(weight_no_dummy_variables)

####### Functions #######
def train_with_dummy_variables():
    (features, outputs) = get_data_arrays("data/housing_train.txt")
    features = add_dummy_variables(features)
    return train(features, outputs)

def train_without_dummy_variables():
    (features, outputs) = get_data_arrays("data/housing_train.txt")
    return train(features, outputs)

def get_data_arrays(filename):
    file = open(filename, 'r')
    (features, outputs) = build_data_arrays(file)
    file.close()
    return features, outputs

def train(features, outputs):
    weight = calculate_weight_vector(features, outputs)
    print "Training SSE: ", calculate_sse(features, outputs, weight)
    return weight

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

def calculate_weight_vector(X, y):
    # Formula for weight vector from slides
    return inv(X.T.dot(X)).dot(X.T).dot(y)

def test_with_dummy_variables(weight):
    (features, outputs) = get_data_arrays("data/housing_test.txt")
    features = add_dummy_variables(features)
    print "Testing SSE: ", calculate_sse(features, outputs, weight)

def test_without_dummy_variables(weight):
    (features, outputs) = get_data_arrays("data/housing_test.txt")
    print "Testing SSE: ", calculate_sse(features, outputs, weight)

def add_dummy_variables(features):
    return np.hstack((np.ones((1, len(features)), dtype=float).T, features))

def calculate_sse(X, y, w):
    # Formula for grad(E(w))) (i.e. SSE) from slides
    return (y-X.dot(w)).T.dot(y-X.dot(w))

main()

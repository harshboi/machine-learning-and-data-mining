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
    weight_vector = train_model()

    print "Weight Vector:"
    print weight_vector

    test_model(weight_vector)

####### Functions #######
def train_model():
    training_file = open("data/housing_train.txt", 'r')
    (features, outputs) = build_data_arrays(training_file)
    weight = compute_weight_vector(features, outputs)
    print "Training SSE: ", calculate_sse(features, outputs, weight)
    return weight

def test_model(weight):
    testing_file = open("data/housing_test.txt", "r")
    (features, outputs) = build_data_arrays(testing_file)
    print "Testing SSE: ", calculate_sse(features, outputs, weight)

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
    # Instert dummy variable at the front of the array to calculate constant
    features_and_output.insert(0, 1)
    return features_and_output[0:-1], features_and_output[-1]

def compute_weight_vector(X, y):
    # Formula for weight vector
	w = inv(X.T.dot(X)).dot(X.T).dot(y)
	return w

def calculate_sse(X, y, w):
    sse = 0
    for i, y_value in enumerate(y):
        sse += (y_value - X[i].dot(w))**2
    return sse

main()

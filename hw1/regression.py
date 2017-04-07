################################################
# CS434 Machine Learning and Data Mining
# Assignment 1
# Nathan Brahmstadt and Jordan Crane
################################################
import numpy as np
from numpy.linalg import inv

####### Part1 ########
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
    
    #Instert the dummy variable at the beginning of the array so that we can get a weight for the constant later
    features_and_outputs.insert(0,1)
    
    return features_and_output[0:-1], features_and_output[-1]


def compute_weight_vector(X, y):
    # Formula for weight vector
	w = inv(X.T.dot(X)).dot(X.T).dot(y)
	return w

    
def train_model():
    training_file = open("data/housing_train.txt", 'r')
    (features, outputs) = build_data_arrays(training_file)
    weight_vector = compute_weight_vector(features, outputs)
    #TODO calculate SSE
    return weight_vector


def test_model(weight_vector):
    testing_file = open("data/housing_test.txt", "r")
    (features, outputs) = build_data_arrays(testing_file)
    #TODO run data through model
    #TODO calculate SSE

#Don't print in scientific notation :)
np.set_printoptions(suppress=True)
weight_vector = train_model()

print "Weight Vector:"
print weight_vector

test_model(weight_vector)


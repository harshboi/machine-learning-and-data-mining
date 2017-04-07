################################################
# CS434 Machine Learning and Data Mining
# Assignment 1
# Nathan Brahmstadt and Jordan Crane
################################################
import numpy as np
from numpy.linalg import inv

####### Part1 ########
def build_training_data_arrays(training_file):
    (features, outputs) = build_training_data_lists(training_file)
    # Arrays are equivalent to matrices and vectors in Python
    return np.array(features, dtype=float), np.array(outputs, dtype=float)

def build_training_data_lists(training_file):
    features = []
    outputs = []
    for line in training_file:
        (line_features, line_output) = extract_features_and_output(line)
        features.append(line_features)
        outputs.append(line_output)
    return features, outputs

def extract_features_and_output(line):
    split_line = line.split()
    #Instert the dummy variable at the beginning of the array so that we can get a weight for the constant later
    split_line.insert(0, 1)
    
    return split_line[0:-1], split_line[-1]

def compute_weight_vector(X, y):
    # Formula for weight vector
	w = inv(X.T.dot(X)).dot(X.T).dot(y)
	return w

    
def train_model():
    training_file = open("housing_train.txt", 'r')
    (features, outputs) = build_training_data_arrays(training_file)
    weight_vector = compute_weight_vector(features, outputs)
    return weight_vector

#Don't print in scientific notation :)
np.set_printoptions(suppress=True)

weight_vector = train_model()

print "Weight Vector:"
print weight_vector

testing_file = open("housing_test.txt", "r")


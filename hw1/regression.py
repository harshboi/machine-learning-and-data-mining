################################################
# CS434 Machine Learning and Data Mining
# Assignment 1
# Nathan Brahmstadt and Jordan Crane
################################################
import numpy as np
import random
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
    
    return features_and_output[0:-1], features_and_output[-1]


def compute_weight_vector(X, y):
    # Formula for weight vector
	w = inv(X.T.dot(X)).dot(X.T).dot(y)
	return w

def get_training_data():
    training_file = open("data/housing_train.txt", 'r')
    return build_data_arrays(training_file)
    
def get_testing_data():
    testing_file = open("data/housing_test.txt", "r")
    return build_data_arrays(testing_file)
    
def train_model(features, outputs):
    weight_vector = compute_weight_vector(features, outputs)
    print("\nWeight Vector:")
    print weight_vector
    print "\nTraining Model SSE:"
    print calc_sse(features, weight_vector, outputs)
    return weight_vector

def calc_sse(x, w, y):
    total_sse = 0
    for i, y_val in enumerate(y):
        running_sum = 0
        for j, w_val in enumerate(w):
         
            running_sum += w_val*x[i][j]

        total_sse += (y_val - running_sum)**2
  
    return total_sse
    
def test_model(features, outputs, weight_vector):
    print "\nTest SSE:"
    print calc_sse(features, weight_vector, outputs)
    
def insert_feature_data(data_array, value):
    data_array = data_array.T
    row, col = data_array.shape
    
    data_array = np.resize(data_array, (row+1, col))
  
    for i, val in enumerate(data_array[row]):
        data_array[row][i] = value
    return data_array.T
   
    
def insert_random_feature_data(data_array, max_random_value):
    data_array = data_array.T
    row, col = data_array.shape
    
    data_array = np.resize(data_array, (row+1, col))

    for i, val in enumerate(data_array[row]):
        data_array[row][i] = random.uniform(0, max_random_value)
    return data_array.T
    
def train_and_test(use_dummy_var_flag, random_features_to_add):
    features, outputs = get_training_data()
    if(use_dummy_var_flag):
        features = insert_feature_data(features, 1)
    weight_vector = train_model(features, outputs)
    features, outputs = get_testing_data()
    if(use_dummy_var_flag):
        features = insert_feature_data(features, 1)
    test_model(features, outputs, weight_vector)

#Don't print in scientific notation :)
np.set_printoptions(suppress=True)
random.seed()

print("\n--------\nNo Dummy Variable\n--------")
train_and_test(0, 0)

print("\n--------\nWith Dummy Variable\n--------")
train_and_test(1, 0)

    


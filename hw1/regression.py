################################################
# CS434 Machine Learning and Data Mining
# Assignment 1
# Nathan Brahmstadt and Jordan Crane
################################################
import numpy as np
import random
from numpy.linalg import inv

####### Main ########
def main():
    # Don't print in scientific notation :)
    np.set_printoptions(suppress=True)
    #Setup random number generator
    random.seed()
    
    #No dummy variable means no constant in the linear regression
    print "\n--------\nNo Dummy Variable\n--------"
    train_and_test(0, 0)

    print "\n--------\nWith Dummy Variable\n--------"
    train_and_test(1, 0)

    #For problem 5, add 2, 4, 6, 8, and 10 features with random values 
    for i in range(5):
        features_to_add = 2+(2*i)
        print "\n--------\nAdding " + str(features_to_add) + " random features\n--------"
        train_and_test(1, features_to_add)
        
####### Functions #######

#Used to train and test, can decide to add a constant or random features(Needed for problem 5)
def train_and_test(use_dummy_var_flag, random_features_to_add):
    #Get the raw text data as a matrix
    features, outputs = get_training_data()
    
    #Adds a 1 to the begining of all samples
    if(use_dummy_var_flag):
        features = insert_feature_data(features, 1)
    #Adds the specified number of features to each sample
    if(random_features_to_add > 0):
        for i in range(random_features_to_add):
            #Max randomized value possible is different for each feature, as specified in assignment
            features = insert_random_feature_data(features, random.uniform(0, 100))
    weight_vector = train_model(features, outputs)
    features, outputs = get_testing_data()
    
    if(use_dummy_var_flag):
        features = insert_feature_data(features, 1)
    if(random_features_to_add > 0):
        for i in range(random_features_to_add):
            features = insert_random_feature_data(features, random.uniform(0, 100))
            
    test_model(features, outputs, weight_vector)
    
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

def get_training_data():
    training_file = open("data/housing_train.txt", 'r')
    return build_data_arrays(training_file)
    
def get_testing_data():
    testing_file = open("data/housing_test.txt", "r")
    return build_data_arrays(testing_file)
    
def train_model(features, outputs):
    weight = calculate_weight_vector(features, outputs)
    print "Weight Vector: ", weight
    print "Training SSE: ", calculate_sse(features, outputs, weight)
    return weight
    
def calculate_weight_vector(X, y):
    # Formula for weight vector from slides
    return inv(X.T.dot(X)).dot(X.T).dot(y)
    
def calculate_sse(X, y, w):
    # Formula for grad(E(w))) (i.e. SSE) from slides
    return (y-X.dot(w)).T.dot(y-X.dot(w))
    
def test_model(features, outputs, weight):
    print "Testing SSE: ", calculate_sse(features, outputs, weight)
    
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
   

main()

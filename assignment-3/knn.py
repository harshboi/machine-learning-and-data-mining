#!/usr/bin/env python2
################################################
# CS434 Machine Learning and Data Mining       #
# Assignment 3: K-nearest Neighbors            #
# Nathan Brahmstadt and Jordan Crane           #
################################################

import math
import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Data:
    def __init__(self, filename):
        file = open(filename)
        (self.features, self.outputs) = self.build_data_arrays(file)
        file.close()

    def build_data_arrays(self, file):
        (features, outputs) = self.build_data_lists(file)
        return (np.array(features, dtype=float), np.array(outputs, dtype=int))

    def build_data_lists(self, file):
        (features, outputs) = ([], [])
        for line in file:
            (line_features, line_output) = self.extract_features_and_output(line)
            features.append(line_features)
            outputs.append(line_output)
        return (features, outputs)

    def extract_features_and_output(self, line):
        features_and_output = line.split(',')
        return (features_and_output[1:], features_and_output[0])

    def normalize_features(self):
        features_max = np.amax(self.features, axis=0)
        features_min = np.amin(self.features, axis=0)
        features_spread = features_max - features_min
        self.features = (self.features - features_min) / features_spread

training_data = Data('data/knn_train.csv')
testing_data = Data('data/knn_test.csv')
testing_data.normalize_features()
print np.amax(testing_data.features, axis=0)

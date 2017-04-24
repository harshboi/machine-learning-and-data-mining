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
from operator import attrgetter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

DistOutputPair = namedtuple('DistOutputPair', 'distance output')

class Data:
    def __init__(self, filename):
        file = open(filename)
        (self.features, self.outputs) = self._build_data_arrays(file)
        file.close()

    def _build_data_arrays(self, file):
        (features, outputs) = self._build_data_lists(file)
        return (np.array(features, dtype=float), np.array(outputs, dtype=int))

    def _build_data_lists(self, file):
        (features, outputs) = ([], [])
        for line in file:
            (line_features, line_output) = self._extract_features_and_output(line)
            features.append(line_features)
            outputs.append(line_output)
        return (features, outputs)

    def _extract_features_and_output(self, line):
        features_and_output = line.split(',')
        return (features_and_output[1:], features_and_output[0])

    def normalize_features(self):
        features_max = np.amax(self.features, axis=0)
        features_min = np.amin(self.features, axis=0)
        features_spread = features_max - features_min
        self.features = (self.features - features_min) / features_spread

class Knn:
    def __init__(self, training_data):
        self.training_data = training_data

    def test(self, data, k=3):
        outputs = map(lambda instance: self._classify(instance, k), data.features)
        print outputs

    def _classify(self, test_instance, k):
        neighbors = self._get_nearest_neighbors(test_instance)
        result = sum([tuple.output for tuple in neighbors[:k]])
        if result < 0: return -1
        else: return 1

    def _get_nearest_neighbors(self, test_instance):
        distances = np.sqrt(np.sum(
                (test_instance - self.training_data.features)**2,
                axis=1
                ))

        neigbors = map(DistOutputPair, distances,
            self.training_data.outputs)

        return sorted(neigbors, key=attrgetter('distance'))

training_data = Data('data/knn_train.csv')
testing_data = Data('data/knn_test.csv')
training_data.normalize_features()
testing_data.normalize_features()

model = Knn(training_data)
model.test(testing_data)

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

DistanceOutputPair = namedtuple('DistanceOutputPair', 'distance output')
FeatureOutputPair = namedtuple('FeatureOutputPair', 'features output')

class Data:
    def __init__(self, filename=None, features=None, outputs=None):
        if filename:
            file = open(filename)
            (self.features, self.outputs) = self._build_data_arrays(file)
            file.close()
        elif features and outputs:
            self.features = features
            self.outputs = outputs
        self._features = self.features
        self._outputs = self.outputs

    def normalize(self):
        features_max = np.amax(self.features, axis=0)
        features_min = np.amin(self.features, axis=0)
        features_spread = features_max - features_min
        self.features = (self.features - features_min) / features_spread
        self._features = self.features

    def leave_out(self, index):
        self.features = np.delete(self._features, index, axis=0)
        self.outputs = np.delete(self._outputs, index, axis=0)
        return FeatureOutputPair(self._features[index], self._outputs[index])

    def include_all(self):
        self.features = self._features
        self.outputs = self._outputs

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

class Knn:
    def __init__(self, training_data):
        self.training_data = training_data

    def get_testing_error(self, data, k=1):
        outputs = self._test(data, k)
        return (sum(abs(outputs - data.outputs)) / float(2 * len(data.outputs)))

    def get_training_error(self, k=1):
        outputs = self._test(self.training_data, k)
        return (sum(abs(outputs - self.training_data.outputs)) /
                float(2 * len(self.training_data.outputs)))

    def get_cross_validation_error(self, k=1):
        total_error = 0
        for i in range(len(self.training_data.features)):
            left_out = self.training_data.leave_out(i)
            result = self._classify(left_out.features, k)
            total_error += abs(left_out.output - result)/2
        self.training_data.include_all()
        return float(total_error) / len(self.training_data.outputs)

    def _test(self, data, k):
        return map(lambda instance: self._classify(instance, k), data.features)

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

        neigbors = map(DistanceOutputPair, distances,
            self.training_data.outputs)

        return sorted(neigbors, key=attrgetter('distance'))

class Main:
    def __init__(self):
        self.training_data = Data('data/knn_train.csv')
        self.testing_data = Data('data/knn_test.csv')
        self.training_data.normalize()
        self.testing_data.normalize()

    def run(self):
        self._part_1()
        self._part_2()
        self._extra_credit()

    def _part_1(self):
        model = Knn(self.training_data)
        for k in range(1, 52, 2):
            print "---------------------------"
            print "K-value: ", k
            print "".join(["Training Error: ",
                str(round(model.get_training_error(k=k), 4) * 100),
                "%"]
                )
            print "".join(["Leave-One-Out Error: ",
                str(round(model.get_cross_validation_error(k=k), 4) * 100),
                "%"]
                )
            print "".join([
                "Testing Error: ",
                str(round(
                    model.get_testing_error(self.testing_data, k=k), 4) * 100),
                "%"]
                )

    def _part_2(self):
        return

    def _extra_credit(self):
        return

np.set_printoptions(suppress=True)
main = Main()
main.run()

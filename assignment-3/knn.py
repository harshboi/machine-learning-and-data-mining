#!/usr/bin/env python2
################################################
# CS434 Machine Learning and Data Mining       #
# Assignment 3: K-nearest Neighbors            #
# Nathan Brahmstadt and Jordan Crane           #
################################################

import math
import numpy as np
from collections import namedtuple
from operator import attrgetter

DistanceOutputPair = namedtuple('DistanceOutputPair', 'distance output')
FeatureOutputPair = namedtuple('FeatureOutputPair', 'features output')

class CsvPrinter:
    def __init__(self, filename, labels=[], delimiter=','):
        if labels:
            self.file = open(filename, 'w')
            self.columns = len(labels)
            labels.append('\n')
            self.delimiter = delimiter
            self.file.write(self.delimiter.join(labels))
        else:
            print "CsvPrinter: Please provide column labels"
            exit()

    def writerow(self, data):
        if len(data) == self.columns:
            data.append('\n')
            self.file.write(self.delimiter.join([str(x) for x in data]))
        else:
            print "CsvPrinter: Data length should match number of labels"
            exit()

    def close(self):
        self.file.close()

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
            output = self._classify(left_out.features, k)
            total_error += abs(left_out.output - output)/2
        self.training_data.include_all()
        return float(total_error) / len(self.training_data.outputs)

    def _test(self, data, k):
        return map(lambda instance: self._classify(instance, k), data.features)

    def _classify(self, test_instance, k):
        neighbors = self._get_nearest_neighbors(test_instance)
        output = sum([neigbor.output for neigbor in neighbors[:k]])
        if output < 0: return -1
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
        labels=['k', 'Training Error', 'Testing Error', 'Leave-one-out Error']
        csv = CsvPrinter('reports/part_1.csv', labels)
        model = Knn(self.training_data)
        for k in range(1, 52, 2):
            training_err = model.get_training_error(k=k)
            testing_err = model.get_testing_error(self.testing_data, k=k)
            cv_err = model.get_cross_validation_error(k=k)
            print "---------------------------"
            print "K-value: ", k
            print "".join(["Training Error: ",
                str(round(training_err, 4) * 100), "%"]
                )
            print "".join(["Leave-One-Out Error: ",
                str(round(cv_err, 4) * 100), "%"]
                )
            print "".join(["Testing Error: ",
                str(round(testing_err, 4) * 100), "%"]
                )
            csv.writerow([k, training_err, testing_err, cv_err])
        print "---------------------------"
        csv.close()

    def _part_2(self):
        return

    def _extra_credit(self):
        return

main = Main()
main.run()
#!/usr/bin/env python2
################################################
# CS434 Machine Learning and Data Mining       #
# Assignment 3: K-nearest Neighbors            #
# Nathan Brahmstadt and Jordan Crane           #
################################################

import math
from math import log
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

    # Specifically for leave-one-out cross-validation
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

class Stump:    
    
    def __init__(self, training_data):
        self.training_data = training_data
        self.trained_information_gain = None
        self.trained_feature = None
        self.trained_split = None
    
    def train_stump(self):
        data_points, feature_number = self.training_data.features.shape
        self.trained_information_gain = 0
        self.trained_feature = 0
        self.trained_split = 0
        #increment through each feature
        for i in range(0,feature_number):
   
            #test information gain using data point as a boundary
            for j in self.training_data.features.T[i]:
                boundary_value = j
                cur_information_gain = self._information_gain(i, j)
                if self.trained_information_gain < cur_information_gain:
                    self.trained_information_gain = cur_information_gain
                    self.trained_feature= i
                    self.trained_split = j
                

      
        
    def _information_gain(self, feature, boundary_value):
    
        initial_uncertainty = self._initial_uncertainty()
    
        data_points, feature_number = self.training_data.features.shape
        greater_than_split_pos = 0
        greater_than_split_neg = 0
        less_than_split_pos = 0
        less_than_split_neg = 0
        
       
        for j, value in enumerate(self.training_data.features.T[feature]):
            if value > boundary_value:
                if self.training_data.outputs[j] > 0:
                    greater_than_split_pos += 1
                else:
                    greater_than_split_neg += 1
            else:
                if self.training_data.outputs[j] > 0:
                    less_than_split_pos += 1
                else:
                    less_than_split_neg += 1
        greater_entropy = self._entropy(greater_than_split_pos, greater_than_split_neg)
        greater_p = ((float)(greater_than_split_pos+greater_than_split_neg)/data_points)*greater_entropy
        
        less_entropy = self._entropy(less_than_split_pos, less_than_split_neg)
        less_p = ((float)(less_than_split_pos+less_than_split_neg)/data_points)*less_entropy
        
        gain = initial_uncertainty - greater_p - less_p
    
        return gain
        
    def _entropy(self, pos, neg):
        sum = pos+neg
        if pos == 0 or neg == 0:
            return 0
        else:
            return -(((float)(pos)/sum)*log(((float)(pos)/sum),2)) - (((float)(neg)/sum)*log(((float)(neg)/sum),2))
     
    def _initial_uncertainty(self):
        pos = 0
        neg = 0
        for i in self.training_data.outputs:
            if i > 0:
                pos += 1
            else:
                neg += 1
        return self._entropy(pos, neg)
     
    def accuracy(self, data):
        prediction = 0
        correct = 0
        wrong = 0
        for i, x in enumerate(data.features):
            if x[self.trained_feature] > self.trained_split:
                prediction = 1
            else:
                prediction = -1
              
            if prediction == data.outputs[i]:
                correct += 1
            else:
                wrong += 1
                
        return float(correct)/(correct+wrong)
            
                
     
training_data = Data('data/knn_train.csv')
testing_data = Data('data/knn_test.csv')
training_data.normalize()
testing_data.normalize()

def main():
    #part_1()
    part_2()
    extra_credit()

def part_1():
    labels=['k', 'Training Error', 'Testing Error', 'Leave-one-out Error']
    csv = CsvPrinter('reports/part_1.csv', labels)
    model = Knn(training_data)

    def _print_results():
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

    for k in range(1, 52, 2):
        training_err = model.get_training_error(k=k)
        testing_err = model.get_testing_error(testing_data, k=k)
        cv_err = model.get_cross_validation_error(k=k)
        _print_results()
        csv.writerow([k, training_err, testing_err, cv_err])
    print "---------------------------"
    csv.close()

def part_2():
    model = Stump(training_data)
    model.train_stump()
    print "----------Part 2-----------"
    print "Split if Value is > " + str(model.trained_split) + " On feature " + str(model.trained_feature)
    print "Information Gain of " + str(model.trained_information_gain)
    
    print "Training Accuracy: " + str(model.accuracy(training_data))
    print "Testing Accuracy: " + str(model.accuracy(testing_data))
    print "---------------------------"
    return

def extra_credit():
    return

if __name__ == '__main__':main()
